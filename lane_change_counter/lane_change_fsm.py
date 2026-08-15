"""Deterministic, timestamp-based lane-change event state machine."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from itertools import pairwise

from lane_change_counter.config import PipelineConfig
from lane_change_counter.models import Lane, LaneAssignment, LaneChangeEvent, SceneGeometry

_EPSILON = 1e-9


class _State(Enum):
    UNSTABLE = auto()
    STABLE_ORIGIN = auto()
    CANDIDATE_TRANSITION = auto()
    AWAITING_TARGET_CONFIRMATION = auto()
    COUNTED = auto()
    STABLE_NEW_ORIGIN = auto()


@dataclass(slots=True)
class _TrackContext:
    state: _State = _State.UNSTABLE
    track_start_time: float | None = None
    origin_lane_id: str | None = None
    origin_start_time: float | None = None
    candidate_target_lane_id: str | None = None
    candidate_start_time: float | None = None
    candidate_start_frame: int | None = None
    maximum_lateral_displacement: float = 0.0
    origin_reference_lateral: float = 0.0
    target_observations: int = 0
    valid_confirmation_observations: int = 0
    last_candidate_target_frame: int | None = None
    last_candidate_target_time: float | None = None
    last_candidate_target_at_frame_edge: bool = False
    last_timestamp: float | None = None


def detect_lane_changes(
    assignments: Sequence[LaneAssignment],
    scene: SceneGeometry,
    config: PipelineConfig,
) -> tuple[LaneChangeEvent, ...]:
    """Return confirmed adjacent-lane transitions for every independent track.

    Records from different tracks may be interleaved in any order. Records for
    one track must have nondecreasing timestamps; silently sorting them could
    manufacture a plausible transition from invalid temporal evidence.
    """
    assignments_by_track: dict[int, list[LaneAssignment]] = defaultdict(list)
    for assignment in assignments:
        assignments_by_track[assignment.track_id].append(assignment)

    lanes = {lane.lane_id: lane for lane in scene.lanes}
    lane_centers = {lane.lane_id: float(index) for index, lane in enumerate(scene.lanes)}
    events: list[LaneChangeEvent] = []
    for track in assignments_by_track.values():
        _require_sorted(track)
        events.extend(_detect_track(track, lanes, lane_centers, config))

    return tuple(
        sorted(
            events,
            key=lambda event: (
                event.confirm_time_s,
                event.track_id,
                event.confirm_frame,
                event.origin_lane_id,
                event.target_lane_id,
            ),
        )
    )


def _detect_track(
    track: Sequence[LaneAssignment],
    lanes: dict[str, Lane],
    lane_centers: dict[str, float],
    config: PipelineConfig,
) -> list[LaneChangeEvent]:
    context = _TrackContext()
    events: list[LaneChangeEvent] = []
    for assignment in track:
        if _gap_exceeded(context, assignment, config):
            _restart_segment(context, assignment.timestamp_s)

        event = _advance(context, assignment, lanes, lane_centers, config)
        if event is not None:
            events.append(event)
        context.last_timestamp = assignment.timestamp_s
    return events


def _advance(
    context: _TrackContext,
    assignment: LaneAssignment,
    lanes: dict[str, Lane],
    lane_centers: dict[str, float],
    config: PipelineConfig,
) -> LaneChangeEvent | None:
    if context.track_start_time is None:
        context.track_start_time = assignment.timestamp_s

    if context.state is _State.UNSTABLE:
        _advance_unstable(context, assignment, lanes, config)
        return None
    if context.state in (_State.STABLE_ORIGIN, _State.STABLE_NEW_ORIGIN):
        return _advance_stable_origin(context, assignment, lanes, lane_centers, config)
    if context.state in (_State.CANDIDATE_TRANSITION, _State.AWAITING_TARGET_CONFIRMATION):
        return _advance_candidate(context, assignment, lanes, lane_centers, config)
    if context.state is _State.COUNTED:
        _advance_counted(context, assignment, config)
        return None
    raise AssertionError(f"unhandled lane-change state: {context.state}")


def _advance_unstable(
    context: _TrackContext,
    assignment: LaneAssignment,
    lanes: dict[str, Lane],
    config: PipelineConfig,
) -> None:
    if assignment.lane_id is None or assignment.lane_id not in lanes:
        _clear_origin(context)
        return

    if context.origin_lane_id != assignment.lane_id or context.origin_start_time is None:
        context.origin_lane_id = assignment.lane_id
        context.origin_start_time = assignment.timestamp_s
    context.origin_reference_lateral = assignment.lateral_lane_widths
    if assignment.timestamp_s - context.origin_start_time + _EPSILON >= config.origin_stable_s:
        context.state = _State.STABLE_ORIGIN


def _advance_stable_origin(
    context: _TrackContext,
    assignment: LaneAssignment,
    lanes: dict[str, Lane],
    lane_centers: dict[str, float],
    config: PipelineConfig,
) -> LaneChangeEvent | None:
    origin_lane_id = context.origin_lane_id
    if origin_lane_id is None:
        _cancel_and_seed(context, assignment, lanes, config)
        return None
    if assignment.lane_id == origin_lane_id:
        return None
    if assignment.lane_id is None:
        _cancel(context)
        return None

    target = lanes.get(assignment.lane_id)
    origin = lanes.get(origin_lane_id)
    if origin is None or target is None or not _eligible_transition(origin, target, config):
        _cancel_and_seed(context, assignment, lanes, config)
        return None

    context.state = _State.CANDIDATE_TRANSITION
    context.candidate_target_lane_id = target.lane_id
    context.candidate_start_time = assignment.timestamp_s
    context.candidate_start_frame = assignment.frame_index
    context.maximum_lateral_displacement = _crossing_displacement(
        context.origin_reference_lateral,
        assignment.lateral_lane_widths,
        lane_centers[origin_lane_id],
        lane_centers[target.lane_id],
    )
    context.target_observations = 1
    context.valid_confirmation_observations = 1
    _remember_candidate_target(context, assignment)
    if (
        context.maximum_lateral_displacement + _EPSILON
        >= config.min_lateral_lane_widths
    ):
        context.state = _State.AWAITING_TARGET_CONFIRMATION
    return _maybe_confirm(context, assignment, config)


def _advance_candidate(
    context: _TrackContext,
    assignment: LaneAssignment,
    lanes: dict[str, Lane],
    lane_centers: dict[str, float],
    config: PipelineConfig,
) -> LaneChangeEvent | None:
    origin_lane_id = context.origin_lane_id
    candidate_lane_id = context.candidate_target_lane_id
    if origin_lane_id is None or candidate_lane_id is None:
        _cancel_and_seed(context, assignment, lanes, config)
        return None
    if assignment.lane_id is None:
        edge_exit_event = _maybe_confirm_edge_exit(context, assignment, config)
        if edge_exit_event is not None:
            return edge_exit_event
        _cancel(context)
        return None
    if assignment.lane_id == origin_lane_id:
        _cancel_to_stable_origin(context)
        return None

    origin = lanes.get(origin_lane_id)
    observed_lane = lanes.get(assignment.lane_id)
    if origin is None or observed_lane is None or not _eligible_transition(origin, observed_lane, config):
        _cancel_and_seed(context, assignment, lanes, config)
        return None

    context.valid_confirmation_observations += 1
    if assignment.lane_id == candidate_lane_id:
        context.target_observations += 1
        _remember_candidate_target(context, assignment)
        context.maximum_lateral_displacement = max(
            context.maximum_lateral_displacement,
            _crossing_displacement(
                context.origin_reference_lateral,
                assignment.lateral_lane_widths,
                lane_centers[origin_lane_id],
                lane_centers[candidate_lane_id],
            ),
        )
        if (
            context.maximum_lateral_displacement + _EPSILON
            >= config.min_lateral_lane_widths
        ):
            context.state = _State.AWAITING_TARGET_CONFIRMATION
    return _maybe_confirm(context, assignment, config)


def _remember_candidate_target(context: _TrackContext, assignment: LaneAssignment) -> None:
    context.last_candidate_target_frame = assignment.frame_index
    context.last_candidate_target_time = assignment.timestamp_s
    context.last_candidate_target_at_frame_edge = assignment.at_frame_edge


def _advance_counted(
    context: _TrackContext,
    assignment: LaneAssignment,
    config: PipelineConfig,
) -> None:
    if assignment.lane_id != context.origin_lane_id:
        context.origin_start_time = None
        return

    if context.origin_start_time is None:
        context.origin_start_time = assignment.timestamp_s
    context.origin_reference_lateral = assignment.lateral_lane_widths
    if (
        context.origin_start_time is not None
        and assignment.timestamp_s - context.origin_start_time + _EPSILON
        >= config.origin_stable_s
    ):
        context.state = _State.STABLE_NEW_ORIGIN


def _maybe_confirm(
    context: _TrackContext,
    assignment: LaneAssignment,
    config: PipelineConfig,
) -> LaneChangeEvent | None:
    if context.state is not _State.AWAITING_TARGET_CONFIRMATION:
        return None
    if assignment.lane_id != context.candidate_target_lane_id:
        return None
    if context.candidate_start_time is None or context.track_start_time is None:
        return None
    target_duration = assignment.timestamp_s - context.candidate_start_time
    track_duration = assignment.timestamp_s - context.track_start_time
    target_ratio = context.target_observations / context.valid_confirmation_observations
    if target_duration + _EPSILON < config.target_stable_s:
        return None
    if track_duration + _EPSILON < config.min_track_s:
        return None
    if target_ratio + _EPSILON < config.target_majority:
        return None

    origin_lane_id = context.origin_lane_id
    target_lane_id = context.candidate_target_lane_id
    start_frame = context.candidate_start_frame
    if origin_lane_id is None or target_lane_id is None or start_frame is None:
        return None
    event = LaneChangeEvent(
        track_id=assignment.track_id,
        origin_lane_id=origin_lane_id,
        target_lane_id=target_lane_id,
        start_frame=start_frame,
        confirm_frame=assignment.frame_index,
        confirm_time_s=assignment.timestamp_s,
    )
    context.state = _State.COUNTED
    context.origin_lane_id = target_lane_id
    context.origin_start_time = assignment.timestamp_s
    context.origin_reference_lateral = assignment.lateral_lane_widths
    _clear_candidate(context)
    return event


def _maybe_confirm_edge_exit(
    context: _TrackContext,
    exit_assignment: LaneAssignment,
    config: PipelineConfig,
) -> LaneChangeEvent | None:
    """Confirm a sufficiently evidenced target crossing that immediately leaves view.

    This narrow alternative applies only after the normal candidate has crossed
    the lateral threshold, its last target contact reaches the frame edge, and
    the next sample is unassigned. Ordinary short target visits still require
    the normal target-stability duration.
    """
    if context.state is not _State.AWAITING_TARGET_CONFIRMATION:
        return None
    if not context.last_candidate_target_at_frame_edge:
        return None
    if context.target_observations < config.edge_exit_min_target_observations:
        return None
    if (
        context.candidate_start_time is None
        or context.track_start_time is None
        or context.last_candidate_target_time is None
        or context.last_candidate_target_frame is None
    ):
        return None
    target_duration = context.last_candidate_target_time - context.candidate_start_time
    track_duration = context.last_candidate_target_time - context.track_start_time
    if target_duration + _EPSILON < config.edge_exit_target_stable_s:
        return None
    if track_duration + _EPSILON < config.min_track_s:
        return None
    origin_lane_id = context.origin_lane_id
    target_lane_id = context.candidate_target_lane_id
    start_frame = context.candidate_start_frame
    if origin_lane_id is None or target_lane_id is None or start_frame is None:
        return None
    event = LaneChangeEvent(
        track_id=exit_assignment.track_id,
        origin_lane_id=origin_lane_id,
        target_lane_id=target_lane_id,
        start_frame=start_frame,
        confirm_frame=context.last_candidate_target_frame,
        confirm_time_s=context.last_candidate_target_time,
        confirmation="edge_exit",
    )
    context.state = _State.COUNTED
    context.origin_lane_id = target_lane_id
    context.origin_start_time = context.last_candidate_target_time
    _clear_candidate(context)
    return event


def _eligible_transition(origin: Lane, target: Lane, config: PipelineConfig) -> bool:
    return (
        target.lane_id in origin.neighbor_ids
        and _direction_difference(origin.direction_deg, target.direction_deg)
        <= config.lane_change_direction_tolerance_deg + _EPSILON
    )


def _direction_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _crossing_displacement(
    origin_lateral: float,
    target_lateral: float,
    origin_center: float,
    target_center: float,
) -> float:
    direction = target_center - origin_center
    if direction == 0.0:
        return 0.0
    boundary = (origin_center + target_center) / 2.0
    if (origin_lateral - boundary) * direction > _EPSILON:
        return 0.0
    if (target_lateral - boundary) * direction < -_EPSILON:
        return 0.0
    return abs(target_lateral - origin_lateral)


def _gap_exceeded(
    context: _TrackContext,
    assignment: LaneAssignment,
    config: PipelineConfig,
) -> bool:
    return (
        context.last_timestamp is not None
        and assignment.timestamp_s - context.last_timestamp > config.max_gap_s + _EPSILON
    )


def _cancel_and_seed(
    context: _TrackContext,
    assignment: LaneAssignment,
    lanes: dict[str, Lane],
    config: PipelineConfig,
) -> None:
    _cancel(context)
    _advance_unstable(context, assignment, lanes, config)


def _restart_segment(context: _TrackContext, timestamp_s: float) -> None:
    _cancel(context)
    context.track_start_time = timestamp_s


def _cancel(context: _TrackContext) -> None:
    context.state = _State.UNSTABLE
    _clear_origin(context)
    _clear_candidate(context)


def _clear_origin(context: _TrackContext) -> None:
    context.origin_lane_id = None
    context.origin_start_time = None
    context.origin_reference_lateral = 0.0


def _clear_candidate(context: _TrackContext) -> None:
    context.candidate_target_lane_id = None
    context.candidate_start_time = None
    context.candidate_start_frame = None
    context.maximum_lateral_displacement = 0.0
    context.target_observations = 0
    context.valid_confirmation_observations = 0
    context.last_candidate_target_frame = None
    context.last_candidate_target_time = None
    context.last_candidate_target_at_frame_edge = False


def _cancel_to_stable_origin(context: _TrackContext) -> None:
    context.state = _State.STABLE_ORIGIN
    _clear_candidate(context)


def _require_sorted(track: Sequence[LaneAssignment]) -> None:
    if any(
        current.timestamp_s < previous.timestamp_s
        or current.frame_index < previous.frame_index
        for previous, current in pairwise(track)
    ):
        raise ValueError("assignments for each track must be sorted by timestamp and frame index")
