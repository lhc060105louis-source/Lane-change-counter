"""Lane assignment from vehicle contact points and timestamp-based jitter suppression."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from itertools import pairwise
from math import isfinite

import cv2
import numpy as np

from lane_change_counter.config import PipelineConfig
from lane_change_counter.models import Lane, LaneAssignment, SceneGeometry, TrackObservation

_BOUNDARY_UNCERTAIN_WIDTHS = 0.10
_FRAME_EDGE_MARGIN_PX = 2.0


def assign_observation(scene: SceneGeometry, observation: TrackObservation) -> LaneAssignment:
    """Assign an observation using its bottom-center road contact point.

    Exclusions take precedence over lanes.  Confidence combines detector and
    geometry confidence and falls linearly to zero in the outer tenth of a
    local lane width, so boundary contacts remain visible but uncertain. The
    lateral value uses one scene-wide lane-width coordinate: ordered lane
    centers are integers and shared boundaries meet at half-integers.
    """
    contact = observation.box.bottom_center
    if any(_signed_distance(exclusion, contact) >= 0.0 for exclusion in scene.exclusions):
        return _unassigned(observation)

    for lane_index, lane in enumerate(scene.lanes):
        boundary_distance = _signed_distance(lane.polygon, contact)
        if boundary_distance >= 0.0:
            cross_section = _local_lane_width_and_lateral(lane, contact)
            if cross_section is None:
                return _unassigned(observation)
            width, local_lateral = cross_section
            lateral = _common_lateral_coordinate(lane_index, local_lateral)
            boundary_factor = min(1.0, boundary_distance / (width * _BOUNDARY_UNCERTAIN_WIDTHS))
            confidence = scene.confidence * observation.score * boundary_factor
            return LaneAssignment(
                track_id=observation.track_id,
                frame_index=observation.frame_index,
                timestamp_s=observation.timestamp_s,
                lane_id=lane.lane_id,
                confidence=confidence,
                lateral_lane_widths=lateral,
                at_frame_edge=_at_frame_edge(contact, scene.frame_size),
            )
    return _unassigned(observation)


def smooth_assignments(
    assignments: Sequence[LaneAssignment], fps: float
) -> tuple[LaneAssignment, ...]:
    """Suppress transient lane labels using each track's 0.6-second origin history.

    The input's ordering is retained, but records for an individual track must
    have nondecreasing timestamps.  A non-null label is replaced only after a
    full origin window has elapsed and the duration-weighted majority of that
    preceding window names another lane.  ``None`` marks uncertain geometry and
    is returned unchanged.  Replacement records retain their raw confidence and
    lateral diagnostic values; input records are never modified.
    """
    if not isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be a positive finite number")

    indexes_by_track: dict[int, list[int]] = defaultdict(list)
    for index, assignment in enumerate(assignments):
        indexes_by_track[assignment.track_id].append(index)

    output = list(assignments)
    config = PipelineConfig()
    for indexes in indexes_by_track.values():
        track = [assignments[index] for index in indexes]
        _require_sorted_timestamps(track)
        for position, current in enumerate(track):
            if current.lane_id is None:
                continue
            origin_lane = _origin_majority(track, position, config)
            if origin_lane is not None and origin_lane != current.lane_id:
                output[indexes[position]] = replace(current, lane_id=origin_lane)
    return tuple(output)


def _unassigned(observation: TrackObservation) -> LaneAssignment:
    return LaneAssignment(
        track_id=observation.track_id,
        frame_index=observation.frame_index,
        timestamp_s=observation.timestamp_s,
        lane_id=None,
        confidence=0.0,
        lateral_lane_widths=0.0,
        at_frame_edge=False,
    )


def _at_frame_edge(contact: tuple[float, float], frame_size: tuple[int, int]) -> bool:
    """Return whether the vehicle road-contact point reaches the visible frame edge."""
    width, height = frame_size
    x, y = contact
    return (
        x <= _FRAME_EDGE_MARGIN_PX
        or x >= width - _FRAME_EDGE_MARGIN_PX
        or y <= _FRAME_EDGE_MARGIN_PX
        or y >= height - _FRAME_EDGE_MARGIN_PX
    )


def _signed_distance(
    polygon: tuple[tuple[float, float], ...], point: tuple[float, float]
) -> float:
    contour = np.asarray(polygon, dtype=np.float32)
    return float(cv2.pointPolygonTest(contour, point, True))


def _local_lane_width_and_lateral(
    lane: Lane,
    point: tuple[float, float],
) -> tuple[float, float] | None:
    """Measure the polygon cross-section on the contact point's image row."""
    scan_y = point[1]
    intersections: list[float] = []
    for start, end in pairwise((*lane.polygon, lane.polygon[0])):
        delta_y = end[1] - start[1]
        if abs(delta_y) <= 1e-9:
            if abs(scan_y - start[1]) <= 1e-9:
                intersections.extend((start[0], end[0]))
            continue
        fraction = (scan_y - start[1]) / delta_y
        if -1e-9 <= fraction <= 1.0 + 1e-9:
            intersections.append(start[0] + fraction * (end[0] - start[0]))

    unique_intersections: list[float] = []
    for intersection in sorted(intersections):
        if not unique_intersections or abs(intersection - unique_intersections[-1]) > 1e-7:
            unique_intersections.append(intersection)
    if len(unique_intersections) < 2:
        return None

    left, right = unique_intersections[0], unique_intersections[-1]
    width = right - left
    if width <= 1e-9:
        return None
    midpoint = (left + right) / 2.0
    return width, (point[0] - midpoint) / width


def _common_lateral_coordinate(
    lane_index: int,
    local_lateral: float,
) -> float:
    """Map a lane-local value into one winding-independent scene coordinate.

    Scene lanes are ordered across the road. Their centers occupy integer
    coordinates, and each consistently oriented local coordinate spans roughly
    ``[-0.5, 0.5]``. Adjacent representations of a shared boundary therefore
    meet at the same half-integer coordinate.
    """
    return float(lane_index) + local_lateral


def _require_sorted_timestamps(track: Sequence[LaneAssignment]) -> None:
    if any(current.timestamp_s < previous.timestamp_s for previous, current in pairwise(track)):
        raise ValueError("assignments for each track must be sorted by timestamp")


def _origin_majority(
    track: Sequence[LaneAssignment], position: int, config: PipelineConfig
) -> str | None:
    current_time = track[position].timestamp_s
    window_start = current_time - config.origin_stable_s
    if position == 0 or track[0].timestamp_s > window_start + 1e-9:
        return None

    durations: dict[str, float] = defaultdict(float)
    observed_duration = 0.0
    for index in range(position):
        start = max(track[index].timestamp_s, window_start)
        end = min(
            track[index + 1].timestamp_s,
            track[index].timestamp_s + config.max_gap_s,
            current_time,
        )
        lane_id = track[index].lane_id
        if end <= start:
            continue
        observed_duration += end - start
        if lane_id is not None:
            durations[lane_id] += end - start
    if not durations:
        return None
    minimum_coverage = config.origin_stable_s * config.target_majority
    if observed_duration + 1e-9 < minimum_coverage:
        return None
    majority_lane = max(durations, key=durations.__getitem__)
    lane_evidence_duration = sum(durations.values())
    if durations[majority_lane] / lane_evidence_duration < config.target_majority:
        return None
    return majority_lane
