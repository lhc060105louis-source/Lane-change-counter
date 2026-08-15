"""Second-pass evidence rendering from already-finalized pipeline records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from lane_change_counter.models import (
    LaneAssignment,
    LaneChangeEvent,
    SceneGeometry,
    TrackObservation,
)
from lane_change_counter.video_io import AtomicVideoWriter, iter_frames, probe_video

_LANE_COLORS = ((70, 210, 90), (60, 180, 240), (220, 170, 60), (220, 100, 210))
_EXCLUSION_COLOR = (35, 35, 235)


def running_total_at_frame(events: Sequence[LaneChangeEvent], frame_index: int) -> int:
    """Return the count of supplied events confirmed no later than ``frame_index``."""
    _require_frame_index(frame_index)
    _validate_events(events)
    return sum(event.confirm_frame <= frame_index for event in events)


def annotate_frame(
    frame: np.ndarray,
    scene: SceneGeometry,
    tracks: Sequence[TrackObservation],
    assignments: Sequence[LaneAssignment],
    events: Sequence[LaneChangeEvent],
    frame_index: int,
    *,
    timestamp_s: float | None = None,
    exclusion_hatches: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Draw evidence from supplied records onto a copy of one decoded frame.

    This function intentionally receives finalized events rather than deriving
    any.  Paths are drawn only from supplied observations at or before this
    frame, so future tracking evidence cannot leak into the annotation.
    """
    _require_color_frame(frame)
    _require_frame_index(frame_index)
    _validate_scene_for_frame(scene, frame)
    _validate_tracks(tracks)
    _validate_assignments(assignments)
    _validate_events(events)
    if timestamp_s is not None and timestamp_s < 0:
        raise ValueError("timestamp must be non-negative")

    rendered = frame.copy()
    overlay = rendered.copy()
    scale = _font_scale(rendered.shape[0])
    thickness = max(1, round(scale * 2))
    _draw_geometry(rendered, overlay, scene, scale, thickness, exclusion_hatches)
    _draw_tracks(rendered, tracks, assignments, frame_index, scale, thickness)
    _draw_counts_and_events(rendered, events, frame_index, timestamp_s, scale, thickness)
    return rendered


def render_annotated_video(
    source: Path,
    target: Path,
    scene: SceneGeometry,
    tracks: Sequence[TrackObservation],
    assignments: Sequence[LaneAssignment],
    events: Sequence[LaneChangeEvent],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically render finalized evidence while preserving source video timing."""
    metadata = probe_video(source)
    if scene.frame_size != (metadata.width, metadata.height):
        raise ValueError("scene geometry dimensions do not match source video")
    _validate_tracks(tracks, frame_count=metadata.frame_count)
    _validate_assignments(assignments, frame_count=metadata.frame_count)
    _validate_events(events, frame_count=metadata.frame_count)
    scale = _font_scale(metadata.height)
    exclusion_hatches = tuple(
        _hatching_mask(
            _polygon(exclusion), (metadata.height, metadata.width), scale
        )
        for exclusion in scene.exclusions
    )

    with AtomicVideoWriter(target, metadata.width, metadata.height, metadata.fps, overwrite=overwrite) as writer:
        decoded_count = 0
        for frame_index, timestamp_s, frame in iter_frames(source):
            writer.write(
                annotate_frame(
                    frame,
                    scene,
                    tracks,
                    assignments,
                    events,
                    frame_index,
                    timestamp_s=timestamp_s,
                    exclusion_hatches=exclusion_hatches,
                )
            )
            decoded_count += 1
        if decoded_count <= 0:
            raise ValueError("source video did not yield any frames while rendering")
    return target


def _draw_geometry(
    rendered: np.ndarray,
    overlay: np.ndarray,
    scene: SceneGeometry,
    scale: float,
    thickness: int,
    exclusion_hatches: Sequence[np.ndarray] | None = None,
) -> None:
    for index, lane in enumerate(scene.lanes):
        polygon = _polygon(lane.polygon)
        color = _LANE_COLORS[index % len(_LANE_COLORS)]
        cv2.fillPoly(overlay, [polygon], color, cv2.LINE_AA)
    for exclusion in scene.exclusions:
        polygon = _polygon(exclusion)
        cv2.fillPoly(overlay, [polygon], _EXCLUSION_COLOR, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, rendered, 0.75, 0.0, rendered)

    for index, lane in enumerate(scene.lanes):
        polygon = _polygon(lane.polygon)
        color = _LANE_COLORS[index % len(_LANE_COLORS)]
        cv2.polylines(rendered, [polygon], True, color, thickness, cv2.LINE_AA)
        _text(rendered, lane.lane_id, _polygon_center(polygon), scale, (255, 255, 255), thickness)
    for index, exclusion in enumerate(scene.exclusions):
        polygon = _polygon(exclusion)
        cv2.polylines(rendered, [polygon], True, _EXCLUSION_COLOR, thickness + 1, cv2.LINE_AA)
        hatch = exclusion_hatches[index] if exclusion_hatches is not None else None
        _draw_hatching(rendered, polygon, scale, hatch)


def _draw_tracks(
    rendered: np.ndarray,
    tracks: Sequence[TrackObservation],
    assignments: Sequence[LaneAssignment],
    frame_index: int,
    scale: float,
    thickness: int,
) -> None:
    assignment_by_track = {
        assignment.track_id: assignment
        for assignment in assignments
        if assignment.frame_index == frame_index
    }
    paths: dict[int, list[tuple[int, int]]] = defaultdict(list)
    current: list[TrackObservation] = []
    for observation in tracks:
        if observation.frame_index <= frame_index:
            paths[observation.track_id].append(tuple(round(value) for value in observation.box.bottom_center))
        if observation.frame_index == frame_index:
            current.append(observation)
    for points in paths.values():
        if len(points) > 1:
            cv2.polylines(rendered, [np.asarray(points[-30:], dtype=np.int32)], False, (255, 255, 255), thickness, cv2.LINE_AA)
    for observation in current:
        box = observation.box
        top_left = (round(box.x1), round(box.y1))
        bottom_right = (round(box.x2), round(box.y2))
        cv2.rectangle(rendered, top_left, bottom_right, (255, 255, 255), thickness, cv2.LINE_AA)
        assignment = assignment_by_track.get(observation.track_id)
        label = f"#{observation.track_id} {observation.class_name}"
        if assignment is not None:
            lane_name = assignment.lane_id if assignment.lane_id is not None else "excluded"
            label += f" {lane_name} {assignment.confidence:.2f}"
        _text(rendered, label, (top_left[0], max(round(18 * scale), top_left[1] - round(6 * scale))), scale, (255, 255, 255), thickness)


def _draw_counts_and_events(
    rendered: np.ndarray,
    events: Sequence[LaneChangeEvent],
    frame_index: int,
    timestamp_s: float | None,
    scale: float,
    thickness: int,
) -> None:
    total = running_total_at_frame(events, frame_index)
    _text(rendered, f"Lane changes: {total} / final {len(events)}", (10, round(28 * scale)), scale, (255, 255, 255), thickness)
    notices = [
        event
        for event in events
        if event.confirm_frame <= frame_index
        and (timestamp_s is None or timestamp_s - event.confirm_time_s <= 1.0 + 1e-9)
    ]
    for index, event in enumerate(notices):
        _text(
            rendered,
            (
                f"#{event.track_id}: {event.origin_lane_id} -> {event.target_lane_id}"
                f" ({event.confirmation})"
            ),
            (10, round((54 + 26 * index) * scale)),
            scale,
            (0, 255, 255),
            thickness,
        )


def _hatching_mask(
    polygon: np.ndarray, frame_shape: tuple[int, int], scale: float
) -> np.ndarray:
    polygon_mask = np.zeros(frame_shape, dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [polygon], 255)
    hatch = np.zeros(frame_shape, dtype=np.uint8)
    step = max(8, round(18 * scale))
    for offset in range(-frame_shape[0], frame_shape[1], step):
        cv2.line(hatch, (offset, 0), (offset + frame_shape[0], frame_shape[0]), 255, 1, cv2.LINE_AA)
    return cv2.bitwise_and(hatch, polygon_mask)


def _draw_hatching(
    frame: np.ndarray,
    polygon: np.ndarray,
    scale: float,
    mask: np.ndarray | None = None,
) -> None:
    if mask is None:
        mask = _hatching_mask(polygon, frame.shape[:2], scale)
    if mask.shape != frame.shape[:2]:
        raise ValueError("exclusion hatching dimensions do not match frame")
    frame[mask > 0] = _EXCLUSION_COLOR


def _text(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _font_scale(height: int) -> float:
    return max(0.38, min(1.2, height / 720.0))


def _polygon(points: Sequence[tuple[float, float]]) -> np.ndarray:
    return np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32)


def _polygon_center(polygon: np.ndarray) -> tuple[int, int]:
    return tuple(np.rint(np.mean(polygon, axis=0)).astype(int))  # type: ignore[return-value]


def _require_color_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("frame must be an uint8 BGR image")


def _validate_scene_for_frame(scene: SceneGeometry, frame: np.ndarray) -> None:
    if scene.frame_size != (frame.shape[1], frame.shape[0]):
        raise ValueError("frame dimensions do not match scene geometry")


def _require_frame_index(frame_index: int) -> None:
    if not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("frame index must be a non-negative integer")


def _validate_tracks(tracks: Sequence[TrackObservation], frame_count: int | None = None) -> None:
    for track in tracks:
        _validate_record_frame(track.frame_index, frame_count)


def _validate_assignments(assignments: Sequence[LaneAssignment], frame_count: int | None = None) -> None:
    for assignment in assignments:
        _validate_record_frame(assignment.frame_index, frame_count)


def _validate_events(events: Sequence[LaneChangeEvent], frame_count: int | None = None) -> None:
    for event in events:
        _validate_record_frame(event.start_frame, frame_count)
        _validate_record_frame(event.confirm_frame, frame_count)
        if event.confirm_frame < event.start_frame:
            raise ValueError("event confirmation frame precedes start frame")


def _validate_record_frame(frame_index: int, frame_count: int | None) -> None:
    _require_frame_index(frame_index)
    if frame_count is not None and frame_index >= frame_count:
        raise ValueError("record frame index is outside source video")
