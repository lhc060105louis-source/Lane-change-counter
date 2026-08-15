from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import supervision as sv

from lane_change_counter.models import Detection, TrackObservation


class VehicleTracker:
    """Adapt typed vehicle detections to Supervision's ByteTrack interface."""

    def __init__(self, *, fps: float, max_gap_s: float) -> None:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a positive finite number")
        if not np.isfinite(max_gap_s) or max_gap_s < 0:
            raise ValueError("max_gap_s must be a non-negative finite number")

        self._lost_track_buffer = round(max_gap_s * fps)
        self._empty_frames = 0
        self._reset_pending = False
        self._track_id_offset = 0
        self._max_emitted_track_id = 0
        self._tracker = self._new_tracker()

    def _new_tracker(self) -> sv.ByteTrack:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The `ByteTrack` was deprecated.*",
                category=FutureWarning,
            )
            # Supervision scales lost_track_buffer relative to frame_rate / 30.
            # Passing the reference rate keeps our already frame-based value exact.
            return sv.ByteTrack(
                lost_track_buffer=self._lost_track_buffer,
                frame_rate=30,
            )

    def update(
        self,
        detections: Sequence[Detection],
        frame: np.ndarray,
    ) -> tuple[TrackObservation, ...]:
        del frame  # Kept in the public interface for tracker adapters that consume pixels.

        source = tuple(detections)
        if not source:
            self._empty_frames += 1
            if self._reset_pending:
                return ()
        elif self._reset_pending:
            # All ByteTrack state has expired. A fresh tracker makes this the first
            # frame of a new ID epoch, without matching it to any expired track.
            self._track_id_offset = self._max_emitted_track_id
            self._tracker = self._new_tracker()
            self._empty_frames = 0
            self._reset_pending = False
        else:
            self._empty_frames = 0

        supervision_detections = sv.Detections(
            xyxy=np.asarray(
                [[item.box.x1, item.box.y1, item.box.x2, item.box.y2] for item in source],
                dtype=float,
            ).reshape((-1, 4)),
            confidence=np.asarray([item.score for item in source], dtype=float),
            class_id=np.asarray([item.class_id for item in source], dtype=int),
            data={"source_index": np.arange(len(source), dtype=int)},
        )
        tracked = self._tracker.update_with_detections(supervision_detections)
        if not source and self._empty_frames > self._lost_track_buffer:
            self._reset_pending = True

        if not tracked or tracked.tracker_id is None:
            return ()
        source_indexes = tracked.data.get("source_index")
        if source_indexes is None:
            raise RuntimeError("Supervision discarded tracker source indexes")

        observations = tuple(
            TrackObservation(
                track_id=self._track_id_offset + int(track_id),
                frame_index=source[int(source_index)].frame_index,
                timestamp_s=source[int(source_index)].timestamp_s,
                box=source[int(source_index)].box,
                class_name=source[int(source_index)].class_name,
                score=source[int(source_index)].score,
            )
            for source_index, track_id in zip(source_indexes, tracked.tracker_id, strict=True)
        )
        self._max_emitted_track_id = max(
            self._max_emitted_track_id,
            *(observation.track_id for observation in observations),
        )
        return observations
