from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    sha256: str

    @property
    def basename(self) -> str:
        return self.path.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, self.y2)

    def to_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass(frozen=True, slots=True)
class Detection:
    frame_index: int
    timestamp_s: float
    box: Box
    class_id: int
    class_name: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "box": self.box.to_dict(),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class TrackObservation:
    track_id: int
    frame_index: int
    timestamp_s: float
    box: Box
    class_name: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "box": self.box.to_dict(),
            "class_name": self.class_name,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class Lane:
    lane_id: str
    polygon: tuple[tuple[float, float], ...]
    direction_deg: float
    neighbor_ids: tuple[str, ...]
    boundary_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "polygon": _json_safe(self.polygon),
            "direction_deg": self.direction_deg,
            "neighbor_ids": _json_safe(self.neighbor_ids),
            "boundary_ids": _json_safe(self.boundary_ids),
        }


@dataclass(frozen=True, slots=True)
class SceneBoundary:
    boundary_id: str
    points: tuple[tuple[float, float], ...]
    kind: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "points": _json_safe(self.points),
            "kind": self.kind,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class SceneRegion:
    region_id: str
    polygon: tuple[tuple[float, float], ...]
    kind: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "polygon": _json_safe(self.polygon),
            "kind": self.kind,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class SceneGeometry:
    frame_size: tuple[int, int]
    lanes: tuple[Lane, ...]
    exclusions: tuple[tuple[tuple[float, float], ...], ...]
    confidence: float
    diagnostics: tuple[str, ...]
    boundaries: tuple[SceneBoundary, ...] = ()
    regions: tuple[SceneRegion, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_size": _json_safe(self.frame_size),
            "lanes": _json_safe(self.lanes),
            "exclusions": _json_safe(self.exclusions),
            "confidence": self.confidence,
            "diagnostics": _json_safe(self.diagnostics),
            "boundaries": _json_safe(self.boundaries),
            "regions": _json_safe(self.regions),
        }


@dataclass(frozen=True, slots=True)
class LaneAssignment:
    track_id: int
    frame_index: int
    timestamp_s: float
    lane_id: str | None
    confidence: float
    lateral_lane_widths: float
    at_frame_edge: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "lane_id": self.lane_id,
            "confidence": self.confidence,
            "lateral_lane_widths": self.lateral_lane_widths,
            "at_frame_edge": self.at_frame_edge,
        }


@dataclass(frozen=True, slots=True)
class LaneChangeEvent:
    track_id: int
    origin_lane_id: str
    target_lane_id: str
    start_frame: int
    confirm_frame: int
    confirm_time_s: float
    confirmation: str = "stable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "origin_lane_id": self.origin_lane_id,
            "target_lane_id": self.target_lane_id,
            "start_frame": self.start_frame,
            "confirm_frame": self.confirm_frame,
            "confirm_time_s": self.confirm_time_s,
            "confirmation": self.confirmation,
        }


@dataclass(frozen=True, slots=True)
class CacheManifest:
    video_sha256: str
    schema_version: str
    model_id: str
    weight_sha256: str
    config_sha256: str
    metadata: VideoMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_sha256": self.video_sha256,
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "weight_sha256": self.weight_sha256,
            "config_sha256": self.config_sha256,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RunProvenance:
    pipeline_schema_version: str
    package_version: str
    model_id: str
    weight_sha256: str
    config_sha256: str
    python_version: str
    numpy_version: str
    opencv_version: str
    psutil_version: str
    supervision_version: str
    torch_version: str
    ultralytics_version: str

    def to_dict(self) -> dict[str, str]:
        return {
            "pipeline_schema_version": self.pipeline_schema_version,
            "package_version": self.package_version,
            "model_id": self.model_id,
            "weight_sha256": self.weight_sha256,
            "config_sha256": self.config_sha256,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "opencv_version": self.opencv_version,
            "psutil_version": self.psutil_version,
            "supervision_version": self.supervision_version,
            "torch_version": self.torch_version,
            "ultralytics_version": self.ultralytics_version,
        }


@dataclass(frozen=True, slots=True)
class RunThresholds:
    origin_stable_s: float
    min_track_s: float
    target_stable_s: float
    max_gap_s: float
    direction_tolerance_deg: float
    lane_change_direction_tolerance_deg: float
    min_lateral_lane_widths: float
    target_majority: float
    edge_exit_target_stable_s: float
    edge_exit_min_target_observations: int
    detector_confidence: float
    detector_iou: float
    detector_image_size: int
    geometry_sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "origin_stable_s": self.origin_stable_s,
            "min_track_s": self.min_track_s,
            "target_stable_s": self.target_stable_s,
            "max_gap_s": self.max_gap_s,
            "direction_tolerance_deg": self.direction_tolerance_deg,
            "lane_change_direction_tolerance_deg": self.lane_change_direction_tolerance_deg,
            "min_lateral_lane_widths": self.min_lateral_lane_widths,
            "target_majority": self.target_majority,
            "edge_exit_target_stable_s": self.edge_exit_target_stable_s,
            "edge_exit_min_target_observations": self.edge_exit_min_target_observations,
            "detector_confidence": self.detector_confidence,
            "detector_iou": self.detector_iou,
            "detector_image_size": self.detector_image_size,
            "geometry_sample_count": self.geometry_sample_count,
        }


@dataclass(frozen=True, slots=True)
class RunSummary:
    input_basename: str
    device: str
    warnings: tuple[str, ...]
    events: tuple[LaneChangeEvent, ...]
    elapsed_s: float
    peak_rss_bytes: int
    status: str
    provenance: RunProvenance | None = None
    thresholds: RunThresholds | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_basename": self.input_basename,
            "device": self.device,
            "warnings": _json_safe(self.warnings),
            "events": _json_safe(self.events),
            "elapsed_s": self.elapsed_s,
            "peak_rss_bytes": self.peak_rss_bytes,
            "status": self.status,
            "provenance": _json_safe(self.provenance),
            "thresholds": _json_safe(self.thresholds),
        }
