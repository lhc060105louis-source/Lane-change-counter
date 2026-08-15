from dataclasses import dataclass
from enum import StrEnum


class DeviceRequest(StrEnum):
    AUTO = "auto"
    MPS = "mps"
    CPU = "cpu"


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    origin_stable_s: float = 0.6
    min_track_s: float = 1.2
    target_stable_s: float = 0.7
    max_gap_s: float = 0.4
    direction_tolerance_deg: float = 20.0
    lane_change_direction_tolerance_deg: float = 45.0
    min_lateral_lane_widths: float = 0.55
    target_majority: float = 0.70
    edge_exit_target_stable_s: float = 0.10
    edge_exit_min_target_observations: int = 4
    detector_confidence: float = 0.25
    detector_iou: float = 0.70
    detector_image_size: int = 960
    geometry_sample_count: int = 41
