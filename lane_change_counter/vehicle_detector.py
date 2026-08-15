from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from platformdirs import user_cache_path

from lane_change_counter.config import DeviceRequest, PipelineConfig
from lane_change_counter.models import Box, Detection

ELIGIBLE_COCO_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
_MPS_ERROR_MARKERS = (
    "mps",
    "metal",
    "not implemented for",
    "not supported on this device",
)
_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "weights.lock.json"


class DeviceUnavailableError(RuntimeError):
    """Raised when an explicitly requested inference device is unavailable."""


class DetectorInferenceError(RuntimeError):
    """Raised when detector inference fails without a safe fallback."""


class DetectorRestartRequired(DetectorInferenceError):
    """Signals that auto fallback succeeded and the caller must restart its video pass."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.restart_frame = 0
        self.device = "cpu"


def resolve_device(request: DeviceRequest) -> str:
    if request is DeviceRequest.CPU:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if request is DeviceRequest.MPS:
        raise DeviceUnavailableError("MPS was explicitly requested but is unavailable")
    return "cpu"


def _load_locked_weight_path() -> Path:
    raw = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    path = user_cache_path("lane-change-counter") / "weights" / f"{raw['model_id']}.pt"
    expected_size = int(raw["size_bytes"])
    expected_sha256 = str(raw["sha256"])

    try:
        if path.stat().st_size != expected_size:
            raise OSError("wrong file size")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise OSError("wrong checksum")
    except OSError as error:
        raise FileNotFoundError(
            "verified detector weights are unavailable; run "
            f"`uv run python scripts/fetch_weights.py` (expected SHA-256 {expected_sha256})"
        ) from error
    return path


def _is_mps_device_error(error: Exception) -> bool:
    message = str(error).lower()
    return isinstance(error, RuntimeError) and any(
        marker in message for marker in _MPS_ERROR_MARKERS
    )


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


class VehicleDetector:
    def __init__(
        self,
        model: Any,
        *,
        device: str,
        config: PipelineConfig,
        request: DeviceRequest | None = None,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self._model = model
        self.device = device
        self._config = config
        self._request = request or (DeviceRequest.MPS if device == "mps" else DeviceRequest.CPU)
        self._model_loader = model_loader
        self._fallback_used = False

    def close(self) -> None:
        """Release the loaded model when a sequential input finishes."""
        self._model = None

    def detect(
        self,
        frame_index: int,
        timestamp_s: float,
        frame: np.ndarray,
    ) -> tuple[Detection, ...]:
        try:
            results = self._model(
                frame,
                device=self.device,
                imgsz=self._config.detector_image_size,
                conf=self._config.detector_confidence,
                iou=self._config.detector_iou,
                verbose=False,
            )
        except Exception as error:
            if (
                self._request is DeviceRequest.AUTO
                and self.device == "mps"
                and not self._fallback_used
                and _is_mps_device_error(error)
            ):
                self._fallback_used = True
                if self._model_loader is None:
                    raise DetectorInferenceError(
                        "MPS inference failed and no CPU model loader is available"
                    ) from error
                try:
                    cpu_model = self._model_loader("cpu")
                except Exception as load_error:
                    raise DetectorInferenceError(
                        "MPS inference failed and the CPU detector could not be loaded"
                    ) from load_error
                self._model = cpu_model
                self.device = "cpu"
                raise DetectorRestartRequired(
                    "MPS inference failed; CPU fallback is ready. Discard all detections "
                    "from this pass and restart video iteration at frame zero."
                ) from error
            raise DetectorInferenceError(
                f"detector inference failed on {self.device}: {error}"
            ) from error

        if not results or results[0].boxes is None:
            return ()

        boxes = results[0].boxes
        coordinates = _as_list(boxes.xyxy)
        class_ids = _as_list(boxes.cls)
        confidences = _as_list(boxes.conf)
        detections = [
            Detection(
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                box=Box(*(float(value) for value in coordinates[index])),
                class_id=class_id,
                class_name=ELIGIBLE_COCO_CLASSES[class_id],
                score=float(confidences[index]),
            )
            for index, raw_class_id in enumerate(class_ids)
            if (class_id := int(raw_class_id)) in ELIGIBLE_COCO_CLASSES
        ]
        detections.sort(
            key=lambda detection: (detection.box.x1, detection.box.y1, detection.class_id)
        )
        return tuple(detections)


def load_detector(config: PipelineConfig, request: DeviceRequest) -> VehicleDetector:
    from ultralytics import YOLO

    device = resolve_device(request)
    weight_path = _load_locked_weight_path()

    def load_model(_device: str) -> Any:
        return YOLO(str(weight_path))

    return VehicleDetector(
        load_model(device),
        device=device,
        config=config,
        request=request,
        model_loader=load_model,
    )
