from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Self

import cv2
import numpy as np

from lane_change_counter.models import VideoMetadata


def fingerprint_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> VideoMetadata:
    """Validate a readable video and return its stable metadata."""
    if not path.is_file():
        raise FileNotFoundError(path)

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise ValueError(f"video has invalid dimensions: {width}x{height}")
        if not 1.0 <= fps <= 240.0:
            raise ValueError(f"video has invalid fps: {fps}")
        if frame_count <= 0:
            raise ValueError(f"video has no frames: {path}")

        decoded, frame = capture.read()
        if not decoded or frame is None:
            raise ValueError(f"video failed representative-frame decode: {path}")

        return VideoMetadata(
            path=path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_s=frame_count / fps,
            sha256=fingerprint_file(path),
        )
    finally:
        capture.release()


def iter_frames(path: Path) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield every decoded frame with timestamps derived from frame index and FPS."""
    metadata = probe_video(path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {path}")
        frame_index = 0
        while True:
            decoded, frame = capture.read()
            if not decoded or frame is None:
                break
            yield frame_index, frame_index / metadata.fps, frame
            frame_index += 1
    finally:
        capture.release()


class AtomicVideoWriter:
    """Write an MP4 to a sibling partial file and promote it only on success."""

    def __init__(
        self,
        target: Path,
        width: int,
        height: int,
        fps: float,
        *,
        overwrite: bool = False,
    ) -> None:
        self.target = target
        self.partial_path = target.with_name(f"{target.name}.partial.mp4")
        self.width = width
        self.height = height
        self.fps = fps
        self._frames_written = 0
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        if self.partial_path.exists():
            raise FileExistsError(self.partial_path)

        self._writer = cv2.VideoWriter(
            str(self.partial_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not self._writer.isOpened():
            try:
                self._writer.release()
            except OSError:
                pass
            try:
                self.partial_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise ValueError(f"could not create video writer: {self.partial_path}")

    def __enter__(self) -> Self:
        return self

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError("frame dimensions do not match video writer")
        self._writer.write(frame)
        self._frames_written += 1

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        try:
            self._writer.release()
        except OSError:
            if exc_type is None:
                self._remove_partial()
                raise
        if exc_type is not None:
            self._remove_partial()
            return False
        try:
            self._validate_partial()
            os.replace(self.partial_path, self.target)
        except (OSError, ValueError):
            self._remove_partial()
            raise
        return False

    def _validate_partial(self) -> None:
        """Reject a silently incomplete encoder output before promotion."""
        try:
            metadata = probe_video(self.partial_path)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(f"rendered video validation failed: {self.partial_path}") from error
        if self._frames_written <= 0 or metadata.frame_count <= 0:
            raise ValueError(f"rendered video validation failed: {self.partial_path}")
        if (metadata.width, metadata.height, metadata.frame_count) != (
            self.width,
            self.height,
            self._frames_written,
        ) or not math.isclose(metadata.fps, self.fps, rel_tol=0.0, abs_tol=0.01):
            raise ValueError(f"rendered video validation failed: {self.partial_path}")

    def _remove_partial(self) -> None:
        try:
            self.partial_path.unlink(missing_ok=True)
        except OSError:
            pass
