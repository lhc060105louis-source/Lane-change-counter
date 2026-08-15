from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lane_change_counter.models import (
    Box,
    CacheManifest,
    LaneChangeEvent,
    RunProvenance,
    RunSummary,
    RunThresholds,
    TrackObservation,
    VideoMetadata,
)

DEFAULT_CACHE_ROOT = Path(".cache")
_CACHE_NAMESPACE = "lane-change-counter"
_GENERATION_NAME = ".generation"
_MANIFEST_NAME = "manifest.json"
_TRACKS_NAME = "tracks.jsonl"


class InvalidTrackCacheError(RuntimeError):
    """Raised when a first-pass observation cache is absent, stale, or corrupt."""


def _cache_directory(cache_root: Path, video_sha256: str) -> Path:
    if len(video_sha256) != 64 or any(character not in "0123456789abcdef" for character in video_sha256):
        raise ValueError("video_sha256 must be exactly 64 lowercase hexadecimal characters")
    return Path(cache_root) / _CACHE_NAMESPACE / video_sha256


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _manifest_text(manifest: CacheManifest) -> str:
    return f"{_canonical_json(manifest.to_dict())}\n"


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _observation_from_dict(raw: object) -> TrackObservation:
    if not isinstance(raw, dict) or set(raw) != {
        "track_id",
        "frame_index",
        "timestamp_s",
        "box",
        "class_name",
        "score",
    }:
        raise ValueError("track observation has unexpected fields")

    track_id = raw["track_id"]
    frame_index = raw["frame_index"]
    timestamp_s = raw["timestamp_s"]
    box = raw["box"]
    class_name = raw["class_name"]
    score = raw["score"]
    if not isinstance(track_id, int) or isinstance(track_id, bool):
        raise TypeError("track_id must be an integer")
    if not isinstance(frame_index, int) or isinstance(frame_index, bool):
        raise TypeError("frame_index must be an integer")
    if not _is_number(timestamp_s) or not _is_number(score):
        raise ValueError("observation numeric values must be finite")
    if not isinstance(class_name, str):
        raise TypeError("class_name must be a string")
    if not isinstance(box, dict) or set(box) != {"x1", "y1", "x2", "y2"}:
        raise ValueError("box has unexpected fields")
    if not all(_is_number(box[key]) for key in ("x1", "y1", "x2", "y2")):
        raise ValueError("box coordinates must be finite")

    return TrackObservation(
        track_id=track_id,
        frame_index=frame_index,
        timestamp_s=float(timestamp_s),
        box=Box(*(float(box[key]) for key in ("x1", "y1", "x2", "y2"))),
        class_name=class_name,
        score=float(score),
    )


def _read_validated_cache(
    expected_manifest: CacheManifest,
    cache_root: Path,
) -> tuple[TrackObservation, ...]:
    try:
        cache_dir = _cache_directory(cache_root, expected_manifest.video_sha256)
        generation_before = cache_dir.joinpath(_GENERATION_NAME).read_text(encoding="utf-8")
        if (
            len(generation_before) != 33
            or not generation_before.endswith("\n")
            or any(character not in "0123456789abcdef" for character in generation_before[:-1])
        ):
            raise ValueError("cache generation marker is malformed")

        manifest_text = cache_dir.joinpath(_MANIFEST_NAME).read_text(encoding="utf-8")
        if manifest_text != _manifest_text(expected_manifest):
            raise ValueError("cache manifest does not match")

        tracks_text = cache_dir.joinpath(_TRACKS_NAME).read_text(encoding="utf-8")
        generation_after = cache_dir.joinpath(_GENERATION_NAME).read_text(encoding="utf-8")
        if generation_after != generation_before:
            raise ValueError("cache generation changed while it was being read")
        if not tracks_text:
            return ()
        lines = tracks_text.splitlines()
        if not tracks_text.endswith("\n") or any(not line for line in lines):
            raise ValueError("tracks.jsonl is not canonical newline-delimited JSON")
        return tuple(_observation_from_dict(json.loads(line)) for line in lines)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise InvalidTrackCacheError(
            f"invalid track cache for video SHA-256 {expected_manifest.video_sha256!r}"
        ) from error


def cache_is_valid(
    expected_manifest: CacheManifest,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> bool:
    try:
        _read_validated_cache(expected_manifest, cache_root)
    except InvalidTrackCacheError:
        return False
    return True


def read_track_cache(
    expected_manifest: CacheManifest,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> tuple[TrackObservation, ...]:
    return _read_validated_cache(expected_manifest, cache_root)


def _write_text_durably(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _require_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_finite_number(value: object, name: str) -> float:
    if not _is_number(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _require_basename(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or Path(value).name != value:
        raise ValueError(f"{name} must be an exact input basename")
    return value


def _event_to_dict(event: object) -> dict[str, Any]:
    if not isinstance(event, LaneChangeEvent):
        raise TypeError("events must contain LaneChangeEvent values")
    _require_int(event.track_id, "event track_id")
    if not isinstance(event.origin_lane_id, str) or not isinstance(event.target_lane_id, str):
        raise TypeError("event lane identifiers must be strings")
    _require_int(event.start_frame, "event start_frame", minimum=0)
    _require_int(event.confirm_frame, "event confirm_frame", minimum=0)
    _require_finite_number(event.confirm_time_s, "event confirm_time_s")
    if event.confirmation not in {"stable", "edge_exit"}:
        raise ValueError("event confirmation must be stable or edge_exit")
    return event.to_dict()


def _write_json_atomically(path: Path, payload: object, *, overwrite: bool) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = f"{json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)}\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                raise FileExistsError(path) from None
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    else:
        temporary_path.unlink(missing_ok=True)


def write_event_log(
    metadata: VideoMetadata,
    events: Sequence[LaneChangeEvent],
    output_dir: Path,
) -> Path:
    """Write the complete, deterministically ordered event evidence for one input video."""
    if not isinstance(metadata, VideoMetadata):
        raise TypeError("metadata must be VideoMetadata")
    if not isinstance(events, Sequence):
        raise TypeError("events must be a sequence")
    event_records = sorted(
        (_event_to_dict(event) for event in events),
        key=lambda event: (event["confirm_frame"], event["track_id"]),
    )
    payload = {
        "input_basename": _require_basename(metadata.basename, "metadata basename"),
        "total_lane_changes": len(event_records),
        "events": event_records,
    }
    output_path = Path(output_dir) / f"{metadata.path.stem}.events.json"
    _write_json_atomically(output_path, payload, overwrite=True)
    return output_path


def write_run_summary(summary: RunSummary, output_dir: Path) -> Path:
    """Write one JSON-safe run summary without accepting lossy numeric values."""
    if not isinstance(summary, RunSummary):
        raise TypeError("summary must be RunSummary")
    _require_basename(summary.input_basename, "summary input_basename")
    if not isinstance(summary.device, str) or not isinstance(summary.status, str):
        raise TypeError("summary device and status must be strings")
    if not isinstance(summary.warnings, tuple) or not all(
        isinstance(warning, str) for warning in summary.warnings
    ):
        raise TypeError("summary warnings must be a tuple of strings")
    if not isinstance(summary.events, tuple):
        raise TypeError("summary events must be a tuple")
    for event in summary.events:
        _event_to_dict(event)
    if not isinstance(summary.provenance, RunProvenance):
        raise TypeError("summary provenance must be RunProvenance")
    if not all(
        isinstance(value, str) and value
        for value in summary.provenance.to_dict().values()
    ):
        raise ValueError("summary provenance values must be non-empty strings")
    if not isinstance(summary.thresholds, RunThresholds):
        raise TypeError("summary thresholds must be RunThresholds")
    for name, value in summary.thresholds.to_dict().items():
        if name in {"detector_image_size", "geometry_sample_count"}:
            _require_int(value, f"summary threshold {name}", minimum=1)
        else:
            _require_finite_number(value, f"summary threshold {name}")
    _require_finite_number(summary.elapsed_s, "summary elapsed_s")
    _require_int(summary.peak_rss_bytes, "summary peak_rss_bytes", minimum=0)
    output_path = Path(output_dir) / f"{Path(summary.input_basename).stem}.run.json"
    _write_json_atomically(output_path, summary.to_dict(), overwrite=True)
    return output_path


def write_answer_json(answer_path: Path, totals: Mapping[str, int], *, overwrite: bool) -> None:
    """Atomically replace ``answer.json`` from the supplied in-memory totals only."""
    if not isinstance(totals, Mapping):
        raise TypeError("totals must be a mapping")
    answer: dict[str, dict[str, int]] = {}
    for basename, total in totals.items():
        answer[_require_basename(basename, "total basename")] = {
            "total_lane_changes": _require_int(total, "total_lane_changes", minimum=0)
        }
    _write_json_atomically(Path(answer_path), answer, overwrite=overwrite)


def write_track_cache(
    manifest: CacheManifest,
    observations: Sequence[TrackObservation],
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Path:
    if manifest.video_sha256 != manifest.metadata.sha256:
        raise ValueError("manifest video_sha256 must match metadata.sha256")

    cache_dir = _cache_directory(cache_root, manifest.video_sha256)
    namespace_dir = cache_dir.parent
    namespace_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{cache_dir.name}.tmp-", dir=namespace_dir))
    backup_dir: Path | None = None

    try:
        tracks_text = "".join(f"{_canonical_json(item.to_dict())}\n" for item in observations)
        _write_text_durably(staging_dir / _TRACKS_NAME, tracks_text)
        _write_text_durably(staging_dir / _MANIFEST_NAME, _manifest_text(manifest))
        _write_text_durably(staging_dir / _GENERATION_NAME, f"{uuid.uuid4().hex}\n")

        if cache_dir.exists():
            backup_dir = namespace_dir / f".{cache_dir.name}.backup-{uuid.uuid4().hex}"
            os.replace(cache_dir, backup_dir)
        try:
            os.replace(staging_dir, cache_dir)
        except BaseException:
            if backup_dir is not None and backup_dir.exists() and not cache_dir.exists():
                os.replace(backup_dir, cache_dir)
                backup_dir = None
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
            backup_dir = None
        return cache_dir
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if backup_dir is not None and backup_dir.exists():
            if not cache_dir.exists():
                os.replace(backup_dir, cache_dir)
            else:
                shutil.rmtree(backup_dir)
