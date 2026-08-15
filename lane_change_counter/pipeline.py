"""Deterministic orchestration for the two-pass lane-change pipeline."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import psutil

from lane_change_counter.annotator import render_annotated_video
from lane_change_counter.config import DeviceRequest, PipelineConfig
from lane_change_counter.lane_assignment import assign_observation, smooth_assignments
from lane_change_counter.lane_change_fsm import detect_lane_changes
from lane_change_counter.models import (
    CacheManifest,
    Detection,
    LaneAssignment,
    LaneChangeEvent,
    RunProvenance,
    RunSummary,
    RunThresholds,
    SceneGeometry,
    TrackObservation,
    VideoMetadata,
)
from lane_change_counter.reporting import (
    DEFAULT_CACHE_ROOT,
    InvalidTrackCacheError,
    read_track_cache,
    write_answer_json,
    write_event_log,
    write_run_summary,
    write_track_cache,
)
from lane_change_counter.scene_geometry import (
    build_background,
    estimate_geometry,
    render_geometry_diagnostic,
    validate_geometry,
)
from lane_change_counter.vehicle_detector import (
    DetectorInferenceError,
    DetectorRestartRequired,
    load_detector,
    resolve_device,
)
from lane_change_counter.vehicle_tracker import VehicleTracker
from lane_change_counter.video_io import iter_frames, probe_video

PIPELINE_SCHEMA_VERSION = "1"
_MINIMUM_GEOMETRY_CONFIDENCE = 0.70
_WEIGHT_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "weights.lock.json"


class Detector(Protocol):
    device: str

    def detect(
        self, frame_index: int, timestamp_s: float, frame: np.ndarray
    ) -> Sequence[Detection]: ...


class Tracker(Protocol):
    def update(
        self, detections: Sequence[Detection], frame: np.ndarray
    ) -> Sequence[TrackObservation]: ...


DetectorFactory = Callable[[PipelineConfig, DeviceRequest], Detector]
TrackerFactory = Callable[[float, float], Tracker]
GeometryFactory = Callable[[np.ndarray, VideoMetadata], SceneGeometry]
GeometryValidator = Callable[[SceneGeometry, Sequence[TrackObservation]], SceneGeometry]
FrameIterator = Callable[[Path], Iterator[tuple[int, float, np.ndarray]]]


def _default_tracker_factory(fps: float, max_gap_s: float) -> VehicleTracker:
    return VehicleTracker(fps=fps, max_gap_s=max_gap_s)


@dataclass(frozen=True, slots=True)
class RunOptions:
    device: DeviceRequest = DeviceRequest.AUTO
    overwrite: bool = False
    rebuild_cache: bool = False
    geometry_only: bool = False
    log_level: str = "INFO"


@dataclass(frozen=True, slots=True)
class PipelineDependencies:
    """Replaceable system boundaries used by synthetic integration tests."""

    config: PipelineConfig = field(default_factory=PipelineConfig)
    detector_factory: DetectorFactory = load_detector
    tracker_factory: TrackerFactory = _default_tracker_factory
    geometry_factory: GeometryFactory = estimate_geometry
    geometry_validator: GeometryValidator = validate_geometry
    frame_iterator: FrameIterator = iter_frames
    cache_root: Path = DEFAULT_CACHE_ROOT
    renderer: Callable[..., Path] = render_annotated_video
    event_writer: Callable[[VideoMetadata, Sequence[Any], Path], Path] = write_event_log
    summary_writer: Callable[[RunSummary, Path], Path] = write_run_summary
    answer_writer: Callable[..., None] = write_answer_json


class InputProcessingError(RuntimeError):
    """Base class for an isolated, expected failure of one input video."""


class VideoInputError(InputProcessingError):
    """Raised when one source video cannot be read or decoded."""


class DetectorProcessingError(InputProcessingError):
    """Raised when inference fails for one input after allowed fallback handling."""


class GeometryValidationError(InputProcessingError):
    """Raised after diagnostic evidence is saved for unusable scene geometry."""


class ArtifactTransactionError(RuntimeError):
    """Raised when an artifact transaction cannot restore its prior generation."""

    def __init__(
        self,
        message: str,
        *,
        publication_error: BaseException | None = None,
        recovery_errors: Sequence[tuple[Path, BaseException]] = (),
    ) -> None:
        self.publication_error = publication_error
        self.recovery_errors = tuple(recovery_errors)
        details = []
        if publication_error is not None:
            details.append(f"publication error: {publication_error}")
        details.extend(f"cleanup error for {path}: {error}" for path, error in recovery_errors)
        super().__init__(f"{message}; {'; '.join(details)}" if details else message)


@dataclass(slots=True)
class _PerformanceMeter:
    started_at: float
    process: psutil.Process
    peak_rss_bytes: int
    stop_event: threading.Event
    monitor: threading.Thread | None = None
    finished_at: float | None = None

    @classmethod
    def start(cls) -> _PerformanceMeter:
        process = psutil.Process()
        rss = int(process.memory_info().rss)
        meter = cls(time.perf_counter(), process, rss, threading.Event())
        meter.monitor = threading.Thread(
            target=meter._monitor_rss,
            name="lane-change-peak-rss",
            daemon=True,
        )
        meter.monitor.start()
        return meter

    def _monitor_rss(self) -> None:
        while not self.stop_event.wait(0.02):
            self.sample()

    def sample(self) -> None:
        self.peak_rss_bytes = max(self.peak_rss_bytes, int(self.process.memory_info().rss))

    def stop(self) -> None:
        if self.finished_at is not None:
            return
        self.stop_event.set()
        if self.monitor is not None:
            self.monitor.join()
        self.sample()
        self.finished_at = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        finished_at = self.finished_at if self.finished_at is not None else time.perf_counter()
        return max(0.0, finished_at - self.started_at)


def process_video(
    input_path: Path,
    output_dir: Path,
    options: RunOptions,
    *,
    dependencies: PipelineDependencies | None = None,
) -> RunSummary:
    """Process one video and publish its complete answer as the final operation."""
    dependency_set = dependencies or PipelineDependencies()
    summary = _process_video_artifacts(
        Path(input_path),
        Path(output_dir),
        options,
        dependency_set,
        reserve_answer=True,
    )
    if not options.geometry_only:
        dependency_set.answer_writer(
            Path(output_dir) / "answer.json",
            {summary.input_basename: len(summary.events)},
            overwrite=options.overwrite,
        )
    return summary


def process_directory(
    input_dir: Path,
    output_dir: Path,
    options: RunOptions,
    *,
    dependencies: PipelineDependencies | None = None,
) -> tuple[RunSummary, ...]:
    """Process direct-child MP4 inputs sequentially and isolate per-input failures."""
    source_dir = Path(input_dir)
    destination = Path(output_dir)
    dependency_set = dependencies or PipelineDependencies()
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)

    inputs = sorted(
        (path for path in source_dir.glob("*.mp4") if path.is_file()),
        key=lambda path: path.name,
    )
    if not inputs:
        raise FileNotFoundError(f"no direct-child *.mp4 inputs found in {source_dir}")

    answer_path = destination / "answer.json"
    if not options.geometry_only and answer_path.exists() and not options.overwrite:
        raise FileExistsError(answer_path)

    summaries: list[RunSummary] = []
    for source in inputs:
        failure_meter = _PerformanceMeter.start()
        isolated_failure: InputProcessingError | None = None
        try:
            summary = _process_video_artifacts(
                source,
                destination,
                options,
                dependency_set,
                reserve_answer=False,
            )
        except InputProcessingError as error:
            isolated_failure = error
        finally:
            failure_meter.stop()
        if isolated_failure is not None:
            summary = RunSummary(
                input_basename=source.name,
                device=options.device.value,
                warnings=(
                    f"{type(isolated_failure).__name__}: {isolated_failure}",
                ),
                events=(),
                elapsed_s=failure_meter.elapsed_s,
                peak_rss_bytes=failure_meter.peak_rss_bytes,
                status="failed",
            )
        summaries.append(summary)
        gc.collect()

    if not options.geometry_only and all(summary.status == "success" for summary in summaries):
        totals = {
            summary.input_basename: len(summary.events)
            for summary in summaries
            if summary.status == "success"
        }
        dependency_set.answer_writer(answer_path, totals, overwrite=options.overwrite)
    return tuple(summaries)


def _process_video_artifacts(
    input_path: Path,
    output_dir: Path,
    options: RunOptions,
    dependencies: PipelineDependencies,
    *,
    reserve_answer: bool,
) -> RunSummary:
    meter = _PerformanceMeter.start()
    try:
        return _process_video_artifacts_measured(
            input_path,
            output_dir,
            options,
            dependencies,
            reserve_answer=reserve_answer,
            meter=meter,
        )
    finally:
        meter.stop()


def _process_video_artifacts_measured(
    input_path: Path,
    output_dir: Path,
    options: RunOptions,
    dependencies: PipelineDependencies,
    *,
    reserve_answer: bool,
    meter: _PerformanceMeter,
) -> RunSummary:
    try:
        metadata = probe_video(input_path)
    except (OSError, ValueError) as error:
        raise VideoInputError(f"unreadable input video {input_path}: {error}") from error
    meter.sample()
    _refuse_existing_outputs(
        metadata,
        output_dir,
        options,
        include_answer=reserve_answer,
    )

    background_frames = _collect_background_frames(
        metadata, dependencies, dependencies.config
    )
    background = build_background(background_frames, dependencies.config.geometry_sample_count)
    meter.sample()
    candidate_scene = dependencies.geometry_factory(background, metadata)
    meter.sample()

    if options.geometry_only:
        _write_geometry_diagnostic(candidate_scene, background, output_dir, metadata, options)
        meter.sample()
        return RunSummary(
            input_basename=metadata.basename,
            device=options.device.value,
            warnings=tuple(candidate_scene.diagnostics),
            events=(),
            elapsed_s=meter.elapsed_s,
            peak_rss_bytes=meter.peak_rss_bytes,
            status="diagnostic",
        )

    manifest = _cache_manifest(metadata, dependencies.config)
    tracks: tuple[TrackObservation, ...]
    detector_device = options.device.value
    warnings: list[str] = []
    try:
        if options.rebuild_cache:
            raise InvalidTrackCacheError("cache rebuild requested")
        tracks = read_track_cache(manifest, cache_root=dependencies.cache_root)
    except InvalidTrackCacheError:
        try:
            tracks, detector_device, restart_warnings = _create_tracks(
                input_path,
                metadata,
                options,
                dependencies,
                meter,
            )
        except DetectorInferenceError as error:
            raise DetectorProcessingError(f"detector failed for {input_path}: {error}") from error
        warnings.extend(restart_warnings)
        write_track_cache(manifest, tracks, cache_root=dependencies.cache_root)
    else:
        detector_device = resolve_device(options.device)
    meter.sample()

    scene = dependencies.geometry_validator(candidate_scene, tracks)
    meter.sample()
    if scene.confidence < _MINIMUM_GEOMETRY_CONFIDENCE:
        _write_geometry_diagnostic(scene, background, output_dir, metadata, options)
        raise GeometryValidationError(
            f"geometry validation confidence {scene.confidence:.2f} is below "
            f"{_MINIMUM_GEOMETRY_CONFIDENCE:.2f}"
        )

    raw_assignments = tuple(assign_observation(scene, observation) for observation in tracks)
    assignments = smooth_assignments(raw_assignments, metadata.fps)
    events = detect_lane_changes(assignments, scene, dependencies.config)
    meter.sample()

    return _stage_and_publish_artifacts(
        input_path,
        output_dir,
        metadata,
        scene,
        tracks,
        assignments,
        events,
        detector_device,
        tuple(warnings),
        manifest,
        options,
        dependencies,
        meter,
    )


def _stage_and_publish_artifacts(
    input_path: Path,
    output_dir: Path,
    metadata: VideoMetadata,
    scene: SceneGeometry,
    tracks: Sequence[TrackObservation],
    assignments: Sequence[LaneAssignment],
    events: Sequence[LaneChangeEvent],
    detector_device: str,
    warnings: tuple[str, ...],
    manifest: CacheManifest,
    options: RunOptions,
    dependencies: PipelineDependencies,
    meter: _PerformanceMeter,
) -> RunSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{metadata.path.stem}.transaction-", dir=output_dir)
    )
    preserve_staging = False
    try:
        staged_video, staged_events, staged_run = _output_paths(
            metadata, staging_dir, geometry_only=False
        )
        dependencies.renderer(
            input_path,
            staged_video,
            scene,
            tracks,
            assignments,
            events,
            overwrite=False,
        )
        meter.sample()
        dependencies.event_writer(metadata, events, staging_dir)
        meter.sample()
        provenance = _run_provenance(manifest)
        thresholds = RunThresholds(**asdict(dependencies.config))
        meter.sample()
        persisted_summary = RunSummary(
            input_basename=metadata.basename,
            device=detector_device,
            warnings=warnings,
            events=tuple(events),
            elapsed_s=meter.elapsed_s,
            peak_rss_bytes=meter.peak_rss_bytes,
            status="success",
            provenance=provenance,
            thresholds=thresholds,
        )
        dependencies.summary_writer(persisted_summary, staging_dir)
        staged_paths = (staged_video, staged_events, staged_run)
        if not all(path.is_file() for path in staged_paths):
            raise OSError("artifact staging did not produce a complete output set")
        final_paths = _output_paths(metadata, output_dir, geometry_only=False)
        try:
            _publish_artifact_set(staged_paths, final_paths, staging_dir, options.overwrite)
        except ArtifactTransactionError:
            preserve_staging = True
            raise
        meter.sample()
        return replace(
            persisted_summary,
            elapsed_s=meter.elapsed_s,
            peak_rss_bytes=meter.peak_rss_bytes,
        )
    finally:
        if not preserve_staging:
            shutil.rmtree(staging_dir)


def _publish_artifact_set(
    staged_paths: Sequence[Path],
    final_paths: Sequence[Path],
    staging_dir: Path,
    overwrite: bool,
) -> None:
    if not overwrite:
        published: list[Path] = []
        try:
            for staged, final in zip(staged_paths, final_paths, strict=True):
                os.link(staged, final)
                published.append(final)
        except BaseException as publication_error:
            cleanup_errors: list[tuple[Path, BaseException]] = []
            for final in reversed(published):
                try:
                    final.unlink(missing_ok=True)
                except BaseException as cleanup_error:  # noqa: BLE001 - continue recovery.
                    cleanup_errors.append((final, cleanup_error))
            if cleanup_errors:
                raise ArtifactTransactionError(
                    f"no-overwrite artifact publication failed; recovery preserved at {staging_dir}",
                    publication_error=publication_error,
                    recovery_errors=cleanup_errors,
                ) from publication_error
            raise
        return

    backups: list[tuple[Path, Path]] = []
    published = []
    try:
        for index, final in enumerate(final_paths):
            if final.exists():
                backup = staging_dir / f".backup-{index}-{final.name}"
                os.replace(final, backup)
                backups.append((final, backup))
        for staged, final in zip(staged_paths, final_paths, strict=True):
            os.replace(staged, final)
            published.append(final)
    except BaseException as publication_error:
        try:
            for final in reversed(published):
                final.unlink(missing_ok=True)
            for final, backup in reversed(backups):
                os.replace(backup, final)
        except BaseException as restore_error:  # noqa: BLE001 - preserve recovery state.
            raise ArtifactTransactionError(
                f"artifact publication failed and rollback could not restore {staging_dir}",
                publication_error=publication_error,
                recovery_errors=((staging_dir, restore_error),),
            ) from publication_error
        raise


def _collect_background_frames(
    metadata: VideoMetadata,
    dependencies: PipelineDependencies,
    config: PipelineConfig,
) -> tuple[np.ndarray, ...]:
    selected_count = min(config.geometry_sample_count, metadata.frame_count)
    wanted = {
        int(index)
        for index in np.linspace(0, metadata.frame_count - 1, num=selected_count, dtype=int)
    }
    samples: list[np.ndarray] = []
    try:
        frames = dependencies.frame_iterator(metadata.path)
        try:
            for frame_index, _timestamp_s, frame in frames:
                if frame_index in wanted:
                    samples.append(frame)
        finally:
            close = getattr(frames, "close", None)
            if callable(close):
                close()
    except (OSError, ValueError) as error:
        raise VideoInputError(
            f"input video decode failed for {metadata.path}: {error}"
        ) from error
    return tuple(samples)


def _create_tracks(
    input_path: Path,
    metadata: VideoMetadata,
    options: RunOptions,
    dependencies: PipelineDependencies,
    meter: _PerformanceMeter,
) -> tuple[tuple[TrackObservation, ...], str, tuple[str, ...]]:
    detector: Detector | None = None
    warnings: list[str] = []
    try:
        detector = dependencies.detector_factory(dependencies.config, options.device)
        for attempt in range(2):
            tracker: Tracker | None = None
            frames: Iterator[tuple[int, float, np.ndarray]] | None = None
            observations: list[TrackObservation] = []
            try:
                tracker = dependencies.tracker_factory(metadata.fps, dependencies.config.max_gap_s)
                frames = dependencies.frame_iterator(input_path)
                for frame_index, timestamp_s, frame in frames:
                    detections = detector.detect(frame_index, timestamp_s, frame)
                    observations.extend(tracker.update(detections, frame))
                    if frame_index % 30 == 0:
                        meter.sample()
            except DetectorRestartRequired as error:
                if options.device is not DeviceRequest.AUTO or attempt != 0:
                    raise
                warnings.append(str(error))
                continue
            finally:
                if frames is not None:
                    close = getattr(frames, "close", None)
                    if callable(close):
                        close()
                _release_component(tracker)
            observations.sort(
                key=lambda observation: (
                    observation.frame_index,
                    observation.track_id,
                    observation.box.x1,
                    observation.box.y1,
                    observation.class_name,
                )
            )
            return tuple(observations), str(detector.device), tuple(warnings)
        raise AssertionError("detector restart loop exited without a completed pass")
    finally:
        _release_component(detector)


def _release_component(component: object | None) -> None:
    if component is None:
        return
    for method_name in ("close", "release"):
        method = getattr(component, method_name, None)
        if callable(method):
            method()
            break


def _cache_manifest(metadata: VideoMetadata, config: PipelineConfig) -> CacheManifest:
    try:
        weight_manifest = json.loads(_WEIGHT_MANIFEST_PATH.read_text(encoding="utf-8"))
        model_id = str(weight_manifest["model_id"])
        weight_sha256 = str(weight_manifest["sha256"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"invalid detector weight manifest: {_WEIGHT_MANIFEST_PATH}") from error
    config_json = json.dumps(
        asdict(config),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return CacheManifest(
        video_sha256=metadata.sha256,
        schema_version=PIPELINE_SCHEMA_VERSION,
        model_id=model_id,
        weight_sha256=weight_sha256,
        config_sha256=hashlib.sha256(config_json).hexdigest(),
        metadata=metadata,
    )


def _run_provenance(manifest: CacheManifest) -> RunProvenance:
    return RunProvenance(
        pipeline_schema_version=PIPELINE_SCHEMA_VERSION,
        package_version=_package_version("lane-change-counter", fallback="0.1.0"),
        model_id=manifest.model_id,
        weight_sha256=manifest.weight_sha256,
        config_sha256=manifest.config_sha256,
        python_version=platform.python_version(),
        numpy_version=_package_version("numpy"),
        opencv_version=_package_version("opencv-python-headless"),
        psutil_version=_package_version("psutil"),
        supervision_version=_package_version("supervision"),
        torch_version=_package_version("torch"),
        ultralytics_version=_package_version("ultralytics"),
    )


def _package_version(distribution: str, *, fallback: str = "unavailable") -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return fallback


def _output_paths(metadata: VideoMetadata, output_dir: Path, *, geometry_only: bool) -> tuple[Path, ...]:
    if geometry_only:
        return (output_dir / f"{metadata.path.stem}.geometry.png",)
    return (
        output_dir / f"annotated_{metadata.basename}",
        output_dir / f"{metadata.path.stem}.events.json",
        output_dir / f"{metadata.path.stem}.run.json",
    )


def _refuse_existing_outputs(
    metadata: VideoMetadata,
    output_dir: Path,
    options: RunOptions,
    *,
    include_answer: bool,
) -> None:
    if options.overwrite:
        return
    candidates = list(_output_paths(metadata, output_dir, geometry_only=options.geometry_only))
    if include_answer and not options.geometry_only:
        candidates.append(output_dir / "answer.json")
    for path in candidates:
        if path.exists():
            raise FileExistsError(path)


def _write_geometry_diagnostic(
    scene: SceneGeometry,
    background: np.ndarray,
    output_dir: Path,
    metadata: VideoMetadata,
    options: RunOptions,
) -> Path:
    target = output_dir / f"{metadata.path.stem}.geometry.png"
    partial = target.with_name(f".{target.name}.partial.png")
    output_dir.mkdir(parents=True, exist_ok=True)
    if target.exists() and not options.overwrite:
        raise FileExistsError(target)
    if partial.exists():
        raise FileExistsError(partial)
    rendered = render_geometry_diagnostic(scene, background)
    try:
        if not cv2.imwrite(str(partial), rendered):
            raise OSError(f"could not write geometry diagnostic: {partial}")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target
