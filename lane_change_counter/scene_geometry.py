from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from math import atan2, degrees, hypot

import cv2
import numpy as np

from lane_change_counter.config import PipelineConfig
from lane_change_counter.models import (
    Lane,
    SceneBoundary,
    SceneGeometry,
    SceneRegion,
    TrackObservation,
    VideoMetadata,
)

_MIN_BACKGROUND_SAMPLES = 11
_MEDIAN_MAX_DIMENSION = 640
_DIRECTION_TOLERANCE_DEG = PipelineConfig().direction_tolerance_deg
_CARRIAGEWAY_DIRECTION_TOLERANCE_DEG = _DIRECTION_TOLERANCE_DEG + 5.0
_MIN_DIRECTION_AGREEMENT_RATIO = 0.90
_PERSISTENT_STATIC_TRACK_OBSERVATIONS = 24


@dataclass(frozen=True, slots=True)
class _Segment:
    start: tuple[float, float]
    end: tuple[float, float]
    line: tuple[float, float, float]
    length: float
    angle_deg: float


@dataclass(frozen=True, slots=True)
class _Boundary:
    bottom_x: float
    coverage: float
    transitions: int
    support: float

    @property
    def is_dashed(self) -> bool:
        return 0.16 <= self.coverage <= 0.72 and self.transitions >= 3


@dataclass(frozen=True, slots=True)
class _DashComponent:
    x: float
    y: float
    height: float
    slope: float


def build_background(frames: Iterable[np.ndarray], sample_count: int) -> np.ndarray:
    """Build a median background from evenly spaced decoded frames."""
    if sample_count < _MIN_BACKGROUND_SAMPLES:
        raise ValueError("sample_count must be at least 11")

    decoded = list(frames)
    if len(decoded) < _MIN_BACKGROUND_SAMPLES:
        raise ValueError("at least 11 decoded frames are required")

    first = _validate_color_image(decoded[0])
    original_height, original_width = first.shape[:2]
    for frame in decoded[1:]:
        _validate_color_image(frame)
        if frame.shape != first.shape:
            raise ValueError("all decoded frames must have identical dimensions")

    selected_count = min(sample_count, len(decoded))
    indices = np.linspace(0, len(decoded) - 1, num=selected_count, dtype=int)
    scale = min(1.0, _MEDIAN_MAX_DIMENSION / max(original_width, original_height))
    working_size = (
        max(1, round(original_width * scale)),
        max(1, round(original_height * scale)),
    )
    samples = [
        frame
        if working_size == (original_width, original_height)
        else cv2.resize(frame, working_size, interpolation=cv2.INTER_AREA)
        for frame in (decoded[index] for index in indices)
    ]
    median = np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)
    if median.shape[:2] != (original_height, original_width):
        median = cv2.resize(
            median, (original_width, original_height), interpolation=cv2.INTER_LINEAR
        )
    return median


def estimate_geometry(background: np.ndarray, metadata: VideoMetadata) -> SceneGeometry:
    """Estimate main-lane polygons and divergent exclusions from image evidence."""
    image = _validate_color_image(background)
    height, width = image.shape[:2]
    frame_size = (width, height)
    diagnostics: list[str] = []
    if frame_size != (metadata.width, metadata.height):
        diagnostics.append("background dimensions differ from video metadata")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    diagonal = hypot(width, height)
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(18, round(min(width, height) * 0.055)),
        minLineLength=max(10, round(diagonal * 0.025)),
        maxLineGap=max(6, round(diagonal * 0.018)),
    )
    segments = _segments_from_hough(raw_lines)
    perspective_segments = [
        segment for segment in segments if 18.0 <= _undirected_angle(segment.angle_deg) <= 162.0
    ]
    vanishing_point, supported = _dominant_vanishing_point(perspective_segments, width, height)
    if vanishing_point is None or len(supported) < 3:
        curved = _estimate_curved_lane_pair(image, gray, frame_size, perspective_segments)
        if curved is not None:
            return curved
        diagnostics.append("insufficient persistent lane evidence")
        return SceneGeometry(frame_size, (), (), 0.0, tuple(diagnostics))

    if vanishing_point[1] > height * 0.40:
        curved = _estimate_curved_lane_pair(image, gray, frame_size, perspective_segments)
        if curved is not None:
            return curved

    boundaries = _extract_boundaries(
        supported,
        vanishing_point,
        gray,
        width,
        height,
    )
    if len(boundaries) < 3:
        curved = _estimate_curved_lane_pair(image, gray, frame_size, perspective_segments)
        if curved is not None:
            return curved
        diagnostics.append("insufficient persistent lane evidence")
        confidence = min(0.45, 0.12 * len(boundaries))
        return SceneGeometry(frame_size, (), (), confidence, tuple(diagnostics))

    dashed_indices = [index for index, boundary in enumerate(boundaries) if boundary.is_dashed]
    usable_ranges = [
        (index - 1, index, index + 1)
        for index in dashed_indices
        if index > 0 and index + 1 < len(boundaries)
    ]
    if not usable_ranges:
        curved = _estimate_curved_lane_pair(image, gray, frame_size, perspective_segments)
        if curved is not None:
            return curved
        diagnostics.extend(
            ("no shared dashed lane boundary detected", "insufficient persistent lane evidence")
        )
        return SceneGeometry(frame_size, (), (), 0.4, tuple(diagnostics))

    # Prefer the dashed boundary with the strongest neighboring boundary support.
    left_index, _shared_index, right_index = max(
        usable_ranges,
        key=lambda indexes: sum(boundaries[index].support for index in indexes),
    )
    selected = boundaries[left_index : right_index + 1]
    lanes = _lanes_from_boundaries(selected, vanishing_point, width, height)
    exclusions = _geometry_exclusions(
        perspective_segments,
        supported,
        vanishing_point,
        boundaries,
        selected,
        width,
        height,
    )

    support_ratio = sum(segment.length for segment in supported) / max(
        1.0, sum(segment.length for segment in perspective_segments)
    )
    boundary_quality = float(np.mean([min(1.0, boundary.coverage / 0.45) for boundary in selected]))
    confidence = min(0.95, 0.70 + 0.12 * support_ratio + 0.10 * boundary_quality)
    if frame_size != (metadata.width, metadata.height):
        confidence = min(confidence, 0.69)

    diagnostics.extend(
        (
            f"detected {len(selected)} persistent converging boundaries",
            "detected 1 dashed shared boundary",
            f"excluded {len(exclusions)} non-lane region(s) from geometric evidence",
        )
    )
    return SceneGeometry(frame_size, lanes, exclusions, confidence, tuple(diagnostics))


def _estimate_curved_lane_pair(
    image: np.ndarray,
    gray: np.ndarray,
    frame_size: tuple[int, int],
    segments: Sequence[_Segment],
) -> SceneGeometry | None:
    """Recover curved road corridors without treating every detected line as a lane."""
    width, height = frame_size
    components = _dash_components(gray)
    path = _best_dashed_path(components, width, height)
    if path is None:
        return None

    path_y = np.asarray([component.y for component in path], dtype=float)
    path_x = np.asarray([component.x for component in path], dtype=float)
    degree = min(3, len(path) - 1)
    shared_coefficients = np.polyfit(path_y, path_x, degree)
    top_y = 0.0
    bottom_y = float(height - 1)
    sample_y = np.linspace(top_y, bottom_y, num=24)
    detected_shared_x = np.polyval(shared_coefficients, sample_y)
    shared_x = detected_shared_x.copy()
    leading_count = min(4, len(path))
    if leading_count >= 2:
        leading_coefficients = np.polyfit(
            path_y[:leading_count], path_x[:leading_count], 1
        )
        before_first_dash = sample_y < path_y[0]
        shared_x[before_first_dash] = np.polyval(
            leading_coefficients, sample_y[before_first_dash]
        )

    shared_curve = tuple((float(x), float(y)) for x, y in zip(shared_x, sample_y, strict=True))
    boundaries = _detect_ground_boundaries(
        detected_shared_x, sample_y, _longitudinal_boundary_score(gray)
    )
    if len(boundaries) < 2:
        return None
    far_field_correction = shared_x - detected_shared_x
    boundaries = tuple(
        (boundary_x + far_field_correction, quality)
        for boundary_x, quality in boundaries
    )
    ordered = tuple(sorted(boundaries, key=lambda boundary: float(np.median(boundary[0]))))
    boundary_x = tuple(points for points, _quality in ordered)
    shared_index = min(
        range(len(boundary_x)),
        key=lambda index: float(np.mean(np.abs(boundary_x[index] - shared_x))),
    )
    gap_evidence = _classify_boundary_gaps(
        image,
        gray,
        segments,
        boundary_x,
        sample_y,
        _curve_direction(shared_curve),
        shared_index,
    )
    grid_indexes = tuple(
        index for index, evidence in enumerate(gap_evidence) if evidence["kind"] == "grid"
    )
    evidence_road_indexes = tuple(
        index for index, evidence in enumerate(gap_evidence) if evidence["kind"] == "road"
    )
    grid_index = grid_indexes[0] if grid_indexes else len(boundary_x) - 1
    road_indexes = tuple(index for index in evidence_road_indexes if index < grid_index)
    if len(road_indexes) < 2:
        return None

    region_result = _nonroad_regions_and_branch_lane(
        gray,
        boundary_x[grid_index],
        boundary_x[min(len(boundary_x) - 1, grid_index + 1)],
        boundary_x[min(len(boundary_x) - 1, grid_index + 2)],
        sample_y,
    )
    if region_result is None:
        regions = tuple(
            SceneRegion(
                f"grid-{region_index + 1}",
                _gap_polygon(boundary_x[gap_index], boundary_x[gap_index + 1], sample_y),
                "grid-transition",
                float(gap_evidence[gap_index]["confidence"]),
            )
            for region_index, gap_index in enumerate(grid_indexes)
        )
        branch_polygon: tuple[tuple[float, float], ...] = ()
    else:
        regions, branch_polygon = region_result

    used_boundary_indexes = sorted(
        {
            boundary_index
            for gap_index in road_indexes
            for boundary_index in (gap_index, gap_index + 1)
        }
    )
    boundary_ids = {
        boundary_index: f"line-{display_index + 1}"
        for display_index, boundary_index in enumerate(used_boundary_indexes)
    }
    scene_boundaries = tuple(
        SceneBoundary(
            boundary_ids[index],
            tuple(
                (float(x), float(y))
                for x, y in zip(boundary_x[index], sample_y, strict=True)
            ),
            (
                "dashed-marking"
                if index == shared_index
                else "road-edge"
            ),
            float(ordered[index][1]),
        )
        for index in used_boundary_indexes
    )

    exclusions = tuple(region.polygon for region in regions)

    lane_specs: list[tuple[int, tuple[tuple[float, float], ...], float]] = []
    for gap_index in road_indexes:
        # A road gap before the first grid transition is a complete lane between
        # its two continuous boundaries. Far-field texture may be noisy, but it
        # must not shorten an otherwise supported lane corridor.
        left = boundary_x[gap_index]
        right = boundary_x[gap_index + 1]
        ys = sample_y
        polygon = _gap_polygon(left, right, ys)
        center_curve = tuple(
            (float((left_x + right_x) / 2.0), float(y))
            for left_x, right_x, y in zip(left, right, ys, strict=True)
        )
        lane_specs.append((gap_index, polygon, _curve_direction(center_curve)))

    lanes_list = [
        Lane(
            f"lane-{index + 1}",
            polygon,
            direction,
            (),
            (boundary_ids[gap_index], boundary_ids[gap_index + 1]),
        )
        for index, (gap_index, polygon, direction) in enumerate(lane_specs)
    ]
    gap_to_lane = {
        gap_index: lane_index for lane_index, (gap_index, *_rest) in enumerate(lane_specs)
    }
    for lane_index, (gap_index, *_rest) in enumerate(lane_specs):
        neighbors: list[str] = []
        if gap_index - 1 in gap_to_lane:
            neighbors.append(lanes_list[gap_to_lane[gap_index - 1]].lane_id)
        if gap_index + 1 in gap_to_lane:
            neighbors.append(lanes_list[gap_to_lane[gap_index + 1]].lane_id)
        lanes_list[lane_index] = replace(lanes_list[lane_index], neighbor_ids=tuple(neighbors))
    if branch_polygon:
        branch_points = np.asarray(branch_polygon, dtype=float)
        top_limit = float(np.percentile(branch_points[:, 1], 15.0))
        bottom_limit = float(np.percentile(branch_points[:, 1], 85.0))
        top_points = branch_points[branch_points[:, 1] <= top_limit]
        bottom_points = branch_points[branch_points[:, 1] >= bottom_limit]
        center_curve = (
            (float(np.mean(top_points[:, 0])), float(np.mean(top_points[:, 1]))),
            (float(np.mean(bottom_points[:, 0])), float(np.mean(bottom_points[:, 1]))),
        )
        lanes_list.append(
            Lane(
                f"lane-{len(lanes_list) + 1}",
                branch_polygon,
                _curve_direction(center_curve),
                (),
                (),
            )
        )
    lanes = tuple(lanes_list)

    span_ratio = (path_y.max() - path_y.min()) / max(1.0, height)
    size_correlation = float(
        np.corrcoef(path_y, np.asarray([component.height for component in path]))[0, 1]
    )
    confidence = min(
        0.93,
        0.72
        + 0.10 * min(1.0, span_ratio)
        + 0.04 * max(0.0, size_correlation)
        + 0.08 * min(
            1.0,
            float(np.mean([ordered[index][1] for index in used_boundary_indexes])),
        ),
    )
    diagnostics = (
        f"detected curved dashed boundary across {span_ratio:.0%} of frame height",
        f"detected {len(scene_boundaries)} continuous ground boundaries",
        f"formed {len(lanes)} travel lanes between adjacent markings",
        (
            f"detected {len(regions)} arbitrary-shape non-road region(s), "
            f"including {sum(region.kind == 'grid-transition' for region in regions)} grid area(s)"
        ),
        "road gaps were selected from smooth asphalt and non-road texture evidence",
    )
    return SceneGeometry(
        frame_size,
        lanes,
        exclusions,
        confidence,
        diagnostics,
        scene_boundaries,
        regions,
    )


def _nonroad_regions_and_branch_lane(
    gray: np.ndarray,
    road_edge_x: np.ndarray,
    grid_outer_hint_x: np.ndarray,
    nonroad_outer_hint_x: np.ndarray,
    sample_y: np.ndarray,
) -> tuple[tuple[SceneRegion, ...], tuple[tuple[float, float], ...]] | None:
    """Aggregate dense non-road line evidence and recover the asphalt gap between it."""
    height, width = gray.shape
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(enhanced, (5, 5), 0), 50, 150)
    diagonal = hypot(width, height)
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360.0,
        threshold=max(18, round(min(width, height) * 0.042)),
        minLineLength=max(10, round(diagonal * 0.032)),
        maxLineGap=max(6, round(diagonal * 0.016)),
    )
    segments = _segments_from_hough(raw_lines)
    if not segments:
        return None

    right_roi = np.zeros(gray.shape, dtype=np.uint8)
    roi_polygon = np.vstack(
        (
            np.column_stack((road_edge_x, sample_y)),
            np.asarray(((width - 1, height - 1), (width - 1, 0)), dtype=float),
        )
    )
    cv2.fillPoly(right_roi, [np.rint(roi_polygon).astype(np.int32)], 1)

    line_evidence = np.zeros(gray.shape, dtype=np.uint8)
    maximum_x_margin = width * 0.16
    for segment in segments:
        midpoint_x = (segment.start[0] + segment.end[0]) / 2.0
        midpoint_y = (segment.start[1] + segment.end[1]) / 2.0
        minimum_x = float(np.interp(midpoint_y, sample_y, road_edge_x))
        maximum_x = float(np.interp(midpoint_y, sample_y, nonroad_outer_hint_x))
        angle = _undirected_angle(segment.angle_deg)
        if not (
            segment.length >= diagonal * 0.04
            and minimum_x < midpoint_x < maximum_x + maximum_x_margin
            and 15.0 < angle < 165.0
        ):
            continue
        cv2.line(
            line_evidence,
            tuple(round(value) for value in segment.start),
            tuple(round(value) for value in segment.end),
            255,
            max(3, round(min(width, height) * 0.0045)),
            cv2.LINE_AA,
        )
    line_evidence[right_roi == 0] = 0

    dilation = max(9, round(min(width, height) * 0.014))
    closing = max(17, round(min(width, height) * 0.029))
    nonroad_mask = cv2.dilate(
        line_evidence, np.ones((dilation, dilation), dtype=np.uint8)
    )
    nonroad_mask = cv2.morphologyEx(
        nonroad_mask,
        cv2.MORPH_CLOSE,
        np.ones((closing, closing), dtype=np.uint8),
    )
    nonroad_mask = cv2.morphologyEx(
        nonroad_mask, cv2.MORPH_OPEN, np.ones((9, 9), dtype=np.uint8)
    )

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (nonroad_mask > 0).astype(np.uint8), 8
    )
    kept = np.zeros(gray.shape, dtype=np.uint8)
    kept_labels = [
        label
        for label in range(1, count)
        if stats[label, cv2.CC_STAT_AREA] >= width * height * 0.004
    ]
    if not kept_labels:
        return None
    for label in kept_labels:
        kept[labels == label] = 255

    grid_seed = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(
        grid_seed,
        [
            np.rint(
                np.asarray(_gap_polygon(road_edge_x, grid_outer_hint_x, sample_y))
            ).astype(np.int32)
        ],
        255,
    )
    grid_label = max(
        kept_labels,
        key=lambda label: int(np.count_nonzero((labels == label) & (grid_seed > 0))),
    )
    ordered_labels = (grid_label,) + tuple(
        label for label in kept_labels if label != grid_label
    )
    regions: list[SceneRegion] = []
    for region_index, label in enumerate(ordered_labels):
        component = ((labels == label).astype(np.uint8) * 255)
        contours, _hierarchy = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        epsilon = max(2.0, cv2.arcLength(contour, True) * 0.003)
        outline = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(outline) < 5:
            continue
        kind = "grid-transition" if label == grid_label else "non-road"
        regions.append(
            SceneRegion(
                f"region-{region_index + 1}",
                tuple((float(x), float(y)) for x, y in outline),
                kind,
                0.86 if label == grid_label else 0.78,
            )
        )

    road_mask = ((right_roi > 0) & (kept == 0)).astype(np.uint8)
    road_mask = cv2.morphologyEx(
        road_mask, cv2.MORPH_OPEN, np.ones((13, 13), dtype=np.uint8)
    )
    road_count, road_labels, road_stats, _road_centroids = cv2.connectedComponentsWithStats(
        road_mask, 8
    )
    road_choices = [
        label
        for label in range(1, road_count)
        if road_stats[label, cv2.CC_STAT_TOP] + road_stats[label, cv2.CC_STAT_HEIGHT]
        >= height - 2
        and road_stats[label, cv2.CC_STAT_AREA] >= width * height * 0.01
    ]
    if not road_choices:
        return None
    road_label = max(
        road_choices, key=lambda label: road_stats[label, cv2.CC_STAT_AREA]
    )
    road_component = ((road_labels == road_label).astype(np.uint8) * 255)
    road_contours, _hierarchy = cv2.findContours(
        road_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    road_contour = max(road_contours, key=cv2.contourArea)
    road_epsilon = max(2.0, cv2.arcLength(road_contour, True) * 0.002)
    road_outline = cv2.approxPolyDP(road_contour, road_epsilon, True).reshape(-1, 2)
    if len(road_outline) < 8:
        return None
    branch_polygon = tuple((float(x), float(y)) for x, y in road_outline)
    return tuple(regions), branch_polygon


def _gap_polygon(
    left_x: np.ndarray,
    right_x: np.ndarray,
    sample_y: np.ndarray,
) -> tuple[tuple[float, float], ...]:
    left = tuple((float(x), float(y)) for x, y in zip(left_x, sample_y, strict=True))
    right = tuple(
        (float(x), float(y))
        for x, y in zip(right_x[::-1], sample_y[::-1], strict=True)
    )
    return left + right


def _classify_boundary_gaps(
    image: np.ndarray,
    gray: np.ndarray,
    segments: Sequence[_Segment],
    boundary_x: Sequence[np.ndarray],
    sample_y: np.ndarray,
    main_direction: float,
    shared_index: int,
) -> tuple[dict[str, float | int | str], ...]:
    """Classify adjacent curves using positive asphalt and negative texture evidence."""
    height, width = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 90)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    local_background = cv2.GaussianBlur(gray, (0, 0), max(5.0, min(gray.shape) / 72.0))
    local_delta = gray.astype(np.int16) - local_background.astype(np.int16)
    white_marking = (
        (gray >= 135)
        & (local_delta >= 8)
        & (hsv[:, :, 1] <= 100)
    ).astype(np.uint8)
    white_marking = cv2.morphologyEx(
        white_marking, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
    )

    off_direction = np.zeros(gray.shape, dtype=np.uint8)
    maximum_length = hypot(width, height) * 0.13
    for segment in segments:
        if (
            segment.length <= maximum_length
            and _angle_distance(segment.angle_deg, main_direction) > 25.0
        ):
            cv2.line(
                off_direction,
                tuple(round(value) for value in segment.start),
                tuple(round(value) for value in segment.end),
                255,
                2,
                cv2.LINE_AA,
            )

    rows = np.indices(gray.shape)[0]
    metrics: list[dict[str, float | int | str]] = []
    for gap_index, (left, right) in enumerate(pairwise(boundary_x)):
        polygon = _gap_polygon(left, right, sample_y)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(np.asarray(polygon)).astype(np.int32)], 255)
        valid = mask > 0
        lower = valid & (rows >= height * 0.45)
        visible_bottom_width = max(
            0.0,
            min(float(width - 1), float(max(left[-1], right[-1])))
            - max(0.0, float(min(left[-1], right[-1]))),
        )
        metrics.append(
            {
                "kind": "unknown",
                "white": float(np.mean(white_marking[valid])) if np.any(valid) else 0.0,
                "edge": float(np.mean(edges[valid] > 0)) if np.any(valid) else 1.0,
                "lower_edge": (
                    float(np.mean(edges[lower] > 0)) if np.any(lower) else 1.0
                ),
                "off_direction": (
                    float(np.mean(off_direction[valid] > 0)) if np.any(valid) else 1.0
                ),
                "visible_bottom_width": visible_bottom_width,
                "area": float(np.count_nonzero(valid)),
                "confidence": 0.0,
            }
        )

    white_values = np.asarray([float(metric["white"]) for metric in metrics])
    edge_values = np.asarray([float(metric["edge"]) for metric in metrics])
    grid_score = white_values + edge_values * 0.8
    grid_index = int(np.argmax(grid_score))
    white_baseline = float(np.median(white_values))
    edge_baseline = float(np.median(edge_values))
    if (
        white_values[grid_index] >= max(0.12, white_baseline + 0.045)
        and edge_values[grid_index] >= edge_baseline + 0.008
    ):
        metrics[grid_index]["kind"] = "grid"
        metrics[grid_index]["confidence"] = min(
            0.98,
            0.62 + white_values[grid_index] + edge_values[grid_index],
        )

    for gap_index, metric in enumerate(metrics):
        if metric["kind"] == "grid":
            continue
        anchor_road = gap_index in (shared_index - 1, shared_index)
        smooth_road = (
            float(metric["lower_edge"]) < 0.05
            and float(metric["visible_bottom_width"]) >= width * 0.025
            and float(metric["area"]) >= width * height * 0.015
        )
        if not (anchor_road or smooth_road):
            metric["kind"] = "non-road"
            metric["confidence"] = min(
                0.95,
                0.55 + float(metric["edge"]) + float(metric["off_direction"]),
            )
            continue

        metric["kind"] = "road"
        metric["confidence"] = min(
            0.97,
            0.70 + max(0.0, 0.06 - float(metric["lower_edge"])) * 3.0,
        )
    return tuple(metrics)


def _dash_components(gray: np.ndarray) -> tuple[_DashComponent, ...]:
    height, width = gray.shape
    local_background = cv2.GaussianBlur(gray, (0, 0), max(3.0, min(width, height) / 72.0))
    contrast = cv2.subtract(gray, local_background)
    mask = (contrast >= 12).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    components: list[_DashComponent] = []
    minimum_area = max(8, round(width * height * 0.00001))
    maximum_area = width * height * 0.003
    for label in range(1, count):
        _x, _y, component_width, component_height, area = stats[label]
        center_x, center_y = centroids[label]
        if not (
            minimum_area <= area <= maximum_area
            and height * 0.007 <= component_height <= height * 0.21
            and component_height >= component_width * 1.05
            and height * 0.04 <= center_y <= height * 0.99
        ):
            continue

        points_y, points_x = np.nonzero(labels == label)
        points = np.column_stack((points_x, points_y)).astype(float)
        covariance = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major = eigenvectors[:, int(np.argmax(eigenvalues))]
        elongation = float(np.max(eigenvalues) / max(1e-6, np.min(eigenvalues)))
        if abs(major[1]) < 0.60 or elongation < 2.0:
            continue
        components.append(
            _DashComponent(
                float(center_x),
                float(center_y),
                float(component_height),
                float(major[0] / major[1]),
            )
        )
    return tuple(sorted(components, key=lambda component: (component.y, component.x)))


def _best_dashed_path(
    components: Sequence[_DashComponent], width: int, height: int
) -> tuple[_DashComponent, ...] | None:
    if len(components) < 5:
        return None

    paths: list[tuple[float, tuple[int, ...]]] = []
    for index, component in enumerate(components):
        best_score = 0.0
        best_path = (index,)
        for previous_index, previous in enumerate(components[:index]):
            delta_y = component.y - previous.y
            if not height * 0.018 <= delta_y <= height * 0.30:
                continue
            connection_slope = (component.x - previous.x) / delta_y
            if (
                abs(connection_slope) > 1.10
                or abs(connection_slope - previous.slope) > 0.75
                or abs(connection_slope - component.slope) > 0.75
                or component.height < previous.height * 0.55
            ):
                continue
            previous_score, previous_path = paths[previous_index]
            direction_mismatch = abs(
                connection_slope - (previous.slope + component.slope) / 2.0
            )
            shrink_penalty = max(0.0, previous.height - component.height) / previous.height
            score = (
                previous_score
                + delta_y / height * 8.0
                + min(1.0, component.height / (height * 0.08))
                - direction_mismatch * 0.4
                - shrink_penalty * 2.0
            )
            if score > best_score:
                best_score = score
                best_path = previous_path + (index,)
        paths.append((best_score, best_path))

    candidates: list[tuple[float, tuple[_DashComponent, ...]]] = []
    for _path_score, indexes in paths:
        path = tuple(components[index] for index in indexes)
        if len(path) < 5:
            continue
        y_values = np.asarray([component.y for component in path], dtype=float)
        heights = np.asarray([component.height for component in path], dtype=float)
        span = float(y_values[-1] - y_values[0])
        coverage = float(np.sum(heights) / max(1.0, span))
        gaps = np.diff(y_values)
        if (
            span < height * 0.55
            or coverage > 0.65
            or np.count_nonzero(gaps > height * 0.045) < 3
        ):
            continue
        correlation = float(np.corrcoef(y_values, heights)[0, 1])
        if not np.isfinite(correlation):
            continue
        degree = min(3, len(path) - 1)
        coefficients = np.polyfit(
            y_values,
            np.asarray([component.x for component in path], dtype=float),
            degree,
        )
        residual = np.asarray([component.x for component in path]) - np.polyval(
            coefficients, y_values
        )
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        score = (
            span / height * 10.0
            + len(path)
            - coverage * 2.0
            + correlation * 8.0
            - rmse / width * 180.0
        )
        candidates.append((score, path))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _longitudinal_boundary_score(gray: np.ndarray) -> np.ndarray:
    horizontal_gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    local_contrast = cv2.subtract(
        gray, cv2.GaussianBlur(gray, (0, 0), max(3.0, min(gray.shape) / 72.0))
    ).astype(np.float32)
    score = cv2.GaussianBlur(horizontal_gradient + local_contrast * 2.0, (0, 0), 3.0)
    scale = max(1.0, float(np.percentile(score, 99.0)))
    return score / scale


def _search_parallel_boundary(
    shared_x: np.ndarray,
    sample_y: np.ndarray,
    score: np.ndarray,
    *,
    side: int,
) -> tuple[np.ndarray | None, float]:
    height, width = score.shape
    progress = (sample_y - sample_y[0]) / max(1.0, sample_y[-1] - sample_y[0])
    best_quality = -1.0
    best_x: np.ndarray | None = None
    for top_fraction in np.linspace(0.01, 0.12, num=12):
        for bottom_fraction in np.linspace(0.10, 0.38, num=29):
            if bottom_fraction < top_fraction * 1.5:
                continue
            for power in (0.7, 0.9, 1.1, 1.3, 1.5):
                offset = width * (
                    top_fraction
                    + (bottom_fraction - top_fraction) * np.power(progress, power)
                )
                candidate_x = np.rint(shared_x + side * offset).astype(int)
                candidate_y = np.rint(sample_y).astype(int)
                valid = (
                    (candidate_x >= 0)
                    & (candidate_x < width)
                    & (candidate_y >= 0)
                    & (candidate_y < height)
                )
                if float(np.mean(valid)) < 0.65:
                    continue
                values = score[candidate_y[valid], candidate_x[valid]]
                quality = float(np.mean(np.clip(values, 0.0, 2.0))) + 0.30 * float(
                    np.percentile(values, 70.0)
                )
                if quality > best_quality:
                    best_quality = quality
                    best_x = shared_x + side * offset
    return best_x, best_quality


def _snap_boundary_to_markings(
    candidate_x: np.ndarray,
    sample_y: np.ndarray,
    score: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Snap a smooth boundary proposal onto the strongest continuous marking path."""
    height, width = score.shape
    radius = max(4, round(width * 0.025))
    offsets = np.arange(-radius, radius + 1, 2, dtype=int)
    sampled_y = np.clip(np.rint(sample_y).astype(int), 0, height - 1)
    sampled_x = np.rint(candidate_x[:, None] + offsets[None, :]).astype(int)
    valid = (sampled_x >= 0) & (sampled_x < width)
    evidence = np.full(sampled_x.shape, -4.0, dtype=float)
    rows = np.broadcast_to(sampled_y[:, None], sampled_x.shape)
    evidence[valid] = score[rows[valid], sampled_x[valid]]

    cumulative = evidence.copy()
    parents = np.zeros(sampled_x.shape, dtype=int)
    for row in range(1, len(sample_y)):
        for column, offset in enumerate(offsets):
            transition = cumulative[row - 1] - 0.035 * np.abs(offsets - offset)
            parent = int(np.argmax(transition))
            cumulative[row, column] += transition[parent]
            parents[row, column] = parent

    indexes = np.empty(len(sample_y), dtype=int)
    indexes[-1] = int(np.argmax(cumulative[-1]))
    for row in range(len(sample_y) - 1, 0, -1):
        indexes[row - 1] = parents[row, indexes[row]]
    snapped_x = candidate_x + offsets[indexes]
    selected_evidence = evidence[np.arange(len(sample_y)), indexes]
    observed = selected_evidence > 0.05
    if np.count_nonzero(observed) >= 4:
        degree = min(3, int(np.count_nonzero(observed)) - 1)
        snapped_x = np.polyval(
            np.polyfit(
                sample_y[observed],
                snapped_x[observed],
                degree,
                w=np.clip(selected_evidence[observed], 0.05, None),
            ),
            sample_y,
        )
    marking_evidence = float(np.mean(selected_evidence[observed])) if np.any(observed) else 0.0
    return snapped_x, marking_evidence


def _detect_ground_boundaries(
    shared_x: np.ndarray,
    sample_y: np.ndarray,
    score: np.ndarray,
) -> tuple[tuple[np.ndarray, float], ...]:
    """Discover an ordered set of road markings without assuming a lane count."""
    boundaries: list[tuple[np.ndarray, float]] = [(shared_x, 1.0)]
    for side in (-1, 1):
        available = score.copy()
        reference = shared_x
        for _ in range(8):  # Safety limit only; evidence determines the count.
            proposal, coarse_quality = _search_parallel_boundary(
                reference, sample_y, available, side=side
            )
            if proposal is None or coarse_quality < 0.20:
                break
            snapped, marking_quality = _snap_boundary_to_markings(proposal, sample_y, score)
            separation = float(np.median(np.abs(snapped - reference)))
            if marking_quality < 0.20 or separation < score.shape[1] * 0.025:
                break
            boundaries.append((snapped, marking_quality))
            _suppress_boundary_evidence(available, snapped, sample_y)
            reference = snapped
    return tuple(boundaries)


def _suppress_boundary_evidence(
    score: np.ndarray, boundary_x: np.ndarray, sample_y: np.ndarray
) -> None:
    """Prevent a second search from rediscovering the same physical marking."""
    mask = np.zeros(score.shape, dtype=np.uint8)
    points = np.column_stack((np.rint(boundary_x), np.rint(sample_y))).astype(np.int32)
    cv2.polylines(mask, [points], False, 255, max(5, round(score.shape[1] * 0.018)))
    score[mask > 0] = 0.0


def _curve_direction(curve: Sequence[tuple[float, float]]) -> float:
    start, end = curve[0], curve[-1]
    return degrees(atan2(start[1] - end[1], start[0] - end[0]))


def validate_geometry(
    scene: SceneGeometry, observations: Sequence[TrackObservation]
) -> SceneGeometry:
    """Return a validated copy whose confidence reflects independent track evidence."""
    diagnostics = list(scene.diagnostics)
    confident = [observation for observation in observations if observation.score >= 0.5]
    failed = scene.confidence < 0.70
    if failed:
        diagnostics.append("candidate geometry confidence below 70%")

    relevant_observations = _main_carriageway_observations(confident, scene)
    valid_observations = [
        observation
        for observation in relevant_observations
        if _point_in_valid_lane(observation.box.bottom_center, scene)
    ]
    coverage = (
        len(valid_observations) / len(relevant_observations)
        if relevant_observations
        else 0.0
    )
    if coverage < 0.70:
        diagnostics.append("track coverage below 70%")
        failed = True

    validated_lanes, direction_status = _infer_lane_directions(
        valid_observations, scene.lanes, scene.frame_size
    )
    if direction_status == "inconsistent":
        diagnostics.append("inconsistent track travel direction")
        failed = True
    elif direction_status == "insufficient":
        diagnostics.append("insufficient directional track evidence")
        failed = True

    if not confident:
        diagnostics.append("insufficient confident track observations")

    confidence = scene.confidence
    if failed or not confident:
        confidence = min(confidence * 0.65, 0.69)
    else:
        confidence = min(0.99, confidence + 0.04 * coverage)
        diagnostics.append(f"track coverage validated at {coverage:.0%}")
    return replace(
        scene,
        lanes=validated_lanes,
        confidence=confidence,
        diagnostics=tuple(diagnostics),
    )


def render_geometry_diagnostic(scene: SceneGeometry, background: np.ndarray) -> np.ndarray:
    """Render lane, adjacency, exclusion, confidence, and diagnostic evidence."""
    source = _validate_color_image(background)
    if (source.shape[1], source.shape[0]) != scene.frame_size:
        raise ValueError("background dimensions do not match scene geometry")

    rendered = source.copy()
    overlay = rendered.copy()
    lane_colors = ((70, 210, 90), (60, 180, 240), (220, 170, 60))
    for index, lane in enumerate(scene.lanes):
        polygon = np.rint(np.asarray(lane.polygon, dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(overlay, [polygon], lane_colors[index % len(lane_colors)], cv2.LINE_AA)
    region_polygons = {region.polygon for region in scene.regions}
    for exclusion in scene.exclusions:
        if exclusion in region_polygons:
            continue
        polygon = np.rint(np.asarray(exclusion, dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(overlay, [polygon], (40, 40, 230), cv2.LINE_AA)
    for region in scene.regions:
        polygon = np.rint(np.asarray(region.polygon, dtype=np.float32)).astype(np.int32)
        color = (30, 110, 235) if region.kind == "grid-transition" else (40, 40, 230)
        cv2.fillPoly(overlay, [polygon], color, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.28, rendered, 0.72, 0.0, rendered)

    scale = max(0.45, min(scene.frame_size) / 720.0)
    thickness = max(1, round(scale * 2))
    for lane in scene.lanes:
        polygon = np.rint(np.asarray(lane.polygon, dtype=np.float32)).astype(np.int32)
        cv2.polylines(rendered, [polygon], True, (80, 255, 100), thickness, cv2.LINE_AA)
        center = tuple(np.rint(np.mean(polygon, axis=0)).astype(int))
        cv2.putText(
            rendered,
            lane.lane_id,
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    for exclusion in scene.exclusions:
        if exclusion in region_polygons:
            continue
        polygon = np.rint(np.asarray(exclusion, dtype=np.float32)).astype(np.int32)
        cv2.polylines(rendered, [polygon], True, (40, 40, 255), thickness + 1, cv2.LINE_AA)
        center = tuple(np.rint(np.mean(polygon, axis=0)).astype(int))
        cv2.putText(
            rendered,
            "EXCLUDED",
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    for region in scene.regions:
        polygon = np.rint(np.asarray(region.polygon, dtype=np.float32)).astype(np.int32)
        cv2.polylines(rendered, [polygon], True, (40, 160, 255), thickness + 2, cv2.LINE_AA)
        center = tuple(np.rint(np.mean(polygon, axis=0)).astype(int))
        label = "GRID AREA" if region.kind == "grid-transition" else region.kind.upper()
        (label_width, label_height), _baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        label_origin = (
            int(np.clip(center[0] - label_width / 2, 5, scene.frame_size[0] - label_width - 5)),
            int(np.clip(center[1] + label_height / 2, label_height + 5, scene.frame_size[1] - 5)),
        )
        cv2.putText(
            rendered,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    for boundary in scene.boundaries:
        points = np.rint(np.asarray(boundary.points, dtype=np.float32)).astype(np.int32)
        cv2.polylines(rendered, [points], False, (30, 255, 255), thickness + 1, cv2.LINE_AA)
        label_index = min(len(points) - 1, max(0, round(len(points) * 0.58)))
        label_point = tuple(points[label_index])
        cv2.putText(
            rendered,
            boundary.boundary_id,
            label_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.8,
            (20, 20, 20),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            boundary.boundary_id,
            label_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.8,
            (40, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    confidence_label = f"geometry confidence {scene.confidence:.2f}"
    (confidence_width, confidence_height), _baseline = cv2.getTextSize(
        confidence_label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    confidence_origin = (
        max(5, scene.frame_size[0] - confidence_width - round(10 * scale)),
        max(confidence_height + 5, round(30 * scale)),
    )
    cv2.putText(
        rendered,
        confidence_label,
        confidence_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        rendered,
        confidence_label,
        confidence_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return rendered


def _validate_color_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("frame must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("frame must be a non-empty BGR image")
    if image.dtype != np.uint8:
        raise ValueError("frame must use uint8 pixels")
    return image


def _segments_from_hough(lines: np.ndarray | None) -> list[_Segment]:
    if lines is None:
        return []
    segments: list[_Segment] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = hypot(dx, dy)
        if length == 0.0:
            continue
        a, b = dy / length, -dx / length
        c = -(a * float(x1) + b * float(y1))
        segments.append(
            _Segment(
                (float(x1), float(y1)),
                (float(x2), float(y2)),
                (a, b, c),
                length,
                degrees(atan2(dy, dx)),
            )
        )
    return segments


def _dominant_vanishing_point(
    segments: Sequence[_Segment], width: int, height: int
) -> tuple[tuple[float, float] | None, tuple[_Segment, ...]]:
    candidates: list[tuple[float, float]] = []
    for first_index, first in enumerate(segments):
        for second in segments[first_index + 1 :]:
            if _angle_distance(first.angle_deg, second.angle_deg) < 10.0:
                continue
            intersection = _line_intersection(first.line, second.line)
            if intersection is None:
                continue
            x, y = intersection
            if -0.25 * width <= x <= 1.25 * width and -0.35 * height <= y <= 0.70 * height:
                candidates.append(intersection)
    if not candidates:
        return None, ()

    tolerance = max(7.0, min(width, height) * 0.025)
    best_point: tuple[float, float] | None = None
    best_supported: list[_Segment] = []
    best_score = -1.0
    length_cap = hypot(width, height) * 0.28
    for candidate in candidates:
        supported = [
            segment
            for segment in segments
            if _point_line_distance(candidate, segment.line) <= tolerance
        ]
        score = sum(min(segment.length, length_cap) for segment in supported)
        if score > best_score:
            best_point, best_supported, best_score = candidate, supported, score
    if best_point is None or len(best_supported) < 2:
        return None, ()

    coefficients = np.asarray([segment.line[:2] for segment in best_supported], dtype=float)
    constants = -np.asarray([segment.line[2] for segment in best_supported], dtype=float)
    refined, _, rank, _ = np.linalg.lstsq(coefficients, constants, rcond=None)
    point = (float(refined[0]), float(refined[1])) if rank == 2 else best_point
    supported = tuple(
        segment
        for segment in segments
        if _point_line_distance(point, segment.line) <= tolerance * 1.35
    )
    return point, supported


def _extract_boundaries(
    supported: Sequence[_Segment],
    vanishing_point: tuple[float, float],
    gray: np.ndarray,
    width: int,
    height: int,
) -> tuple[_Boundary, ...]:
    bottom_y = float(height - 1)
    projected: list[tuple[float, _Segment]] = []
    for segment in supported:
        x = _line_x_at_y(segment.line, bottom_y)
        if x is not None and -0.15 * width <= x <= 1.15 * width:
            projected.append((x, segment))
    projected.sort(key=lambda item: item[0])

    groups: list[list[tuple[float, _Segment]]] = []
    cluster_distance = max(8.0, width * 0.035)
    for item in projected:
        if (
            not groups
            or abs(item[0] - np.median([value for value, _ in groups[-1]])) > cluster_distance
        ):
            groups.append([item])
        else:
            groups[-1].append(item)

    threshold = max(90.0, float(cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[0]) + 18.0)
    candidates: list[_Boundary] = []
    for group in groups:
        weights = np.asarray([segment.length for _, segment in group], dtype=float)
        values = np.asarray([value for value, _ in group], dtype=float)
        bottom_x = float(np.average(values, weights=weights))
        coverage, transitions = _boundary_coverage(gray, vanishing_point, bottom_x, threshold)
        support = float(np.sum(np.minimum(weights, hypot(width, height) * 0.25)))
        if coverage >= 0.12 and support >= min(width, height) * 0.07:
            candidates.append(_Boundary(bottom_x, coverage, transitions, support))

    candidates.sort(key=lambda boundary: boundary.bottom_x)
    separated: list[_Boundary] = []
    minimum_lane_width = width * 0.11
    for candidate in candidates:
        if not separated or candidate.bottom_x - separated[-1].bottom_x >= minimum_lane_width:
            separated.append(candidate)
        elif candidate.support > separated[-1].support:
            separated[-1] = candidate
    return tuple(separated)


def _boundary_coverage(
    gray: np.ndarray,
    vanishing_point: tuple[float, float],
    bottom_x: float,
    threshold: float,
) -> tuple[float, int]:
    height, width = gray.shape
    start_y = int(np.clip(vanishing_point[1] + height * 0.06, 0, height - 2))
    end_y = max(start_y + 1, height - 2)
    ys = np.arange(start_y, end_y + 1)
    denominator = max(1.0, (height - 1) - vanishing_point[1])
    xs = vanishing_point[0] + (bottom_x - vanishing_point[0]) * (
        (ys - vanishing_point[1]) / denominator
    )
    radius = max(2, round(min(width, height) * 0.009))
    visible = np.zeros(len(ys), dtype=bool)
    for index, (x, y) in enumerate(zip(xs, ys, strict=True)):
        center_x = round(x)
        left, right = max(0, center_x - radius), min(width, center_x + radius + 1)
        if left < right:
            visible[index] = bool(np.max(gray[int(y), left:right]) >= threshold)
    if len(visible) >= 5:
        kernel = np.ones(3, dtype=np.uint8)
        visible = cv2.morphologyEx(
            visible.astype(np.uint8)[None, :], cv2.MORPH_CLOSE, kernel[None, :]
        )[0].astype(bool)
    transitions = int(np.count_nonzero(visible[1:] != visible[:-1]))
    return float(np.mean(visible)), transitions


def _lanes_from_boundaries(
    boundaries: Sequence[_Boundary],
    vanishing_point: tuple[float, float],
    width: int,
    height: int,
) -> tuple[Lane, ...]:
    top_y = float(np.clip(vanishing_point[1] + height * 0.08, 0, height - 2))
    bottom_y = float(height - 1)
    top_xs = [
        vanishing_point[0]
        + (boundary.bottom_x - vanishing_point[0])
        * ((top_y - vanishing_point[1]) / max(1.0, bottom_y - vanishing_point[1]))
        for boundary in boundaries
    ]
    lanes: list[Lane] = []
    for index, (left, right) in enumerate(pairwise(boundaries)):
        lane_id = f"lane-{index + 1}"
        polygon = (
            (top_xs[index], top_y),
            (top_xs[index + 1], top_y),
            (right.bottom_x, bottom_y),
            (left.bottom_x, bottom_y),
        )
        bottom_center = (left.bottom_x + right.bottom_x) / 2.0
        direction = degrees(
            atan2(vanishing_point[1] - bottom_y, vanishing_point[0] - bottom_center)
        )
        lanes.append(Lane(lane_id, polygon, direction, ()))

    for index, lane in enumerate(lanes):
        neighbors: list[str] = []
        if index > 0 and boundaries[index].is_dashed:
            neighbors.append(lanes[index - 1].lane_id)
        if index + 1 < len(lanes) and boundaries[index + 1].is_dashed:
            neighbors.append(lanes[index + 1].lane_id)
        lanes[index] = replace(lane, neighbor_ids=tuple(neighbors))
    return tuple(lanes)


def _geometry_exclusions(
    segments: Sequence[_Segment],
    supported: Sequence[_Segment],
    vanishing_point: tuple[float, float],
    all_boundaries: Sequence[_Boundary],
    lane_boundaries: Sequence[_Boundary],
    width: int,
    height: int,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    exclusions = _wide_solid_gap_exclusions(
        all_boundaries, lane_boundaries, vanishing_point, height
    )
    supported_ids = {id(segment) for segment in supported}
    bottom_center = (lane_boundaries[0].bottom_x + lane_boundaries[-1].bottom_x) / 2.0
    main_angle = degrees(
        atan2(vanishing_point[1] - (height - 1), vanishing_point[0] - bottom_center)
    )
    minimum_length = hypot(width, height) * 0.10
    divergent = [
        segment
        for segment in segments
        if id(segment) not in supported_ids
        and segment.length >= minimum_length
        and _angle_distance(segment.angle_deg, main_angle) > 20.0
    ]
    if divergent:
        # Orientation clustering prevents unrelated edge fragments from creating
        # one enormous exclusion. The strongest family represents a branch.
        clusters = _orientation_clusters(divergent)
        strongest = max(clusters, key=lambda cluster: sum(segment.length for segment in cluster))
        polygon = _segment_family_polygon(strongest, width, height)
        if polygon:
            exclusions.append(polygon)

    maximum_stripe_length = hypot(width, height) * 0.10
    minimum_stripe_length = hypot(width, height) * 0.018
    stripe_candidates = [
        segment
        for segment in segments
        if id(segment) not in supported_ids
        and minimum_stripe_length <= segment.length < maximum_stripe_length
        and _angle_distance(segment.angle_deg, main_angle) > 20.0
    ]
    stripe_clusters = [
        cluster for cluster in _orientation_clusters(stripe_candidates) if len(cluster) >= 3
    ]
    if stripe_clusters:
        strongest_stripes = max(
            stripe_clusters, key=lambda cluster: sum(segment.length for segment in cluster)
        )
        polygon = _segment_family_polygon(strongest_stripes, width, height)
        if polygon:
            exclusions.append(polygon)
    return tuple(exclusions)


def _wide_solid_gap_exclusions(
    all_boundaries: Sequence[_Boundary],
    lane_boundaries: Sequence[_Boundary],
    vanishing_point: tuple[float, float],
    height: int,
) -> list[tuple[tuple[float, float], ...]]:
    lane_gaps = [right.bottom_x - left.bottom_x for left, right in pairwise(lane_boundaries)]
    if not lane_gaps:
        return []
    reference_width = float(np.median(lane_gaps))
    lane_pairs = {(id(left), id(right)) for left, right in pairwise(lane_boundaries)}
    top_y = float(np.clip(vanishing_point[1] + height * 0.08, 0, height - 2))
    bottom_y = float(height - 1)
    exclusions: list[tuple[tuple[float, float], ...]] = []
    for left, right in pairwise(all_boundaries):
        if (id(left), id(right)) in lane_pairs:
            continue
        if left.is_dashed or right.is_dashed:
            continue
        if right.bottom_x - left.bottom_x < reference_width * 1.35:
            continue
        top_scale = (top_y - vanishing_point[1]) / max(1.0, bottom_y - vanishing_point[1])
        left_top = vanishing_point[0] + (left.bottom_x - vanishing_point[0]) * top_scale
        right_top = vanishing_point[0] + (right.bottom_x - vanishing_point[0]) * top_scale
        exclusions.append(
            (
                (left_top, top_y),
                (right_top, top_y),
                (right.bottom_x, bottom_y),
                (left.bottom_x, bottom_y),
            )
        )
    return exclusions


def _orientation_clusters(segments: Sequence[_Segment]) -> list[list[_Segment]]:
    clusters: list[list[_Segment]] = []
    for segment in sorted(segments, key=lambda item: _undirected_angle(item.angle_deg)):
        if not clusters or _angle_distance(segment.angle_deg, clusters[-1][-1].angle_deg) > 12.0:
            clusters.append([segment])
        else:
            clusters[-1].append(segment)
    return clusters


def _segment_family_polygon(
    segments: Sequence[_Segment], width: int, height: int
) -> tuple[tuple[float, float], ...]:
    points = np.asarray(
        [point for segment in segments for point in (segment.start, segment.end)],
        dtype=np.float32,
    )
    if len(points) < 3:
        return ()
    hull = cv2.convexHull(points).reshape(-1, 2)
    if abs(cv2.contourArea(hull)) < width * height * 0.001:
        center, size, angle = cv2.minAreaRect(points)
        minimum_width = min(width, height) * 0.035
        rectangle = cv2.boxPoints(
            (center, (max(size[0], minimum_width), max(size[1], minimum_width)), angle)
        )
        hull = rectangle
    polygon = tuple((float(x), float(y)) for x, y in hull)
    return polygon if len(polygon) >= 3 else ()


def _point_in_valid_lane(point: tuple[float, float], scene: SceneGeometry) -> bool:
    in_lane = any(
        cv2.pointPolygonTest(np.asarray(lane.polygon, dtype=np.float32), point, False) >= 0
        for lane in scene.lanes
    )
    in_exclusion = any(
        cv2.pointPolygonTest(np.asarray(polygon, dtype=np.float32), point, False) >= 0
        for polygon in scene.exclusions
    )
    return in_lane and not in_exclusion


def _main_carriageway_observations(
    observations: Sequence[TrackObservation], scene: SceneGeometry
) -> tuple[TrackObservation, ...]:
    """Keep ambiguous tracks but exclude reliably moving traffic on another roadway."""
    by_track: dict[int, list[TrackObservation]] = defaultdict(list)
    for observation in observations:
        by_track[observation.track_id].append(observation)

    minimum_motion = max(8.0, hypot(*scene.frame_size) * 0.025)
    relevant: list[TrackObservation] = []
    for track in by_track.values():
        direction = _track_direction(track, minimum_motion)
        if direction is None:
            if len(track) < _PERSISTENT_STATIC_TRACK_OBSERVATIONS:
                relevant.extend(track)
            continue
        if any(
            _angle_distance(direction, lane.direction_deg)
            <= _CARRIAGEWAY_DIRECTION_TOLERANCE_DEG
            for lane in scene.lanes
        ):
            relevant.extend(track)
    return tuple(relevant)


def _track_direction(
    observations: Sequence[TrackObservation], minimum_motion: float
) -> float | None:
    if len(observations) < 3:
        return None
    ordered = sorted(observations, key=lambda item: (item.frame_index, item.timestamp_s))
    start, end = ordered[0].box.bottom_center, ordered[-1].box.bottom_center
    delta_x, delta_y = end[0] - start[0], end[1] - start[1]
    if hypot(delta_x, delta_y) < minimum_motion:
        return None
    return degrees(atan2(delta_y, delta_x))


def _infer_lane_directions(
    observations: Sequence[TrackObservation],
    lanes: Sequence[Lane],
    frame_size: tuple[int, int],
) -> tuple[tuple[Lane, ...], str]:
    by_track: dict[int, list[TrackObservation]] = defaultdict(list)
    for observation in observations:
        by_track[observation.track_id].append(observation)

    validated: list[Lane] = []
    status = "valid"
    for lane in lanes:
        polygon = np.asarray(lane.polygon, dtype=np.float32)
        angles: list[float] = []
        for track in by_track.values():
            in_lane = [
                observation
                for observation in track
                if cv2.pointPolygonTest(polygon, observation.box.bottom_center, False) >= 0
            ]
            direction = _track_direction(
                in_lane, max(8.0, hypot(*frame_size) * 0.025)
            )
            if direction is not None:
                angles.append(direction)

        if not angles:
            validated.append(lane)
            if status != "inconsistent":
                status = "insufficient"
            continue

        dominant = max(
            angles,
            key=lambda candidate: sum(
                _directed_angle_distance(candidate, angle) <= _DIRECTION_TOLERANCE_DEG
                for angle in angles
            ),
        )
        agreeing = [
            angle
            for angle in angles
            if _directed_angle_distance(dominant, angle) <= _DIRECTION_TOLERANCE_DEG
        ]
        if len(agreeing) / len(angles) < _MIN_DIRECTION_AGREEMENT_RATIO:
            validated.append(lane)
            status = "inconsistent"
            continue

        radians = np.deg2rad(agreeing)
        direction = degrees(atan2(float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians)))))
        validated.append(replace(lane, direction_deg=direction))
    if not lanes and status == "valid":
        status = "insufficient"
    return tuple(validated), status


def _line_intersection(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> tuple[float, float] | None:
    a1, b1, c1 = first
    a2, b2, c2 = second
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) < 1e-6:
        return None
    return ((b1 * c2 - b2 * c1) / determinant, (c1 * a2 - c2 * a1) / determinant)


def _point_line_distance(point: tuple[float, float], line: tuple[float, float, float]) -> float:
    return abs(line[0] * point[0] + line[1] * point[1] + line[2])


def _line_x_at_y(line: tuple[float, float, float], y: float) -> float | None:
    a, b, c = line
    return None if abs(a) < 1e-6 else -(b * y + c) / a


def _undirected_angle(angle: float) -> float:
    return angle % 180.0


def _angle_distance(first: float, second: float) -> float:
    difference = abs(_undirected_angle(first) - _undirected_angle(second))
    return min(difference, 180.0 - difference)


def _directed_angle_distance(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)
