"""Command-line entry point for the lane-change counting pipeline."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from lane_change_counter.config import DeviceRequest
from lane_change_counter.pipeline import RunOptions, process_directory, process_video

_LOGGER = logging.getLogger(__name__)
_SUCCESS_STATUSES = frozenset({"success", "diagnostic"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count completed lane changes in MP4 video")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="one input MP4")
    inputs.add_argument("--input-dir", type=Path, help="directory of direct-child MP4 files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=tuple(request.value for request in DeviceRequest),
        default=DeviceRequest.AUTO.value,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--log-level", choices=("INFO", "DEBUG"), default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(levelname)s: %(message)s",
    )
    options = RunOptions(
        device=DeviceRequest(arguments.device),
        overwrite=arguments.overwrite,
        rebuild_cache=arguments.rebuild_cache,
        geometry_only=arguments.geometry_only,
        log_level=arguments.log_level,
    )
    try:
        if arguments.input is not None:
            summary = process_video(arguments.input, arguments.output_dir, options)
            return 0 if summary.status in _SUCCESS_STATUSES else 1
        summaries = process_directory(arguments.input_dir, arguments.output_dir, options)
        return 0 if all(summary.status in _SUCCESS_STATUSES for summary in summaries) else 1
    except Exception as error:  # noqa: BLE001 - the CLI converts package failures to status 1.
        _LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
