from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from platformdirs import user_cache_path

CHUNK_SIZE = 1024 * 1024
REQUIRED_MANIFEST_FIELDS = {"model_id", "url", "sha256", "size_bytes"}


class WeightFetchError(RuntimeError):
    """Raised when a pinned model asset cannot be safely acquired."""


@dataclass(frozen=True, slots=True)
class WeightManifest:
    model_id: str
    url: str
    sha256: str
    size_bytes: int


def load_manifest(path: Path) -> WeightManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WeightFetchError(f"could not read weight manifest {path}: {error}") from error

    if not isinstance(raw, dict) or set(raw) != REQUIRED_MANIFEST_FIELDS:
        raise WeightFetchError(
            "weight manifest must contain exactly model_id, url, sha256, and size_bytes"
        )
    if (
        not isinstance(raw["model_id"], str)
        or not raw["model_id"]
        or not isinstance(raw["url"], str)
        or not raw["url"]
        or not isinstance(raw["sha256"], str)
        or len(raw["sha256"]) != 64
        or not isinstance(raw["size_bytes"], int)
        or isinstance(raw["size_bytes"], bool)
        or raw["size_bytes"] < 0
    ):
        raise WeightFetchError("weight manifest contains invalid field values")
    return WeightManifest(**raw)


def _digest_stream(source: BinaryIO, destination: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while chunk := source.read(CHUNK_SIZE):
        destination.write(chunk)
        digest.update(chunk)
        size += len(chunk)
    destination.flush()
    os.fsync(destination.fileno())
    return size, digest.hexdigest()


def _matches_manifest(path: Path, manifest: WeightManifest) -> bool:
    try:
        if path.stat().st_size != manifest.size_bytes:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(CHUNK_SIZE):
                digest.update(chunk)
        return digest.hexdigest() == manifest.sha256
    except OSError:
        return False


def fetch_weights(manifest: WeightManifest, cache_dir: Path) -> Path:
    weights_dir = cache_dir / "weights"
    target = weights_dir / f"{manifest.model_id}.pt"
    if _matches_manifest(target, manifest):
        return target

    weights_dir.mkdir(parents=True, exist_ok=True)
    partial_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".partial",
            dir=weights_dir,
            delete=False,
        ) as partial:
            partial_path = Path(partial.name)
            with urllib.request.urlopen(manifest.url) as response:
                size, sha256 = _digest_stream(response, partial)

        if size != manifest.size_bytes:
            raise WeightFetchError(
                f"weight size mismatch: expected {manifest.size_bytes}, received {size}"
            )
        if sha256 != manifest.sha256:
            raise WeightFetchError(
                f"weight checksum mismatch: expected {manifest.sha256}, received {sha256}"
            )

        os.replace(partial_path, target)
        partial_path = None
        return target
    except WeightFetchError:
        raise
    except Exception as error:
        raise WeightFetchError(f"could not download pinned weights: {error}") from error
    finally:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download and verify pinned detector weights")
    parser.add_argument("--manifest", type=Path, default=root / "weights.lock.json")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=user_cache_path("lane-change-counter"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        manifest = load_manifest(args.manifest)
        path = fetch_weights(manifest, args.cache_dir)
    except WeightFetchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
