# Lane Change Counter

This submission contains one standalone computer-vision pipeline for counting
completed vehicle lane changes in an input MP4.  The same code and default
parameters are used for every video; the pipeline does not branch on a file
name, read `answer.json`, or contain video-specific event frames, tracks, or
lane coordinates.

## Requirements and setup

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- The locked `yolo11n.pt` model weight described by `weights.lock.json`

Create the locked environment from this directory:

```bash
uv sync
```

Fetch and verify the required model weight (internet access is required for
this one-time step):

```bash
uv run python scripts/fetch_weights.py
```

The script stores the weight in the platform cache and verifies its SHA-256
against `weights.lock.json`.  No credentials are required.  CPU inference is
the portable default; `--device auto` also uses MPS when it is available and
falls back safely to CPU if necessary.

## Run the pipeline

Run one video:

```bash
uv run python run.py --input /path/to/input.mp4 --output-dir output --device auto --overwrite
```

Run all MP4 files directly inside a directory:

```bash
uv run python run.py --input-dir /path/to/videos --output-dir output --device auto --overwrite
```

For each input the pipeline:

1. estimates scene geometry from the decoded frames;
2. detects vehicles with YOLO11n and tracks them with ByteTrack;
3. assigns track contact points to automatically inferred, adjacent travel
   lanes and uses temporal smoothing plus a finite-state rule to confirm only
   completed lane changes; and
4. writes an event log, run provenance, updated `answer.json`, and an
   annotated video from the pipeline's own detections, assignments, and
   confirmed events.

If geometry or inference is unreliable, the pipeline fails rather than
inventing a zero count.  The annotated video shows inferred lanes, vehicle
tracks and lane assignments, recently confirmed events, a running count, and
the final total.  It is generated directly from the finalized pipeline
records; it is not manually edited after inference.

## Submitted results

`answer.json` reports the counts generated for the three supplied videos. The
adjacent `*.events.json` and `*.run.json` files provide machine-readable event
evidence and provenance (including the applied thresholds, package versions,
weight hash, and actual device). The locally generated `annotated_*.mp4`
renderings are intentionally not committed because each exceeds GitHub's
normal per-file size limit.

Counting convention: a lane change is recorded only after a tracked vehicle
moves between adjacent travel lanes in the same direction and satisfies the
pipeline's temporal confirmation rule.  In-lane motion, short boundary
contacts, grid/non-road regions, and joining traffic are excluded.  The narrow
edge-exit confirmation rule is documented in the emitted event metadata and
only applies when a vehicle has demonstrated a transition into the adjacent
lane immediately before it leaves the frame.

## Included files

- `run.py`, `lane_change_counter/`, and `scripts/fetch_weights.py`: pipeline
  and required helper code.
- `pyproject.toml` and `uv.lock`: reproducible environment definition.
- `weights.lock.json`: required model identifier, source, and integrity hash.
- `answer.json`, `*.events.json`, and `*.run.json`: generated submission
  results and machine-readable evidence.
