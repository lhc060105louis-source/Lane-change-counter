# Lane Change Counter

## Why this project

Lane-change counting looks simple until vehicle occlusion, perspective, road
geometry, and brief boundary contacts turn a visual impression into an
ambiguous event. This project implements a reproducible computer-vision
pipeline that counts *completed* lane changes in MP4 traffic footage. One
pipeline and one set of default parameters are applied uniformly to MP4 inputs:
the code does not branch on file names, read `answer.json`, or encode
video-specific lanes, tracks, or event frames.

The repository is structured as a small research-engineering artifact: it
separates scene understanding from event logic, records provenance with each
run, and keeps machine-readable evidence beside the submitted counts.

## Method at a glance

For each video, the pipeline:

1. Decodes a representative set of frames and estimates scene geometry,
   including travel-lane polygons, adjacency, direction, and excluded non-road
   regions.
2. Detects vehicles with the locked YOLO11n model and maintains identities with
   ByteTrack.
3. Assigns each tracked vehicle's bottom-center contact point to an inferred
   lane, then temporally smooths assignments to suppress transient labels near
   lane boundaries.
4. Runs a deterministic finite-state confirmation rule per track, recording
   only sustained transitions between adjacent, same-direction travel lanes.

The result is an event log rather than a frame-by-frame heuristic count. A
run that cannot establish reliable geometry produces diagnostic evidence
instead of inventing a zero count.

## Reproduce the pipeline

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/). The project
declares its runtime and development tools in `pyproject.toml`; `uv.lock`
pins the resolved environment.

Create the environment:

```bash
uv sync
```

Fetch and integrity-check the required model once (internet access required):

```bash
uv run python scripts/fetch_weights.py
```

`weights.lock.json` locks the model to `yolo11n`, including its download URL,
SHA-256 (`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`),
and size (5,613,764 bytes). The fetch script stores the weight in the platform
cache and verifies that hash. CPU is the portable default; `--device auto`
uses MPS when available and otherwise falls back to CPU.

Process one MP4:

```bash
uv run python run.py --input /path/to/input.mp4 --output-dir output --device auto --overwrite
```

Process all direct-child MP4 files in a directory:

```bash
uv run python run.py --input-dir /path/to/videos --output-dir output --device auto --overwrite
```

Each successful input produces `annotated_<input>.mp4`, `<input>.events.json`,
and `<input>.run.json`; the output directory also receives an updated
`answer.json`. Use `--geometry-only` to emit a geometry diagnostic without
running detection and counting.

## Results and evidence

The committed `answer.json` records the pipeline output for the supplied
videos:

| Input | Completed lane changes |
| --- | ---: |
| `lane_change_count.mp4` | 2 |
| `lane_change_count_2.mp4` | 5 |
| `lane_change_count_3.mp4` | 2 |

For every result, committed `evidence/*.events.json` files list confirmed
events with track IDs, origin and target lanes, frames, timestamps, and
confirmation type. The paired `evidence/*.run.json` files capture run status,
applied thresholds, package versions, the model ID and weight hash, device,
and resource/runtime provenance.

The annotated MP4 renderings are local, size-excluded outputs and are not
committed. They are generated directly from finalized detections, assignments,
and confirmed events; they visualize inferred lanes, vehicle tracks and lane
assignments, recent confirmations, the running count, and the final total.

## Scope and decision rule

A counted event requires a tracked vehicle to establish an origin lane, move
far enough laterally into an adjacent same-direction lane, and remain in that
target lane long enough to satisfy temporal confirmation. The pipeline
therefore excludes in-lane motion, short boundary contacts, transitions
through excluded grid or non-road regions, and joining traffic. A narrow
edge-exit rule can confirm an event only after the track has already shown the
required transition into the adjacent lane; its outcome is marked in the event
evidence.

This is a lane-change counter for compatible traffic footage, not a general
traffic analytics system. Its geometry estimation and conservative decision
rule are designed to make outputs inspectable and reproducible; footage with
insufficient persistent lane evidence is reported diagnostically rather than
assigned a count.

## Repository layout

```text
run.py                    CLI entry point for one MP4 or a directory of MP4s
lane_change_counter/      Geometry, detection, tracking, assignment, FSM, and reporting
scripts/fetch_weights.py  Locked-weight fetch and SHA-256 verification helper
tests/                    Automated unit and integration coverage
evidence/                 Committed event logs and run provenance for submitted outputs
answer.json               Submitted per-video lane-change totals
weights.lock.json         YOLO11n source, SHA-256, and expected size
pyproject.toml, uv.lock   Python 3.11 project metadata and locked environment
```
