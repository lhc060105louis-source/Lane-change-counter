# README Research-Engineering Positioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Present the lane-change counter as a reproducible applied computer-vision project for technical recruiters and research supervisors.

**Architecture:** Rewrite `README.md` as a concise English project narrative, then align the public GitHub description to it. Validate both local documentation and remote metadata after publishing.

**Tech Stack:** Markdown, Git, GitHub CLI, Python 3.11.

## Global Constraints

- Preserve the factual behavior of the existing pipeline.
- Keep the README in English for an international technical and academic audience.
- Do not claim that the large annotated MP4 renderings are stored in GitHub.
- Keep the quick-start path short and executable.

---

### Task 1: Rewrite and validate the README

**Files:**
- Modify: `README.md`
- Reference: `pyproject.toml`, `run.py`, `weights.lock.json`, `evidence/*.json`

**Interfaces:**
- Consumes: Existing CLI commands, evidence artifacts, and the model-weight lock.
- Produces: An accurate README with overview, credibility, workflow, quick start, results, scope, and repository layout.

- [ ] **Step 1: Inspect source-facing facts before editing**

Run:

```bash
sed -n '1,220p' pyproject.toml
sed -n '1,260p' run.py
sed -n '1,160p' weights.lock.json
```

Expected: The documented runtime, commands, and model lock match the source.

- [ ] **Step 2: Replace the README with a research-engineering narrative**

Use these exact top-level sections:

```markdown
# Lane Change Counter
## Why this project
## Method at a glance
## Reproduce the pipeline
## Results and evidence
## Scope and decision rule
## Repository layout
```

State that one pipeline is applied uniformly to MP4 inputs. Describe geometry
estimation, YOLO11n detection, ByteTrack tracking, lane assignment, temporal
smoothing, finite-state confirmation, and the distinction between committed
JSON evidence and local size-excluded MP4 renderings.

- [ ] **Step 3: Validate Markdown and references**

Run:

```bash
rg -n '^#|^##|uv (sync|run)|annotated_.*mp4|answer\.json|events\.json|run\.json' README.md
git diff --check
```

Expected: Required sections and commands are present; no whitespace errors.

- [ ] **Step 4: Commit the README update**

Run:

```bash
git add README.md
git commit -m "Improve project documentation"
```

Expected: The commit changes only `README.md` at this task boundary.

### Task 2: Publish and verify repository metadata

**Files:**
- Modify: GitHub description for `lhc060105louis-source/Lane-change-counter`
- Verify: `README.md` and remote `main`

**Interfaces:**
- Consumes: The completed README and fixed description.
- Produces: GitHub metadata and remote content that match the project positioning.

- [ ] **Step 1: Set the repository description**

Run:

```bash
gh repo edit lhc060105louis-source/Lane-change-counter --description "Reproducible computer-vision pipeline for counting completed vehicle lane changes from video."
```

Expected: The repository overview shows the exact sentence.

- [ ] **Step 2: Push the documentation commit**

Run:

```bash
git push origin main
```

Expected: The remote `main` branch advances to the documentation commit.

- [ ] **Step 3: Verify local and remote state**

Run:

```bash
git status --short --branch
git ls-remote origin refs/heads/main
gh repo view lhc060105louis-source/Lane-change-counter --json description,url
```

Expected: The worktree is clean, remote `main` equals local `HEAD`, and the description is exact.
