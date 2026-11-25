# BrainLens (`brainlens` v1.0.0)

AI co-review for EEG — from scalp EEG to SEEG: evidence-grounded EEG/iEEG/SEEG neurophysiology workbench with admission timelines, quantitative navigation trends, native vendor formats, PyHFO screening, cross-shaft synchrony stratification, Clinical Triad verification, clinician-owned verdicts, one-click MDT slides, and a 7-tool MCP server.

> **Research & Clinical Decision-Support Scope**: BrainLens is **not a medical device** and does not establish seizure onset zones (SOZ) or surgical resection targets. Automated HFO counts and model scores require visual verification of raw traces, filtered waveforms, and time-frequency maps by qualified clinicians. Clinician sign-offs are strictly separated from researcher, engineering rehearsal, and AI agent notes.

## Installation

```bash
pip install brainlens
# Or from a local source checkout:
pip install ./workbench
```

Requires Python 3.11–3.13 (`numpy>=1.26,<3`, `mne`, `edfio`, `scipy`, `matplotlib`, `pydantic`, `filelock`, `mcp>=1.30,<2`).

### Optional PyHFO Detection & ONNX Classification Engine

Signal inspection, QC, Pro Waveform Viewer, and Triad verification work out of the box with `--no-detector` (`detector_enabled=False`). For HFO detection, install PyHFO 4.x, the PyHFO team's packaged engine. Fetch its pinned ONNX classifiers (`artifact`, `spike`, `eHFO`) once on a connected machine. BrainLens detects the engine in its own interpreter and workers use the verified models offline.

```bash
pip install "brainlens[pyhfo]" && brainlens configure-engine --fetch-models
brainlens inspect "*"
```

With the pre-release wheel, run `pip install "./workbench[pyhfo]" build/pyhfo-4.0.0-py3-none-any.whl` from the repository root, then run the fetch command. To select a separate package interpreter, pass `--python /path/to/venv/bin/python` to `configure-engine`.

#### Advanced / legacy source engine

For the legacy upstream checkout at commit `c1c7d3fec990e8661604080664766dc1297cbfdc`, register its source, interpreter, and model directory.

```bash
brainlens configure-engine \
  --source ~/pyHFO \
  --python ~/pyhfo-venv/bin/python \
  --model-cache ~/pyhfo-onnx-models
```

Alternatively, configure via environment variables: `BRAINLENS_PYHFO_SOURCE`, `BRAINLENS_PYHFO_PYTHON`, and `BRAINLENS_PYHFO_MODEL_CACHE` (legacy `BTK_PYHFO_*` env vars and `~/.braintoolkit/engine.json` are also read as fallbacks).

## Quick Start

```bash
# Inspect registered cases, local recordings, and PyHFO engine health
brainlens inspect "*"

# Run baseline HFO screening on an EDF/BDF/FIFF/BrainVision file or BIDS folder
brainlens analyze /path/to/recording.edf

# Export this run for an EEG reader
brainlens export-edf <case> --recording <recording> --run-name baseline

# Launch the local interactive BrainLens Neurophysiology Workbench (loopback only)
brainlens-workbench

# Start the 7-tool stdio MCP server
brainlens-mcp --workspace ./workbench_results
```

The **Export EDF+ for EEG reader** button exports EDF/EDF+C with the original clinical identity, original annotations, and BrainLens events. BDF, FIF, and BrainVision readers can use the accompanying `onset,duration,description` CSV (seconds from data start). Each annotation export includes a JSON provenance sidecar.

Use **De-identify for sharing** for a shareable EDF/CSV, BIDS export, or signed MDT case sheet. An optional pseudonym labels the files and their contents, for example `brainlens export-edf HUP134 --recording run-02 --deidentify --pseudonym P001`. The local workbench shows clinical identity. Agent-facing MCP responses always mask names and IDs obtained from native recording headers.

## Security Model & Practical Scale Guidance

- **Single-User Loopback Design**: `brainlens-workbench` binds to `127.0.0.1` and uses a per-session CSRF token (`X-BrainLens-Token`) injected into the local UI. Read-only `GET` API endpoints on loopback are unauthenticated; **do not expose the workbench port to a network or run it on an untrusted multi-user host**.
- **Untrusted Workspace Protection**: Workspace-local `<workspace>/engine.json` files are ignored by default (`BRAINLENS_ALLOW_WORKSPACE_ENGINE=1` required to opt in). Exported MDT case sheets are signed against an out-of-workspace HMAC-SHA256 trust store (`~/.brainlens/`).
- **Recording File Size Guidance**: To prevent silent mid-review source mutation, the interactive Waveform Reader performs full SHA-256 source integrity verification around disk reads. Interactive paging is tuned for standard review segments (typically **<= ~1 GB per recording**, e.g. 50–500 MB interictal SEEG clips); for multi-gigabyte continuous long-term recordings (1–10+ GB), segment into review windows before interactive browsing.

## License

MIT License (see `LICENSE`).
