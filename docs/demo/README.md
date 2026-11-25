# BrainLens 1.0 Launch Video

Real-motion product launch video for **BrainLens 1.0 — AI co-review for EEG** (~84 s, 1920×1080, H.264, 30 fps, full-bleed real UI with hard cuts, deliberate steady punch-ins at key reading moments, and a plain narrative subtitle on every beat so the silent cut can be followed without sound; muxed with `build/voiceover.wav` when present).

## Deliverables

- `build/brainlens_1_0_launch.mp4` — 1920×1080 H.264 launch video (untracked; silent cut unless `build/voiceover.wav` is present).
- `docs/demo/brainlens_1_0_poster.png` — 1920×1080 full-bleed hero poster frame captured directly from the live workbench.
- `docs/demo/voiceover.md` — Calm, dry ~122-word voice-over script with timecodes for recording `build/voiceover.wav`.
- `docs/demo/record_launch.py` — End-to-end reproducible pipeline (`prepare` → `capture` → `render`).
- `build/demo-captures/contact_sheet.png` & `build/demo-captures/beat_frames/` — 18-frame contact sheet and 22 full-resolution beat QA frames.

## Re-render in One Command

From the repository root, in a Python 3.11–3.13 environment with `pip install "brainlens[pyhfo]"` and `python-pptx` installed (headless Chrome / `chrome-headless-shell` available, and `imageio` + `imageio-ffmpeg` + `Pillow` in `build/video-venv`), with `inputs/HUP134/run-01.edf` and `run-02.edf` present:

```bash
BRAINLENS_TRUST_DIR="$PWD/build/trust" \
MPLCONFIGDIR="$PWD/build/mpl" \
python docs/demo/record_launch.py
```

If `build/voiceover.wav` exists when `--stage render` runs, `record_launch.py` automatically muxes it (AAC 192 kbps) into `build/brainlens_1_0_launch.mp4`.

### Optional Environment Variables

With nothing set, BrainLens's own engine auto-detection is used. Set these only to point at a specific engine checkout/interpreter/cache, headless Chrome binary, or extra import paths:

- `BRAINLENS_PYHFO_SOURCE` — path to a PyHFO source checkout.
- `BRAINLENS_PYHFO_PYTHON` — Python interpreter of the PyHFO engine virtual environment.
- `BRAINLENS_PYHFO_MODEL_CACHE` — directory holding the pre-fetched ONNX model cache.
- `BRAINLENS_DEMO_EXTRA_PATH` — `os.pathsep`-separated extra import paths appended to `sys.path`/`PYTHONPATH` for `record_launch.py` and spawned MCP subprocesses.
- `BRAINLENS_CHROME_PATH` — path to a headless Chrome / `chrome-headless-shell` executable if not auto-discovered.

`record_launch.py` prepends `./workbench/src` and appends `build/video-venv`'s `site-packages`, keeps all temporary/downloaded data and rendered video outputs under `build/`, never places CHB-MIT data under `inputs/`, and never touches `./workbench_results`.

### Stage Options

- Full clean rebuild of workspace and captures (preserves verified CHB-MIT EDFs in `build/demo-data/`): `python docs/demo/record_launch.py --clean`
- Individual stages:
  - `--stage prepare` — Ensures CHB-MIT Scalp EEG Database v1.0.0 (`chb01`, 42 EDFs + `chb01-summary.txt`) is present and SHA-256 verified in `build/demo-data/chbmit/chb01/`, writes `PROVENANCE.json` and per-file sidecars (`modality: "scalp"`, `reference: "bipolar"`, 7 expert seizure annotations), inspects `chb01` into `build/demo-workspace`, pre-warms `Line length` trends caches, and inspects/analyzes real public CC0 SEEG `inputs/HUP134` (`run-01` & `run-02`, `baseline` with PyHFO STE detection + ONNX `artifact`/`spike` scoring) while resetting mutable interactive state.
  - `--stage capture` — Launches local `WorkbenchHTTPServer`, headless Chrome (`1440×810` CSS @ DPR=2 so `Page.startScreencast` captures native `2880×1620` Retina frames), and a live stdio MCP session (`brainlens.mcp_server`). Drives real `Input.dispatchMouseEvent` / `mouseWheel` / `dispatchKeyEvent` (48–105 ms/char jittered human typing) in chronological order so the cold open has zero verdicts, captures every shot's screencast frames, exact dispatched cursor trajectory, and measured UI bounding boxes, and renders the real exported `.pptx` slides at `2880×1620`.
  - `--stage render` — Resamples recorded screencast shots to 30 fps, composites a plain macOS arrow cursor along the exact dispatched coordinates, cuts cleanly into and out of steady Retina punch-in crops at key reading moments (typed questions, replies, ~230 Hz island, verdict panel, exported slide — no slow drifting zooms), keeps minimal neutral typography (plain title card, two quiet case tags, one plain narrative subtitle per beat saying what the clinician or agent is doing (hard on/off, with short holds on real frames for reading time), full-bleed real `.pptx` slides, plain end card), muxes `build/voiceover.wav` if present, and exports `docs/demo/brainlens_1_0_poster.png`, `build/demo-captures/contact_sheet.png`, and `build/demo-captures/beat_frames/`.

## Story (~84 s)

Two distinct real public datasets representing two separate patients (tagged quietly at the start of each case, with full credits in the end-card footer):

1. **0:00–0:22 · Cold open (`HUP134` live MDT case sheet, before any verdicts exist)** — The clinician drags to highlight `LB10`, types *"Does it recur in run-02?"* (steady punch-in), and clicks **Send to Agent**. Real MCP calls move the open reader by itself to `run-02 / LB10` (steady punch-in on the ~230 Hz island) and post a short plain reply in the inbox (steady punch-in), honestly noting that the artifact model scores both events as artifact and they need clinician eyes.
2. **0:22–0:25 · Title card** — Plain neutral card: *BrainLens 1.0* / *AI co-review for EEG*.
3. **0:25–0:51 · Case 1 · Scalp EEG · CHB-MIT (`chb01`)** — Opens the 42-file admission timeline with admission-relative clocks, collapsed gaps, and a compact recording selector, then opens `chb01_03` in the Waveform Reader (23-channel double-banana acquisition order, `1–70 Hz`, `Notch 60 Hz`, `10 s` pages, triad collapsed). The clinician types *"Where are the seizures?"* into the reader's **Ask agent** box (steady punch-in); real MCP calls (`open_workbench(action="status")`, `query_evidence(mode="timeline", text="seizure")`, `open_workbench(action="focus", recording="chb01_03", start_seconds=2996, view="reader")`, and `resolve_actions`) move the reader live to the first annotated seizure onset at `49:56` and display the reply beside the double-banana waveforms (steady punch-in), followed by the `Line length` navigation trend.
4. **0:51–1:11 · Case 2 · SEEG · OpenNeuro HUP134** — Separates widespread multi-shaft bursts from focal candidates (`LB10`, `LC7`, `LC8`), clicks a burst to show shafts firing together, then inspects `LB10`'s Clinical Triad beside its locked-Y quiet control. A steady punch-in on the Clinician review panel shows `Artifact model: artifact (real-HFO p 0.00)` and `Agent: artifact` as the clinician presses `3` (`Uncertain`), keeping both verdict entries visible and separate.
5. **1:11–1:20 · MDT slides (`HUP134`)** — One click on **MDT slides**, cutting directly to the real exported `Leading focal contacts` slide (showing only roles with verdicts) and `Focal candidate` Clinical Triad slide, full bleed, with a steady punch-in on the model label and role-separated verdicts.
6. **1:20–1:24 · End card** — Plain neutral card: *BrainLens 1.0*, *AI co-review for EEG*, *Open source · from the PyHFO team, Roychowdhury Lab, UCLA*, `github.com/zyp5511/BrainLens`, *Works with Claude Code, Codex, Cursor and Antigravity*, and dataset credits footer.

Every beat carries one plain subtitle (the same sentences as `voiceover.md`), so a viewer with the sound off and no knowledge of the UI can follow who is doing what and why.

## Data Provenance & Credits

- **Scalp EEG (`chb01`):** CHB-MIT Scalp EEG Database v1.0.0 (`chb01`, 42 EDF files, 23 bipolar double-banana channels, 256 Hz, 7 expert-annotated seizures from `chb01-summary.txt`), PhysioNet (`https://physionet.org/content/chbmit/1.0.0/`), **ODC-By 1.0**.
  - Shoeb AH. *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment.* PhD Thesis, Massachusetts Institute of Technology, 2009.
  - Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. *PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals.* Circulation 101(23):e215–e220, 2000.
- **SEEG (`HUP134`):** HUP iEEG Epilepsy Dataset (`sub-HUP134`, `run-01.edf` and `run-02.edf`, 1024 Hz, 83 SEEG + 2 EKG), OpenNeuro [`ds004100`](https://openneuro.org/datasets/ds004100/versions/1.1.3) v1.1.3 (DOI `10.18112/openneuro.ds004100.v1.1.3`), **CC0**.
- **Detection & Classification Engine:** PyHFO 4.0 by the Roychowdhury Lab, UCLA.
