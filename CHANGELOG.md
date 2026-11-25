# Changelog

All notable changes to BrainLens are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning.

## [1.0.0] - 2026-11-24

First official release. BrainLens is an evidence-grounded SEEG/iEEG/EEG neurophysiology workbench and AI co-review engine for Python 3.11–3.13, released under the MIT license.

### Added

**Signal ingestion, QC and detection**

- Readers for `.edf`, `.bdf`, `.fif`/`.fif.gz`, BrainVision (`.vhdr`/`.eeg`/`.vmrk`) and BIDS iEEG/EEG directories, with verified units, shafts, coordinates and per-recording provenance.
- Native vendor readers for Nihon Kohden, Persyst, EEGLAB, Neuroscan CNT, Compumedics Curry 7/8, GDF and Nicolet, with companion files bound into integrity checks and vendor header identity masked in agent output; Micromed `.TRC` through the optional `brainlens[micromed]` extra (python-neo). Unsupported vendor formats (Natus `.erd`, EGI `.mff`, NicoletOne `.e`) are refused with an export path, and unreadable vendor headers never echo header text.
- Long recordings: multi-gigabyte files are hashed once per stat signature and read by window with sampled re-verification; EDF+D / BDF+D and gapped EDF+C are mapped to true time with explicit gaps; detection is chunked and bounded (`analysis_window_too_large`).
- Trend overview (`query_evidence(mode="trends")`, Trends panel): per-channel band power, line length, RMS and HFO rate per epoch, for navigation only. Admission timeline (`mode="timeline"`) across a case's files, with admission-relative times for agents and unknown offsets for placeholder dates. Multi-segment analysis within one recording from explicit or marker-derived segments (`overrides.segments`, `overrides.marker_segments`).
- Full-coverage QC and zero-phase SOS preparation, with immutable per-stage caching (`qc`, `prepare`, `detect`) under MVCC analysis generations. Exclusion branches rerun QC, preparation and detection over the masked spans; downstream-only branches (for example ONNX scoring) reuse the cached stages.
- PyHFO 4.x package engine (`pip install "brainlens[pyhfo]"`) for `STE` / `MNI` / `HIL` HFO detection and the three ONNX classifiers (`artifact`, `spike`, `eHFO`), run in a separate worker process. `brainlens configure-engine --fetch-models` downloads the pinned models once and verifies them against PyHFO's registry; analysis workers run offline. A pinned legacy source checkout (`c1c7d3fec990e8661604080664766dc1297cbfdc`) is supported as the advanced path.
- Cross-shaft synchrony classification of every detection as `focal`, `regional` or `widespread_burst`: simultaneous events on `≥5` shafts within `±30 ms` are grouped into burst episodes, so common-reference transients and arousal bursts stay out of focal rankings.
- Signal guardrails: fast ripple (`250–450 Hz`) is offered only when both the Nyquist limit and the documented hardware low-pass allow it; ONNX `eHFO` is reported `unavailable` (never `0`) when the sampling rate cannot support it; bipolar recordings are protected against double-differencing; 0-event channels are excluded from rankings; rates are never pooled across recordings.
- Sensitivity branch comparison, unpooled cross-recording concordance, quiet control windows and cohort summaries (`brainlens query`, `brainlens cohort`).

**Evidence, anatomy and ictal analysis**

- Per-contact multi-biomarker evidence map: focal ripple, focal fast ripple, ONNX spike-HFO and IED candidate rates with within-recording percentiles and a descriptive convergence count (not a validated predictor). The IED candidate detector (`brainlens.spikes.detect_ieds`) is a band-limited envelope detector with documented parameters.
- Ictal Epileptogenicity Index after Bartolomei et al. 2008 for one explicitly analyzed seizure, with an explicit or marker-derived onset (never inferred) and per-channel reasons for missing values.
- Case-level anatomical label overlays (`anatomy_labels.json`) imported from UTF-8 CSV/TSV or BIDS `*_electrodes.tsv`, with source, atlas and `patient_native` provenance. Labels are shown in the overview, reader gutter, triad, 3D tooltip, comparisons, case sheet, BIDS events and cohort CSV, and never affect recording, stage or analysis fingerprints.
- Source-cited external SOZ / Resected overlays for concordance display.

**Clinical review workflow**

- Persistent loopback workbench UI (light and dark themes, live SSE sync) with a case triage overview, a multi-channel waveform reader with Clinical Triad (raw trace, ripple/fast-ripple envelope, Morlet spectrogram) and locked-Y quiet control, a 3D depth-shaft viewer, and an interactive MDT case sheet with an Agent Sync inbox.
- Clinical reader display conventions: sensitivity in µV/mm (17 presets with per-browser screen calibration, calibration state labels and a physical calibration pulse), `15` / `30` / `60 mm/s` timebases, explicit LFF/HFF display filters clamped to per-trace Nyquist and acquisition low-pass, PageUp/PageDown paging and a `?` shortcut overlay.
- English / 简体中文 workbench display language, persisted per browser with English fallback. Exported clinical documents, channel names and units stay English.
- Role-attributed verdicts (`clinician` / `researcher` / `engineering_rehearsal` / `agent`), clips and clinical questions, kept strictly separate.
- Ask the agent from where you are reading: the MDT case sheet, the waveform reader and the admission timeline (also before any analysis), anchored to the recording, time window, channel or marker; agents read questions with `open_workbench(action="status")` and answer inline with `resolve_actions`.
- Scalp EEG reading defaults: acquired montage order (e.g. double banana), 1–70 Hz with line-frequency notch, 10 s pages. Model scores are labelled by meaning (for example `Artifact model: artifact (real-HFO p 0.00)`), clinical exports list only roles that gave verdicts, and implausible header dates fall back to admission-relative times.
- Blinded reader review sets (all events or a seeded stratified sample) with a deterministic per-reader shuffled order. Agreement reporting: percent agreement, 3-category and binary Cohen's κ with 95% CI, confusion matrix, prevalence/bias indices, Fleiss' κ for ≥3 readers and a disagreement list.

**Exports**

- Print-ready MDT Case Discussion Sheet (HTML/PDF), signed with HMAC-SHA256 against an out-of-workspace trust store.
- MDT slide deck (16:9 PPTX) built from the same evidence snapshot and card selection as the case sheet, with the supporting numbers in speaker notes.
- BIDS `*_events.tsv` + `*_events.json` with `GeneratedBy` provenance.
- Annotated EDF+ export on a lossless, sample-verified copy of the recording, carrying BrainLens events, burst episodes and human clinician verdicts (agent verdicts are never labelled clinical) and preserving source annotations. Every export writes an annotation CSV and a provenance JSON for all supported input formats.
- Opt-in de-identification (`--deidentify`, optional `--pseudonym`) for EDF/CSV/JSON, case sheets, BIDS exports, file names and case names. MCP responses always mask header-derived identity.
- Reusable group protocols and cohort CSV summaries.

**Agent, CLI and deployment**

- 7-tool stdio MCP server (`brainlens-mcp`: `inspect_case`, `analyze_case`, `query_evidence`, `render_evidence_card`, `open_workbench`, `record_judgment`, `export_or_protocol`) and one canonical Open Agent Skill (`skills/brainlens/SKILL.md`) for Claude Code, Codex, Cursor and Antigravity. Tools run through an async shim in a worker thread with MCP progress notifications for `analyze_case` / `run_cohort`; `open_workbench` waits in Co-Review standby (`wait_action`).
- CLI: `inspect`, `analyze`, `query`, `evidence`, `ei`, `card`, `export`, `export-slides`, `export-bids`, `export-edf`, `review-set`, `agreement`, `cohort`, `overlay`, `anatomy`, `protocol`, `workbench`, `watch-actions`, `resolve-actions`, `configure-engine`.
- Single-user loopback security model: per-session CSRF token (`X-BrainLens-Token`) on mutating routes, loopback and port-matching `Host` checks, workspace-local `engine.json` ignored unless `BRAINLENS_ALLOW_WORKSPACE_ENGINE=1`, and SHA-256 source verification on interactive reads (full verify on first read, stat-signature fast path with a 60 s full re-verify).
- Docker image (`docker/Dockerfile`: multi-stage, non-root, optional pre-fetched PyHFO) with a loopback-only `compose.yaml`, plus `scripts/build_offline_bundle.sh` and `install_offline.sh` for SHA256SUMS-verified air-gapped installation.
- Documentation: `README.md`, `workbench/README.md`, `docs/anatomy-labels.md`, `docs/reader-conventions.md`, `docs/evidence-map.md`, `docs/reader-agreement.md`, `docs/mdt-slides.md`, `docs/i18n.md` and `docs/deployment.md`.
