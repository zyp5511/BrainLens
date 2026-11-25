# BrainLens Hospital & Air-Gapped Workstation Deployment Guide

This guide covers reproducible installation of **BrainLens** on clinical research
and hospital workstations—both online and completely air-gapped—via Python
virtual environments or Docker.

> **Clinical Scope**: BrainLens is **not a medical device** and does not
> establish seizure onset zones (SOZ) or standalone surgical resection
> boundaries. All automated counts and model scores require visual verification
> alongside ictal EEG, MRI/PET anatomy, neuropsychology, and electrode coverage
> by qualified clinicians. Clinician, researcher, engineering rehearsal, and AI
> agent judgments remain strictly separated, and separate recordings are never
> pooled.

> **Security Boundary**: BrainLens is a **single-user, loopback-only**
> workstation workbench (`127.0.0.1`). Multi-user access, authentication/SSO,
> RBAC, and centralized audit logging are **out of scope**. Do **not** expose
> the workbench port to a hospital LAN/WAN or run it on an untrusted multi-user
> host.

The Browser UI (`workbench/src/brainlens/assets/`) has **no external CDN,
analytics, or web-font dependencies** and runs 100% offline.

---

## 1. Workstation Installation (Online)

### Prerequisites
- Python **3.11, 3.12, or 3.13** (`>=3.11,<3.14`).
- Local read access to SEEG/iEEG/EEG recordings (`.edf`, `.bdf`, `.fif`/`.fif.gz`,
  BrainVision `.vhdr`/`.eeg`/`.vmrk`, or BIDS directories) and write access to a
  local workspace directory.

### Install into an isolated virtual environment

```bash
python3.12 -m venv /opt/brainlens-venv
/opt/brainlens-venv/bin/pip install --upgrade pip
/opt/brainlens-venv/bin/pip install ./workbench

# Optional: install python-pptx for MDT PowerPoint slide export (`brainlens export-slides`)
/opt/brainlens-venv/bin/pip install "python-pptx>=1.0,<2"
```

This installs three entry points into `/opt/brainlens-venv/bin/`:
- `brainlens` — CLI (`inspect`, `analyze`, `query`, `card`, `export`, `export-edf`, `export-slides`, `export-bids`, `anatomy`, etc.)
- `brainlens-workbench` — local single-user loopback HTTP UI (binds `127.0.0.1:8765` by default)
- `brainlens-mcp` — 7-tool stdio MCP server

### Source-checkout launchers (`workbench/bin/*`)

If running directly from a git checkout, `workbench/bin/{brainlens,brainlens-mcp,brainlens-workbench}`
select the Python runtime in this order:
1. `$BRAINLENS_PYTHON` (or legacy `$BTK_PYTHON`) — fails fast if set to a non-executable path.
2. `<repo>/.venv/bin/python3` (or `python`).
3. `python3` from `PATH`.

Before launching, each wrapper verifies that the chosen interpreter is Python
3.11–3.13 and can import all required BrainLens runtime dependencies, printing
an actionable error to `stderr` if anything is missing.

### Optional PyHFO detection and ONNX classifiers

Without PyHFO, inspection, full-coverage QC, Pro Waveform Reader, Clinical Triad
verification, and case sheets work immediately with `--no-detector`
(`detector_enabled=False`).

PyHFO 4.x is the PyHFO team's packaged engine for `STE`, `MNI`, and `HIL`
detection. Install the extra in the BrainLens venv and fetch all three classifiers
once while connected.

```bash
/opt/brainlens-venv/bin/pip install "brainlens[pyhfo]"
/opt/brainlens-venv/bin/brainlens configure-engine --fetch-models
/opt/brainlens-venv/bin/brainlens inspect "*"
```

Until PyHFO 4.x is on your index, use
`/opt/brainlens-venv/bin/pip install "./workbench[pyhfo]" build/pyhfo-4.0.0-py3-none-any.whl`
from this checkout for the install step. BrainLens discovers the package in its
current interpreter after checking explicit engine settings. To register a
different interpreter, pass `--python /path/to/venv/bin/python` to
`configure-engine`.

The fetch command uses PyHFO's registry to verify the SHA-256 pins for the three
ONNX weights and their specs. To choose a portable cache, set `HF_HUB_CACHE`
to an absolute directory for both the fetch and subsequent BrainLens sessions.
Workers set `HF_HUB_OFFLINE=1` during analysis. The `engine` section of
`inspect "*"` reports the mode, version and available classifiers.

#### Advanced / legacy source engine

Deployments using the legacy source layout pin upstream PyHFO to commit
`c1c7d3fec990e8661604080664766dc1297cbfdc` and register a dedicated interpreter.

```bash
git clone https://github.com/roychowdhuryresearch/pyHFO.git /opt/pyhfo
git -C /opt/pyhfo checkout c1c7d3fec990e8661604080664766dc1297cbfdc

python3.12 -m venv /opt/pyhfo-venv
/opt/pyhfo-venv/bin/pip install \
  "HFODetector>=0.0.23,<0.1" "mne>=1.7,<2" "numpy>=1.26,<2" "scipy>=1.11,<2" \
  "pandas>=2,<3" "p_tqdm>=1.4,<2" "onnxruntime>=1.18,<2" \
  "matplotlib>=3.8,<4" "openpyxl>=3.1,<4"
```

Download the pinned ONNX weights (`model_a`, `model_s`, `model_e`):

```bash
mkdir -p /opt/pyhfo-onnx-models/{model_a,model_s,model_e}
curl -fsSL "https://huggingface.co/roychowdhuryresearch/HFO-artifact/resolve/7f95298b60deff4aae8b1ddfe64acda32a1c1fa7/model.onnx" -o /opt/pyhfo-onnx-models/model_a/model.onnx
curl -fsSL "https://huggingface.co/roychowdhuryresearch/HFO-artifact/resolve/7f95298b60deff4aae8b1ddfe64acda32a1c1fa7/model.spec.json" -o /opt/pyhfo-onnx-models/model_a/model.spec.json
curl -fsSL "https://huggingface.co/roychowdhuryresearch/HFO-spkHFO/resolve/de371a79569e81921e0174a4e440df2383dfdb8b/model.onnx" -o /opt/pyhfo-onnx-models/model_s/model.onnx
curl -fsSL "https://huggingface.co/roychowdhuryresearch/HFO-spkHFO/resolve/de371a79569e81921e0174a4e440df2383dfdb8b/model.spec.json" -o /opt/pyhfo-onnx-models/model_s/model.spec.json
curl -fsSL "https://huggingface.co/roychowdhuryresearch/HFO-eHFO/resolve/2e59449dd81f588bf1a49bde647cd0e6d1db0a8c/model.onnx" -o /opt/pyhfo-onnx-models/model_e/model.onnx
curl -fsSL "https://huggingface.co/roychowdhuryresearch/HFO-eHFO/resolve/2e59449dd81f588bf1a49bde647cd0e6d1db0a8c/model.spec.json" -o /opt/pyhfo-onnx-models/model_e/model.spec.json
```

Register the engine in `~/.brainlens/engine.json` (or via environment variables
`BRAINLENS_PYHFO_SOURCE`, `BRAINLENS_PYHFO_PYTHON`, and
`BRAINLENS_PYHFO_MODEL_CACHE`):

```bash
/opt/brainlens-venv/bin/brainlens configure-engine \
  --source /opt/pyhfo \
  --python /opt/pyhfo-venv/bin/python \
  --model-cache /opt/pyhfo-onnx-models

/opt/brainlens-venv/bin/brainlens inspect "*"
```

PyHFO worker subprocesses automatically set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` so no runtime network calls occur.

---

## 2. Offline / Air-Gapped Workstation Installation

### Bundle BrainLens with the PyHFO 4.x package

On a connected machine matching the target OS, architecture and Python version,
build all wheels together so pip resolves BrainLens and PyHFO against the same
NumPy. From the repository root, use a fresh staging venv and bundle directory.

```bash
python3.12 -m venv build/engine-staging
py="$PWD/build/engine-staging/bin/python"
bundle="$PWD/dist/brainlens-pyhfo"
mkdir -p "$bundle/wheels"
"$py" -m pip wheel --wheel-dir "$bundle/wheels" \
  "./workbench[pyhfo]" build/pyhfo-4.0.0-py3-none-any.whl "python-pptx>=1.0,<2"
"$py" -m pip install --no-index --find-links "$bundle/wheels" "brainlens[pyhfo]"
HF_HUB_CACHE="$bundle/hub" build/engine-staging/bin/brainlens configure-engine --fetch-models
tar -C "$bundle" -czf dist/brainlens-pyhfo.tar.gz wheels hub
(cd dist && shasum -a 256 brainlens-pyhfo.tar.gz > SHA256SUMS.pyhfo)
```

Once your index carries the wheel, omit its local filename from the
`pip wheel` command. The archive preserves the Hugging Face snapshot links
and their target files. Transfer the archive and its checksum together. On the
air-gapped workstation, run the following from the transfer directory.

```bash
shasum -a 256 -c SHA256SUMS.pyhfo
mkdir -p /opt/brainlens-bundle
tar -C /opt/brainlens-bundle -xzf brainlens-pyhfo.tar.gz
python3.12 -m venv /opt/brainlens-venv
/opt/brainlens-venv/bin/pip install --no-index --find-links /opt/brainlens-bundle/wheels \
  "brainlens[pyhfo]" "python-pptx>=1.0,<2"
export HF_HUB_CACHE=/opt/brainlens-bundle/hub
/opt/brainlens-venv/bin/brainlens inspect "*"
```

Set the same `HF_HUB_CACHE` in the workbench service or launch environment.
On Linux, `sha256sum` can replace `shasum -a 256` in both checksum commands.

### Base BrainLens and legacy source bundles

Use `scripts/build_offline_bundle.sh` on a connected staging machine to
build a base BrainLens bundle with `install_offline.sh`. Its optional
`--with-pyhfo` path supports existing legacy source deployments.

#### Step 1 — Build the bundle on a connected machine

For a target workstation matching the build host OS/architecture/Python:

```bash
./scripts/build_offline_bundle.sh --clean
```

For a specific target platform and Python version (for example, an air-gapped
Linux x86_64 workstation running Python 3.12), optionally including PyHFO
engine wheels, checkout, and ONNX weights:

```bash
./scripts/build_offline_bundle.sh \
  --python-version 3.12 \
  --platform manylinux_2_28_x86_64 \
  --with-pyhfo \
  --pyhfo-source /path/to/pyHFO \
  --onnx-models /path/to/pyhfo-onnx-models \
  --clean
```

> **Note on cross-platform PyHFO downloads**: All BrainLens core dependencies
> and `python-pptx` publish binary/pure wheels. If a transitive PyHFO dependency
> only publishes a source distribution for your target platform, run
> `build_offline_bundle.sh` on a connected machine matching the target OS and
> architecture (without `--platform`).

By default the script creates `dist/offline-bundle/` containing:
- `wheels/` — built `brainlens-*.whl`, all runtime dependencies, and `python-pptx>=1.0,<2` (omit with `--no-pptx`).
- `wheels-pyhfo/` — isolated PyHFO dependency wheels (when `--with-pyhfo` is passed).
- `pyhfo/` and `onnx-models/` — pre-fetched engine source and weights (when `--pyhfo-source` / `--onnx-models` are passed).
- `install_offline.sh` — offline installer (`pip install --no-index --find-links`).
- `BUNDLE_MANIFEST.txt` — build metadata.
- `SHA256SUMS` — SHA-256 digests of every file in the bundle.

#### Step 2 — Transfer and verify on the air-gapped workstation

Transfer `dist/offline-bundle/` via approved institutional media, then verify
integrity and install without network access:

```bash
cd offline-bundle

# Optional standalone integrity check
./install_offline.sh --verify-only

# Install BrainLens strictly offline
./install_offline.sh --python python3.12 --venv /opt/brainlens-venv

# Or install both BrainLens and the isolated PyHFO engine venv offline:
./install_offline.sh \
  --python python3.12 \
  --venv /opt/brainlens-venv \
  --with-pyhfo \
  --pyhfo-venv /opt/pyhfo-venv
```

If `pyhfo/` and `onnx-models/` were bundled, copy them to a permanent local
path (e.g., `/opt/pyhfo` and `/opt/pyhfo-onnx-models`) and register them with
`brainlens configure-engine`.

Verify the offline install:

```bash
/opt/brainlens-venv/bin/brainlens inspect "*"
/opt/brainlens-venv/bin/brainlens-workbench
```

---

## 3. Docker Deployment

`docker/Dockerfile` is a multi-stage `python:3.12-slim` image that:
- Installs the built `brainlens` wheel, its runtime dependencies, and `python-pptx>=1.0,<2` into `/opt/brainlens-venv`.
- Runs as a dedicated non-root user (`brainlens`, UID/GID `10001`).
- Persists workspaces under `/data` (`BRAINLENS_WORKSPACE=/data/workbench_results`) and keeps the HMAC sheet trust store separate at `BRAINLENS_TRUST_DIR=/home/brainlens/.brainlens`.
- Expects clinical recordings mounted read-only at `/recordings`.
- Sets `BRAINLENS_NO_BROWSER=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `MPLBACKEND=Agg`.
- Uses a `HEALTHCHECK` against `http://127.0.0.1:8765/api/health`.
- Launches `brainlens-workbench --host 0.0.0.0 --port 8765` as PID 1.

### Loopback-only publishing & port-matching security invariant

Inside a container, the process must bind `0.0.0.0` so Docker's port forwarder
can reach it across the container network namespace. Therefore, **the only safe
host publish is strictly to host loopback on the matching port**:

```bash
-p 127.0.0.1:8765:8765
```

**Why mismatched ports and non-loopback hosts are rejected by design**:
`WorkbenchHTTPServer._check_host` only accepts requests whose `Host` header is
`127.0.0.1`, `localhost`, or `[::1]` **and** whose port matches the port the
server is listening on (`8765`).
- Publishing `-p 127.0.0.1:9000:8765` causes the browser to send
  `Host: 127.0.0.1:9000`, which BrainLens rejects with `403 forbidden_host`.
- Accessing the container via a LAN IP or hostname sends a non-loopback `Host`
  header, which is also rejected with `403 forbidden_host`.
- All mutating `POST` routes additionally require a matching loopback `Origin`,
  `Content-Type: application/json`, and the per-session `X-BrainLens-Token`.

If you must run on a different local port (for example `9000`), match **both**
the loopback publish and the container `--port` argument:

```bash
docker run --rm -p 127.0.0.1:9000:9000 \
  -v brainlens-data:/data \
  -v brainlens-trust:/home/brainlens/.brainlens \
  -v /path/to/recordings:/recordings:ro \
  brainlens:1.0.0 --port 9000
```

### Quick start with Docker Compose

`docker/compose.yaml` enforces `- "127.0.0.1:8765:8765"`, mounts recordings
read-only, isolates the HMAC trust store from `/data`, and applies
`no-new-privileges:true`:

```bash
BRAINLENS_RECORDINGS_DIR=/path/to/hospital/recordings \
  docker compose -f docker/compose.yaml up --build -d
```

Open `http://127.0.0.1:8765/` in your local browser.

### Standalone `docker build` and `docker run`

```bash
# Build base image (no PyHFO engine; QC, reader, triad, exports work out of the box)
docker build -f docker/Dockerfile -t brainlens:1.0.0 .

# Run with loopback-only publish, persistent workspace & trust store, read-only recordings
docker run -d --name brainlens-workbench \
  --security-opt no-new-privileges:true \
  -p 127.0.0.1:8765:8765 \
  -v brainlens-data:/data \
  -v brainlens-trust:/home/brainlens/.brainlens \
  -v /path/to/recordings:/recordings:ro \
  brainlens:1.0.0
```

> **Host bind-mount permissions (Linux)**: Named Docker volumes (`brainlens-data`,
> `brainlens-trust`) inherit `brainlens:brainlens` (`10001:10001`) ownership
> automatically. If you bind-mount host directories for `/data` or
> `/home/brainlens/.brainlens`, ensure they are writable by UID/GID `10001`
> (e.g., `chown -R 10001:10001 /srv/brainlens/data` and `chmod 0700` on the
> trust directory).

### Docker image with the PyHFO 4.x package

Build the base image from this checkout, then install the PyHFO team's wheel
into its BrainLens environment. On the connected build host, fetch the pinned
models as the same user that will run the workbench.

```bash
docker build -f docker/Dockerfile -t brainlens:1.0.0 .
mkdir -p build/engine-image
cp build/pyhfo-4.0.0-py3-none-any.whl build/engine-image/
docker build -t brainlens:1.0.0-pyhfo -f - build/engine-image <<'DOCKERFILE'
FROM brainlens:1.0.0
USER root
COPY pyhfo-*.whl /tmp/
RUN /opt/brainlens-venv/bin/pip install --no-cache-dir /tmp/pyhfo-*.whl "brainlens[pyhfo]"
USER brainlens
ENV HF_HUB_CACHE=/home/brainlens/.pyhfo/cache/huggingface/hub
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 brainlens configure-engine --fetch-models
DOCKERFILE

docker run --rm --network none --entrypoint brainlens brainlens:1.0.0-pyhfo inspect "*"
```

Once PyHFO 4.x is on the container's package index, omit the wheel copy and
install `"brainlens[pyhfo]"` directly. The final image includes the cache and
inherits the base image's offline worker settings, non-root user and entrypoint.

For advanced deployments using the pinned legacy checkout, the repository
Dockerfile still accepts `--build-arg WITH_PYHFO=1`. Supply the source tree and
local ONNX bundle under `docker/vendor/` as described in the
[vendor instructions](../docker/vendor/README.md).

### Air-gapped Docker transfer (`docker save` / `docker load`)

Build on a connected machine, export a checksummed archive, and load on the
offline workstation:

```bash
# Connected machine
docker save brainlens:1.0.0-pyhfo | gzip > brainlens-1.0.0-pyhfo.tar.gz
sha256sum brainlens-1.0.0-pyhfo.tar.gz > brainlens-1.0.0-pyhfo.tar.gz.sha256

# Air-gapped workstation
sha256sum -c brainlens-1.0.0-pyhfo.tar.gz.sha256
docker load -i brainlens-1.0.0-pyhfo.tar.gz
```

### Running CLI commands inside the container

```bash
# Inspect mounted recordings and engine health inside the running container
docker exec brainlens-workbench brainlens inspect "*"
docker exec brainlens-workbench brainlens analyze /recordings/case01/run-01.edf --run-name baseline

# Or run a one-shot CLI command using the same volumes
docker run --rm --entrypoint brainlens \
  -v brainlens-data:/data \
  -v brainlens-trust:/home/brainlens/.brainlens \
  -v /path/to/recordings:/recordings:ro \
  brainlens:1.0.0 inspect "*"
```

---

## 4. Clinical & De-Identified Exports and Anatomy Import

All exports are written under `<workspace>/cases/<case>/exports/` and retain
BrainLens clinical scope disclaimers and English as the canonical export
language.

| Capability | CLI Command | MCP Action (`export_or_protocol`) | Notes |
| :--- | :--- | :--- | :--- |
| **MDT Case Discussion Sheet** | `brainlens export <case> --primary-run baseline` | `action="export_case_sheet"` | Print-ready HTML/PDF signed with HMAC-SHA256 against the out-of-workspace trust store (`BRAINLENS_TRUST_DIR`). |
| **Annotated EDF+ & CSV** | `brainlens export-edf <case> ... [--deidentify] [--pseudonym P001]` | `action="export_edf_annotations"` (optional `deidentify`, `pseudonym`) | Writes an annotation CSV and, for continuous EDF sources, a lossless annotated EDF+ signal copy. By default the original header is kept for in-hospital use; `--deidentify` strips header identity and the recording date for sharing, and `--pseudonym` replaces case/recording names. |
| **MDT Conference Slides (PPTX)** | `brainlens export-slides <case> ...` | `action="export_mdt_slides"` | Generates a PowerPoint slide deck for epilepsy surgery conferences. Requires `python-pptx>=1.0,<2` (pre-installed in Docker and default offline bundle). |
| **BIDS Events TSV/JSON** | `brainlens export-bids <case> ...` | `action="export_bids_events"` | Exports BIDS-compatible `*_events.tsv` and `*_events.json`. |
| **Anatomy Labels Import** | `brainlens anatomy <case> <file> --source ...` | *(CLI import)* | Attaches source-cited anatomical region labels to contacts. Template (`fsaverage`) coordinates are never presented as patient anatomy. |

> **De-identification reminder**: Before moving any exported EDF+, CSV, HTML,
> or PPTX outside the hospital network or clinical enclave, verify outputs
> against institutional HIPAA / GDPR / IRB de-identification policies (for
> example, confirming free-text clinician notes, filenames, and case slugs do
> not contain PHI).

---

## 5. What Is Not Supported

The following scenarios are **explicitly unsupported** in this release:
- **Multi-user hosting, authentication, SSO, or RBAC**: BrainLens has no user
  accounts or login flow. Read-only `GET` endpoints on loopback are
  unauthenticated.
- **Hospital LAN/WAN exposure or multi-user reverse proxies**: Never bind the
  host port to `0.0.0.0`, never forward external traffic to `8765`, and never
  run the workbench on a shared untrusted multi-user host.
- **Enterprise audit logging**: BrainLens maintains append-only per-case
  `judgments.jsonl` and HMAC-signed sheet manifests for single-user integrity,
  not centralized SIEM/HIPAA audit trails.
- **Untrusted shared workspaces executing custom engines**: Workspace-local
  `<workspace>/engine.json` is ignored by default. Do **not** set
  `BRAINLENS_ALLOW_WORKSPACE_ENGINE=1` on shared or untrusted workspaces.

### Large recordings and analysis windows

A recording can exceed 4 GiB. Registration streams the whole source and its
signal companions through SHA-256. For a file of at least 1 GiB, later window
reads in the same process check the file identity and sample its contents. The
threshold is configurable with `BRAINLENS_LARGE_FILE_THRESHOLD_BYTES`.

The check compares the device, inode, size, nanosecond mtime and ctime. It also
covers the first and last 1 MiB and eight evenly spaced interior 1 MiB blocks.
A cold process, a changed file signature, a sampled digest mismatch, or an
unsettled timestamp at either the cached or the current verification triggers
a full hash. The timestamp guard requires both mtime and ctime to be more
than two seconds old. A
size mismatch against registration can be rejected immediately. A metadata-only
touch causes re-verification. A change during a window read or analysis
invalidates that operation. Sampling alone cannot detect a rewrite confined
to unsampled bytes if the filesystem also conceals all identity and timestamp
changes. Use a stable filesystem for source recordings.

The default prepared-data budget is 4 GiB across selected output channels,
including the float64 signal and its validity mask. Configure the limit with
`BRAINLENS_PREPARED_DATA_BUDGET_BYTES`. Set
`overrides.start_seconds` and `overrides.end_seconds` to analyze a shorter
interval, or pass `overrides.segments` to select disjoint windows in one
recording. The budget counts the acquired samples across those windows.
Budget separately for worker memory and analysis artifacts. Quantitative
trends read the full source in bounded blocks, even for a file that would
exceed the detection budget. Allow approximately 72 bytes per channel per
epoch on disk for the nine float64 signal measures, plus the manifest.
At 128 channels and 10-second epochs this is about 76 MiB per 24 hours.
Additional epoch lengths and HFO-rate runs have separate caches.
[Long recording support](long-recordings.md) covers the admission clock,
trend measures, segment selection, platform requirements and exports.

---

## 6. Workspace Backup, Restore & Upgrade

### What to back up

1. **Workspace directory** (`./workbench_results` or Docker volume `brainlens-data` -> `/data/workbench_results`):
   - **Critical human review & export state** (irreplaceable):
     - `cases/<case>/judgments.jsonl` (append-only role-attributed verdicts, clips, notes, questions)
     - `cases/<case>/case.json`, `clinical_overlay.json`, `report_actions*.json*`
     - `cases/<case>/recordings/<rec>/recording.json` and `runs/`
     - `cases/<case>/exports/` and `cards/`
     - `protocols/` and `cohorts/`
   - **Recomputable cache** (optional to exclude on space-constrained backups):
     - `cases/<case>/recordings/<rec>/_stage_cache/` (rebuilt automatically if re-analyzed with source recordings available).
2. **Out-of-workspace HMAC trust store** (`~/.brainlens/` or Docker volume `brainlens-trust` -> `/home/brainlens/.brainlens`):
   - Contains `sheet_hmac.key`, `verified_sheets.json`, and optional `engine.json`.
   - Back up **separately** from the workspace directory and preserve strict
     permissions (`0700` directory, `0600` files). If lost, previously exported
     HTML MDT sheets will show as `unverified` until re-exported.
3. **Source recordings** (read-only clinical archive):
   - BrainLens never modifies source recordings, but `recording.json` stores
     paths and SHA-256 fingerprints so interactive waveform inspection requires
     the original unmodified files at their mounted paths.

### Example Docker backup (stop server first for a clean snapshot)

```bash
docker stop brainlens-workbench

docker run --rm \
  -v brainlens-data:/data:ro \
  -v "$(pwd)/backups":/backup \
  python:3.12-slim \
  tar -czf /backup/brainlens-workspace-$(date +%Y%m%d).tar.gz -C /data workbench_results

docker run --rm \
  -v brainlens-trust:/trust:ro \
  -v "$(pwd)/backups":/backup \
  python:3.12-slim \
  tar -czf /backup/brainlens-trust-$(date +%Y%m%d).tar.gz -C /trust .

chmod 0600 backups/brainlens-trust-*.tar.gz
docker start brainlens-workbench
```

### Safe upgrade procedure

BrainLens preserves recording fingerprints, stage fingerprints, and HMAC sheet
verification formats across compatible upgrades so existing workspaces and runs
continue working without re-analysis.

**Virtualenv workstation**:
1. Stop any running `brainlens-workbench` or `brainlens-mcp` processes.
2. Back up the workspace directory and `~/.brainlens/`.
3. Upgrade the package (online `pip install --upgrade ./workbench` or offline `./install_offline.sh --venv /opt/brainlens-venv`).
4. Verify with `brainlens inspect "*"` and restart `brainlens-workbench`.

**Docker workstation**:
1. Stop the container (`docker compose -f docker/compose.yaml down` or `docker stop brainlens-workbench`) without removing the `brainlens-data` and `brainlens-trust` volumes.
2. Back up both volumes as shown above.
3. Load or build the new image tag and start the container again.
4. Confirm health via `curl -fsS http://127.0.0.1:8765/api/health` and verify registered cases in the UI.
