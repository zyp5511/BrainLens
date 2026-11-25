#!/usr/bin/env python3
"""Reproducible real-motion launch video pipeline for BrainLens 1.0 ("AI co-review for EEG").

Produces:
  - build/brainlens_1_0_launch.mp4        (1920x1080, H.264, 30 fps, ~84 s; silent or muxed with build/voiceover.wav)
  - docs/demo/brainlens_1_0_poster.png    (1920x1080 full-bleed hero poster frame)
  - build/demo-captures/contact_sheet.png (18-frame review contact sheet of the final MP4)
  - build/demo-captures/beat_frames/      (full-resolution 1920x1080 QA frames at each beat)

Stages (run all by default, or selectively with --stage):
  1. prepare  — Verify/fetch real public pediatric scalp EEG CHB-MIT chb01 (42 EDFs,
                23 bipolar channels, 256 Hz, PhysioNet ODC-By 1.0) into
                build/demo-data/chbmit/chb01/, verify SHA-256 against SHA256SUMS.txt,
                write PROVENANCE.json + per-file <stem>.json sidecars (modality=scalp,
                reference=bipolar, expert seizure markers from chb01-summary.txt), and
                inspect chb01 + inspect/analyze real public CC0 SEEG HUP134 (run-01 &
                run-02) in build/demo-workspace (never touches ./workbench_results or
                puts CHB-MIT under inputs/).
  2. capture  — Launch local WorkbenchHTTPServer + headless Chrome over CDP
                (1440x810 CSS @ DPR=2 -> native 2880x1620 Page.startScreencast frames) +
                live stdio MCP client session (brainlens.mcp_server). Drives real input
                (Input.dispatchMouseEvent / mouseWheel / dispatchKeyEvent with jittered
                48-105 ms human typing) and records timestamped screencast frames, exact
                dispatched cursor tracks, and measured UI bounding boxes for every shot,
                in strict chronological state order (cold open filmed before any verdict
                exists on HUP134; Case 1 scalp EEG uses the reader's Ask agent box).
  3. render   — Resamples every recorded screencast shot to 30 fps, composites a plain
                macOS arrow cursor along the exact dispatched coordinates, cuts cleanly
                into and out of steady retina punch-in crops at key reading moments (no
                slow drifting zooms), keeps minimal neutral typography (plain title card,
                two quiet case tags, one plain narrative subtitle per beat, full-bleed real MDT
                .pptx slides, plain end card) with hard cuts between shots, muxes
                build/voiceover.wav when present, and exports poster, beat frames, and
                18-frame contact sheet.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import os
import random
import re
import select
import shutil
import socket
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths & Environment Bootstrap
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
WORKBENCH_SRC = REPO_ROOT / "workbench" / "src"
BUILD_DIR = REPO_ROOT / "build"
DEMO_WS = BUILD_DIR / "demo-workspace"
DEMO_INPUTS = BUILD_DIR / "demo-inputs"
CHBMIT_DIR = BUILD_DIR / "demo-data" / "chbmit" / "chb01"
CHBMIT_BASE_URL = "https://physionet.org/files/chbmit/1.0.0"
CAPTURE_DIR = BUILD_DIR / "demo-captures"
SHOTS_DIR = CAPTURE_DIR / "shots"
TRUST_DIR = BUILD_DIR / "trust"
MPL_DIR = BUILD_DIR / "mpl"
OUT_DIR = REPO_ROOT / "docs" / "demo"
MP4_OUT_PATH = BUILD_DIR / "brainlens_1_0_launch.mp4"
VOICEOVER_WAV_PATH = BUILD_DIR / "voiceover.wav"
POSTER_OUT_PATH = OUT_DIR / "brainlens_1_0_poster.png"

# Ensure build/mpl and build/trust are set before importing matplotlib or brainlens.
TRUST_DIR.mkdir(parents=True, exist_ok=True)
MPL_DIR.mkdir(parents=True, exist_ok=True)
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("BRAINLENS_TRUST_DIR", str(TRUST_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("BRAINLENS_NO_BROWSER", "1")


def _demo_extra_paths() -> list[Path]:
    """Optional extra import paths, portable & machine-independent."""
    candidates: list[Path] = [REPO_ROOT / ".venv-devdeps"]
    for entry in os.environ.get("BRAINLENS_DEMO_EXTRA_PATH", "").split(os.pathsep):
        if entry:
            candidates.append(Path(entry).expanduser())
    return [c for c in candidates if c.is_dir()]


if WORKBENCH_SRC.is_dir():
    if str(WORKBENCH_SRC) in sys.path:
        sys.path.remove(str(WORKBENCH_SRC))
    sys.path.insert(0, str(WORKBENCH_SRC))
for candidate in (
    *_demo_extra_paths(),
    *sorted((BUILD_DIR / "video-venv" / "lib").glob("python*/site-packages")),
):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.append(str(candidate))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Viewport & Video Dimensions
VP_W, VP_H = 1440, 810
CAP_W, CAP_H = VP_W * 2, VP_H * 2  # Native DPR=2 retina capture (2880x1620) for razor-sharp punch-ins
W, H = 1920, 1080
FPS = 30
CSS_TO_PX_X = W / VP_W
CSS_TO_PX_Y = H / VP_H

# Neutral Palette (no radial glows, no neon, no glowing borders)
BG_FLAT = (14, 17, 22)
INK_PRIMARY = (242, 245, 248)
INK_SECONDARY = (182, 190, 202)
INK_MUTED = (130, 140, 154)
INK_FOOTER = (118, 128, 142)


# ---------------------------------------------------------------------------
# Stage 1: Prepare Real Public Datasets — CHB-MIT chb01 (Scalp EEG) & HUP134 (SEEG)
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "BrainLens-Demo/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)
    os.replace(tmp, dest)


def _write_json_if_changed(path: Path, payload: dict[str, Any]) -> bool:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def _parse_chb01_summary(summary_path: Path) -> dict[str, list[tuple[int, int]]]:
    """Parse expert-annotated seizure intervals (start_s, end_s) per EDF from chb01-summary.txt."""
    seizures_by_file: dict[str, list[tuple[int, int]]] = {}
    cur_file: str | None = None
    pending_start: int | None = None
    for raw_line in summary_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        m_file = re.match(r"^File Name:\s*(\S+)", line)
        if m_file:
            cur_file = m_file.group(1)
            seizures_by_file.setdefault(cur_file, [])
            pending_start = None
            continue
        m_start = re.match(r"^Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds", line)
        if m_start and cur_file:
            pending_start = int(m_start.group(1))
            continue
        m_end = re.match(r"^Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds", line)
        if m_end and cur_file and pending_start is not None:
            seizures_by_file[cur_file].append((pending_start, int(m_end.group(1))))
            pending_start = None
    return seizures_by_file


def _prepare_chbmit_chb01() -> dict[str, Any]:
    """Ensure CHB-MIT chb01 EDFs + summary exist, verify SHA-256, and write PROVENANCE + sidecars."""
    CHBMIT_DIR.mkdir(parents=True, exist_ok=True)
    sums_path = CHBMIT_DIR / "SHA256SUMS.txt"
    if not sums_path.is_file():
        print(f"[prepare] Downloading PhysioNet SHA256SUMS.txt into {CHBMIT_DIR} ...")
        _download_file(f"{CHBMIT_BASE_URL}/SHA256SUMS.txt", sums_path)

    expected_hashes: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and parts[1].startswith("chb01/"):
            fname = parts[1].split("/", 1)[1]
            if fname == "chb01-summary.txt" or fname.endswith(".edf"):
                expected_hashes[fname] = parts[0].lower()

    edf_names = sorted(fn for fn in expected_hashes if fn.endswith(".edf"))
    assert len(edf_names) == 42, f"Expected 42 chb01 EDF files in SHA256SUMS.txt, found {len(edf_names)}"

    required_files = ["chb01-summary.txt", *edf_names]
    downloaded = 0
    for fname in required_files:
        fpath = CHBMIT_DIR / fname
        if not fpath.is_file():
            print(f"[prepare] Downloading {fname} from PhysioNet CHB-MIT v1.0.0 ...")
            _download_file(f"{CHBMIT_BASE_URL}/chb01/{fname}", fpath)
            downloaded += 1
        actual = _sha256_file(fpath)
        if actual != expected_hashes[fname]:
            raise RuntimeError(
                f"SHA-256 mismatch for {fpath.name}: expected {expected_hashes[fname]}, got {actual}"
            )
    print(f"[prepare] CHB-MIT chb01: {len(edf_names)} EDFs + chb01-summary.txt SHA-256 verified (downloaded={downloaded}).")

    summary_path = CHBMIT_DIR / "chb01-summary.txt"
    seizures_by_file = _parse_chb01_summary(summary_path)
    total_seizures = sum(len(v) for v in seizures_by_file.values())
    assert total_seizures == 7, f"Expected 7 expert-annotated seizures in chb01-summary.txt, found {total_seizures}"

    provenance_doc = {
        "dataset": "CHB-MIT Scalp EEG Database",
        "version": "1.0.0",
        "url": "https://physionet.org/content/chbmit/1.0.0/",
        "files_url": f"{CHBMIT_BASE_URL}/chb01/",
        "license": "Open Data Commons Attribution License v1.0 (ODC-By 1.0)",
        "citation": [
            "Shoeb AH. Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment. PhD Thesis, Massachusetts Institute of Technology, 2009.",
            "Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220, 2000.",
        ],
        "subject": "chb01",
        "description": "real public pediatric scalp EEG",
        "same_subject_association": {
            "name": "CHB-MIT Scalp EEG Database v1.0.0",
            "public_url": "https://physionet.org/content/chbmit/1.0.0/",
            "license": "ODC-By 1.0",
            "participant_identity": {
                "participant_id": "chb01",
                "modality": "scalp",
            },
            "same_patient_basis": (
                "All 42 EDF recordings belong to subject chb01 in the published CHB-MIT Scalp EEG Database v1.0.0 "
                "(PhysioNet, Children's Hospital Boston). Real public pediatric scalp EEG, distinct from HUP134."
            ),
        },
        "limits": [
            "Real public pediatric scalp EEG (23 bipolar double-banana channels, 256 Hz); surrogate EDF dates preserve relative admission timing and inter-file gaps.",
            "Seizure markers come from CHB-MIT chb01-summary.txt expert annotations (not automated seizure detection); quantitative trends are for navigation only.",
        ],
    }
    _write_json_if_changed(CHBMIT_DIR / "PROVENANCE.json", provenance_doc)

    sidecars_written = 0
    for fname in edf_names:
        stem = Path(fname).stem
        markers = [
            {
                "start": float(s0),
                "end": float(s1),
                "label": "Seizure",
                "source": "CHB-MIT chb01-summary.txt expert annotation",
            }
            for s0, s1 in seizures_by_file.get(fname, [])
        ]
        sidecar_doc = {
            "modality": "scalp",
            "reference": "bipolar",
            "reference_description": "CHB-MIT 23-channel bipolar double-banana scalp montage (256 Hz)",
            "line_freq": 60,
            "markers": markers,
        }
        if _write_json_if_changed(CHBMIT_DIR / f"{stem}.json", sidecar_doc):
            sidecars_written += 1

    print(
        f"[prepare] CHB-MIT chb01 metadata ready: PROVENANCE.json + {len(edf_names)} sidecars "
        f"({sidecars_written} updated, {total_seizures} expert-annotated seizures)."
    )
    return {
        "edf_count": len(edf_names),
        "seizures_by_file": {k: [[s0, s1] for s0, s1 in v] for k, v in seizures_by_file.items() if v},
        "total_seizures": total_seizures,
    }


def _reset_demo_interactive_state(ws: Path) -> None:
    """Clear mutable judgments, report actions, open questions, and exports on demo cases.

    Keeps the cached PyHFO baseline analysis runs intact so Stage 2 (`capture`) starts
    with zero verdicts and zero pending/resolved actions every time on both HUP134 and chb01.
    """
    from brainlens.pipeline import refresh_verdict_tallies
    from brainlens.storage import WorkbenchStore

    store = WorkbenchStore(ws)
    for case_slug in ("HUP134", "chb01"):
        if not store.has_case(case_slug):
            continue
        cdir = store.case_dir(case_slug)
        for fname in (
            "judgments.jsonl",
            "report_actions.jsonl",
            "report_actions_cursor.json",
            "anatomy_labels.json",
            "clinical_overlay.json",
            "question_ref.json",
        ):
            p = cdir / fname
            if p.exists():
                p.unlink()
        for sub in ("exports", "cards", "review_sets"):
            sd = cdir / sub
            if sd.is_dir():
                shutil.rmtree(sd)
        case_json = cdir / "case.json"
        if case_json.is_file():
            doc = json.loads(case_json.read_text(encoding="utf-8"))
            doc["open_questions"] = []
            case_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    for fname in ("report_actions.jsonl", "report_actions_state.json", "ui_state.json"):
        p = ws / fname
        if p.exists():
            p.unlink()
    if store.has_case("HUP134"):
        for rec in ("run-01", "run-02"):
            if store.has_run("HUP134", rec, "baseline"):
                refresh_verdict_tallies(store, "HUP134", rec, "baseline")


def prepare_workspace(clean: bool = False) -> dict[str, Any]:
    """Stage 1: prepare real CHB-MIT chb01 (scalp EEG) and real HUP134 (SEEG) in build/demo-workspace."""
    from brainlens.io_readers import inspect_case
    from brainlens.pipeline import analyze_case
    from brainlens.storage import WorkbenchStore
    from brainlens.timeline import admission_timeline
    from brainlens.trends import recording_trends

    if clean:
        for p in (DEMO_WS, DEMO_INPUTS, CAPTURE_DIR):
            if p.exists():
                shutil.rmtree(p)
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    DEMO_WS.mkdir(parents=True, exist_ok=True)
    store = WorkbenchStore(DEMO_WS)

    # 1A. CHB-MIT chb01 (42 EDF files, ~40.55 h recorded)
    chb_meta = _prepare_chbmit_chb01()
    existing_chb_recs = store.list_recordings("chb01") if store.has_case("chb01") else []
    if len(existing_chb_recs) != 42:
        t0_chb = time.monotonic()
        print("[prepare] Inspecting real scalp EEG case chb01 (42 EDFs) into build/demo-workspace ...")
        inspect_case(str(CHBMIT_DIR), workspace_dir=DEMO_WS, case_name="chb01", persist=True)
        print(f"[prepare] Inspected chb01 (42 recordings) in {time.monotonic() - t0_chb:.1f}s")
    else:
        print("[prepare] Reusing registered chb01 case (42 recordings) in build/demo-workspace")

    chb_tl = admission_timeline(store, "chb01", text="seizure")
    total_rec_s = sum(float(r["duration_seconds"]) for r in chb_tl["recordings"])
    max_end_s = max(
        (float(r["offset_seconds"]) + float(r["duration_seconds"]) for r in chb_tl["recordings"] if r.get("offset_seconds") is not None),
        default=total_rec_s,
    )
    print(
        f"[prepare] chb01 admission timeline: {len(chb_tl['recordings'])} files, "
        f"{total_rec_s / 3600:.1f}h recorded across {max_end_s / 3600:.1f}h span, "
        f"{len(chb_tl['gaps'])} inter-file gaps, {len(chb_tl['events'])} seizure markers."
    )

    for sz_rec in ("chb01_03", "chb01_16"):
        if sz_rec in store.list_recordings("chb01"):
            recording_trends(store, "chb01", sz_rec, measure="line_length", epoch_seconds=10)

    # 1B. HUP134 (run-01 & run-02)
    hup_dir = REPO_ROOT / "inputs" / "HUP134"
    if not (hup_dir / "run-01.edf").is_file() or not (hup_dir / "run-02.edf").is_file():
        raise FileNotFoundError(
            f"Real HUP134 EDF recordings not found in {hup_dir} (expected run-01.edf and run-02.edf)."
        )

    print("[prepare] Inspecting & analyzing real SEEG case inputs/HUP134 (run-01 & run-02) ...")
    inspect_case(str(hup_dir), workspace_dir=DEMO_WS, persist=True)

    analyze_elapsed: dict[str, float] = {}
    for rec in ("run-01", "run-02"):
        if not store.has_run("HUP134", rec, "baseline"):
            t0 = time.monotonic()
            analyze_case(
                f"HUP134/{rec}",
                workspace_dir=DEMO_WS,
                run_name="baseline",
                overrides={"classifier_enabled": True, "classifier_stages": ["artifact", "spike"]},
            )
            elapsed = round(time.monotonic() - t0, 2)
            analyze_elapsed[rec] = elapsed
            print(f"[prepare] HUP134/{rec} baseline completed in {elapsed:.1f}s")
        else:
            b_existing = store.load_run("HUP134", rec, "baseline")
            analyze_elapsed[rec] = round(float(b_existing["meta"].get("elapsed_seconds") or 18.0), 2)
            print(f"[prepare] Reusing cached HUP134/{rec} baseline run")

    _reset_demo_interactive_state(DEMO_WS)

    b_r1 = store.load_run("HUP134", "run-01", "baseline")
    b_r2 = store.load_run("HUP134", "run-02", "baseline")
    focal_r1 = [e for e in b_r1["events"] if e.get("synchrony_class") == "focal"]
    focal_r2 = [e for e in b_r2["events"] if e.get("synchrony_class") == "focal"]
    burst_windows_r1 = b_r1["summary"].get("synchronous_burst_windows") or []

    manifest_path = CAPTURE_DIR / "manifest.json"
    prev_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest = {
        **prev_manifest,
        "case": "HUP134",
        "scalp_case": "chb01",
        "chb01": {
            "edf_count": len(chb_tl["recordings"]),
            "recorded_hours": round(total_rec_s / 3600.0, 2),
            "span_hours": round(max_end_s / 3600.0, 2),
            "inter_file_gap_count": len(chb_tl["gaps"]),
            "seizure_events": chb_tl["events"],
            "seizures_by_file": chb_meta["seizures_by_file"],
        },
        "analyze_elapsed_s": analyze_elapsed,
        "hup134_run01_counts": b_r1["summary"]["counts"],
        "hup134_run02_counts": b_r2["summary"]["counts"],
        "hup134_run01_leaders": [c["channel_name"] for c in b_r1["summary"].get("leading_focal_channels") or []],
        "hup134_run02_leaders": [c["channel_name"] for c in b_r2["summary"].get("leading_focal_channels") or []],
        "hup134_run01_focal_events": [
            {
                "id": e["id"],
                "channel": e["channel_name"],
                "start_seconds": e["start_seconds"],
                "end_seconds": e["end_seconds"],
                "artifact_score": e.get("artifact_score"),
                "artifact_label": e.get("artifact_label"),
            }
            for e in focal_r1
        ],
        "hup134_run02_focal_events": [
            {
                "id": e["id"],
                "channel": e["channel_name"],
                "start_seconds": e["start_seconds"],
                "end_seconds": e["end_seconds"],
                "artifact_score": e.get("artifact_score"),
                "artifact_label": e.get("artifact_label"),
            }
            for e in focal_r2
        ],
        "hup134_run01_burst_windows": burst_windows_r1,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[prepare] Stage 1 complete: chb01={len(chb_tl['recordings'])} files / {len(chb_tl['events'])} seizures; "
        f"HUP134 run-01={b_r1['summary']['counts']['total_raw_events']} raw "
        f"({b_r1['summary']['counts']['widespread_burst_episodes']} bursts, "
        f"{b_r1['summary']['counts']['focal_events']} focal: {manifest['hup134_run01_leaders']}); "
        f"run-02={b_r2['summary']['counts']['focal_events']} focal ({manifest['hup134_run02_leaders']})."
    )
    return manifest


# ---------------------------------------------------------------------------
# Stage 2: Headless Chrome CDP Screencast + Real Input + Live Stdio MCP
# ---------------------------------------------------------------------------

def _find_chrome() -> str:
    pw_shells = (
        sorted(
            (Path.home() / "Library" / "Caches" / "ms-playwright").glob(
                "chromium_headless_shell-*/**/chrome-headless-shell"
            ),
            reverse=True,
        )
        if (Path.home() / "Library" / "Caches" / "ms-playwright").is_dir()
        else []
    )
    cands = [
        os.environ.get("BRAINLENS_CHROME_PATH", ""),
        *(str(p) for p in pw_shells),
        shutil.which("chrome-headless-shell") or "",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    for c in cands:
        if c and Path(c).is_file():
            return c
    raise FileNotFoundError("Headless Chrome / chrome-headless-shell not found.")


def _ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 4 * t * t * t if t < 0.5 else 1.0 - ((-2 * t + 2) ** 3) / 2.0


def _ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


class _MiniCDP:
    """Raw-socket CDP client supporting asynchronous Page.startScreencast & real input."""

    def __init__(self, ws_url: str) -> None:
        rest = ws_url[5:]
        host_port, path = rest.split("/", 1)
        host, port_s = host_port.split(":")
        self.sock = socket.create_connection((host, int(port_s)), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET /{path} HTTP/1.1\r\nHost: {host_port}\r\n"
            f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.sock.recv(4096)
        assert b"101 " in buf.split(b"\r\n", 1)[0], buf
        self._extra = buf.split(b"\r\n\r\n", 1)[1]
        self._id = 0
        self.exceptions: list[str] = []
        self.console_errors: list[str] = []

        # Active screencast shot state
        self._recording = False
        self._shot_t0 = 0.0
        self._shot_dir: Path | None = None
        self._shot_frames: list[tuple[float, str]] = []
        self._shot_cursor: list[tuple[float, float, float, bool]] = []
        self._shot_marks: dict[str, float] = {}
        self._shot_boxes: dict[str, dict[str, float]] = {}
        self.cur_x: float = VP_W * 0.5
        self.cur_y: float = VP_H * 0.5
        self.cur_visible: bool = True

    def _recv_exact(self, n: int) -> bytes:
        while len(self._extra) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EOFError("CDP websocket closed")
            self._extra += chunk
        out = self._extra[:n]
        self._extra = self._extra[n:]
        return out

    def _read_frame(self) -> tuple[int, bytes]:
        b1, b2 = self._recv_exact(2)
        opcode = b1 & 0x0F
        masked = (b2 & 0x80) != 0
        length = b2 & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        mask = self._recv_exact(4) if masked else b""
        payload = self._recv_exact(length)
        if masked:
            payload = bytes(payload[i] ^ mask[i % 4] for i in range(length))
        return opcode, payload

    def _send_text(self, text: str) -> None:
        data = text.encode("utf-8")
        mask = os.urandom(4)
        header = bytearray([0x81])
        n = len(data)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", n))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", n))
        header.extend(mask)
        masked = bytes(data[i] ^ mask[i % 4] for i in range(n))
        self.sock.sendall(bytes(header) + masked)

    def _handle_event(self, msg: dict[str, Any]) -> None:
        m = msg.get("method")
        p = msg.get("params") or {}
        if m == "Page.screencastFrame":
            sid = p["sessionId"]
            if self._recording and self._shot_dir is not None:
                t_rel = max(0.0, time.monotonic() - self._shot_t0)
                idx = len(self._shot_frames)
                fname = f"f_{idx:05d}.png"
                (self._shot_dir / fname).write_bytes(base64.b64decode(p["data"]))
                self._shot_frames.append((t_rel, fname))
            self._id += 1
            self._send_text(
                json.dumps({"id": self._id, "method": "Page.screencastFrameAck", "params": {"sessionId": sid}})
            )
        elif m == "Runtime.exceptionThrown":
            det = p.get("exceptionDetails") or {}
            self.exceptions.append(f"{det.get('text', '')}: {(det.get('exception') or {}).get('description', '')}")
        elif m == "Runtime.consoleAPICalled" and p.get("type") == "error":
            args = [str(a.get("value", a.get("description", ""))) for a in p.get("args", [])]
            self.console_errors.append(" ".join(args))

    def _drain_ready(self) -> None:
        """Non-blocking drain of any pending websocket frames already available."""
        while True:
            if not self._extra:
                r, _, _ = select.select([self.sock], [], [], 0.0)
                if not r:
                    return
            op, payload = self._read_frame()
            if op == 8:
                raise EOFError("CDP closed")
            if op == 9:
                self.sock.sendall(b"\x8a\x80\x00\x00\x00\x00")
                continue
            if op == 1:
                msg = json.loads(payload.decode("utf-8"))
                if "method" in msg:
                    self._handle_event(msg)

    def pump(self, duration: float) -> None:
        """Sleep for `duration` seconds while actively pumping & acking CDP screencast frames."""
        end = time.monotonic() + max(0.0, duration)
        while True:
            self._drain_ready()
            rem = end - time.monotonic()
            if rem <= 0:
                break
            r, _, _ = select.select([self.sock], [], [], min(0.015, rem))
            if r:
                self._drain_ready()

    def send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._id += 1
        cid = self._id
        self._send_text(json.dumps({"id": cid, "method": method, "params": params or {}}))
        while True:
            op, payload = self._read_frame()
            if op == 8:
                raise EOFError("CDP closed")
            if op == 9:
                self.sock.sendall(b"\x8a\x80\x00\x00\x00\x00")
                continue
            if op != 1:
                continue
            msg = json.loads(payload.decode("utf-8"))
            if msg.get("id") == cid:
                if "error" in msg:
                    raise RuntimeError(f"CDP error {method}: {msg['error']}")
                return msg.get("result", {})
            if "method" in msg:
                self._handle_event(msg)

    def eval(self, expr: str) -> Any:
        res = self.send("Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True})
        if res.get("exceptionDetails"):
            raise RuntimeError(f"JS exception: {res['exceptionDetails']}")
        return (res.get("result") or {}).get("value")

    def wait_for(self, expr: str, timeout: float = 20.0, interval: float = 0.08) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.eval(expr):
                return
            self.pump(interval)
        raise TimeoutError(f"Timed out waiting for JS condition: {expr}")

    def settle(self, extra_s: float = 0.20) -> None:
        self.eval("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
        if extra_s > 0:
            self.pump(extra_s)

    def _capture_native_png_bytes(self) -> bytes:
        res = self.send("Page.captureScreenshot", {"format": "png", "fromSurface": True})
        return base64.b64decode(res["data"])

    def _log_cursor(self, x: float, y: float, visible: bool | None = None) -> None:
        self.cur_x = float(x)
        self.cur_y = float(y)
        if visible is not None:
            self.cur_visible = bool(visible)
        if self._recording:
            t_rel = max(0.0, time.monotonic() - self._shot_t0)
            self._shot_cursor.append((t_rel, self.cur_x, self.cur_y, self.cur_visible))

    def record_box(self, name: str, rect: dict[str, Any] | None) -> None:
        if not rect:
            return
        self._shot_boxes[name] = {
            "x": round(float(rect["x"]), 2),
            "y": round(float(rect["y"]), 2),
            "w": round(float(rect["w"]), 2),
            "h": round(float(rect["h"]), 2),
        }

    # --- Shot lifecycle ---

    def start_shot(
        self,
        shot_id: str,
        start_cursor: tuple[float, float] | None = None,
        cursor_visible: bool = True,
    ) -> None:
        sdir = SHOTS_DIR / shot_id
        if sdir.exists():
            shutil.rmtree(sdir)
        sdir.mkdir(parents=True, exist_ok=True)
        self._shot_dir = sdir
        self._shot_frames = []
        self._shot_cursor = []
        self._shot_marks = {}
        self._shot_boxes = {}
        self.cur_visible = cursor_visible
        if start_cursor is not None:
            self.cur_x, self.cur_y = float(start_cursor[0]), float(start_cursor[1])
            self.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": self.cur_x, "y": self.cur_y})
            self.pump(0.04)

        # Guarantee crisp native retina frame at t=0.0 before starting screencast
        self.settle(0.05)
        f0 = self._capture_native_png_bytes()
        (sdir / "f_00000.png").write_bytes(f0)
        self._shot_frames.append((0.0, "f_00000.png"))

        self._shot_t0 = time.monotonic()
        self._recording = True
        self._shot_cursor.append((0.0, self.cur_x, self.cur_y, self.cur_visible))
        self.send(
            "Page.startScreencast",
            {"format": "png", "maxWidth": CAP_W, "maxHeight": CAP_H, "everyNthFrame": 1},
        )
        print(f"[shot] START {shot_id}")

    def mark(self, name: str) -> float:
        t_rel = max(0.0, time.monotonic() - self._shot_t0) if self._recording else 0.0
        self._shot_marks[name] = round(t_rel, 4)
        return t_rel

    def stop_shot(self) -> dict[str, Any]:
        assert self._recording and self._shot_dir is not None
        self.pump(0.08)
        dur = max(0.1, time.monotonic() - self._shot_t0)
        self._shot_cursor.append((dur, self.cur_x, self.cur_y, self.cur_visible))
        # Capture one final crisp native state frame to guarantee latest paint is present
        f_end = self._capture_native_png_bytes()
        end_name = f"f_{len(self._shot_frames):05d}.png"
        (self._shot_dir / end_name).write_bytes(f_end)
        self._shot_frames.append((dur, end_name))
        self.send("Page.stopScreencast")
        self._recording = False
        self.pump(0.05)

        meta = {
            "shot_id": self._shot_dir.name,
            "dir": str(self._shot_dir),
            "duration_s": round(dur, 4),
            "frame_count": len(self._shot_frames),
            "frames": [[round(t, 4), fn] for t, fn in self._shot_frames],
            "cursor": [[round(t, 4), round(x, 2), round(y, 2), vis] for t, x, y, vis in self._shot_cursor],
            "marks": dict(self._shot_marks),
            "boxes": dict(self._shot_boxes),
        }
        (self._shot_dir / "shot.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(
            f"[shot] STOP  {self._shot_dir.name}: {dur:.2f}s, {len(self._shot_frames)} frames "
            f"({len(self._shot_frames) / max(0.05, dur):.1f} fps), marks={self._shot_marks}"
        )
        self._shot_dir = None
        return meta

    # --- Real Human-Paced Input Primitives ---

    def hold(self, seconds: float) -> None:
        self._log_cursor(self.cur_x, self.cur_y)
        self.pump(seconds)
        self._log_cursor(self.cur_x, self.cur_y)

    def move_to(
        self,
        tx: float,
        ty: float,
        duration: float = 0.55,
        buttons: int = 0,
        curve_sign: float = 1.0,
    ) -> None:
        """Move cursor from current position to (tx, ty) on a subtly curved cubic-eased path."""
        sx, sy = self.cur_x, self.cur_y
        dx, dy = tx - sx, ty - sy
        dist = math.hypot(dx, dy)
        steps = max(8, int(round(duration * 55)))
        # Subtle quadratic Bézier arc (~3.5% of distance, capped at 26 CSS px)
        arc = min(26.0, dist * 0.035) * curve_sign
        if dist > 1.0:
            nx, ny = -dy / dist, dx / dist
        else:
            nx, ny = 0.0, 0.0
        mx = (sx + tx) * 0.5 + nx * arc
        my = (sy + ty) * 0.5 + ny * arc

        t_start = time.monotonic()
        for k in range(1, steps + 1):
            u = _ease_in_out_cubic(k / steps)
            om = 1.0 - u
            cx = om * om * sx + 2 * om * u * mx + u * u * tx
            cy = om * om * sy + 2 * om * u * my + u * u * ty
            ev: dict[str, Any] = {"type": "mouseMoved", "x": cx, "y": cy}
            if buttons:
                ev["button"] = "left"
                ev["buttons"] = buttons
            self.send("Input.dispatchMouseEvent", ev)
            self._log_cursor(cx, cy, visible=True)
            target_t = t_start + duration * (k / steps)
            rem = target_t - time.monotonic()
            if rem > 0.002:
                self.pump(rem)

    def click_at(self, x: float, y: float, move_dur: float = 0.50, pre_pause: float = 0.06, hold_s: float = 0.065) -> None:
        self.move_to(x, y, duration=move_dur)
        if pre_pause > 0:
            self.pump(pre_pause)
        self.send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": x, "y": y, "button": "left", "buttons": 1, "clickCount": 1})
        self._log_cursor(x, y, visible=True)
        self.pump(hold_s)
        self.send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x, "y": y, "button": "left", "buttons": 0, "clickCount": 1})
        self._log_cursor(x, y, visible=True)

    def drag_select(self, x0: float, y0: float, x1: float, y1: float, move_dur: float = 0.50, drag_dur: float = 0.40) -> None:
        self.move_to(x0, y0, duration=move_dur)
        self.pump(0.08)
        self.send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": x0, "y": y0, "button": "left", "buttons": 1, "clickCount": 1})
        self._log_cursor(x0, y0, visible=True)
        self.pump(0.04)
        self.move_to(x1, y1, duration=drag_dur, buttons=1, curve_sign=0.0)
        self.pump(0.04)
        self.send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x1, "y": y1, "button": "left", "buttons": 0, "clickCount": 1})
        self._log_cursor(x1, y1, visible=True)

    def type_text(self, text: str, rng: random.Random, min_ms: float = 45.0, max_ms: float = 110.0) -> None:
        """Type text character by character via Input.dispatchKeyEvent with jittered human timing."""
        for ch in text:
            self.send("Input.dispatchKeyEvent", {"type": "keyDown", "text": ch, "unmodifiedText": ch, "key": ch})
            self.send("Input.dispatchKeyEvent", {"type": "keyUp", "key": ch})
            delay_ms = rng.uniform(min_ms, max_ms)
            if ch in " ?-":
                delay_ms = min(max_ms, delay_ms + rng.uniform(8.0, 22.0))
            self.pump(delay_ms / 1000.0)

    def press_key(self, key: str, code: str, vk: int, text: str = "") -> None:
        kd: dict[str, Any] = {"type": "keyDown", "key": key, "code": code, "windowsVirtualKeyCode": vk}
        if text:
            kd["text"] = text
            kd["unmodifiedText"] = text
        self.send("Input.dispatchKeyEvent", kd)
        self.pump(0.06)
        self.send("Input.dispatchKeyEvent", {"type": "keyUp", "key": key, "code": code, "windowsVirtualKeyCode": vk})

    def wheel_scroll(self, x: float, y: float, total_delta_y: float, duration: float = 1.4, steps: int = 32) -> None:
        """Smooth eased vertical mouseWheel scroll at (x, y)."""
        self.move_to(x, y, duration=0.35)
        t_start = time.monotonic()
        prev_u = 0.0
        for k in range(1, steps + 1):
            u = _ease_in_out_cubic(k / steps)
            dy = (u - prev_u) * total_delta_y
            prev_u = u
            if abs(dy) > 0.1:
                self.send("Input.dispatchMouseEvent", {"type": "mouseWheel", "x": x, "y": y, "deltaX": 0, "deltaY": dy})
            target_t = t_start + duration * (k / steps)
            rem = target_t - time.monotonic()
            if rem > 0.002:
                self.pump(rem)

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Font Helpers & Real PPTX -> 1920x1080 Slide Renderer
# ---------------------------------------------------------------------------

_SF_PRO = "/System/Library/Fonts/SFNS.ttf"
_SF_MONO = "/System/Library/Fonts/SFNSMono.ttf"
_FONT_CACHE: dict[tuple[str, int, int], ImageFont.FreeTypeFont] = {}


def get_font(family: str, size: int, weight: int = 400) -> ImageFont.FreeTypeFont:
    key = (family, size, weight)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    if family == "mono" and Path(_SF_MONO).is_file():
        f = ImageFont.truetype(_SF_MONO, size)
        try:
            w = max(294, min(900, weight))
            f.set_variation_by_axes([w, w])
        except Exception:
            pass
    elif Path(_SF_PRO).is_file():
        f = ImageFont.truetype(_SF_PRO, size)
        try:
            opsz = 96 if size >= 44 else (48 if size >= 28 else 20)
            f.set_variation_by_axes([100, opsz, 400, max(100, min(900, weight))])
        except Exception:
            pass
    else:
        f = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", size)
    _FONT_CACHE[key] = f
    return f


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    out: list[str] = []
    for para in text.splitlines() or [""]:
        words = para.split(" ")
        cur = ""
        for w in words:
            cand = f"{cur} {w}".strip() if cur else w
            if draw.textlength(cand, font=font) <= max_w or not cur:
                cur = cand
            else:
                out.append(cur)
                cur = w
        out.append(cur)
    return out


def render_pptx_slides(pptx_path: Path, out_dir: Path) -> list[dict[str, Any]]:
    """Render every slide of a real BrainLens MDT .pptx deck to native 2880x1620 PNGs."""
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    out_dir.mkdir(parents=True, exist_ok=True)
    prs = Presentation(str(pptx_path))
    W_s, H_s = CAP_W, CAP_H
    sc = W_s / W
    sw = float(prs.slide_width)
    sh = float(prs.slide_height)
    sx = W_s / sw
    sy = H_s / sh
    pt_to_px = (W_s / 13.333333) / 72.0

    rendered: list[dict[str, Any]] = []
    for idx, slide in enumerate(prs.slides, start=1):
        img = Image.new("RGB", (W_s, H_s), (252, 253, 254))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, W_s, int(round(8 * sc))], fill=(20, 110, 113))
        draw.line(
            [(int(round(80 * sc)), int(round(106 * sc))), (W_s - int(round(80 * sc)), int(round(106 * sc)))],
            fill=(226, 232, 240),
            width=max(2, int(round(2 * sc))),
        )

        slide_title = ""
        slide_context = ""
        body_texts: list[str] = []

        for shape in slide.shapes:
            x0 = int(round(shape.left * sx))
            y0 = int(round(shape.top * sy))
            w0 = int(round(shape.width * sx))
            h0 = int(round(shape.height * sy))

            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                pic = Image.open(io.BytesIO(shape.image.blob)).convert("RGBA")
                pic_resized = pic.resize((max(1, w0), max(1, h0)), Image.Resampling.LANCZOS)
                img.paste(pic_resized, (x0, y0), pic_resized)
            elif shape.has_table:
                tbl = shape.table
                col_ws = [int(round(c.width * sx)) for c in tbl.columns]
                n_rows = len(tbl.rows)
                row_h = max(int(round(56 * sc)), int(round(h0 / max(1, n_rows))))
                cy = y0
                for r_i, row in enumerate(tbl.rows):
                    cx = x0
                    for c_i, cell in enumerate(row.cells):
                        cw = col_ws[c_i]
                        fill_rgb = (23, 45, 67) if r_i == 0 else ((234, 241, 242) if r_i % 2 else (248, 250, 250))
                        try:
                            fc = cell.fill.fore_color.rgb
                            if fc is not None:
                                fill_rgb = (fc[0], fc[1], fc[2])
                        except Exception:
                            pass
                        draw.rectangle(
                            [cx, cy, cx + cw, cy + row_h],
                            fill=fill_rgb,
                            outline=(214, 224, 228),
                            width=max(1, int(round(sc))),
                        )
                        txt = cell.text.strip()
                        fg = (255, 255, 255) if r_i == 0 else (23, 45, 67)
                        fsize = int(round((14.0 if r_i == 0 else 15.0) * pt_to_px))
                        font = get_font("sans", max(int(round(14 * sc)), fsize), 700 if r_i == 0 else 500)
                        lines = _wrap_text(draw, txt, font, max(int(round(30 * sc)), cw - int(round(18 * sc))))
                        ty = cy + max(int(round(6 * sc)), (row_h - len(lines) * int(fsize * 1.24)) // 2)
                        for ln in lines[:3]:
                            draw.text((cx + int(round(10 * sc)), ty), ln, font=font, fill=fg)
                            ty += int(fsize * 1.22)
                        cx += cw
                    cy += row_h
            elif shape.has_text_frame:
                tf = shape.text_frame
                cy = y0
                full_t = tf.text.strip()
                if full_t:
                    body_texts.append(full_t)
                for p in tf.paragraphs:
                    txt = p.text
                    if not txt.strip():
                        cy += int(round(22 * sc))
                        continue
                    pt_sz = p.font.size.pt if p.font.size else 20.0
                    px_sz = max(int(round(16 * sc)), int(round(pt_sz * pt_to_px)))
                    is_bold = bool(p.font.bold)
                    if not slide_context and y0 < int(round(90 * sc)) and pt_sz <= 18:
                        slide_context = txt.strip()
                    if is_bold and pt_sz >= 30 and not slide_title:
                        slide_title = txt.strip()
                    fg = (23, 45, 67)
                    try:
                        if p.font.color and p.font.color.rgb:
                            rgb = p.font.color.rgb
                            fg = (rgb[0], rgb[1], rgb[2])
                    except Exception:
                        pass
                    font = get_font("sans", px_sz, 700 if is_bold else 400)
                    for ln in _wrap_text(draw, txt, font, max(int(round(80 * sc)), w0)):
                        if cy + px_sz > H_s - int(round(24 * sc)):
                            break
                        draw.text((x0, cy), ln, font=font, fill=fg)
                        cy += int(px_sz * 1.28)
                    cy += int(px_sz * 0.22)

        out_p = out_dir / f"slide_{idx:02d}.png"
        img.save(out_p)
        rendered.append(
            {
                "index": idx,
                "title": slide_title,
                "context": slide_context,
                "text_preview": " | ".join(body_texts)[:800],
                "path": str(out_p),
            }
        )
    return rendered


# ---------------------------------------------------------------------------
# Stage 2 Execution: Live Chronological Co-Review Screencast Capture
# ---------------------------------------------------------------------------

async def _run_live_capture_session(manifest: dict[str, Any]) -> dict[str, Any]:
    """Run WorkbenchHTTPServer + Chrome CDP screencast + live stdio MCP in chronological order."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from brainlens.workbench_http import WorkbenchHTTPServer

    # Reset HUP134 so Beat 1 (cold open) starts with ZERO verdicts and ZERO pending actions!
    _reset_demo_interactive_state(DEMO_WS)

    if SHOTS_DIR.exists():
        shutil.rmtree(SHOTS_DIR)
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)

    chrome_bin = _find_chrome()
    srv = WorkbenchHTTPServer(workspace_dir=DEMO_WS, port=0)
    base_url = srv.start().rstrip("/")
    print(f"[capture] Live WorkbenchHTTPServer listening at {base_url}")

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dbg_port = s.getsockname()[1]

    user_data = BUILD_DIR / "chrome-demo-profile"
    if user_data.exists():
        shutil.rmtree(user_data)
    user_data.mkdir(parents=True, exist_ok=True)

    # Launch headless Chrome at 1440x810 CSS @ DPR=2 so Page.startScreencast delivers native 1920x1080 PNGs
    chrome = subprocess.Popen(
        [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--no-sandbox",
            f"--remote-debugging-port={dbg_port}",
            f"--user-data-dir={user_data}",
            f"--window-size={VP_W},{VP_H}",
            "--force-device-scale-factor=2",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    cdp: _MiniCDP | None = None
    shots: dict[str, Any] = {}
    mcp_log: dict[str, Any] = {"server": "brainlens-mcp (stdio)", "calls": []}
    rng = random.Random(20261124)

    mcp_env = dict(os.environ)
    mcp_env["PYTHONPATH"] = os.pathsep.join(
        [str(WORKBENCH_SRC), *[str(p) for p in _demo_extra_paths()]]
    )
    mcp_env["BRAINLENS_TRUST_DIR"] = str(TRUST_DIR)
    mcp_env["MPLCONFIGDIR"] = str(MPL_DIR)
    mcp_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "brainlens.mcp_server", "--workspace", str(DEMO_WS)],
        env=mcp_env,
    )

    try:
        ws_url = None
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{dbg_port}/json/list", timeout=1.0) as r:
                    tabs = json.loads(r.read().decode("utf-8"))
                page_tab = next((t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")), None)
                if page_tab:
                    ws_url = page_tab["webSocketDebuggerUrl"]
                    break
            except Exception:
                await asyncio.sleep(0.15)
        assert ws_url, "Chrome CDP endpoint did not become ready"

        cdp = _MiniCDP(ws_url)
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")

        def nav(url: str) -> None:
            cdp.send("Page.navigate", {"url": url})
            cdp.wait_for("document.readyState === 'complete'")

        async with stdio_client(mcp_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_resp = await session.list_tools()
                mcp_log["tools"] = [t.name for t in tools_resp.tools]
                print(f"[capture] Connected to live stdio MCP server ({len(mcp_log['tools'])} tools)")

                async def call_mcp(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
                    t0 = time.monotonic()
                    res = await session.call_tool(name, arguments)
                    elapsed = round(time.monotonic() - t0, 3)
                    parsed = json.loads(res.content[0].text)
                    mcp_log["calls"].append(
                        {"tool": name, "arguments": arguments, "elapsed_s": elapsed, "result": parsed}
                    )
                    if isinstance(parsed, dict) and parsed.get("error"):
                        raise RuntimeError(f"MCP tool {name} failed: {parsed}")
                    print(f"[mcp] {name}({list(arguments.keys())}) -> OK in {elapsed:.2f}s")
                    return parsed

                # ==================================================================
                # SCENE 1: Cold Open on HUP134 live MDT Case Sheet
                # Filmed BEFORE any verdict exists (0 of 3 focal candidates reviewed, zero agent verdicts).
                # ==================================================================
                print("[capture] Scene 1: Cold open — HUP134 MDT Case Sheet (no verdicts) -> highlight LB10 -> ask -> live SSE jump ...")
                nav(f"{base_url}/?case=HUP134&recording=run-01&run=baseline&view=sheet&theme=dark")
                cdp.wait_for(
                    "document.getElementById('caseSelect')?.value === 'HUP134' && "
                    "(document.getElementById('agentSyncBadge')?.textContent || '').includes('Linked') && "
                    "document.getElementById('sheetFrame')?.contentDocument?.body?.classList?.contains('live-connected') && "
                    "Boolean(document.getElementById('sheetFrame')?.contentDocument?.querySelector('tr.ch-row[data-channel=\"LB10\"]'))",
                    timeout=35.0,
                )
                cdp.eval("localStorage.setItem('brainlens_language', 'en');")
                # Scroll sheet so run-01/run-02 summary cards (0 of 3 focal candidates) and LB10 table are both visible
                sheet_check = cdp.eval(
                    """(() => {
                        const fr = document.getElementById('sheetFrame');
                        const doc = fr.contentDocument;
                        const row = doc.querySelector('tr.ch-row[data-channel="LB10"]');
                        if (row) row.scrollIntoView({ behavior: 'instant', block: 'center' });
                        const txt = doc.body.innerText || '';
                        return {
                            hasZeroReviewed: txt.includes('0 of 3 focal candidates') || txt.includes('0/3 focal reviewed'),
                            hasUncertain: txt.includes('1 uncertain') || txt.includes('Clinician: uncertain'),
                            hasEngRehearsal: txt.includes('Engineering rehearsal'),
                            hasLegacyHfoP: txt.includes('HFO p='),
                        };
                    })()"""
                )
                assert sheet_check["hasZeroReviewed"] and not sheet_check["hasUncertain"], (
                    f"Cold open must have zero verdicts, got: {sheet_check}"
                )
                assert not sheet_check["hasEngRehearsal"] and not sheet_check["hasLegacyHfoP"], (
                    f"Cold open case sheet contained forbidden legacy labels: {sheet_check}"
                )
                # Wait for transient "Case Discussion Sheet updated" toast to clear before rolling camera
                cdp.pump(2.6)

                cdp.start_shot("s01_cold_open", start_cursor=(980, 340), cursor_visible=True)
                cdp.hold(0.55)

                # Measure exact LB10 text bounds inside sheetFrame
                lb10_rect = cdp.eval(
                    """(() => {
                        const fr = document.getElementById('sheetFrame');
                        const frR = fr.getBoundingClientRect();
                        const doc = fr.contentDocument;
                        const strong = doc.querySelector('tr.ch-row[data-channel="LB10"] strong');
                        const range = doc.createRange();
                        range.selectNodeContents(strong);
                        const r = range.getBoundingClientRect();
                        return { x0: frR.left + r.left, x1: frR.left + r.right, y: frR.top + r.top + r.height / 2 };
                    })()"""
                )
                cdp.drag_select(
                    lb10_rect["x0"] + 1.0,
                    lb10_rect["y"],
                    lb10_rect["x1"] + 3.0,
                    lb10_rect["y"],
                    move_dur=0.52,
                    drag_dur=0.38,
                )
                cdp.wait_for(
                    "document.getElementById('sheetFrame')?.contentDocument?.getElementById('selPopBtn')?.classList?.contains('open')",
                    timeout=8.0,
                )
                cdp.mark("highlighted")
                cdp.hold(0.30)

                # Click the real "Annotate / Ask Agent" popup button
                pop_pt = cdp.eval(
                    """(() => {
                        const fr = document.getElementById('sheetFrame');
                        const frR = fr.getBoundingClientRect();
                        const r = fr.contentDocument.getElementById('selPopBtn').getBoundingClientRect();
                        return { x: frR.left + r.left + r.width / 2, y: frR.top + r.top + r.height / 2 };
                    })()"""
                )
                cdp.click_at(pop_pt["x"], pop_pt["y"], move_dur=0.34)
                cdp.wait_for(
                    "document.getElementById('sheetFrame')?.contentDocument?.getElementById('composerBackdrop')?.classList?.contains('open')",
                    timeout=8.0,
                )
                composer_box = cdp.eval(
                    """(() => {
                        const fr = document.getElementById('sheetFrame');
                        const frR = fr.getBoundingClientRect();
                        const r = fr.contentDocument.querySelector('#composerBackdrop .composer-card').getBoundingClientRect();
                        return { x: frR.left + r.left, y: frR.top + r.top, w: r.width, h: r.height };
                    })()"""
                )
                cdp.record_box("composer_card", composer_box)
                cdp.mark("composer_open")
                cdp.hold(0.26)

                # Type question with real Input.dispatchKeyEvent (48-105 ms/char jittered)
                cdp.mark("typing_start")
                cdp.type_text("Does it recur in run-02?", rng, min_ms=48.0, max_ms=105.0)
                cdp.mark("typing_end")
                cdp.hold(0.32)

                # Click "Send to Agent" (real POST /api/report-action)
                send_pt = cdp.eval(
                    """(() => {
                        const fr = document.getElementById('sheetFrame');
                        const frR = fr.getBoundingClientRect();
                        const r = fr.contentDocument.getElementById('composerSendBtn').getBoundingClientRect();
                        return { x: frR.left + r.left + r.width / 2, y: frR.top + r.top + r.height / 2 };
                    })()"""
                )
                cdp.click_at(send_pt["x"], send_pt["y"], move_dur=0.34)
                cdp.wait_for(
                    "!document.getElementById('sheetFrame')?.contentDocument?.getElementById('composerBackdrop')?.classList?.contains('open') && "
                    "(document.getElementById('sheetFrame')?.contentDocument?.getElementById('agentTrayCount')?.textContent || '').includes('pending')",
                    timeout=10.0,
                )
                cdp.mark("sent")
                # Move cursor aside to a calm resting spot, then stay STILL while agent jumps reader
                cdp.move_to(900, 560, duration=0.28)
                cdp.hold(0.32)

                # REAL MCP CALL 1: open_workbench(status) reads clinician's pending question
                status_res = await call_mcp("open_workbench", {"action": "status", "case": "HUP134"})
                mcp_log["status_result"] = status_res
                cdp.pump(0.03)
                pending_acts = status_res.get("pending_report_actions") or []
                assert pending_acts, "Expected pending report action from MDT case sheet!"
                target_act = next((a for a in pending_acts if "run-02" in (a.get("comment") or "")), pending_acts[-1])
                act_id = target_act["id"]

                # REAL MCP CALL 2: query_evidence(compare_recordings)
                cmp_res = await call_mcp(
                    "query_evidence",
                    {"target": "HUP134", "mode": "compare_recordings", "run_a": "baseline"},
                )
                mcp_log["compare_recordings_result"] = cmp_res
                cdp.pump(0.03)

                # REAL MCP CALL 3: open_workbench(focus -> run-02 / LB10 / reader)
                # The open browser tab jumps BY ITSELF via SSE while the mouse stays still!
                focus_r2_res = await call_mcp(
                    "open_workbench",
                    {
                        "action": "focus",
                        "case": "HUP134",
                        "recording": "run-02",
                        "run_name": "baseline",
                        "channel": "LB10",
                        "queue_mode": "focal",
                        "view": "reader",
                    },
                )
                mcp_log["focus_run02_result"] = focus_r2_res
                cdp.wait_for(
                    "document.getElementById('view-reader')?.classList.contains('active') && "
                    "(document.getElementById('triadHeaderText')?.textContent || '').includes('run-02') && "
                    "(document.getElementById('triadHeaderText')?.textContent || '').includes('LB10') && "
                    "(document.getElementById('triadHeaderText')?.textContent || '').includes('230') && "
                    "(document.getElementById('verdictEventMeta')?.textContent || '').includes('Artifact model: artifact')",
                    timeout=20.0,
                )
                cdp.mark("reader_jumped")
                r2_boxes = cdp.eval(
                    """(() => {
                        const tl = document.querySelector('#triadDock .triad-left')?.getBoundingClientRect();
                        const vp = document.querySelector('#triadDock .verdict-panel')?.getBoundingClientRect();
                        const td = document.getElementById('triadDock')?.getBoundingClientRect();
                        const meta = document.getElementById('verdictEventMeta')?.textContent || '';
                        return {
                            triad_left: tl ? { x: tl.left, y: tl.top, w: tl.width, h: tl.height } : null,
                            verdict_panel: vp ? { x: vp.left, y: vp.top, w: vp.width, h: vp.height } : null,
                            triad_dock: td ? { x: td.left, y: td.top, w: td.width, h: td.height } : null,
                            hasLegacyHfoP: meta.includes('HFO p='),
                        };
                    })()"""
                )
                assert not r2_boxes["hasLegacyHfoP"], "run-02 LB10 verdict meta still shows legacy 'HFO p='!"
                cdp.record_box("triad_left", r2_boxes["triad_left"])
                cdp.record_box("verdict_panel", r2_boxes["verdict_panel"])
                cdp.record_box("triad_dock", r2_boxes["triad_dock"])
                # Hold live on run-02 / LB10 (~230 Hz island) long enough for full-window + toast fade + steady punch-in
                cdp.hold(5.2)
                shots["s01_cold_open"] = cdp.stop_shot()

                # REAL MCP CALL 4: resolve_actions with honest plain clinical prose about PyHFO's artifact model
                agent_reply_text = (
                    "Yes, twice in run-02, peaking near 230 Hz. "
                    "The artifact model scores both as artifact, so this needs your eyes. "
                    "Opened run-02 at 179.4 s."
                )
                resolve_res = await call_mcp(
                    "open_workbench",
                    {
                        "action": "resolve_actions",
                        "case": "HUP134",
                        "action_ids": [act_id],
                        "agent_reply": agent_reply_text,
                    },
                )
                mcp_log["resolve_actions_result"] = resolve_res
                mcp_log["resolved_action_id"] = act_id
                mcp_log["agent_reply_text"] = agent_reply_text

                # Shot 2: Live MDT Case Sheet -> open Agent Review Inbox -> reply labelled Agent
                print("[capture] Scene 1b: Inbox reply labelled Agent on live MDT Case Sheet ...")
                cdp.eval("document.querySelector('#viewNav button[data-view=\"sheet\"]')?.click()")
                cdp.wait_for(
                    "document.getElementById('view-sheet')?.classList.contains('active') && "
                    "document.getElementById('sheetFrame')?.contentDocument?.body?.classList?.contains('live-connected')",
                    timeout=25.0,
                )
                cdp.eval(
                    """(() => {
                        const doc = document.getElementById('sheetFrame').contentDocument;
                        const row = doc.querySelector('tr.ch-row[data-channel="LB10"]');
                        if (row) row.scrollIntoView({ behavior: 'instant', block: 'center' });
                    })()"""
                )
                # Wait for sheet refresh toast to fade before rolling Shot 2
                cdp.pump(2.6)

                cdp.start_shot("s02_inbox_reply", start_cursor=(1120, 620), cursor_visible=True)
                cdp.hold(0.25)
                tray_pt = cdp.eval(
                    """(() => {
                        const fr = document.getElementById('sheetFrame');
                        const frR = fr.getBoundingClientRect();
                        const r = fr.contentDocument.getElementById('agentTrayBtn').getBoundingClientRect();
                        return { x: frR.left + r.left + r.width / 2, y: frR.top + r.top + r.height / 2 };
                    })()"""
                )
                cdp.click_at(tray_pt["x"], tray_pt["y"], move_dur=0.40)
                cdp.wait_for(
                    "document.getElementById('sheetFrame')?.contentDocument?.getElementById('agentTrayDrawer')?.classList?.contains('open') && "
                    "(document.getElementById('sheetFrame')?.contentDocument?.getElementById('agentTrayList')?.textContent || '').includes('artifact model')",
                    timeout=15.0,
                )
                inbox_box = cdp.eval(
                    """(() => {
                        const fr = document.getElementById('sheetFrame');
                        const frR = fr.getBoundingClientRect();
                        const r = fr.contentDocument.getElementById('agentTrayDrawer').getBoundingClientRect();
                        return { x: frR.left + r.left, y: frR.top + r.top, w: r.width, h: r.height };
                    })()"""
                )
                cdp.record_box("inbox_drawer", inbox_box)
                cdp.mark("inbox_open")
                # Move cursor cleanly off the inbox drawer so the reply punch-in is spotless
                cdp.move_to(720, 440, duration=0.42)
                cdp.hold(3.8)
                shots["s02_inbox_reply"] = cdp.stop_shot()

                # ==================================================================
                # SCENE 3: Case 1 — Scalp EEG (CHB-MIT chb01)
                # Compact picker + 42-file admission timeline (no 2076, collapsed gaps) ->
                # Reader Ask agent ("Where are the seizures?") -> real MCP status + timeline search
                # + live focus to chb01_03 at 2996 s + reply next to question on double-banana 1-70 Hz ->
                # Line length navigation trend
                # ==================================================================
                print("[capture] Scene 3: Case 1 Scalp EEG (CHB-MIT chb01) timeline -> Reader Ask agent -> MCP jump & reply -> Line length ...")
                nav(f"{base_url}/?case=chb01&recording=chb01_03&view=overview&theme=dark")
                cdp.wait_for(
                    "document.getElementById('caseSelect')?.value === 'chb01' && "
                    "document.querySelector('.recording-picker summary')?.textContent?.trim() === 'chb01_03' && "
                    "!document.getElementById('admissionPanel')?.hidden",
                    timeout=20.0,
                )
                cdp.eval(
                    """(() => {
                        const ap = document.getElementById('admissionPanel');
                        ap.open = true;
                        ap.dispatchEvent(new Event('toggle'));
                    })()"""
                )
                cdp.wait_for(
                    "document.querySelectorAll('#admissionFiles .admission-file').length >= 42 && "
                    "document.querySelectorAll('#admissionEvents .admission-event').length >= 7",
                    timeout=20.0,
                )
                tl_check = cdp.eval(
                    """(() => {
                        const sc = document.querySelector('#view-overview .overview-scroll');
                        const ap = document.getElementById('admissionPanel');
                        if (sc && ap) sc.scrollTop = Math.max(0, sc.scrollTop + ap.getBoundingClientRect().top - 58);
                        const af = document.getElementById('admissionFiles');
                        const afR = af.getBoundingClientRect();
                        const topbarOk = document.querySelector('.topbar').scrollWidth <= innerWidth;
                        const gapSummary = document.querySelector('#admissionGaps summary')?.textContent || '';
                        const gapsOpen = Boolean(document.querySelector('#admissionGaps details')?.open);
                        const pageTxt = document.getElementById('view-overview').innerText || '';
                        return {
                            has2076: pageTxt.includes('2076'),
                            topbarOk,
                            gapSummary,
                            gapsOpen,
                            afCenter: { x: afR.left + afR.width * 0.5, y: afR.top + afR.height * 0.5 },
                        };
                    })()"""
                )
                assert not tl_check["has2076"], f"chb01 overview/timeline still contains 2076: {tl_check}"
                assert tl_check["topbarOk"], f"Topbar overflowed on 42-recording chb01 case: {tl_check}"
                assert not tl_check["gapsOpen"], f"Timeline gaps should be collapsed by default: {tl_check}"
                cdp.pump(0.45)

                cdp.start_shot("s03_chb01_timeline", start_cursor=(760, 260), cursor_visible=True)
                cdp.hold(0.90)
                # Smoothly scroll inside the compact #admissionFiles list to show the 42 hourly files spanning ~40h
                cdp.wheel_scroll(
                    tl_check["afCenter"]["x"],
                    tl_check["afCenter"]["y"],
                    total_delta_y=520.0,
                    duration=1.35,
                    steps=28,
                )
                cdp.mark("scrolled_files")
                cdp.hold(0.45)

                # Click the Waveform Reader tab to open chb01_03 with scalp defaults
                reader_tab_pt = cdp.eval(
                    """(() => {
                        const r = document.querySelector('#viewNav button[data-view="reader"]').getBoundingClientRect();
                        return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
                    })()"""
                )
                cdp.click_at(reader_tab_pt["x"], reader_tab_pt["y"], move_dur=0.45)
                cdp.mark("clicked_reader")
                cdp.wait_for(
                    "document.getElementById('view-reader')?.classList.contains('active') && "
                    "document.getElementById('pageLabel')?.textContent === 'All (23 ch)' && "
                    "Boolean(document.querySelector('#readerAskAgent button[type=submit]:enabled')?.offsetWidth)",
                    timeout=20.0,
                )
                cdp.hold(0.30)
                shots["s03_chb01_timeline"] = cdp.stop_shot()

                # Shot 4: Scalp Reader Ask Agent ("Where are the seizures?") -> real MCP status + timeline + focus(2996) + reply!
                scalp_check = cdp.eval(
                    """(async () => {
                        const page = await (await fetch('/api/multichannel?case=chb01&recording=chb01_03&start=0&duration=10&band=eeg_1_70')).json();
                        const labels = (page.traces || []).map(t => t.channel_name);
                        return {
                            pageLabel: document.getElementById('pageLabel')?.textContent,
                            dur: document.getElementById('readerDurSelect')?.value,
                            filter: document.getElementById('dispFilterSelect')?.value,
                            notch: document.getElementById('displayNotchChk')?.checked,
                            notchLbl: document.getElementById('notchLbl')?.textContent,
                            triadExpanded: document.getElementById('toggleTriadBtn')?.getAttribute('aria-expanded'),
                            first8: labels.slice(0, 8),
                            count: labels.length,
                        };
                    })()"""
                )
                expected_first8 = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1"]
                assert scalp_check["first8"] == expected_first8, f"Expected double-banana scalp order, got: {scalp_check}"
                assert scalp_check["filter"] == "eeg_1_70" and scalp_check["notch"] and scalp_check["dur"] == "10" and scalp_check["triadExpanded"] == "false", (
                    f"Expected scalp defaults (1-70 Hz, notch, 10 s, triad collapsed), got: {scalp_check}"
                )
                cdp.pump(0.45)

                ask_ui = cdp.eval(
                    """(() => {
                        const box = document.getElementById('readerAskAgent');
                        const br = box.getBoundingClientRect();
                        const inp = box.querySelector('input[name="question"]').getBoundingClientRect();
                        const btn = box.querySelector('button[type="submit"]').getBoundingClientRect();
                        return {
                            box: { x: br.left, y: br.top, w: br.width, h: br.height },
                            inputPt: { x: inp.left + 44, y: inp.top + inp.height / 2 },
                            btnPt: { x: btn.left + btn.width / 2, y: btn.top + btn.height / 2 },
                        };
                    })()"""
                )

                cdp.start_shot("s04_chb01_reader_agent", start_cursor=(560, 460), cursor_visible=True)
                cdp.record_box("reader_ask_initial", ask_ui["box"])
                cdp.hold(0.65)

                # Click the Reader's Ask agent input
                cdp.click_at(ask_ui["inputPt"]["x"], ask_ui["inputPt"]["y"], move_dur=0.45)
                cdp.mark("ask_focused")
                cdp.hold(0.22)

                # Type "Where are the seizures?" with real human keystrokes
                cdp.mark("typing_start")
                cdp.type_text("Where are the seizures?", rng, min_ms=48.0, max_ms=105.0)
                cdp.mark("typing_end")
                cdp.hold(0.26)

                # Click "Ask agent" submit button
                cdp.click_at(ask_ui["btnPt"]["x"], ask_ui["btnPt"]["y"], move_dur=0.30)
                cdp.wait_for(
                    "(document.querySelector('#readerAskAgent .ask-thread')?.textContent || '').includes('Where are the seizures?')",
                    timeout=10.0,
                )
                cdp.mark("question_sent")
                # Move cursor gently onto a calm resting spot on the waveforms and stay STILL while MCP acts
                cdp.move_to(960, 440, duration=0.35)
                cdp.hold(0.32)

                # REAL MCP CALL 1: open_workbench(action="status", case="chb01") reads pending scalp question
                chb_status_res = await call_mcp("open_workbench", {"action": "status", "case": "chb01"})
                mcp_log["chb01_status_result"] = chb_status_res
                cdp.pump(0.03)
                chb_pending = chb_status_res.get("pending_report_actions") or []
                assert chb_pending, f"Expected pending report action on chb01, got: {chb_status_res}"
                chb_act = next((a for a in chb_pending if "seizure" in (a.get("comment") or "").lower()), chb_pending[-1])
                chb_act_id = chb_act["id"]
                chb_case_ref = chb_act.get("case_ref") or "chb01"

                # REAL MCP CALL 2: query_evidence(target="chb01", mode="timeline", text="seizure")
                chb_tl_res = await call_mcp(
                    "query_evidence",
                    {"target": "chb01", "mode": "timeline", "text": "seizure"},
                )
                mcp_log["chb01_timeline_result"] = chb_tl_res
                cdp.pump(0.03)
                assert len(chb_tl_res.get("events") or []) == 7, f"Expected 7 seizure markers from MCP timeline query: {chb_tl_res}"

                # REAL MCP CALL 3: open_workbench(action="focus", case="chb01", recording="chb01_03", start_seconds=2996, view="reader")
                chb_focus_res = await call_mcp(
                    "open_workbench",
                    {
                        "action": "focus",
                        "case": "chb01",
                        "recording": "chb01_03",
                        "start_seconds": 2996,
                        "view": "reader",
                    },
                )
                mcp_log["chb01_focus_result"] = chb_focus_res
                cdp.wait_for(
                    "document.getElementById('view-reader')?.classList.contains('active') && "
                    "parseFloat(document.getElementById('readerStartInput')?.value || '0') >= 2995 && "
                    "document.getElementById('pageLabel')?.textContent === 'All (23 ch)' && "
                    "Boolean(document.querySelector('#readerAskAgent button[type=submit]:enabled')?.offsetWidth)",
                    timeout=20.0,
                )

                # REAL MCP CALL 4: open_workbench(action="resolve_actions", ...) -> reply appears next to question!
                chb_reply_text = "Seven annotated seizures over about 40 hours. Opened the first one, chb01_03 at 49:56."
                chb_resolve_res = await call_mcp(
                    "open_workbench",
                    {
                        "action": "resolve_actions",
                        "case": chb_case_ref,
                        "action_ids": [chb_act_id],
                        "agent_reply": chb_reply_text,
                    },
                )
                mcp_log["chb01_resolve_result"] = chb_resolve_res
                mcp_log["chb01_reply_text"] = chb_reply_text
                cdp.wait_for(
                    "(document.querySelector('#readerAskAgent .ask-thread')?.textContent || '').includes('Seven annotated seizures') && "
                    "(document.querySelector('#readerAskAgent .ask-thread')?.textContent || '').includes('Resolved')",
                    timeout=15.0,
                )
                cdp.mark("replied_and_focused")
                ask_replied_box = cdp.eval(
                    """(() => {
                        const box = document.getElementById('readerAskAgent');
                        const br = box.getBoundingClientRect();
                        const th = box.querySelector('.ask-thread');
                        const thFits = th ? (th.scrollHeight <= th.clientHeight + 2) : true;
                        return {
                            box: { x: br.left, y: br.top, w: br.width, h: br.height },
                            thFits,
                            pageLabel: document.getElementById('pageLabel')?.textContent,
                            filter: document.getElementById('dispFilterSelect')?.value,
                            startVal: document.getElementById('readerStartInput')?.value,
                        };
                    })()"""
                )
                assert ask_replied_box["pageLabel"] == "All (23 ch)", f"All (23 ch) lost after MCP focus: {ask_replied_box}"
                assert ask_replied_box["filter"] == "eeg_1_70", f"1-70 Hz filter lost after MCP focus: {ask_replied_box}"
                cdp.record_box("reader_ask_replied", ask_replied_box["box"])
                print(f"[capture] chb01 reader reply box={ask_replied_box['box']} thFits={ask_replied_box['thFits']} start={ask_replied_box['startVal']}")
                # Hold long enough for full-window live jump + toast fade + steady punch-in on reply + full-window onset
                cdp.hold(6.2)
                shots["s04_chb01_reader_agent"] = cdp.stop_shot()

                # Shot 5: Line length trend view on chb01_03 (navigation trend; hover across 0:49:56 band)
                nav(f"{base_url}/?case=chb01&recording=chb01_03&view=overview&theme=dark")
                cdp.wait_for("document.getElementById('caseSelect')?.value === 'chb01' && !document.getElementById('trendPanel')?.hidden")
                cdp.eval(
                    """(() => {
                        document.getElementById('admissionPanel').open = false;
                        const tp = document.getElementById('trendPanel');
                        const ms = document.getElementById('trendMeasure');
                        ms.value = 'line_length';
                        tp.open = true;
                        tp.dispatchEvent(new Event('toggle'));
                        ms.dispatchEvent(new Event('change', { bubbles: true }));
                    })()"""
                )
                cdp.wait_for("!document.getElementById('trendPlot')?.hidden && document.getElementById('trendHeatmap')?.width > 100")
                cdp.pump(0.55)
                trend_box = cdp.eval(
                    """(() => {
                        const tp = document.getElementById('trendPanel');
                        tp.scrollIntoView({ behavior: 'instant', block: 'start' });
                        const r = document.getElementById('trendHeatmap').getBoundingClientRect();
                        return { x: r.left, y: r.top, w: r.width, h: r.height };
                    })()"""
                )
                cdp.pump(0.35)

                # Compute exact pixel coordinate of t = 3000 s (bright seizure band at 0:49:56-0:50:36) on FP2-F4 (row 8)
                left_pad = 88.0
                sz_x = trend_box["x"] + left_pad + (3000.0 / 3600.0) * (trend_box["w"] - left_pad)
                sz_y = trend_box["y"] + 8.5 * 20.0
                mid_x = trend_box["x"] + left_pad + 0.45 * (trend_box["w"] - left_pad)

                cdp.start_shot("s05_chb01_trends", start_cursor=(mid_x, sz_y + 40), cursor_visible=True)
                cdp.hold(0.55)
                # Real pointer hover across the Line length heatmap onto the 0:49:56 band -> updates live readout!
                cdp.move_to(sz_x, sz_y, duration=0.95, curve_sign=-0.4)
                cdp.mark("hovered_seizure_band")
                cdp.hold(2.4)
                shots["s05_chb01_trends"] = cdp.stop_shot()

                # ==================================================================
                # SCENE 4: Case 2 — SEEG (OpenNeuro HUP134)
                # Bursts vs focal -> click burst -> shafts firing together ->
                # LB10 Triad + locked-Y quiet control + Artifact model label + Agent: artifact + Key 3: Uncertain
                # ==================================================================
                print("[capture] Scene 4: Case 2 SEEG (HUP134) overview burst click, multi-shaft burst, Triad + quiet control + key 3 ...")
                q_summary = await call_mcp(
                    "query_evidence",
                    {"target": "HUP134/run-01", "mode": "recording_summary", "run_a": "baseline"},
                )
                mcp_log["recording_summary"] = q_summary

                # Record the Agent's honest assessment on LB10 (run-01) as ARTIFACT (citing PyHFO's artifact model)
                r1_lb10_ev = next(e for e in manifest["hup134_run01_focal_events"] if e["channel"] == "LB10")
                await call_mcp(
                    "record_judgment",
                    {
                        "case": "HUP134",
                        "recording": "run-01",
                        "run_name": "baseline",
                        "channel": "LB10",
                        "event_id": r1_lb10_ev["id"],
                        "verdict": "artifact",
                        "author": "agent",
                        "role": "agent",
                        "note": "Artifact model scores this event as artifact (real-HFO p 0.00); flagged for visual review.",
                    },
                )

                nav(f"{base_url}/?case=HUP134&recording=run-01&run=baseline&view=overview&theme=dark")
                cdp.wait_for(
                    "document.getElementById('caseSelect')?.value === 'HUP134' && "
                    "document.getElementById('view-overview')?.classList.contains('active') && "
                    "document.querySelectorAll('#kpiGrid .kpi-box').length >= 4 && "
                    "document.querySelectorAll('#activeChannelsTableWrap tr').length >= 3",
                    timeout=20.0,
                )
                # Frame KPIs + Event Synchrony Timeline + Cross-Snapshot Active Channels cleanly
                burst_target = cdp.eval(
                    """(() => {
                        const sc = document.querySelector('#view-overview .overview-scroll');
                        const kpi = document.getElementById('kpiGrid');
                        if (sc && kpi) sc.scrollTop = Math.max(0, sc.scrollTop + kpi.getBoundingClientRect().top - 64);
                        const tl = document.getElementById('overviewTimelineCanvas').getBoundingClientRect();
                        const burstX = tl.left + 42.0 + (55.0 / 300.0) * (tl.width - 54.0);
                        const burstY = tl.top + tl.height * 0.52;
                        return { x: burstX, y: burstY };
                    })()"""
                )
                cdp.pump(0.55)

                cdp.start_shot("s06_hup134_overview", start_cursor=(580, 150), cursor_visible=True)
                cdp.hold(1.7)
                # Real click on the first coral burst bar (~50.5 s) on the Overview timeline!
                cdp.click_at(burst_target["x"], burst_target["y"], move_dur=0.80)
                cdp.mark("clicked_burst")
                cdp.wait_for("document.getElementById('view-reader')?.classList.contains('active')", timeout=15.0)
                cdp.hold(0.40)
                shots["s06_hup134_overview"] = cdp.stop_shot()

                # Shot 7: Multi-shaft burst view (many shafts firing together simultaneously at 50.50 s)
                cdp.eval(
                    """(() => {
                        if (document.getElementById('toggleTriadBtn')?.getAttribute('aria-expanded') === 'false') {
                            document.getElementById('toggleTriadBtn')?.click();
                        }
                        document.querySelector('#queueTabs button[data-qmode="bursts"]')?.click();
                    })()"""
                )
                cdp.wait_for("document.querySelectorAll('#queueList .q-item').length >= 5")
                cdp.pump(0.30)
                cdp.eval("""(() => { document.querySelector('#queueList .q-item')?.click(); })()""")
                cdp.pump(0.50)
                cdp.eval(
                    """(() => {
                        const btns = [...document.querySelectorAll('#shaftPills button')];
                        const p1 = btns.find(x => x.textContent.trim() === 'P1');
                        if (p1) p1.click();
                        const dur = document.getElementById('readerDurSelect');
                        dur.value = '4'; dur.dispatchEvent(new Event('change', { bubbles: true }));
                        const gain = document.getElementById('readerGainSelect');
                        gain.value = '2'; gain.dispatchEvent(new Event('change', { bubbles: true }));
                        const rh = document.getElementById('rowHeightSelect');
                        rh.value = '28'; rh.dispatchEvent(new Event('change', { bubbles: true }));
                    })()"""
                )
                cdp.pump(0.75)
                burst_badge_ok = cdp.eval(
                    """(() => {
                        const b = document.getElementById('syncClassBadge');
                        const p = b.closest('.verdict-panel').getBoundingClientRect();
                        const r = b.getBoundingClientRect();
                        return r.left >= p.left && r.right <= p.right && b.clientWidth >= b.scrollWidth;
                    })()"""
                )
                assert burst_badge_ok, "syncClassBadge clipped on multi-shaft burst view!"

                cdp.start_shot("s07_hup134_burst", start_cursor=(860, 310), cursor_visible=True)
                cdp.hold(0.95)
                # Gently scroll down across contacts so LA5-LA8 & LB1-LB5 shafts firing together are clearly framed
                for _ in range(2):
                    cdp.press_key("ArrowDown", "ArrowDown", 40)
                    cdp.pump(0.35)
                cdp.mark("scrolled_burst")
                cdp.hold(1.9)
                shots["s07_hup134_burst"] = cdp.stop_shot()

                # Reset reader display defaults before focusing LB10 focal candidate
                cdp.eval(
                    """(() => {
                        const dur = document.getElementById('readerDurSelect');
                        dur.value = '8'; dur.dispatchEvent(new Event('change', { bubbles: true }));
                        const gain = document.getElementById('readerGainSelect');
                        gain.value = '1'; gain.dispatchEvent(new Event('change', { bubbles: true }));
                        const rh = document.getElementById('rowHeightSelect');
                        rh.value = 'fit'; rh.dispatchEvent(new Event('change', { bubbles: true }));
                        document.querySelector('#queueTabs button[data-qmode="focal"]')?.click();
                    })()"""
                )
                cdp.pump(0.35)

                # Shot 8: LB10 Clinical Triad + Locked-Y Side-by-Side Quiet Control + Model Label + Key 3 (Uncertain)
                await call_mcp(
                    "open_workbench",
                    {
                        "action": "focus",
                        "case": "HUP134",
                        "recording": "run-01",
                        "run_name": "baseline",
                        "channel": "LB10",
                        "queue_mode": "focal",
                        "view": "reader",
                    },
                )
                cdp.wait_for(
                    "document.getElementById('view-reader')?.classList.contains('active') && "
                    "(document.getElementById('triadHeaderText')?.textContent || '').includes('LB10') && "
                    "(document.getElementById('verdictEventMeta')?.textContent || '').includes('Artifact model: artifact (real-HFO p 0.00)') && "
                    "(document.getElementById('verdictEventMeta')?.textContent || '').includes('Agent: artifact')",
                    timeout=20.0,
                )
                cdp.eval(
                    """(() => {
                        const chk = document.getElementById('autoNextChk');
                        if (chk && chk.checked) chk.click();
                        if (document.activeElement) document.activeElement.blur();
                    })()"""
                )
                # Wait for "Synced with agent" toast to fade cleanly
                cdp.pump(2.5)

                s8_geom = cdp.eval(
                    """(() => {
                        const sbs = document.getElementById('sideBySideBtn').getBoundingClientRect();
                        const vp = document.querySelector('#triadDock .verdict-panel').getBoundingClientRect();
                        const td = document.getElementById('triadDock').getBoundingClientRect();
                        const meta = document.getElementById('verdictEventMeta')?.textContent || '';
                        return {
                            sbsPt: { x: sbs.left + sbs.width / 2, y: sbs.top + sbs.height / 2 },
                            verdict_panel: { x: vp.left, y: vp.top, w: vp.width, h: vp.height },
                            triad_dock: { x: td.left, y: td.top, w: td.width, h: td.height },
                            hasLegacyHfoP: meta.includes('HFO p='),
                        };
                    })()"""
                )
                assert not s8_geom["hasLegacyHfoP"], "run-01 LB10 verdict meta still shows legacy 'HFO p='!"

                cdp.start_shot("s08_hup134_triad_verdict", start_cursor=(720, 620), cursor_visible=True)
                cdp.record_box("verdict_panel", s8_geom["verdict_panel"])
                cdp.record_box("triad_dock", s8_geom["triad_dock"])
                cdp.hold(0.60)
                # Click Side-by-Side Control to show locked-Y quiet control beside LB10
                cdp.click_at(s8_geom["sbsPt"]["x"], s8_geom["sbsPt"]["y"], move_dur=0.55)
                cdp.wait_for("document.getElementById('sideBySideBtn')?.classList.contains('active')", timeout=10.0)
                cdp.mark("sbs_open")
                # Move cursor gently back onto the waveform area (away from verdict buttons) so key '3' is keyboard-driven
                cdp.move_to(620, 330, duration=0.65, curve_sign=-0.35)
                cdp.hold(1.90)

                # Mark right before pressing key '3' so the steady punch-in on the verdict panel catches the live flip!
                cdp.mark("pre_key3")
                cdp.hold(0.65)
                cdp.mark("key3")
                cdp.press_key("3", "Digit3", 51, text="3")
                cdp.wait_for(
                    "document.getElementById('vUncertainBtn')?.classList.contains('active') && "
                    "(document.getElementById('verdictEventMeta')?.textContent || '').toLowerCase().includes('uncertain') && "
                    "(document.getElementById('verdictEventMeta')?.textContent || '').includes('Agent: artifact') && "
                    "(document.getElementById('verdictEventMeta')?.textContent || '').includes('Artifact model: artifact (real-HFO p 0.00)')",
                    timeout=10.0,
                )
                cdp.mark("verdict_saved")
                # Hold long enough for both separate verdicts + model label to be read comfortably (and toast to fade)
                cdp.hold(3.8)
                shots["s08_hup134_triad_verdict"] = cdp.stop_shot()

                # Turn side-by-side off cleanly
                cdp.eval("if (document.getElementById('sideBySideBtn')?.classList.contains('active')) document.getElementById('sideBySideBtn').click()")
                cdp.pump(0.25)

                # ==================================================================
                # SCENE 5: One click on MDT slides -> real .pptx slides
                # ==================================================================
                print("[capture] Scene 5: One click on MDT slides & real .pptx export ...")
                cdp.eval("document.querySelector('#viewNav button[data-view=\"sheet\"]')?.click()")
                cdp.wait_for(
                    "document.getElementById('view-sheet')?.classList.contains('active') && "
                    "document.getElementById('sheetFrame')?.contentDocument?.body?.classList?.contains('live-connected')",
                    timeout=30.0,
                )
                cdp.eval(
                    """(() => {
                        const doc = document.getElementById('sheetFrame').contentDocument;
                        const fig = doc.querySelector('.fig-card[data-channel="LB10"]');
                        if (fig) fig.scrollIntoView({ behavior: 'instant', block: 'start' });
                    })()"""
                )
                cdp.pump(2.6)

                slides_btn_pt = cdp.eval(
                    """(() => {
                        const r = document.getElementById('exportSlidesBtn').getBoundingClientRect();
                        return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
                    })()"""
                )

                cdp.start_shot("s09_mdt_slides_click", start_cursor=(760, 220), cursor_visible=True)
                cdp.hold(0.45)
                cdp.click_at(slides_btn_pt["x"], slides_btn_pt["y"], move_dur=0.55)
                cdp.mark("clicked_slides")
                cdp.wait_for(
                    "(document.getElementById('exportDownloads')?.textContent || '').includes('Download MDT slides')",
                    timeout=60.0,
                )
                cdp.mark("slides_ready")
                cdp.hold(1.1)
                shots["s09_mdt_slides_click"] = cdp.stop_shot()

                print(
                    f"[capture] Live capture complete! JS exceptions={len(cdp.exceptions)}, "
                    f"console errors={len(cdp.console_errors)}"
                )
                if cdp.exceptions:
                    print("  WARNING JS exceptions:", cdp.exceptions)
                if cdp.console_errors:
                    print("  WARNING Console errors:", cdp.console_errors)

    finally:
        if cdp is not None:
            cdp.close()
        chrome.terminate()
        try:
            chrome.wait(timeout=4)
        except Exception:
            chrome.kill()
        srv.stop()

    # Render real slides from the exported .pptx created by clicking "MDT slides"
    pptx_path = DEMO_WS / "cases" / "HUP134" / "exports" / "HUP134_baseline_mdt_slides.pptx"
    assert pptx_path.is_file(), f"Expected exported MDT PPTX at {pptx_path}"
    slides_dir = CAPTURE_DIR / "slides_mdt"
    if slides_dir.exists():
        shutil.rmtree(slides_dir)
    rendered_slides = render_pptx_slides(pptx_path, slides_dir)
    print(f"[capture] Rendered {len(rendered_slides)} real slides from {pptx_path.name}")
    for sl in rendered_slides:
        print(f"  Slide {sl['index']:02d}: {sl['title']!r} ({sl['context']!r})")
    # Verify clinical display fixes inside the real exported PPTX
    from pptx import Presentation
    deck_txt = "\n".join(
        t for slide in Presentation(str(pptx_path)).slides for t in slide._element.xpath(".//a:t/text()")
    )
    assert "Engineering rehearsal" not in deck_txt, "Exported MDT slides still contain 'Engineering rehearsal'!"
    assert "HFO p=" not in deck_txt, "Exported MDT slides still contain legacy 'HFO p='!"
    assert "Artifact model: artifact (real-HFO p 0.00)" in deck_txt, "Exported MDT slides missing plain model label!"
    assert "Clinician: uncertain" in deck_txt and "Agent: artifact" in deck_txt, (
        "Exported MDT slides missing role-separated Clinician/Agent verdicts!"
    )

    manifest["shots"] = shots
    manifest["slides_pptx_path"] = str(pptx_path)
    manifest["rendered_slides"] = rendered_slides
    (CAPTURE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (CAPTURE_DIR / "mcp_transcript.json").write_text(json.dumps(mcp_log, indent=2), encoding="utf-8")
    return manifest


def capture_ui_and_slides() -> dict[str, Any]:
    """Stage 2 entrypoint."""
    manifest_path = CAPTURE_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    return asyncio.run(_run_live_capture_session(manifest))


# ---------------------------------------------------------------------------
# Stage 3: Real-Motion 30 fps Compositor (Hard Cuts, macOS Cursor, Minimal Text)
# ---------------------------------------------------------------------------

_CURSOR_IMG_CACHE: dict[float, tuple[Image.Image, int, int]] = {}


def _get_macos_arrow_cursor(scale_mul: float = 1.0) -> tuple[Image.Image, int, int]:
    """Return (RGBA cursor image, hotspot_x, hotspot_y) — plain macOS arrow pointer."""
    key = round(float(scale_mul), 2)
    if key in _CURSOR_IMG_CACHE:
        return _CURSOR_IMG_CACHE[key]

    oversample = 4
    # Classic macOS pointer geometry (~27 px tall at 1920x1080 when scale_mul=1.0)
    unit = 1.32 * key
    pad = int(round(8 * key))
    raw_pts = [
        (0.0, 0.0),
        (0.0, 17.6),
        (4.1, 13.7),
        (7.0, 20.5),
        (9.8, 19.3),
        (6.9, 12.6),
        (12.7, 12.6),
    ]
    cw = int(math.ceil((14.0 * unit + pad * 2)))
    ch = int(math.ceil((22.0 * unit + pad * 2)))
    pts_hi = [((pad + x * unit) * oversample, (pad + y * unit) * oversample) for x, y in raw_pts]

    # Soft downward drop shadow
    sh = Image.new("RGBA", (cw * oversample, ch * oversample), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sh_pts = [(x + 1.0 * key * oversample, y + 2.0 * key * oversample) for x, y in pts_hi]
    sd.polygon(sh_pts, fill=(0, 0, 0, 105))
    sh = sh.filter(ImageFilter.GaussianBlur(radius=2.2 * key * oversample))

    # Crisp white outline (drawn via thick stroke + bevelled polygon)
    fg = Image.new("RGBA", (cw * oversample, ch * oversample), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fg)
    ring_w = max(2, int(round(3.4 * key * oversample)))
    for i in range(len(pts_hi)):
        p0 = pts_hi[i]
        p1 = pts_hi[(i + 1) % len(pts_hi)]
        fd.line([p0, p1], fill=(255, 255, 255, 255), width=ring_w)
        r = ring_w / 2.0
        fd.ellipse([p0[0] - r, p0[1] - r, p0[0] + r, p0[1] + r], fill=(255, 255, 255, 255))
    fd.polygon(pts_hi, fill=(14, 15, 18, 255))

    combined = Image.alpha_composite(sh, fg)
    final = combined.resize((cw, ch), Image.Resampling.LANCZOS)
    _CURSOR_IMG_CACHE[key] = (final, pad, pad)
    return _CURSOR_IMG_CACHE[key]


def _composite_cursor(
    frame: Image.Image,
    x_css: float,
    y_css: float,
    crop_css: tuple[float, float, float, float] = (0.0, 0.0, float(VP_W), float(VP_H)),
) -> Image.Image:
    cx0, cy0, cx1, cy1 = crop_css
    cw_css = max(1.0, cx1 - cx0)
    ch_css = max(1.0, cy1 - cy0)
    if x_css < cx0 - 20.0 or x_css > cx1 + 20.0 or y_css < cy0 - 20.0 or y_css > cy1 + 20.0:
        return frame.copy()
    zoom = VP_W / cw_css
    cursor_im, hx, hy = _get_macos_arrow_cursor(min(1.28, zoom ** 0.35))
    px = int(round((x_css - cx0) / cw_css * W)) - hx
    py = int(round((y_css - cy0) / ch_css * H)) - hy
    out = frame.copy()
    out.paste(cursor_im, (px, py), cursor_im)
    return out


def _fit_crop_16_9(
    rect: dict[str, float] | None,
    *,
    pad_x: float = 36.0,
    pad_y: float = 24.0,
    min_w: float = 680.0,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    fallback: tuple[float, float, float, float] = (0.0, 0.0, float(VP_W), float(VP_H)),
) -> tuple[float, float, float, float]:
    """Compute a steady 16:9 CSS-pixel crop box around `rect`, clamped inside [0, VP_W] x [0, VP_H]."""
    if not rect:
        return fallback
    x0 = float(rect["x"]) - pad_x
    y0 = float(rect["y"]) - pad_y
    x1 = float(rect["x"]) + float(rect["w"]) + pad_x
    y1 = float(rect["y"]) + float(rect["h"]) + pad_y
    w = max(min_w, x1 - x0, (y1 - y0) * (16.0 / 9.0))
    w = min(float(VP_W), w)
    h = w * (9.0 / 16.0)
    cx = (x0 + x1) * 0.5 + shift_x
    cy = (y0 + y1) * 0.5 + shift_y
    nx0 = max(0.0, min(float(VP_W) - w, cx - w * 0.5))
    ny0 = max(0.0, min(float(VP_H) - h, cy - h * 0.5))
    return (round(nx0, 2), round(ny0, 2), round(nx0 + w, 2), round(ny0 + h, 2))


def _crop_and_resize_to_1080p(im: Image.Image, crop_css: tuple[float, float, float, float]) -> Image.Image:
    cx0, cy0, cx1, cy1 = crop_css
    full_box = (0.0, 0.0, float(VP_W), float(VP_H))
    if crop_css == full_box:
        return im if im.size == (W, H) else im.resize((W, H), Image.Resampling.LANCZOS)
    sx = im.width / float(VP_W)
    sy = im.height / float(VP_H)
    px0 = max(0, min(im.width - 2, int(round(cx0 * sx))))
    py0 = max(0, min(im.height - 2, int(round(cy0 * sy))))
    px1 = max(px0 + 2, min(im.width, int(round(cx1 * sx))))
    py1 = max(py0 + 2, min(im.height, int(round(cy1 * sy))))
    cropped = im.crop((px0, py0, px1, py1))
    return cropped if cropped.size == (W, H) else cropped.resize((W, H), Image.Resampling.LANCZOS)


def _draw_caption_bottom_center(frame: Image.Image, text: str, alpha: float = 1.0) -> Image.Image:
    """Plain narrative subtitle at bottom centre (up to two lines) on a neutral dark backing.

    The silent cut is read, not heard: every beat carries one plain sentence saying what the
    clinician or the agent is doing and why, so a viewer who knows nothing about the UI can follow.
    """
    if not text or alpha <= 0.01:
        return frame
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    font = get_font("sans", 46, 600)
    max_w = 1560
    lines = [text]
    if d.textlength(text, font=font) > max_w:
        # Balanced two-line split (no orphan words): minimise the longer line
        words = text.split()
        best: tuple[float, list[str]] | None = None
        for k in range(1, len(words)):
            a, b = " ".join(words[:k]), " ".join(words[k:])
            wa, wb = d.textlength(a, font=font), d.textlength(b, font=font)
            if max(wa, wb) <= max_w and (best is None or max(wa, wb) < best[0]):
                best = (max(wa, wb), [a, b])
        lines = best[1] if best else _wrap_text(d, text, font, max_w)[:2]
    line_h = 60
    pad_x, pad_y = 34, 16
    tw = max(d.textlength(ln, font=font) for ln in lines)
    bw = int(round(tw + pad_x * 2))
    bh = line_h * len(lines) + pad_y * 2
    x0 = (W - bw) // 2
    y0 = H - 44 - bh
    d.rounded_rectangle(
        [x0, y0, x0 + bw, y0 + bh],
        radius=10,
        fill=(0, 0, 0, int(238 * alpha)),
    )
    for i, ln in enumerate(lines):
        d.text(
            (W // 2, y0 + pad_y + line_h * i + line_h // 2),
            ln,
            font=font,
            fill=(255, 255, 255, int(255 * alpha)),
            anchor="mm",
        )
    return Image.alpha_composite(frame.convert("RGBA"), ov).convert("RGB")


def _draw_quiet_case_tag(frame: Image.Image, text: str, alpha: float = 1.0) -> Image.Image:
    """Quiet dataset honesty tag at the start of a case (bottom-left, clear of the centred subtitle)."""
    if not text or alpha <= 0.01:
        return frame
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    font = get_font("sans", 28, 600)
    tw = d.textlength(text, font=font)
    pad_x = 20
    bw = int(round(tw + pad_x * 2))
    bh = 48
    x0 = 28
    y0 = H - 44 - 152 - 14 - bh
    d.rounded_rectangle(
        [x0, y0, x0 + bw, y0 + bh],
        radius=5,
        fill=(0, 0, 0, int(232 * alpha)),
        outline=(255, 255, 255, int(26 * alpha)),
        width=1,
    )
    d.text((x0 + pad_x, y0 + bh // 2), text, font=font, fill=(*INK_PRIMARY, int(250 * alpha)), anchor="lm")
    return Image.alpha_composite(frame.convert("RGBA"), ov).convert("RGB")


def _render_title_card() -> Image.Image:
    """Plain neutral title card: 'BrainLens 1.0' / 'AI co-review for EEG'."""
    im = Image.new("RGB", (W, H), BG_FLAT)
    d = ImageDraw.Draw(im)
    d.text((W // 2, H // 2 - 34), "BrainLens 1.0", font=get_font("sans", 72, 600), fill=INK_PRIMARY, anchor="mm")
    d.text((W // 2, H // 2 + 44), "AI co-review for EEG", font=get_font("sans", 32, 400), fill=INK_SECONDARY, anchor="mm")
    return im


def _render_end_card() -> Image.Image:
    """Plain neutral end card in the exact agreed order, zero glows/pills/neon."""
    im = Image.new("RGB", (W, H), BG_FLAT)
    d = ImageDraw.Draw(im)

    d.text((W // 2, 278), "BrainLens 1.0", font=get_font("sans", 68, 600), fill=INK_PRIMARY, anchor="mm")
    d.text((W // 2, 352), "AI co-review for EEG", font=get_font("sans", 32, 400), fill=INK_SECONDARY, anchor="mm")
    d.text(
        (W // 2, 456),
        "Open source · from the PyHFO team, Roychowdhury Lab, UCLA",
        font=get_font("sans", 25, 400),
        fill=INK_SECONDARY,
        anchor="mm",
    )
    d.text(
        (W // 2, 548),
        "github.com/zyp5511/BrainLens",
        font=get_font("mono", 29, 500),
        fill=INK_PRIMARY,
        anchor="mm",
    )
    d.text(
        (W // 2, 636),
        "Works with Claude Code, Codex, Cursor and Antigravity",
        font=get_font("sans", 22, 400),
        fill=INK_MUTED,
        anchor="mm",
    )
    d.text(
        (W // 2, 972),
        "Data: CHB-MIT (PhysioNet, ODC-By) · HUP134 (OpenNeuro ds004100, CC0)",
        font=get_font("sans", 19, 400),
        fill=INK_FOOTER,
        anchor="mm",
    )
    return im


def _interp_cursor(track: list[list[Any]], t: float) -> tuple[float, float, bool]:
    if not track:
        return (VP_W * 0.5, VP_H * 0.5, False)
    if t <= track[0][0]:
        return (float(track[0][1]), float(track[0][2]), bool(track[0][3]))
    if t >= track[-1][0]:
        return (float(track[-1][1]), float(track[-1][2]), bool(track[-1][3]))
    # Binary search
    lo, hi = 0, len(track) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if track[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    t0, x0, y0, v0 = track[lo]
    t1, x1, y1, v1 = track[hi]
    if t1 <= t0:
        return (float(x0), float(y0), bool(v0))
    u = (t - t0) / (t1 - t0)
    return (float(x0 + (x1 - x0) * u), float(y0 + (y1 - y0) * u), bool(v0 and v1))


def _resample_shot(
    shot_meta: dict[str, Any],
    segments: list[tuple[float, float]],
    target_duration_s: float | None = None,
    subs: list[tuple[str, float, float]] | None = None,
    case_tag: tuple[str, float, float] | None = None,
    src_crops: list[tuple[float, float, tuple[float, float, float, float]]] | None = None,
    holds: list[tuple[float, float]] | None = None,
) -> list[Image.Image]:
    """Resample one or more real-time (t_start, t_end) slices of a recorded shot at 30 fps.

    Every frame uses the real recorded screencast frame at that exact timestamp plus the
    plain macOS arrow cursor at the exact dispatched coordinate. Optional `src_crops`
    entries `(src_t0, src_t1, (cx0, cy0, cx1, cy1))` perform hard-cut steady 16:9
    punch-ins in CSS coordinates (no slow drifting zooms). `subs` entries
    `(text, src_t0, src_t1)` show a narrative subtitle while the source time is in
    [src_t0, src_t1) (hard on/off, like ordinary subtitles). `holds` entries
    `(src_t, seconds)` freeze the real frame at `src_t` for reading time.
    """
    sdir = Path(shot_meta["dir"])
    raw_frames: list[tuple[float, str]] = [(float(t), str(fn)) for t, fn in shot_meta["frames"]]
    cursor_track = shot_meta.get("cursor") or []
    dur_raw = float(shot_meta["duration_s"])
    full_box = (0.0, 0.0, float(VP_W), float(VP_H))

    # Build piecewise timeline mapping output_t -> source_t
    clamped_segs: list[tuple[float, float]] = []
    for a, b in segments:
        aa = max(0.0, min(dur_raw, float(a)))
        bb = max(aa, min(dur_raw, float(b)))
        if bb > aa:
            clamped_segs.append((aa, bb))
    if not clamped_segs:
        clamped_segs = [(0.0, dur_raw)]

    # Output timeline entries: (src_a, src_b, out_len); a hold has src_a == src_b
    timeline: list[tuple[float, float, float]] = []
    pending_holds = sorted((float(ht), float(hd)) for ht, hd in (holds or []) if hd > 0)
    for a, b in clamped_segs:
        cur = a
        for ht, hd in pending_holds:
            if a <= ht <= b and ht >= cur:
                if ht > cur:
                    timeline.append((cur, ht, ht - cur))
                timeline.append((ht, ht, hd))
                cur = ht
        if b > cur:
            timeline.append((cur, b, b - cur))
    raw_total = sum(n for _, _, n in timeline)
    out_dur = float(target_duration_s) if target_duration_s is not None else raw_total
    n_out = max(1, int(round(out_dur * FPS)))

    # Cache raw PNGs and cropped 1080p renders keyed by (filename, crop_box)
    raw_cache: dict[str, Image.Image] = {}
    cropped_cache: dict[tuple[str, tuple[float, float, float, float]], Image.Image] = {}

    def load_cropped(fn: str, crop_css: tuple[float, float, float, float]) -> Image.Image:
        key = (fn, crop_css)
        out = cropped_cache.get(key)
        if out is None:
            raw = raw_cache.get(fn)
            if raw is None:
                raw = Image.open(sdir / fn).convert("RGB")
                raw_cache[fn] = raw
            out = _crop_and_resize_to_1080p(raw, crop_css)
            cropped_cache[key] = out
        return out

    def crop_at_src_t(st: float) -> tuple[float, float, float, float]:
        if src_crops:
            for c_a, c_b, box in src_crops:
                if c_a <= st < c_b:
                    return box
        return full_box

    def frame_at_src_t(st: float) -> str:
        # Latest recorded screencast frame with t <= st
        lo, hi = 0, len(raw_frames) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if raw_frames[mid][0] <= st + 1e-4:
                lo = mid
            else:
                hi = mid - 1
        return raw_frames[lo][1]

    def map_out_to_src(ot: float) -> float:
        acc = 0.0
        for a, b, n in timeline:
            if ot < acc + n:
                return a if b == a else a + (ot - acc)
            acc += n
        return timeline[-1][1]

    out_frames: list[Image.Image] = []
    for k in range(n_out):
        ot = k / FPS
        st = map_out_to_src(ot)
        fn = frame_at_src_t(st)
        crop_box = crop_at_src_t(st)
        im = load_cropped(fn, crop_box)
        cx, cy, vis = _interp_cursor(cursor_track, st)
        if vis:
            im = _composite_cursor(im, cx, cy, crop_box)
        else:
            im = im.copy()

        if case_tag is not None:
            tag_txt, t_a, t_b = case_tag
            if t_a <= ot <= t_b:
                fade = 1.0
                if ot - t_a < 0.25:
                    fade = _ease_out_cubic((ot - t_a) / 0.25)
                elif t_b - ot < 0.30:
                    fade = _ease_out_cubic((t_b - ot) / 0.30)
                im = _draw_quiet_case_tag(im, tag_txt, alpha=fade)

        for sub_txt, s_a, s_b in subs or []:
            if s_a <= st < s_b:
                im = _draw_caption_bottom_center(im, sub_txt)
                break

        out_frames.append(im)
    return out_frames


class _HardCutAssembler:
    """Streams frames to imageio-ffmpeg with clean hard cuts between shots (no dissolves)."""

    def __init__(self, out_path: Path, fps: int = FPS) -> None:
        import imageio.v2 as imageio

        self.out_path = out_path
        self.fps = fps
        self.writer = imageio.get_writer(
            str(out_path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=None,
            ffmpeg_params=["-crf", "22", "-preset", "slow", "-maxrate", "2600k", "-bufsize", "5200k", "-movflags", "+faststart"],
            macro_block_size=8,
        )
        self.total_frames = 0
        self.timeline_log: list[dict[str, Any]] = []

    def add_clip(self, label: str, frames: list[Image.Image]) -> None:
        if not frames:
            return
        t0 = self.total_frames / self.fps
        for im in frames:
            self.writer.append_data(np.asarray(im, dtype=np.uint8))
            self.total_frames += 1
        t1 = self.total_frames / self.fps
        self.timeline_log.append({"label": label, "start_s": round(t0, 2), "end_s": round(t1, 2), "duration_s": round(t1 - t0, 2)})

    def close(self) -> float:
        self.writer.close()
        return self.total_frames / self.fps


def _mux_voiceover_if_present(silent_mp4: Path, final_mp4: Path, duration_s: float) -> bool:
    """If build/voiceover.wav exists, mux AAC audio into final_mp4; otherwise move silent_mp4 in place."""
    if not VOICEOVER_WAV_PATH.is_file():
        os.replace(silent_mp4, final_mp4)
        return False

    ffmpeg_exe = None
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise RuntimeError(f"Found {VOICEOVER_WAV_PATH}, but ffmpeg binary could not be located to mux it.")

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(silent_mp4),
        "-i", str(VOICEOVER_WAV_PATH),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-af", "apad",
        "-t", f"{duration_s:.3f}",
        "-movflags", "+faststart",
        str(final_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    silent_mp4.unlink(missing_ok=True)
    print(f"[render] Muxed audio from {VOICEOVER_WAV_PATH} into {final_mp4}")
    return True


def _build_contact_sheet(
    mp4_path: Path,
    out_path: Path,
    n_frames: int = 18,
    total_frames: int | None = None,
    beat_frame_times_s: list[tuple[str, float]] | None = None,
    beat_frames_dir: Path | None = None,
) -> None:
    """Sample 18 evenly spaced frames into a 3x6 contact sheet + export full-res beat QA frames."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(mp4_path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", FPS))
    total = int(total_frames) if total_frames else 0
    if total <= 0:
        try:
            total = int(reader.count_frames())
        except Exception:
            total = int(round(float(meta.get("duration", 0.0)) * fps))
    if total <= 0:
        reader.close()
        return

    t_lo = int(round(total * 0.01))
    t_hi = max(t_lo + 1, int(round(total * 0.985)))
    indices = [int(round(t_lo + i * (t_hi - t_lo) / max(1, n_frames - 1))) for i in range(n_frames)]

    beat_index_map: dict[int, list[tuple[str, float]]] = {}
    if beat_frame_times_s and beat_frames_dir is not None:
        beat_frames_dir.mkdir(parents=True, exist_ok=True)
        for label, ts_s in beat_frame_times_s:
            fi = max(0, min(total - 1, int(round(ts_s * fps))))
            beat_index_map.setdefault(fi, []).append((label, ts_s))

    wanted = set(indices) | set(beat_index_map.keys())
    sampled: dict[int, np.ndarray] = {}
    for fi, frame in enumerate(reader):
        if fi in wanted:
            sampled[fi] = frame.copy()
            if len(sampled) >= len(wanted):
                break
    reader.close()
    if not sampled:
        return

    if beat_frames_dir is not None:
        for fi, items in beat_index_map.items():
            if fi not in sampled:
                continue
            im = Image.fromarray(sampled[fi])
            for label, _ in items:
                im.save(beat_frames_dir / f"{label}.png")

    cols = 3
    rows = math.ceil(n_frames / cols)
    tw, th = 560, 315
    hdr_h = 28
    pad = 14
    sheet_w = cols * tw + (cols + 1) * pad
    sheet_h = 46 + pad + rows * (th + hdr_h + pad)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (14, 17, 22))
    d = ImageDraw.Draw(sheet)
    d.text(
        (pad, 13),
        f"BrainLens 1.0 Launch Video Contact Sheet  {total / fps:.1f}s  {total} frames @ {fps:.0f} fps  {mp4_path.stat().st_size / (1024 * 1024):.2f} MB",
        font=get_font("mono", 17, 600),
        fill=INK_PRIMARY,
    )
    f_ts = get_font("mono", 15, 600)
    for k, fi in enumerate(indices):
        if fi not in sampled:
            continue
        r, c = divmod(k, cols)
        x0 = pad + c * (tw + pad)
        y0 = 46 + pad + r * (th + hdr_h + pad)
        d.rectangle([x0, y0, x0 + tw, y0 + hdr_h], fill=(22, 26, 34))
        d.text((x0 + 10, y0 + hdr_h // 2), f"#{k + 1:02d}   {fi / fps:5.1f}s", font=f_ts, fill=INK_SECONDARY, anchor="lm")
        thumb = Image.fromarray(sampled[fi]).resize((tw, th), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x0, y0 + hdr_h))
    sheet.save(out_path)


def _frame_pane_gray(img_path: Path) -> np.ndarray:
    """Load a raw shot frame downsampled to 480x270 grayscale for transition/loading detection."""
    im = Image.open(img_path).convert("RGB").resize((480, 270), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32).mean(axis=2)


def _pane_edge_density(gray: np.ndarray, y0: float, y1: float, x0: float, x1: float) -> float:
    """Fraction of pixels with horizontal or vertical gradient > 8 in the fractional ROI."""
    h, w = gray.shape
    sub = gray[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    gx = np.abs(np.diff(sub, axis=1)) > 8.0
    gy = np.abs(np.diff(sub, axis=0)) > 8.0
    return float(gx.mean() + gy.mean())


def _trim_before_view_switch(shot_meta: dict[str, Any], click_t: float) -> float:
    """Return a cut-out timestamp right after `click_t` and strictly before the incoming view's loading frames.

    Used when a shot clicks a navigation control (e.g., Waveform Reader tab or burst bar) and
    the next shot already starts cleanly on the fully-rendered destination view.
    """
    sdir = Path(shot_meta["dir"])
    raw_frames: list[tuple[float, str]] = [(float(t), str(fn)) for t, fn in shot_meta["frames"]]
    dur_raw = float(shot_meta["duration_s"])
    if not raw_frames:
        return min(dur_raw, click_t)

    # Find the settled pre-click frame (latest frame at or before click_t)
    pre_idx = 0
    for i, (t, _) in enumerate(raw_frames):
        if t <= click_t + 1e-4:
            pre_idx = i
        else:
            break
    pre_gray = _frame_pane_gray(sdir / raw_frames[pre_idx][1])

    # Find the first frame after click_t where the main content pane switches away / blanks
    t_switch = dur_raw
    for t, fn in raw_frames[pre_idx + 1:]:
        g = _frame_pane_gray(sdir / fn)
        diff = float(np.mean(np.abs(g - pre_gray)))
        wave_edge = _pane_edge_density(g, 0.18, 0.68, 0.24, 0.97)
        if diff > 4.0 or wave_edge < 0.03:
            t_switch = t
            break

    return round(max(0.1, min(click_t + 0.015, t_switch - 0.015)), 4)


def _excise_loading_gap(
    shot_meta: dict[str, Any],
    t_after: float,
    *,
    require_triad: bool = False,
    max_pre_hold: float | None = None,
    min_wave_edge: float = 0.25,
    min_triad_edge: float = 0.25,
) -> list[tuple[float, float]]:
    """Excise any intermediate blank/dimmed/loading frames during an in-shot MCP jump.

    Scans recorded frames starting from the settled state at `t_after`:
      1. Finds the first frame that departs from the settled pre-jump view (dimming, blanking, or view swap).
      2. Finds the first subsequent frame where the main waveform pane (and optionally Clinical Triad) is
         fully rendered (`edge >= threshold`).
      3. Cuts cleanly from just before the departure straight to the first fully-rendered frame.
    """
    sdir = Path(shot_meta["dir"])
    raw_frames: list[tuple[float, str]] = [(float(t), str(fn)) for t, fn in shot_meta["frames"]]
    dur_raw = float(shot_meta["duration_s"])
    if not raw_frames:
        return [(0.0, dur_raw)]

    # Settled pre-transition frame at t_after
    base_idx = 0
    for i, (t, _) in enumerate(raw_frames):
        if t <= t_after + 1e-4:
            base_idx = i
        else:
            break
    base_gray = _frame_pane_gray(sdir / raw_frames[base_idx][1])
    base_wave = _pane_edge_density(base_gray, 0.18, 0.68, 0.24, 0.97)

    # 1. Locate the first frame that begins transitioning/dimming/blanking
    leave_idx: int | None = None
    for i in range(base_idx + 1, len(raw_frames)):
        t, fn = raw_frames[i]
        g = _frame_pane_gray(sdir / fn)
        diff = float(np.mean(np.abs(g - base_gray)))
        we = _pane_edge_density(g, 0.18, 0.68, 0.24, 0.97)
        if diff > 4.0 or (base_wave > 0.25 and we < 0.15):
            leave_idx = i
            break
    if leave_idx is None:
        return [(0.0, dur_raw)]

    # 2. Locate the first frame at or after leave_idx where destination waveforms (+ triad) are rendered
    ready_idx: int | None = None
    for i in range(leave_idx, len(raw_frames)):
        _, fn = raw_frames[i]
        g = _frame_pane_gray(sdir / fn)
        we = _pane_edge_density(g, 0.18, 0.68, 0.24, 0.97)
        te = _pane_edge_density(g, 0.68, 0.96, 0.24, 0.97) if require_triad else 1.0
        if we >= min_wave_edge and te >= min_triad_edge:
            ready_idx = i
            break
    if ready_idx is None or ready_idx == leave_idx:
        return [(0.0, dur_raw)]

    t_leave = raw_frames[leave_idx][0]
    t_ready = raw_frames[ready_idx][0]
    cut_out = t_leave - 0.02
    if max_pre_hold is not None:
        cut_out = min(cut_out, t_after + max_pre_hold)
    cut_out = max(t_after, round(cut_out, 4))
    cut_in = min(dur_raw, round(t_ready + 0.005, 4))
    if cut_in <= cut_out + 0.01:
        return [(0.0, dur_raw)]
    return [(0.0, cut_out), (cut_in, dur_raw)]


def render_video() -> dict[str, Any]:
    """Stage 3: Assemble the ~84 s real-motion cut with hard cuts, steady punch-ins, macOS cursor, and narrative subtitles."""
    manifest = json.loads((CAPTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    shots = manifest["shots"]
    slides_meta = manifest["rendered_slides"]

    def load_native_rgb(path_str: str) -> Image.Image:
        return Image.open(path_str).convert("RGB")

    def load_1080p_rgb(path_str: str) -> Image.Image:
        im = load_native_rgb(path_str)
        return im if im.size == (W, H) else im.resize((W, H), Image.Resampling.LANCZOS)

    def find_slide_native(predicate, fallback_idx: int = 0) -> Image.Image:
        for sl in slides_meta:
            if predicate(sl):
                return load_native_rgb(sl["path"])
        return load_native_rgb(slides_meta[min(fallback_idx, len(slides_meta) - 1)]["path"])

    sl_leaders_native = find_slide_native(
        lambda s: "leading focal" in s["title"].lower() and "run-01" in s["context"].lower(),
        fallback_idx=3,
    )
    sl_triad_native = find_slide_native(
        lambda s: (
            "clinical triad" in s["context"].lower()
            and "focal candidate" in s["title"].lower()
            and "run-01" in s["context"].lower()
        ),
        fallback_idx=5,
    )
    sl_leaders_r1 = sl_leaders_native.resize((W, H), Image.Resampling.LANCZOS)
    sl_triad_r1 = sl_triad_native.resize((W, H), Image.Resampling.LANCZOS)
    # Steady punch-in on Triad slide: unclipped right-side LB10 summary (model label + Clinician & Agent verdicts)
    sl_triad_r1_crop = _crop_and_resize_to_1080p(sl_triad_native, (876.0, 170.0, 1416.0, 473.75))

    tmp_mp4 = BUILD_DIR / "brainlens_1_0_launch.silent.tmp.mp4"
    asm = _HardCutAssembler(tmp_mp4, fps=FPS)

    # ----------------------------------------------------------------------
    # SCENE 1 (0–15.5 s): Cold open on live HUP134 MDT Case Sheet
    # Steady punch-ins on: (1) typed question, (2) ~230 Hz island, (3) inbox reply
    # ----------------------------------------------------------------------
    print("[render] Scene 1: Cold open — HUP134 case sheet -> typing punch-in -> reader jump -> 230 Hz punch-in -> inbox reply punch-in ...")
    s01 = shots["s01_cold_open"]
    s01_marks = s01.get("marks") or {}
    s01_boxes = s01.get("boxes") or {}
    t_comp_open = float(s01_marks.get("composer_open", 2.1))
    t_sent = float(s01_marks.get("sent", 4.8))
    t_jumped = float(s01_marks.get("reader_jumped", t_sent + 1.0))
    # Cut directly from settled post-send Case Sheet straight to first fully-rendered reader + triad frame
    s01_segs = _excise_loading_gap(
        s01,
        t_after=t_sent + 0.18,
        require_triad=True,
        max_pre_hold=0.47,
    )
    print(f"[render] s01_cold_open segments (no blank/loading frames): {s01_segs}")

    crop_composer = _fit_crop_16_9(
        s01_boxes.get("composer_card"),
        pad_x=65.0,
        pad_y=42.0,
        min_w=720.0,
        fallback=(660.0, 220.0, 1380.0, 625.0),
    )
    crop_230hz = _fit_crop_16_9(
        s01_boxes.get("triad_left"),
        pad_x=0.0,
        pad_y=20.0,
        min_w=898.0,
        shift_x=0.0,
        shift_y=-24.0,
        fallback=(267.0, 304.88, 1165.0, 810.0),
    )
    t_230 = t_jumped + 2.35
    s01_frames = _resample_shot(
        s01,
        s01_segs,
        target_duration_s=None,
        src_crops=[
            (t_comp_open, t_sent + 0.08, crop_composer),
            (t_230, 999.0, crop_230hz),
        ],
        holds=[
            (float(s01_marks.get("highlighted", 1.6)), 1.3),
            (float(s01_marks.get("typing_end", t_sent - 1.0)) + 0.1, 0.9),
            (t_230 - 0.45, 1.0),
            (t_230 + 1.6, 0.6),
        ],
        subs=[
            ("Before epilepsy surgery, doctors review signals from electrodes inside the brain.", 0.0, t_comp_open),
            ("The doctor asks the AI: does this LB10 signal come back in the next recording?", t_comp_open, t_sent + 0.18),
            ("The AI checks the next recording and opens the reader right at the match.", t_sent + 0.18, t_230),
            ("There it is: a short burst of fast activity, around 230 Hz.", t_230, 999.0),
        ],
    )
    asm.add_clip("scene1a_cold_open_and_jump", s01_frames)

    s02 = shots["s02_inbox_reply"]
    s02_marks = s02.get("marks") or {}
    s02_boxes = s02.get("boxes") or {}
    t_inbox = float(s02_marks.get("inbox_open", 0.85))
    crop_inbox = _fit_crop_16_9(
        s02_boxes.get("inbox_drawer"),
        pad_x=44.0,
        pad_y=32.0,
        min_w=790.0,
        shift_x=-52.0,
        fallback=(650.0, 365.62, 1440.0, 810.0),
    )
    s02_frames = _resample_shot(
        s02,
        [(0.0, float(s02["duration_s"]))],
        target_duration_s=None,
        src_crops=[(t_inbox + 0.45, 999.0, crop_inbox)],
        holds=[(float(s02["duration_s"]) - 0.3, 1.2)],
        subs=[
            ("Its answer: yes, twice. But its own artifact model doubts them, so it asks the doctor to look.", 0.0, 999.0),
        ],
    )
    asm.add_clip("scene1b_inbox_reply", s02_frames)

    # Save full-res 1080p hero poster candidates from authentic live frames
    poster_inbox = load_1080p_rgb(str(Path(s02["dir"]) / s02["frames"][-1][1]))
    poster_reader = load_1080p_rgb(str(Path(s01["dir"]) / s01["frames"][-1][1]))
    poster_inbox.save(CAPTURE_DIR / "poster_cand_inbox_reply.png")
    poster_reader.save(CAPTURE_DIR / "poster_cand_run02_reader.png")

    # ----------------------------------------------------------------------
    # SCENE 2: Plain neutral title card
    # ----------------------------------------------------------------------
    print("[render] Scene 2: Plain title card ...")
    title_img = _render_title_card()
    asm.add_clip("scene2_title_card", [title_img] * int(round(2.6 * FPS)))

    # ----------------------------------------------------------------------
    # SCENE 3: Case 1 — Scalp EEG (CHB-MIT chb01)
    # Timeline -> Reader Ask agent ("Where are the seizures?") -> MCP live jump & reply -> Line length
    # ----------------------------------------------------------------------
    print("[render] Scene 3: Case 1 Scalp EEG (CHB-MIT chb01) timeline -> Reader Ask agent & onset -> Line length ...")
    s03 = shots["s03_chb01_timeline"]
    s03_marks = s03.get("marks") or {}
    t_clicked_reader = float(s03_marks.get("clicked_reader", float(s03["duration_s"])))
    # Cut straight from the timeline click on Waveform Reader to s04 frame 0 (traces already rendered)
    s03_end = _trim_before_view_switch(s03, t_clicked_reader)
    print(f"[render] s03_chb01_timeline trimmed before loading reader: 0.0 -> {s03_end}s (raw {s03['duration_s']}s)")
    s03_hold = 1.5
    s03_frames = _resample_shot(
        s03,
        [(0.0, s03_end)],
        target_duration_s=None,
        case_tag=("Case 1 · Scalp EEG · CHB-MIT", 0.0, s03_end + s03_hold - 0.05),
        holds=[(min(s03_end - 0.2, float(s03_marks.get("scrolled_files", 2.6)) + 0.1), s03_hold)],
        subs=[("About 40 hours of monitoring, split across 42 files, laid out on one timeline.", 0.0, 999.0)],
    )
    asm.add_clip("scene3a_chb01_timeline", s03_frames)

    s04 = shots["s04_chb01_reader_agent"]
    s04_marks = s04.get("marks") or {}
    s04_boxes = s04.get("boxes") or {}
    t_ask_foc = float(s04_marks.get("ask_focused", 1.1))
    t_ask_sent = float(s04_marks.get("question_sent", 3.6))
    t_ask_replied = float(s04_marks.get("replied_and_focused", t_ask_sent + 1.0))
    # Excise any empty waveform canvas frame while MCP jumps from 0 s to 2996 s (49:56)
    s04_segs = _excise_loading_gap(
        s04,
        t_after=t_ask_sent + 0.35,
        require_triad=False,
    )
    print(f"[render] s04_chb01_reader_agent segments (no blank jump frames): {s04_segs}")

    crop_scalp_typing = _fit_crop_16_9(
        s04_boxes.get("reader_ask_initial"),
        pad_x=16.0,
        pad_y=36.0,
        min_w=660.0,
        shift_x=185.0,
        shift_y=-40.0,
        fallback=(0.0, 438.75, 660.0, 810.0),
    )
    crop_scalp_reply = _fit_crop_16_9(
        s04_boxes.get("reader_ask_replied"),
        pad_x=16.0,
        pad_y=30.0,
        min_w=700.0,
        shift_x=205.0,
        shift_y=-28.0,
        fallback=(0.0, 416.25, 700.0, 810.0),
    )
    t_reply_crop = t_ask_replied + 2.40
    t_reply_end = t_ask_replied + 5.20
    s04_frames = _resample_shot(
        s04,
        s04_segs,
        target_duration_s=None,
        src_crops=[
            (t_ask_foc, t_ask_sent + 0.14, crop_scalp_typing),
            (t_reply_crop, t_reply_end, crop_scalp_reply),
        ],
        holds=[
            (max(0.05, t_ask_foc - 0.15), 1.9),
            (t_reply_crop - 0.6, 1.1),
            (t_reply_end - 0.3, 0.5),
            (min(float(s04["duration_s"]) - 0.2, t_reply_end + 0.6), 1.4),
        ],
        subs=[
            ("Instead of scrolling through 40 hours, the doctor just asks.", 0.0, max(0.1, t_ask_foc - 0.1)),
            ("The AI finds the seven seizures marked by experts and opens the first one.", t_ask_sent + 0.35, t_reply_crop),
            ("The first seizure onset, on the standard scalp montage.", t_reply_end, 999.0),
        ],
    )
    asm.add_clip("scene3b_chb01_reader_agent", s04_frames)

    poster_scalp = load_1080p_rgb(str(Path(s04["dir"]) / s04["frames"][-1][1]))
    poster_scalp.save(CAPTURE_DIR / "poster_cand_scalp_agent.png")

    s05 = shots["s05_chb01_trends"]
    s05_frames = _resample_shot(
        s05,
        [(0.0, float(s05["duration_s"]))],
        target_duration_s=None,
        holds=[(float((s05.get("marks") or {}).get("hovered_seizure_band", 1.5)) + 0.1, 0.6)],
        subs=[("Trends show the whole recording at a glance, for quick navigation.", 0.0, 999.0)],
    )
    asm.add_clip("scene3c_chb01_trends", s05_frames)

    # ----------------------------------------------------------------------
    # SCENE 4: Case 2 — SEEG (OpenNeuro HUP134)
    # Bursts vs focal -> click burst -> shafts firing together -> LB10 Triad + steady punch-in on verdicts
    # ----------------------------------------------------------------------
    print("[render] Scene 4: Case 2 SEEG (HUP134) overview -> burst -> Triad + steady verdict punch-in ...")
    s06 = shots["s06_hup134_overview"]
    s06_marks = s06.get("marks") or {}
    t_clicked_burst = float(s06_marks.get("clicked_burst", float(s06["duration_s"])))
    # Cut straight from clicking the coral burst bar on Overview to s07 frame 0 (burst traces + triad rendered)
    s06_end = _trim_before_view_switch(s06, t_clicked_burst)
    print(f"[render] s06_hup134_overview trimmed before loading reader: 0.0 -> {s06_end}s (raw {s06['duration_s']}s)")
    s06_hold = 2.0
    s06_frames = _resample_shot(
        s06,
        [(0.0, s06_end)],
        target_duration_s=None,
        case_tag=("Case 2 · SEEG · OpenNeuro HUP134", 0.0, s06_end + s06_hold - 0.05),
        holds=[(min(s06_end - 0.2, 1.2), s06_hold)],
        subs=[("1,150 automatic detections. 1,147 of them hit many electrodes at the same moment.", 0.0, 999.0)],
    )
    asm.add_clip("scene4a_hup134_overview", s06_frames)

    s07 = shots["s07_hup134_burst"]
    s07_frames = _resample_shot(
        s07,
        [(0.0, float(s07["duration_s"]))],
        target_duration_s=None,
        holds=[(float(s07["duration_s"]) - 0.3, 0.5)],
        subs=[("Bursts like this one are set aside, not ranked as a focus.", 0.0, 999.0)],
    )
    asm.add_clip("scene4b_hup134_burst", s07_frames)

    s08 = shots["s08_hup134_triad_verdict"]
    s08_marks = s08.get("marks") or {}
    s08_boxes = s08.get("boxes") or {}
    t_pre_key3 = float(s08_marks.get("pre_key3", 3.8))
    crop_verdict = _fit_crop_16_9(
        s08_boxes.get("verdict_panel"),
        pad_x=22.0,
        pad_y=18.0,
        min_w=680.0,
        shift_x=-170.0,
        fallback=(760.0, 427.5, 1440.0, 810.0),
    )
    t_saved = float(s08_marks.get("verdict_saved", t_pre_key3 + 1.6))
    t_both = t_saved + 1.1
    s08_frames = _resample_shot(
        s08,
        [(0.0, float(s08["duration_s"]))],
        target_duration_s=None,
        src_crops=[(t_pre_key3, 999.0, crop_verdict)],
        holds=[(t_pre_key3 - 1.0, 0.8), (t_saved + 0.3, 0.9)],
        subs=[
            ("That leaves three local candidates. LB10 sits beside a quiet stretch, on the same scale.", 0.0, t_pre_key3),
            ("The AI calls it artifact. The doctor isn't sure and marks it uncertain.", t_pre_key3, t_both),
            ("Both calls are kept, each labeled with who made it.", t_both, 999.0),
        ],
    )
    asm.add_clip("scene4c_hup134_triad_verdict", s08_frames)

    poster_triad = load_1080p_rgb(str(Path(s08["dir"]) / s08["frames"][-1][1]))
    poster_triad.save(CAPTURE_DIR / "poster_cand_triad_verdict.png")

    # ----------------------------------------------------------------------
    # SCENE 5: One click on MDT slides -> full-bleed slides + steady Triad slide punch-in
    # ----------------------------------------------------------------------
    print("[render] Scene 5: MDT slides click -> full-bleed real slides + steady Triad slide punch-in ...")
    s09 = shots["s09_mdt_slides_click"]
    s09_marks = s09.get("marks") or {}
    t_click = float(s09_marks.get("clicked_slides", 1.1))
    t_ready = float(s09_marks.get("slides_ready", t_click + 0.5))
    if t_ready - t_click > 0.65:
        s09_segs = [(0.0, t_click + 0.25), (t_ready, float(s09["duration_s"]))]
    else:
        s09_segs = [(0.0, float(s09["duration_s"]))]
    s09_frames = _resample_shot(
        s09,
        s09_segs,
        target_duration_s=None,
        holds=[(max(0.0, t_click - 0.4), 0.7)],
        subs=[("One click turns the review into slides for the surgery conference.", 0.0, 999.0)],
    )
    asm.add_clip("scene5a_mdt_slides_click", s09_frames)

    # Real exported slides, FULL BLEED + steady punch-in on Triad slide for legibility
    slide_sub = "The same evidence and both verdicts, ready to present."
    asm.add_clip("scene5b_slide_leaders", [_draw_caption_bottom_center(sl_leaders_r1, slide_sub)] * int(round(2.0 * FPS)))
    asm.add_clip("scene5c_slide_triad_r1", [_draw_caption_bottom_center(sl_triad_r1, slide_sub)] * int(round(1.8 * FPS)))
    asm.add_clip("scene5d_slide_triad_r1_punchin", [sl_triad_r1_crop] * int(round(2.3 * FPS)))

    # ----------------------------------------------------------------------
    # SCENE 6 (58.8–62.8 s): Plain neutral end card
    # ----------------------------------------------------------------------
    print("[render] Scene 6: Plain end card ...")
    end_img = _render_end_card()
    asm.add_clip("scene6_end_card", [end_img] * int(round(4.0 * FPS)))

    duration = asm.close()
    muxed_audio = _mux_voiceover_if_present(tmp_mp4, MP4_OUT_PATH, duration)

    # Hero poster: full-bleed live MDT Case Sheet + 3D shafts + resolved Agent reply
    poster_inbox.save(POSTER_OUT_PATH)
    size_mb = MP4_OUT_PATH.stat().st_size / (1024 * 1024)
    print(
        f"[render] Rendered {MP4_OUT_PATH}: {duration:.2f}s @ {FPS} fps, {W}x{H}, "
        f"{size_mb:.2f} MB (audio_muxed={muxed_audio})"
    )
    print(f"[render] Timeline: {json.dumps(asm.timeline_log)}")

    # Compute exact beat QA timestamps dynamically from asm.timeline_log so every full-window beat AND every punch-in is inspected!
    tl = {c["label"]: c for c in asm.timeline_log}

    def at_frac(label: str, frac: float) -> float:
        c = tl[label]
        return round(c["start_s"] + (c["end_s"] - c["start_s"]) * frac, 2)

    beat_qa_times = [
        ("01_s1_sheet_initial", at_frac("scene1a_cold_open_and_jump", 0.04)),
        ("02_s1_punchin_typing_q", at_frac("scene1a_cold_open_and_jump", 0.36)),
        ("03_s1_reader_run02_full", at_frac("scene1a_cold_open_and_jump", 0.65)),
        ("04_s1_punchin_230hz", at_frac("scene1a_cold_open_and_jump", 0.90)),
        ("05_s1_inbox_full", at_frac("scene1b_inbox_reply", 0.22)),
        ("06_s1_punchin_inbox_reply", at_frac("scene1b_inbox_reply", 0.78)),
        ("07_s2_title_card", at_frac("scene2_title_card", 0.50)),
        ("08_s3_chb01_timeline", at_frac("scene3a_chb01_timeline", 0.35)),
        ("09_s3_chb01_punchin_typing", at_frac("scene3b_chb01_reader_agent", 0.26)),
        ("10_s3_chb01_reader_onset_full", at_frac("scene3b_chb01_reader_agent", 0.55)),
        ("11_s3_chb01_punchin_reply", at_frac("scene3b_chb01_reader_agent", 0.80)),
        ("12_s3_chb01_onset_full_tail", at_frac("scene3b_chb01_reader_agent", 0.96)),
        ("13_s3_chb01_trends_ll", at_frac("scene3c_chb01_trends", 0.80)),
        ("14_s4_hup134_overview", at_frac("scene4a_hup134_overview", 0.45)),
        ("15_s4_hup134_burst", at_frac("scene4b_hup134_burst", 0.72)),
        ("16_s4_hup134_triad_sbs_full", at_frac("scene4c_hup134_triad_verdict", 0.40)),
        ("17_s4_hup134_punchin_verdict", at_frac("scene4c_hup134_triad_verdict", 0.88)),
        ("18_s5_mdt_click", at_frac("scene5a_mdt_slides_click", 0.78)),
        ("19_s5_slide_leaders", at_frac("scene5b_slide_leaders", 0.50)),
        ("20_s5_slide_triad_full", at_frac("scene5c_slide_triad_r1", 0.50)),
        ("21_s5_slide_triad_punchin", at_frac("scene5d_slide_triad_r1_punchin", 0.50)),
        ("22_s6_end_card", at_frac("scene6_end_card", 0.50)),
    ]
    cs_path = CAPTURE_DIR / "contact_sheet.png"
    beat_dir = CAPTURE_DIR / "beat_frames"
    if beat_dir.exists():
        shutil.rmtree(beat_dir)
    _build_contact_sheet(
        MP4_OUT_PATH,
        cs_path,
        n_frames=18,
        total_frames=asm.total_frames,
        beat_frame_times_s=beat_qa_times,
        beat_frames_dir=beat_dir,
    )
    print(f"[render] Saved contact sheet: {cs_path} and {len(beat_qa_times)} beat frames in {beat_dir}")
    return {
        "mp4": str(MP4_OUT_PATH),
        "poster": str(POSTER_OUT_PATH),
        "contact_sheet": str(cs_path),
        "beat_frames_dir": str(beat_dir),
        "duration_s": round(duration, 2),
        "size_mb": round(size_mb, 2),
        "audio_muxed": muxed_audio,
        "timeline": asm.timeline_log,
    }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build and render the BrainLens 1.0 launch video.")
    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "capture", "render"],
        default="all",
        help="Pipeline stage to run (default: all).",
    )
    parser.add_argument("--clean", action="store_true", help="Rebuild build/demo-workspace from scratch.")
    args = parser.parse_args()

    t0 = time.monotonic()
    if args.stage in ("all", "prepare"):
        prepare_workspace(clean=args.clean)
    if args.stage in ("all", "capture"):
        capture_ui_and_slides()
    if args.stage in ("all", "render"):
        res = render_video()
        print(json.dumps(res, indent=2))
    print(f"[done] Total elapsed: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
