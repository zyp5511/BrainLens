"""Release Gate test suite covering all P0 & P1 clinical, signal, security, and packaging invariants."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest

WORKBENCH_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKBENCH_DIR.parent
_REPO_SYNTH = PROJECT_ROOT / "inputs" / "synthetic_seeg.edf"
_FIXTURE_SYNTH = Path(__file__).resolve().parent / "fixtures" / "synthetic_seeg.edf"
SYNTHETIC_EDF = _REPO_SYNTH if _REPO_SYNTH.is_file() else _FIXTURE_SYNTH
SRC_DIR = WORKBENCH_DIR / "src"
if importlib.util.find_spec("brainlens") is None and SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlens.analytics import annotate_event_synchrony, compare_runs, compute_overlay_concordance, query_case, summarize_run
from brainlens.bids_export import export_bids_events
from brainlens.case_sheet import export_case_sheet
from brainlens.cohort import run_cohort
from brainlens.io_readers import inspect_case
from brainlens.models import AnalysisConfig, ChannelSpec, ToolkitError, atomic_write_json, read_json
from brainlens.pipeline import _auto_branch_slug, analyze_case, refresh_verdict_tallies
from brainlens.protocols import load_protocol_config, save_protocol
from brainlens.pyhfo_runner import _sanitized_worker_env, check_engine_health, resolve_engine_settings, save_engine_settings
from brainlens.signal_prep import _build_reference_plan, compute_qc, prepare_signals
from brainlens.storage import WorkbenchStore
from brainlens.triad_views import compute_multichannel_batch, compute_triad_view, render_evidence_card
from brainlens.workbench_http import WorkbenchHTTPServer, is_loopback_http_url


def _write_fif(
    path: Path,
    ch_names: list[str],
    sfreq: float,
    n_samples: int,
    *,
    ch_types: list[str] | str = "seeg",
    seed: int = 0,
    sidecar: dict | None = None,
) -> Path:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 20e-6, size=(len(ch_names), n_samples))
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(path, overwrite=True, verbose="ERROR")
    if sidecar is not None:
        stem = path.name[:-7] if path.name.lower().endswith(".fif.gz") else path.stem
        (path.parent / f"{stem}.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return path


def _write_minimal_bdf_plus(path: Path, sfreq: int = 256, n_records: int = 2) -> Path:
    """Write a minimal valid BDF+ file with 2 EEG signals + 1 'BDF Annotations' signal."""
    labels = ["LA1", "LA2", "BDF Annotations"]
    ns = len(labels)
    n_samps = [sfreq, sfreq, 60]
    hdr = bytearray()
    hdr.extend(b"\xffBIOSEMI")
    hdr.extend(b" " * 80)  # patient id
    hdr.extend(b" " * 80)  # recording id
    hdr.extend(b"01.01.26")
    hdr.extend(b"00.00.00")
    hdr.extend(f"{256 * (ns + 1):<8}".encode("ascii"))
    hdr.extend(f"{'BDF+C':<44}".encode("ascii"))
    hdr.extend(f"{n_records:<8}".encode("ascii"))
    hdr.extend(f"{1:<8}".encode("ascii"))
    hdr.extend(f"{ns:<4}".encode("ascii"))
    for lb in labels:
        hdr.extend(f"{lb:<16}".encode("ascii"))
    for _ in range(ns):
        hdr.extend(b" " * 80)
    for lb in labels:
        unit = "" if "Annotations" in lb else "uV"
        hdr.extend(f"{unit:<8}".encode("ascii"))
    for _ in range(ns):
        hdr.extend(f"{-1000:<8}".encode("ascii"))
    for _ in range(ns):
        hdr.extend(f"{1000:<8}".encode("ascii"))
    for _ in range(ns):
        hdr.extend(f"{-8388608:<8}".encode("ascii"))
    for _ in range(ns):
        hdr.extend(f"{8388607:<8}".encode("ascii"))
    for _ in range(ns):
        hdr.extend(b" " * 80)
    for n in n_samps:
        hdr.extend(f"{n:<8}".encode("ascii"))
    for _ in range(ns):
        hdr.extend(b" " * 32)

    body = bytearray()
    for rec_i in range(n_records):
        for ch_i, n in enumerate(n_samps):
            if ch_i < 2:
                for k in range(n):
                    val = int(round(500 * np.sin(2 * np.pi * 10 * (rec_i * n + k) / sfreq)))
                    body.extend(struct.pack("<i", val)[:3])
            else:
                tal = f"+{rec_i}\x14\x14\x00".encode("ascii")
                tal = tal.ljust(n * 3, b"\x00")
                body.extend(tal)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(hdr) + bytes(body))
    return path


def _has_pyhfo_engine() -> bool:
    return bool(check_engine_health().get("available"))


# ---------------------------------------------------------------------------
# P0 #1: Cross-Case File Name Collision & Recording Fingerprint Reuse Guard
# ---------------------------------------------------------------------------


def test_p0_cross_case_filename_collision(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    f_a = _write_fif(tmp_path / "patientA" / "visit_raw.fif", ["LA1", "LA2"], 512.0, 2048, seed=1)
    f_b = _write_fif(tmp_path / "patientB" / "visit_raw.fif", ["LA1", "LA2"], 512.0, 4096, seed=2)

    res_a = analyze_case(str(f_a), workspace_dir=ws, overrides={"detector_enabled": False})
    res_b = analyze_case(str(f_b), workspace_dir=ws, overrides={"detector_enabled": False})

    assert res_a["case"] != res_b["case"], "Different directories with identical stem must NOT collide into one case"
    assert res_b["runs"][0]["already_existed"] is False

    store = WorkbenchStore(ws)
    rec_a = store.load_recording(res_a["case"], res_a["runs"][0]["recording"])
    rec_b = store.load_recording(res_b["case"], res_b["runs"][0]["recording"])
    assert rec_a.n_samples == 2048
    assert rec_b.n_samples == 4096
    assert rec_a.fingerprint != rec_b.fingerprint

    # Re-analyzing patientB's file finds its disambiguated case deterministically
    res_b2 = analyze_case(str(f_b), workspace_dir=ws, overrides={"detector_enabled": False})
    assert res_b2["case"] == res_b["case"]
    assert res_b2["runs"][0]["already_existed"] is True

    # Explicit case_name collision against a different source path MUST raise ToolkitError('case_name_collision')
    with pytest.raises(ToolkitError) as exc_info:
        inspect_case(str(f_b), workspace_dir=ws, case_name=res_a["case"])
    assert exc_info.value.code == "case_name_collision"

    # Mutating a file in place changes recording.fingerprint and MUST NOT return stale already_existed=True
    _write_fif(f_a, ["LA1", "LA2"], 512.0, 3072, seed=99)
    with pytest.raises(ToolkitError) as exc_conflict:
        analyze_case(str(f_a), workspace_dir=ws, overrides={"detector_enabled": False}, overwrite=False)
    assert exc_conflict.value.code == "run_conflict"


# ---------------------------------------------------------------------------
# P0 #2: ONNX Score Cache Isolation Across Branches (Real PyHFO Engine)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_pyhfo_engine(), reason="PyHFO engine not configured in this environment")
def test_p0_onnx_branch_cache_isolation(tmp_path: Path) -> None:
    syn_edf = SYNTHETIC_EDF
    if not syn_edf.is_file():
        pytest.skip("synthetic_seeg.edf fixture not present")

    # Workspace 1: baseline -> onnx_clf -> bipolar_onnx_tight (parent=onnx_clf)
    ws_branch = tmp_path / "ws_branch"
    analyze_case(str(syn_edf), workspace_dir=ws_branch, run_name="baseline")
    analyze_case(
        "synthetic_seeg",
        workspace_dir=ws_branch,
        run_name="onnx_clf",
        overrides={"classifier_enabled": True, "classifier_stages": ["artifact", "spike", "ehfo"]},
    )
    analyze_case(
        "synthetic_seeg",
        workspace_dir=ws_branch,
        run_name="bipolar_tight",
        parent_run="onnx_clf",
        overrides={
            "reference_mode": "bipolar",
            "classifier_enabled": True,
            "classifier_stages": ["artifact", "spike", "ehfo"],
            "classifier_thresholds": {"artifact": 0.991},
        },
    )

    # Workspace 2: cold bipolar_tight with zero prior ONNX cache
    ws_cold = tmp_path / "ws_cold"
    analyze_case(
        str(syn_edf),
        workspace_dir=ws_cold,
        run_name="bipolar_tight",
        overrides={
            "reference_mode": "bipolar",
            "classifier_enabled": True,
            "classifier_stages": ["artifact", "spike", "ehfo"],
            "classifier_thresholds": {"artifact": 0.991},
        },
    )

    ev_branch = WorkbenchStore(ws_branch).load_run("synthetic_seeg", "synthetic_seeg", "bipolar_tight")["events"]
    ev_cold = WorkbenchStore(ws_cold).load_run("synthetic_seeg", "synthetic_seeg", "bipolar_tight")["events"]
    assert len(ev_branch) == len(ev_cold) and len(ev_cold) > 0
    for eb, ec in zip(ev_branch, ev_cold):
        assert eb["id"] == ec["id"]
        assert eb["labels"] == ec["labels"], f"Branch ONNX label mismatch vs cold run: {eb} != {ec}"
        for st_k, cold_val in (ec.get("scores") or {}).items():
            br_val = (eb.get("scores") or {}).get(st_k)
            if cold_val is None:
                assert br_val is None, f"Expected None score for {st_k} on rejected event, got {br_val}"
            else:
                assert br_val is not None
                assert abs(float(br_val) - float(cold_val)) < 1e-5


# ---------------------------------------------------------------------------
# P1: Immutable Stage Cache, Interrupted Run Safety, Auto-Branch Slugs
# ---------------------------------------------------------------------------


def test_immutable_stage_cache_and_interrupted_runs(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    f = _write_fif(
        tmp_path / "case1.fif",
        ["LA1", "LA2", "LB1", "LB2"],
        512.0,
        512 * 10,
        sidecar={"reference": "common"},
    )
    analyze_case(str(f), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    analyze_case("case1", workspace_dir=ws, run_name="child_copy", parent_run="baseline", overrides={"detector_enabled": False})

    store = WorkbenchStore(ws)
    child_meta_before = store.load_run("case1", "case1", "child_copy")["meta"]
    child_prep_before = read_json(Path(child_meta_before["prepared_json_path"]))
    assert len(child_prep_before["channels"]) == 4

    # Overwrite baseline with excluded channel -> child's donor files MUST remain untouched!
    analyze_case(
        "case1",
        workspace_dir=ws,
        run_name="baseline",
        overrides={"detector_enabled": False, "exclude_channels": ["LA1"]},
        overwrite=True,
    )
    child_prep_after = read_json(Path(child_meta_before["prepared_json_path"]))
    assert child_prep_after == child_prep_before
    # Refreshing child verdict tallies still succeeds against immutable donor files
    refreshed = refresh_verdict_tallies(store, "case1", "case1", "child_copy")
    assert len(refreshed["channels"]) == 4

    # Interrupted run with only summary.json MUST NOT be reported as complete
    broken_dir = store.run_dir("case1", "case1", "interrupted_run")
    broken_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(broken_dir / "summary.json", {"counts": {}})
    assert store.has_run("case1", "case1", "interrupted_run") is False
    assert all(m["run_name"] != "interrupted_run" for m in store.list_runs("case1", "case1"))
    with pytest.raises(ToolkitError):
        store.load_run("case1", "case1", "interrupted_run")

    # Auto branch slugs distinguish start_seconds / end_seconds
    slug1 = _auto_branch_slug({"start_seconds": 0, "end_seconds": 4}, None)
    slug2 = _auto_branch_slug({"start_seconds": 6, "end_seconds": 10}, None)
    assert slug1 != slug2
    r1 = analyze_case("case1", workspace_dir=ws, overrides={"detector_enabled": False, "start_seconds": 0, "end_seconds": 4})
    r2 = analyze_case("case1", workspace_dir=ws, overrides={"detector_enabled": False, "start_seconds": 6, "end_seconds": 10})
    assert r1["run_name"] != r2["run_name"]
    assert r2["runs"][0]["already_existed"] is False


# ---------------------------------------------------------------------------
# P1: Average Reference Shafts & Widespread Burst Threshold Honesty
# ---------------------------------------------------------------------------


def test_average_reference_preserves_shafts_and_burst_threshold_honesty(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    f = _write_fif(
        tmp_path / "five_shafts.fif",
        ["A1", "B1", "C1", "D1", "E1"],
        512.0,
        512 * 6,
        sidecar={"reference": "common"},
    )
    inspect_case(str(f), workspace_dir=ws)
    rec = WorkbenchStore(ws).load_recording("five_shafts", "five_shafts")
    cfg = AnalysisConfig(reference={"mode": "average"}, detector={"enabled": False})
    out_chs, _ = _build_reference_plan(rec, cfg)

    shafts = [c.shaft for c in out_chs]
    assert shafts == ["A", "B", "C", "D", "E"], f"Average reference corrupted shafts: {shafts}"
    assert [c.contacts for c in out_chs] == [["A1"], ["B1"], ["C1"], ["D1"], ["E1"]]

    # On a 5-shaft recording with min_burst_shafts=5:
    # - 3 simultaneous shafts MUST be 'regional' (NOT 'widespread_burst')
    # - 5 simultaneous shafts MUST be 'widespread_burst' (1 burst episode)
    evs_3 = [
        {"id": f"e3_{i}", "channel_id": c.id, "channel_name": c.name, "shaft": c.shaft, "start_sample": 500, "end_sample": 530}
        for i, c in enumerate(out_chs[:3])
    ]
    ann_3, bursts_3 = annotate_event_synchrony(evs_3, out_chs, sfreq=512.0, window_ms=30.0, min_burst_shafts=5)
    assert all(e["synchrony_class"] == "regional" for e in ann_3)
    assert len(bursts_3) == 0

    evs_5 = [
        {"id": f"e5_{i}", "channel_id": c.id, "channel_name": c.name, "shaft": c.shaft, "start_sample": 1000, "end_sample": 1030}
        for i, c in enumerate(out_chs)
    ]
    ann_5, bursts_5 = annotate_event_synchrony(evs_5, out_chs, sfreq=512.0, window_ms=30.0, min_burst_shafts=5)
    assert all(e["synchrony_class"] == "widespread_burst" for e in ann_5)
    assert len(bursts_5) == 1


# ---------------------------------------------------------------------------
# P1: Analytics Stage Nulls, 16th-Rank SOZ Recall, Raw Compare & NumPy 1.26 Compat
# ---------------------------------------------------------------------------


def test_analytics_ehfo_null_soz_recall_raw_compare_and_numpy_compat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = tmp_path / "ws"
    ch_names = [f"LA{i}" for i in range(1, 18)]  # 17 channels!
    f = _write_fif(tmp_path / "sev17.fif", ch_names, 512.0, 512 * 10, sidecar={"reference": "common"})
    inspect_case(str(f), workspace_dir=ws)
    store = WorkbenchStore(ws)
    rec = store.load_recording("sev17", "sev17")
    cfg = AnalysisConfig(detector={"enabled": False})

    qc_dir = tmp_path / "qc"
    prep_dir = tmp_path / "prep"
    qc = compute_qc(rec, cfg, qc_dir)
    prep = prepare_signals(rec, cfg, qc, prep_dir)

    # Simulate NumPy 1.26 environment (has np.trapz, lacks np.trapezoid)
    if hasattr(np, "trapezoid"):
        orig_trap = np.trapezoid
        monkeypatch.setattr(np, "trapz", orig_trap, raising=False)
        monkeypatch.delattr(np, "trapezoid")

    # Create 16 focal-active channels where LA16 is rank #16 (1 focal event) and LA17 has 0 focal + 20 burst events (raw #1!)
    out_chs = [ChannelSpec.model_validate(c) for c in prep["channels"]]
    valid_ivs = {c.id: [[0, 512 * 10]] for c in out_chs}
    events: list[dict] = []
    # Give LA1..LA15 descending focal counts 16..2, LA16 = 1 focal event
    t_cursor = 100
    for idx in range(16):
        ch = out_chs[idx]
        cnt = 16 - idx
        for k in range(cnt):
            s0 = t_cursor
            t_cursor += 200
            events.append({
                "id": f"ev_f_{idx}_{k}",
                "channel_id": ch.id,
                "channel_name": ch.name,
                "shaft": ch.shaft,
                "start_sample": s0,
                "end_sample": s0 + 25,
                "start_seconds": s0 / 512.0,
                "end_seconds": (s0 + 25) / 512.0,
                "scores": {"artifact": 0.9, "spike": 0.8},
                "labels": {"artifact": 1, "spike": 1},
            })
    # Give LA17 20 raw events (simulate widespread burst on LA17? Actually on 1 shaft LA, let's give LA17 25 focal events in run B or test raw vs focal directly)
    det = {"status": "complete", "events": events, "valid_intervals": valid_ivs, "segments": [], "warnings": []}
    clf = {
        **det,
        "classifier_status": "partial",
        "classification": {
            "requested": ["artifact", "spike", "ehfo"],
            "stages": {
                "artifact": {"status": "complete"},
                "spike": {"status": "complete"},
                "ehfo": {"status": "unavailable", "reason": "fs < 2000 Hz"},
            },
        },
    }

    rdir_a = store.run_dir("sev17", "sev17", "run_a")
    rdir_a.mkdir(parents=True, exist_ok=True)
    summary_a, _ = summarize_run(rec, prep, qc, det, clf, {}, cfg, rdir_a)

    # 1) Unavailable ehfo stage MUST yield None (never 0!), while artifact & spike yield int counts
    assert summary_a["counts"]["non_artifact_events"] == len(events)
    assert summary_a["counts"]["spike_hfo_events"] == len(events)
    assert summary_a["counts"]["ehfo_events"] is None
    assert all(r["ehfo_count"] is None and r["ehfo_rate_per_min"] is None for r in summary_a["channels"])

    # 2) Overlay concordance with 16th-ranked SOZ channel LA16 MUST compute recall == 1.0 (not truncated at [:15]!)
    atomic_write_json(rdir_a / "config.json", {"analysis_config": cfg.model_dump(mode="json")})
    atomic_write_json(
        rdir_a / "run_meta.json",
        {
            "run_name": "run_a",
            "status": "complete",
            "detection_json_path": str(tmp_path / "det_a.json"),
            "prepared_json_path": str(prep_dir / "prepared.json"),
        },
    )
    atomic_write_json(tmp_path / "det_a.json", det)
    store.save_clinical_overlay("sev17", {"source": "unit-test", "soz_channels": ["LA16"]})
    conc = compute_overlay_concordance(store, "sev17", rec_slug="sev17", run_name="run_a", top_k=5)
    assert conc["metrics"]["focal_active_channels"] == 16
    assert conc["metrics"]["soz_recall_all_focal"] == 1.0

    # 3) compare_runs(metric="raw") MUST sort top5 by raw_rank, NOT focal_rank!
    # Manually set summary_a's LA17 to focal_rank=None, raw_rank=1 (highest raw rate) and LA1 to focal_rank=1, raw_rank=2
    s_mod = read_json(rdir_a / "summary.json")
    for row in s_mod["channels"]:
        if row["channel_name"] == "LA17":
            row["raw_count"] = 99
            row["raw_rate_per_min"] = 594.0
            row["raw_rank"] = 1
        elif row["raw_rank"] is not None:
            row["raw_rank"] = row["raw_rank"] + 1
    # Keep s_mod['channels'] sorted by focal_rank (LA1 first, LA17 at end!)
    atomic_write_json(rdir_a / "summary.json", s_mod)
    cmp_raw = compare_runs(store, "sev17", "sev17", "run_a", "run_a", metric="raw")
    assert cmp_raw["summary_metrics"]["top5_a"][0] == "LA17"


# ---------------------------------------------------------------------------
# P1: Mixed-Type FIF, BDF+ Annotations, Sidecar Containment & Fingerprint Invalidation
# ---------------------------------------------------------------------------


def test_io_mixed_fif_bdf_plus_sidecar_containment_and_fingerprint(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    # 1) Mixed FIF preserves MNE channel types (seeg vs ecg/emg)
    mix_f = _write_fif(
        tmp_path / "mix" / "mixed_raw.fif",
        ["LA1", "LA2", "AUX_HEART", "AUX_MUSCLE"],
        512.0,
        2048,
        ch_types=["seeg", "seeg", "ecg", "emg"],
    )
    # Place an UNRELATED *_braintoolkit.json in the same folder -> must NOT be attached!
    (tmp_path / "mix" / "unrelated_other_braintoolkit.json").write_text(
        json.dumps({"synthetic": True, "reference": "bipolar"}),
        encoding="utf-8",
    )
    insp_mix = inspect_case(str(mix_f), workspace_dir=ws)
    rec_mix = WorkbenchStore(ws).load_recording(insp_mix["case"], insp_mix["recordings"][0]["recording"])
    assert rec_mix.provenance.is_synthetic is False
    kinds = {c.name: (c.kind, c.is_auxiliary) for c in rec_mix.channels}
    assert kinds["LA1"] == ("seeg", False)
    assert kinds["AUX_HEART"] == ("ecg", True)
    assert kinds["AUX_MUSCLE"] == ("emg", True)

    # 2) Sidecar path traversal in published_bipolar_pairs_file MUST raise unsafe_sidecar_path
    (tmp_path / "escape_secret.json").write_text("[]", encoding="utf-8")
    trav_f = _write_fif(
        tmp_path / "trav" / "rec_trav.fif",
        ["LA1", "LA2"],
        512.0,
        2048,
        sidecar={"provenance": {"published_bipolar_pairs_file": "../escape_secret.json"}},
    )
    with pytest.raises(ToolkitError) as exc_trav:
        inspect_case(str(trav_f), workspace_dir=ws)
    assert exc_trav.value.code == "unsafe_sidecar_path"

    # 3) BDF+ with 'BDF Annotations' channel parses without header_mismatch
    bdf_p = _write_minimal_bdf_plus(tmp_path / "bdfplus" / "sample_plus.bdf")
    insp_bdf = inspect_case(str(bdf_p), workspace_dir=ws)
    assert insp_bdf["recordings"][0]["channel_count"] == 2

    # 4) Adding a BAD marker to sidecar MUST change recording.fingerprint & invalidate QC
    bad_f = _write_fif(tmp_path / "badtest" / "rec_bad.fif", ["LA1", "LA2"], 512.0, 2048, sidecar={"reference": "common"})
    inspect_case(str(bad_f), workspace_dir=ws)
    fp_before = WorkbenchStore(ws).load_recording("rec_bad", "rec_bad").fingerprint
    (tmp_path / "badtest" / "rec_bad.json").write_text(
        json.dumps({
            "reference": "common",
            "markers": [{"start": 0.5, "end": 1.5, "label": "BAD_artifact", "source": "review", "channels": ["LA1"]}],
        }),
        encoding="utf-8",
    )
    inspect_case(str(bad_f), workspace_dir=ws)
    fp_after = WorkbenchStore(ws).load_recording("rec_bad", "rec_bad").fingerprint
    assert fp_after != fp_before


# ---------------------------------------------------------------------------
# P1: Triad Control Guard, Explicit-Channel Non-Jump, Derived QC, Bipolar Guard, 540 Hz FR
# ---------------------------------------------------------------------------


def test_triad_control_guard_derived_qc_bipolar_guard_and_540hz_fr(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    # 1) 540 Hz recording: nyq=270, nyq*0.92=248.4 < 250 -> FR filter must NOT crash and must report broadband!
    f540 = _write_fif(tmp_path / "fs540.fif", ["LA1", "LA2"], 540.0, 540 * 5, sidecar={"reference": "bipolar"})
    inspect_case(str(f540), workspace_dir=ws)
    store = WorkbenchStore(ws)
    batch_fr = compute_multichannel_batch(
        store,
        case_slug="fs540",
        rec_slug="fs540",
        start_seconds=0.0,
        duration_seconds=4.0,
        montage="adjacent_bipolar",
        display_filter="fast_ripple_250_450",
    )
    assert batch_fr["display_filter_available"] is False
    assert batch_fr["effective_display_filter"] == "broadband"
    assert batch_fr["display_filter"] == "broadband"
    # Already-bipolar recording MUST skip adjacent_bipolar double subtraction!
    assert batch_fr["montage"] == "acquired"
    assert "already bipolar" in (batch_fr["montage_notice"] or "").lower()

    # 2) Derived bipolar QC propagation + control window rejection + no cross-channel jump on 0-event channel
    f_com = _write_fif(
        tmp_path / "com_ref.fif",
        ["LA1", "LA2", "LB1", "LB2"],
        512.0,
        512 * 8,
        sidecar={
            "reference": "common",
            "markers": [{"start": 1.0, "end": 2.0, "label": "BAD_seg", "source": "qc", "channels": ["LA1"]}],
        },
    )
    analyze_case(
        str(f_com),
        workspace_dir=ws,
        run_name="bip_run",
        overrides={"reference_mode": "bipolar", "detector_enabled": False},
    )
    # Inject a synthetic event on LA1-LA2 at [4.0, 4.1]s (leaving LB1-LB2 with 0 events)
    # and mark detection complete with full valid coverage so control-overlap guard is tested!
    rdir = store.run_generation_dir("com_ref", "com_ref", "bip_run")
    meta_bip = read_json(rdir / "run_meta.json")
    summary = read_json(rdir / "summary.json")
    la_ch = next(c for c in summary["channels"] if c["channel_name"] == "LA1-LA2")
    lb_ch = next(c for c in summary["channels"] if c["channel_name"] == "LB1-LB2")
    ev_stub = {
        "id": "ev_test_001",
        "channel_id": la_ch["channel_id"],
        "channel_name": la_ch["channel_name"],
        "shaft": "LA",
        "start_sample": int(4.0 * 512),
        "end_sample": int(4.1 * 512),
        "start_seconds": 4.0,
        "end_seconds": 4.1,
        "synchrony_class": "focal",
        "coactive_channel_count": 1,
        "coactive_shaft_count": 1,
    }
    det_bip = read_json(Path(meta_bip["detection_json_path"]))
    det_bip["status"] = "complete"
    det_bip["valid_intervals"] = {c["channel_id"]: [[0, 512 * 8]] for c in summary["channels"]}
    det_bip["events"] = [ev_stub]
    atomic_write_json(Path(meta_bip["detection_json_path"]), det_bip)
    summary["detector_status"] = "complete"
    atomic_write_json(rdir / "summary.json", summary)
    atomic_write_json(rdir / "events.json", [ev_stub])

    # Derived bipolar batch MUST show LA1's [1.0, 2.0]s BAD interval on LA1-LA2!
    batch_bip = compute_multichannel_batch(
        store,
        case_slug="com_ref",
        rec_slug="com_ref",
        run_name="bip_run",
        shaft="ALL",
        start_seconds=0.0,
        duration_seconds=5.0,
    )
    la_trace = next(t for t in batch_bip["traces"] if t["channel_name"] == "LA1-LA2")
    assert la_trace["invalid_seconds"] == [[1.0, 2.0]], la_trace["invalid_seconds"]

    # Control window overlapping ev_test_001 MUST raise control_window_not_clean!
    with pytest.raises(ToolkitError) as exc_ctrl:
        compute_triad_view(
            target="com_ref/com_ref",
            workspace_dir=ws,
            recording="com_ref",
            run_name="bip_run",
            channel="LA1-LA2",
            event_id="control",
            center_seconds=4.05,
            window_seconds=0.6,
        )
    assert exc_ctrl.value.code == "control_window_not_clean"

    # Explicit 0-event channel LB1-LB2 with top_focal MUST NOT jump to LA1-LA2!
    tri_lb = compute_triad_view(
        target="com_ref/com_ref",
        workspace_dir=ws,
        recording="com_ref",
        run_name="bip_run",
        channel="LB1-LB2",
        event_id="top_focal",
    )
    assert tri_lb["channel_name"] == "LB1-LB2"
    assert tri_lb["view_mode"] == "cursor_window"
    assert tri_lb["focus_event"] is None


# ---------------------------------------------------------------------------
# P1: Role Separation, Per-Run Case Sheet, Cohort Single-Recording, Protocol & BIDS
# ---------------------------------------------------------------------------


def test_role_separation_case_sheet_cohort_protocol_and_bids(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    case_dir = tmp_path / "PatientX"
    _write_fif(case_dir / "rec1.fif", ["LA1", "LA2"], 512.0, 512 * 6, sidecar={"reference": "common"})
    _write_fif(case_dir / "rec2.fif", ["LA1", "LA2"], 512.0, 512 * 6, sidecar={"reference": "common"})
    inspect_case(str(case_dir), workspace_dir=ws)
    store = WorkbenchStore(ws)

    # Cohort targeting ONLY PatientX/rec2 MUST run only rec2 (not rec1)!
    coh = run_cohort(["PatientX/rec2"], cohort_name="single_rec_coh", overrides={"detector_enabled": False}, workspace_dir=ws)
    assert coh["total_recordings"] == 1
    assert coh["recordings"][0]["recording"] == "rec2"
    assert not store.has_run("PatientX", "rec1", "baseline")
    assert store.has_run("PatientX", "rec2", "baseline")

    # Protocol save lists average_channels in stripped fields & enforces scalp/seeg modality mismatch
    analyze_case(
        "PatientX/rec2",
        workspace_dir=ws,
        run_name="avg_run",
        overrides={"reference_mode": "average", "average_channels": ["LA1", "LA2"], "detector_enabled": False},
    )
    proto = save_protocol("AvgProto", "PatientX", recording="rec2", run_name="avg_run", workspace_dir=ws)
    assert "average_channels" in proto["stripped_case_specific_fields"]
    with pytest.raises(ToolkitError) as exc_mod:
        load_protocol_config("AvgProto", store, recording_modality="scalp")
    assert exc_mod.value.code == "protocol_modality_mismatch"

    # Inject 2 synthetic events on PatientX/rec2 baseline for role & BIDS checks
    rdir = store.run_generation_dir("PatientX", "rec2", "baseline")
    det_path = Path(read_json(rdir / "run_meta.json")["detection_json_path"])
    det = read_json(det_path)
    ch0 = read_json(rdir / "summary.json")["channels"][0]
    det["status"] = "complete"
    det["valid_intervals"] = {ch0["channel_id"]: [[0, 512 * 6]]}
    det["events"] = [
        {
            "id": "ev_role_1",
            "channel_id": ch0["channel_id"],
            "channel_name": ch0["channel_name"],
            "shaft": "LA",
            "start_sample": 500,
            "end_sample": 540,
            "start_seconds": 500 / 512.0,
            "end_seconds": 540 / 512.0,
            "scores": {"artifact": 0.9, "ehfo": 0.9},
            "labels": {"artifact": 1, "ehfo": 1},
        },
        {
            "id": "ev_role_2",
            "channel_id": ch0["channel_id"],
            "channel_name": ch0["channel_name"],
            "shaft": "LA",
            "start_sample": 1000,
            "end_sample": 1040,
            "start_seconds": 1000 / 512.0,
            "end_seconds": 1040 / 512.0,
            "scores": {"artifact": 0.9, "ehfo": 0.9},
            "labels": {"artifact": 1, "ehfo": 1},
        },
    ]
    atomic_write_json(det_path, det)
    clf_path = Path(read_json(rdir / "run_meta.json")["classification_json_path"])
    atomic_write_json(clf_path, {**det, "classifier_status": "skipped", "classification": {"requested": [], "stages": {}}})

    # Record AGENT verdict on ev_role_1 -> MUST NOT increment clinician_confirmed!
    store.record_judgment(
        "PatientX",
        {
            "kind": "event_verdict",
            "recording": "rec2",
            "run_name": "baseline",
            "channel": ch0["channel_name"],
            "event_id": "ev_role_1",
            "verdict": "confirm",
            "clipped": True,
            "author": "agent",
            "role": "agent",
            "note": "Agent assessment",
        },
    )
    # Record CLINICIAN artifact rejection on ev_role_2
    store.record_judgment(
        "PatientX",
        {
            "kind": "event_verdict",
            "recording": "rec2",
            "run_name": "baseline",
            "channel": ch0["channel_name"],
            "event_id": "ev_role_2",
            "verdict": "artifact",
            "author": "dr_epilepsy",
            "role": "clinician",
            "note": "EMG artifact",
        },
    )
    summary_after = refresh_verdict_tallies(store, "PatientX", "rec2", "baseline")
    assert summary_after["counts"]["clinician_confirmed"] == 0
    assert summary_after["counts"]["clinician_artifact"] == 1
    assert summary_after["counts"]["agent_confirmed"] == 1

    # BIDS: distinct filenames for confirmed+focal vs confirmed+all, AND rejected artifact gets hfo_artifact_rejected!
    b_all = export_bids_events("PatientX", recording="rec2", run_name="baseline", only_confirmed=False, exclude_widespread_bursts=True, workspace_dir=ws)
    tsv_text = Path(b_all["tsv_path"]).read_text(encoding="utf-8")
    assert "hfo_artifact_rejected" in tsv_text

    b_cf = export_bids_events("PatientX", recording="rec2", run_name="baseline", only_confirmed=True, exclude_widespread_bursts=True, workspace_dir=ws)
    b_ca = export_bids_events("PatientX", recording="rec2", run_name="baseline", only_confirmed=True, exclude_widespread_bursts=False, workspace_dir=ws)
    assert Path(b_cf["tsv_path"]).name != Path(b_ca["tsv_path"]).name

    # Per-run Case Sheet: exporting avg_run writes PatientX_avg_run_case_sheet.html and does NOT show up as baseline's sheet!
    exp_avg = export_case_sheet("PatientX", workspace_dir=ws, primary_run="avg_run")
    assert Path(exp_avg["report_path"]).name == "PatientX_avg_run_case_sheet.html"
    srv = WorkbenchHTTPServer(workspace_dir=ws, port=0)
    boot_base = srv._bootstrap({"case": "PatientX", "recording": "rec2", "run_name": "baseline", "lite": "1"})
    boot_avg = srv._bootstrap({"case": "PatientX", "recording": "rec2", "run_name": "avg_run", "lite": "1"})
    assert boot_base["active"]["sheet_url"] is None
    assert boot_avg["active"]["sheet_url"] == "/exports/PatientX/PatientX_avg_run_case_sheet.html"


# ---------------------------------------------------------------------------
# Track 2: Security — GET /api/triad Read-Only, /exports/ Whitelist & Loopback Guard
# ---------------------------------------------------------------------------


def test_security_http_get_readonly_exports_whitelist_and_loopback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    ws = tmp_path / "ws"
    unreg_f = _write_fif(tmp_path / "outside.fif", ["LA1", "LA2"], 512.0, 2048)
    reg_f = _write_fif(tmp_path / "inside.fif", ["LA1", "LA2"], 512.0, 2048)
    analyze_case(str(reg_f), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    real_sheet = export_case_sheet("inside", workspace_dir=ws, primary_run="baseline")
    real_sheet_name = Path(real_sheet["report_path"]).name

    # Write unverified / forged files inside exports/ AND even forge an in-workspace .verified_sheets.json!
    exp_dir = WorkbenchStore(ws).case_dir("inside") / "exports"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "note.txt").write_text("<script>alert(1)</script>", encoding="utf-8")
    (exp_dir / "arbitrary.html").write_text("<html><head></head><body>hi</body></html>", encoding="utf-8")
    forged_html = "<html><head></head><body>forged</body></html>"
    (exp_dir / "forged_case_sheet.html").write_text(forged_html, encoding="utf-8")
    (exp_dir / ".verified_sheets.json").write_text(
        json.dumps({"forged_case_sheet.html": {"sha256": hashlib.sha256(forged_html.encode()).hexdigest()}}),
        encoding="utf-8",
    )
    (exp_dir / "evil.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>", encoding="utf-8")

    srv = WorkbenchHTTPServer(workspace_dir=ws, port=0)
    base_url = srv.start().rstrip("/")
    tok = srv.auth_token
    try:
        # 1) GET /api/triad on unregistered path MUST fail (400) and NOT persist a new case!
        with pytest.raises(urllib.error.HTTPError) as exc_http:
            urllib.request.urlopen(f"{base_url}/api/triad?case={unreg_f}&recording=outside", timeout=5)
        assert exc_http.value.code == 400
        assert WorkbenchStore(ws).list_cases() == [WorkbenchStore(ws).load_case("inside")]

        # 2) Unknown export extension MUST return application/octet-stream + nosniff (never text/html + token!)
        with urllib.request.urlopen(f"{base_url}/exports/inside/note.txt", timeout=5) as r_txt:
            assert r_txt.headers.get("Content-Type") == "application/octet-stream"
            assert r_txt.headers.get("X-Content-Type-Options") == "nosniff"
            body_txt = r_txt.read().decode("utf-8")
            assert tok not in body_txt

        # 3) Unverified HTML (even WITH a forged in-workspace .verified_sheets.json!) & SVG MUST be sandboxed + no token!
        for fname in ("arbitrary.html", "forged_case_sheet.html", "evil.svg"):
            with urllib.request.urlopen(f"{base_url}/exports/inside/{fname}", timeout=5) as r_u:
                assert r_u.headers.get("Content-Security-Policy") == "sandbox"
                assert "attachment" in (r_u.headers.get("Content-Disposition") or "")
                assert r_u.headers.get("X-Content-Type-Options") == "nosniff"
                assert tok not in r_u.read().decode("utf-8")

        # 4) Dotfiles like .verified_sheets.json MUST return 404
        with pytest.raises(urllib.error.HTTPError) as exc_dot:
            urllib.request.urlopen(f"{base_url}/exports/inside/.verified_sheets.json", timeout=5)
        assert exc_dot.value.code == 404

        # 5) Genuine out-of-workspace HMAC-verified case sheet DOES receive dynamic X-BrainLens-Token and renders inline
        with urllib.request.urlopen(f"{base_url}/exports/inside/{real_sheet_name}", timeout=5) as r_sh:
            assert r_sh.headers.get("Content-Security-Policy") is None
            assert f'content="{tok}"' in r_sh.read().decode("utf-8")

        # 6) Copied/untrusted workspace with a different BRAINLENS_TRUST_DIR MUST NOT trust the sheet & bootstrap returns sheet_url=None!
        monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "other_machine_trust"))
        with urllib.request.urlopen(f"{base_url}/exports/inside/{real_sheet_name}", timeout=5) as r_other:
            assert r_other.headers.get("Content-Security-Policy") == "sandbox"
            assert tok not in r_other.read().decode("utf-8")
        boot_untrusted = srv._bootstrap({"case": "inside", "recording": "inside", "run_name": "baseline", "lite": "1"})
        assert boot_untrusted["active"]["sheet_url"] is None
    finally:
        srv.stop()

    # 7) Loopback URL check used by MCP/CLI before sending X-BrainLens-Token
    assert is_loopback_http_url("http://127.0.0.1:8765/api/health") is True
    assert is_loopback_http_url("http://localhost:8765/api/ui-state") is True
    assert is_loopback_http_url("http://[::1]:8765/api/ui-state") is True
    assert is_loopback_http_url("https://127.0.0.1:8765/api/ui-state") is False
    assert is_loopback_http_url("http://evil.example.com:8765/api/ui-state") is False
    assert is_loopback_http_url("file:///etc/passwd") is False


# ---------------------------------------------------------------------------
# Track 3: Portable Engine Env Vars, Live Import Probe, Opt-in & Env Sanitization
# ---------------------------------------------------------------------------


def _make_fake_pyhfo(root: Path, *, complete: bool = True) -> Path:
    mods = ["src/hfo_app.py"]
    if complete:
        mods.extend([
            "src/param/param_detector.py",
            "src/param/param_filter.py",
            "src/utils/utils_feature.py",
        ])
    for rel in mods:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# stub\n", encoding="utf-8")
    return root


def test_engine_env_resolution_sanitization_and_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import brainlens.pyhfo_runner as pr

    pr._ENGINE_PROBE_CACHE.clear()
    # Stub interpreter simulating a broken venv missing p_tqdm
    broken_py = tmp_path / "broken_python"
    broken_py.write_text(
        '#!/bin/sh\necho \'BRAINLENS_PROBE:{"ok": false, "missing": ["p_tqdm: No module named p_tqdm"]}\'\n',
        encoding="utf-8",
    )
    broken_py.chmod(0o755)
    # Stub interpreter simulating a healthy PyHFO venv
    ok_py = tmp_path / "ok_python"
    ok_py.write_text(
        '#!/bin/sh\necho \'BRAINLENS_PROBE:{"ok": true, "missing": []}\'\n',
        encoding="utf-8",
    )
    ok_py.chmod(0o755)

    # 1) Incomplete checkout lacking src/param/param_detector.py MUST fail health & save!
    inc_src = _make_fake_pyhfo(tmp_path / "incomplete_pyhfo", complete=False)
    monkeypatch.setenv("BRAINLENS_PYHFO_SOURCE", str(inc_src))
    monkeypatch.setenv("BRAINLENS_PYHFO_PYTHON", str(ok_py))
    inc_health = pr.check_engine_health(tmp_path / "empty_ws")
    assert inc_health["configured"] is True
    assert inc_health["available"] is False
    assert "src/param/param_detector.py" in inc_health["missing_modules"]
    with pytest.raises(ToolkitError) as exc_inc:
        save_engine_settings(source_root=inc_src, python_executable=ok_py, scope="workspace", workspace_dir=tmp_path / "cfg_ws")
    assert exc_inc.value.code == "engine_incompatible"

    # 2) Complete checkout with interpreter missing p_tqdm MUST report available=False & missing_imports!
    fake_src = _make_fake_pyhfo(tmp_path / "compat_pyhfo", complete=True)
    monkeypatch.setenv("BRAINLENS_PYHFO_SOURCE", str(fake_src))
    monkeypatch.setenv("BRAINLENS_PYHFO_PYTHON", str(broken_py))
    bad_venv_health = pr.check_engine_health(tmp_path / "empty_ws")
    assert bad_venv_health["available"] is False
    assert bad_venv_health["import_ok"] is False
    assert any("p_tqdm" in m for m in bad_venv_health["missing_imports"])

    # 3) Complete checkout + healthy interpreter passes health, env resolution & env sanitization
    monkeypatch.setenv("BRAINLENS_PYHFO_PYTHON", str(ok_py))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-do-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret-do-not-leak")

    cfg = resolve_engine_settings(tmp_path / "empty_ws")
    assert Path(cfg.source_root) == fake_src
    assert Path(cfg.python_executable) == ok_py
    ok_health = pr.check_engine_health(tmp_path / "empty_ws")
    assert ok_health["available"] is True
    assert ok_health["import_ok"] is True

    san = _sanitized_worker_env(fake_src, None)
    assert "OPENAI_API_KEY" not in san
    assert "AWS_SECRET_ACCESS_KEY" not in san
    assert san["PYTHONPATH"] == str(fake_src)
    assert san["HF_HUB_OFFLINE"] == "1"

    # 4) save_engine_settings writes 0600 config; workspace engine.json requires explicit BRAINLENS_ALLOW_WORKSPACE_ENGINE=1!
    cfg_ws = tmp_path / "cfg_ws"
    saved = save_engine_settings(
        source_root=fake_src,
        python_executable=ok_py,
        scope="workspace",
        workspace_dir=cfg_ws,
    )
    saved_p = Path(saved["saved_to"])
    assert saved_p.is_file()
    assert (saved_p.stat().st_mode & 0o777) == 0o600

    for env_k in (
        "BRAINLENS_PYHFO_SOURCE",
        "BRAINLENS_PYHFO_PYTHON",
        "BRAINLENS_ALLOW_WORKSPACE_ENGINE",
        "BTK_PYHFO_SOURCE",
        "BTK_PYHFO_PYTHON",
        "BTK_ALLOW_WORKSPACE_ENGINE",
    ):
        monkeypatch.delenv(env_k, raising=False)
    monkeypatch.setattr(pr, "USER_ENGINE_JSON", tmp_path / "nonexistent_user_engine.json")
    monkeypatch.setattr(pr, "LEGACY_USER_ENGINE_JSON", tmp_path / "nonexistent_legacy_user_engine.json")
    monkeypatch.setattr(pr, "IS_SOURCE_CHECKOUT", False)
    monkeypatch.setattr(pr, "find_spec", lambda name: None)  # Exercise the no-installed-engine case in any venv.
    with pytest.raises(ToolkitError) as exc_ws:
        resolve_engine_settings(cfg_ws)
    assert exc_ws.value.code == "engine_not_configured"
    monkeypatch.setenv("BRAINLENS_ALLOW_WORKSPACE_ENGINE", "1")
    cfg_optin = resolve_engine_settings(cfg_ws)
    assert Path(cfg_optin.source_root) == fake_src


# ---------------------------------------------------------------------------
# Round 3 Adversarial Counterexample Regression Gates (P0 / P1 / P2)
# ---------------------------------------------------------------------------


def test_round3_p0_concurrent_registration_long_stem_and_conflict_rollback(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    ws = tmp_path / "ws_conc"
    fa = _write_fif(tmp_path / "pt_a" / "visit_raw.fif", ["LA1", "LA2"], 512.0, 2048)
    fb = _write_fif(tmp_path / "pt_b" / "visit_raw.fif", ["RA1", "RA2"], 512.0, 4096)

    # 1) Concurrent registration of two different directories with identical filename MUST create 2 distinct cases
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(inspect_case, str(fa), workspace_dir=ws, persist=True)
        f2 = pool.submit(inspect_case, str(fb), workspace_dir=ws, persist=True)
        res_a, res_b = f1.result(), f2.result()
    assert res_a["case"] != res_b["case"]
    assert len(WorkbenchStore(ws).list_cases()) == 2

    # 2) Long >80-char filenames in two directories MUST preserve hash disambiguation (never truncate hash off!)
    long_stem = "patient_extremely_long_recording_identifier_" + ("z" * 50) + ".fif"
    la = _write_fif(tmp_path / "long_a" / long_stem, ["LA1", "LA2"], 512.0, 2048)
    lb = _write_fif(tmp_path / "long_b" / long_stem, ["RB1", "RB2"], 512.0, 4096)
    ws_long = tmp_path / "ws_long"
    ra = inspect_case(str(la), workspace_dir=ws_long, persist=True)
    rb = inspect_case(str(lb), workspace_dir=ws_long, persist=True)
    assert ra["case"] != rb["case"]
    assert len(WorkbenchStore(ws_long).list_cases()) == 2

    # 3) Run conflict on modified file MUST NOT overwrite persisted recording.json!
    ws_conf = tmp_path / "ws_conf"
    mut_f = _write_fif(tmp_path / "mut" / "rec_raw.fif", ["LA1", "LA2"], 512.0, 2048)
    analyze_case(str(mut_f), workspace_dir=ws_conf, run_name="baseline", overrides={"detector_enabled": False, "start_seconds": 0.0, "end_seconds": 4.0})
    store_c = WorkbenchStore(ws_conf)
    orig_rec = store_c.load_recording("rec_raw", "rec_raw")
    assert abs(orig_rec.duration - 4.0) < 0.05
    # Replace file with 8s recording and attempt conflicting baseline run (overwrite=False)
    _write_fif(mut_f, ["LA1", "LA2"], 512.0, 4096)
    with pytest.raises(ToolkitError) as exc_conf:
        analyze_case(str(mut_f), workspace_dir=ws_conf, run_name="baseline", overrides={"detector_enabled": False, "start_seconds": 0.0, "end_seconds": 8.0})
    assert exc_conf.value.code == "run_conflict"
    after_rec = store_c.load_recording("rec_raw", "rec_raw")
    assert abs(after_rec.duration - 4.0) < 0.05
    assert after_rec.fingerprint == orig_rec.fingerprint


def test_round3_p1_atomic_overwrite_and_legacy_ehfo_normalization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import brainlens.pipeline as pipe_mod

    ws = tmp_path / "ws_ow"
    f = _write_fif(tmp_path / "ow_raw.fif", ["LA1", "LA2"], 512.0, 5120)  # 10s
    analyze_case(str(f), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False, "start_seconds": 0.0, "end_seconds": 4.0})
    store = WorkbenchStore(ws)
    orig = store.load_run("ow_raw", "ow_raw", "baseline")
    assert orig["summary"]["selection"]["start_seconds"] == 0.0
    assert orig["summary"]["selection"]["end_seconds"] == 4.0
    orig_gen = orig["meta"]["generation_id"]
    assert orig_gen == orig["summary"]["generation_id"]

    # Simulate I/O crash right before writing staged config.json during overwrite!
    real_write = pipe_mod.atomic_write_json

    def _boom_on_config(path: Path, data: Any, *args: Any, **kwargs: Any) -> None:
        if Path(path).name == "config.json":
            raise OSError("Simulated power loss before config commit")
        real_write(path, data, *args, **kwargs)

    monkeypatch.setattr(pipe_mod, "atomic_write_json", _boom_on_config)
    with pytest.raises(OSError):
        analyze_case(str(f), workspace_dir=ws, run_name="baseline", overwrite=True, overrides={"detector_enabled": False, "start_seconds": 6.0, "end_seconds": 10.0})
    monkeypatch.undo()

    # Original generation MUST remain completely intact and consistent (never mixed 0-4s config + 6-10s summary!)
    assert store.has_run("ow_raw", "ow_raw", "baseline") is True
    after = store.load_run("ow_raw", "ow_raw", "baseline")
    assert after["meta"]["generation_id"] == orig_gen
    assert after["config"]["analysis_config"]["start_seconds"] == 0.0
    assert after["config"]["analysis_config"]["end_seconds"] == 4.0
    assert after["summary"]["selection"]["start_seconds"] == 0.0
    assert after["summary"]["selection"]["end_seconds"] == 4.0
    # Retrying original 0-4s request cleanly returns already_existed=True
    retry = analyze_case(str(f), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False, "start_seconds": 0.0, "end_seconds": 4.0})
    assert retry["runs"][0]["already_existed"] is True

    # Legacy pre-v2 summary with ehfo status=unavailable AND stale ehfo_events=0 MUST normalize to None on load_run!
    rdir = store.run_generation_dir("ow_raw", "ow_raw", "baseline")
    sum_doc = json.loads((rdir / "summary.json").read_text(encoding="utf-8"))
    sum_doc["classification"] = {"requested": ["artifact", "ehfo"], "stages": {"ehfo": {"status": "unavailable"}}}
    sum_doc["counts"]["ehfo_events"] = 0
    for ch_r in sum_doc["channels"]:
        ch_r["ehfo_count"] = 0
        ch_r["ehfo_rate_per_min"] = 0.0
    (rdir / "summary.json").write_text(json.dumps(sum_doc), encoding="utf-8")
    meta_doc = json.loads((rdir / "run_meta.json").read_text(encoding="utf-8"))
    meta_doc["counts"]["ehfo_events"] = 0
    (rdir / "run_meta.json").write_text(json.dumps(meta_doc), encoding="utf-8")
    loaded_norm = store.load_run("ow_raw", "ow_raw", "baseline")
    assert loaded_norm["summary"]["counts"]["ehfo_events"] is None
    assert loaded_norm["meta"]["counts"]["ehfo_events"] is None
    assert store.list_runs("ow_raw", "ow_raw")[0]["counts"]["ehfo_events"] is None
    assert all(r["ehfo_count"] is None and r["ehfo_rate_per_min"] is None for r in loaded_norm["summary"]["channels"])


def test_round3_p1_control_guard_lowpass_fr_and_clip_label(tmp_path: Path) -> None:
    ws = tmp_path / "ws_ctrl_lp"
    # 1) 1024 Hz FIF with explicit acquisition lowpass = 100 Hz -> FR (250-450 Hz) MUST be unavailable!
    rng = np.random.default_rng(7)
    info = mne.create_info(["LA1", "LA2"], sfreq=1024.0, ch_types=["seeg", "seeg"])
    with info._unlock():
        info["lowpass"] = 100.0
    raw = mne.io.RawArray(rng.normal(0.0, 20e-6, size=(2, 4096)), info, verbose="ERROR")
    f_lp = tmp_path / "lp100_raw.fif"
    raw.save(f_lp, overwrite=True, verbose="ERROR")

    analyze_case(str(f_lp), workspace_dir=ws, run_name="disabled", overrides={"detector_enabled": False})

    # detector_enabled=False MUST reject event_id="control" (never certify un-detected window as quiet control!)
    with pytest.raises(ToolkitError) as exc_dis:
        compute_triad_view("lp100_raw", workspace_dir=ws, recording="lp100_raw", run_name="disabled", channel="LA1", center_seconds=2.0, event_id="control")
    assert exc_dis.value.code == "control_requires_detection"

    # Missing run MUST also reject event_id="control"
    with pytest.raises(ToolkitError) as exc_miss:
        compute_triad_view("lp100_raw", workspace_dir=ws, recording="lp100_raw", run_name="does_not_exist", channel="LA1", center_seconds=2.0, event_id="control")
    assert exc_miss.value.code == "control_requires_detection"

    # Reader & Triad MUST report Fast Ripple unavailable when acquisition lowpass is 100 Hz (even at fs=1024 Hz!)
    store = WorkbenchStore(ws)
    mc = compute_multichannel_batch(store, "lp100_raw", "lp100_raw", run_name="disabled", start_seconds=0.0, duration_seconds=2.0, display_filter="fast_ripple_250_450")
    assert mc["display_filter_available"] is False
    assert "100 Hz" in (mc["montage_notice"] or "")

    tr = compute_triad_view("lp100_raw", workspace_dir=ws, recording="lp100_raw", run_name="disabled", channel="LA1", center_seconds=2.0, event_id=None)
    assert tr["bands"]["fast_ripple"]["available"] is False
    assert "100 Hz" in tr["bands"]["fast_ripple"]["reason"]

    # Non-event interval clip exported to Case Sheet MUST be labeled "clipped interval", NOT "control window"!
    store.record_judgment("lp100_raw", {
        "kind": "interval_note",
        "recording": "lp100_raw",
        "run_name": "disabled",
        "channel": "LA1",
        "center_seconds": 2.0,
        "clipped": True,
        "author": "Dr. A",
        "role": "clinician",
        "note": "Interval clip",
    })
    sheet = export_case_sheet("lp100_raw", workspace_dir=ws, primary_run="disabled")
    sheet_html = Path(sheet["report_path"]).read_text(encoding="utf-8")
    assert '<div class="fig-verdicts">Clipped interval</div>' in sheet_html
    assert '<div class="fig-verdicts">Control window</div>' not in sheet_html


def test_round3_p1_inputs_sidecar_modality_bids_and_fingerprint(tmp_path: Path) -> None:
    # 1) run-1 *_braintoolkit.json sidecar MUST NOT match run-10 file
    sdir = tmp_path / "sidecar_test"
    f1 = _write_fif(sdir / "sub-01_run-1_raw.fif", ["LA1", "LA2"], 512.0, 2048)
    f10 = _write_fif(sdir / "sub-01_run-10_raw.fif", ["LA1", "LA2"], 512.0, 2048)
    (sdir / "sub-01_run-1_braintoolkit.json").write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    insp1 = inspect_case(str(f1), workspace_dir=tmp_path / "ws_s1", persist=False)
    insp10 = inspect_case(str(f10), workspace_dir=tmp_path / "ws_s10", persist=False)
    assert insp1["recordings"][0]["provenance"]["is_synthetic"] is True
    assert insp10["recordings"][0]["provenance"]["is_synthetic"] is False

    # 2) All-EEG native FIF MUST be classified as modality="scalp" (and detector disabled by default)
    rng = np.random.default_rng(11)
    eeg_info = mne.create_info(["Fp1", "C3", "O1"], sfreq=512.0, ch_types=["eeg", "eeg", "eeg"])
    eeg_raw = mne.io.RawArray(rng.normal(0.0, 15e-6, size=(3, 2048)), eeg_info, verbose="ERROR")
    f_scalp = tmp_path / "bids_scalp" / "sub-01_task-rest_eeg.fif"
    f_scalp.parent.mkdir(parents=True, exist_ok=True)
    eeg_raw.save(f_scalp, overwrite=True, verbose="ERROR")
    # Standard scalp BIDS *_channels.tsv marking C3 as bad
    (f_scalp.parent / "sub-01_task-rest_channels.tsv").write_text(
        "name\ttype\tstatus\nFp1\tEEG\tgood\nC3\tEEG\tbad\nO1\tEEG\tgood\n",
        encoding="utf-8",
    )
    ws_sc = tmp_path / "ws_scalp"
    insp_sc = inspect_case(str(f_scalp), workspace_dir=ws_sc, persist=True)
    rec_sc = WorkbenchStore(ws_sc).load_recording(insp_sc["case"], insp_sc["recordings"][0]["recording"])
    assert rec_sc.modality == "scalp"
    assert any(m.label == "BAD_channel" and "C3" in (m.channels or []) for m in rec_sc.markers), [m.model_dump() for m in rec_sc.markers]
    ana_sc = analyze_case(str(f_scalp), workspace_dir=ws_sc)
    sc_bundle = WorkbenchStore(ws_sc).load_run(insp_sc["case"], insp_sc["recordings"][0]["recording"], ana_sc["run_name"])
    assert sc_bundle["summary"]["detector_status"] == "skipped"

    # 3) Fingerprint MUST change when contact metadata changes, and BAD channels order is deterministic
    f_det = tmp_path / "bad_order_raw.fif"
    info_b = mne.create_info(["LA1", "LA2", "LB1"], sfreq=512.0, ch_types=["seeg"] * 3)
    raw_b = mne.io.RawArray(rng.normal(0.0, 15e-6, size=(3, 2048)), info_b, verbose="ERROR")
    raw_b.info["bads"] = ["LB1", "LA1"]
    raw_b.save(f_det, overwrite=True, verbose="ERROR")
    rec_b1 = inspect_case(str(f_det), workspace_dir=tmp_path / "ws_b1", persist=True)
    r_obj = WorkbenchStore(tmp_path / "ws_b1").load_recording(rec_b1["case"], rec_b1["recordings"][0]["recording"])
    bad_markers = [tuple(m.channels or []) for m in r_obj.markers if m.label == "BAD_channel"]
    assert bad_markers == [("LA1",), ("LB1",)], bad_markers
    # Updating contacts on a channel via sidecar MUST change recording.fingerprint!
    fp_before = r_obj.fingerprint
    (tmp_path / "bad_order_raw.json").write_text(
        json.dumps({"channels": {"LA1": {"contacts": ["LA1_CORRECTED"]}}}),
        encoding="utf-8",
    )
    rec_b2 = inspect_case(str(f_det), workspace_dir=tmp_path / "ws_b2", persist=True)
    r_obj2 = WorkbenchStore(tmp_path / "ws_b2").load_recording(rec_b2["case"], rec_b2["recordings"][0]["recording"])
    assert r_obj2.channels[0].contacts == ["LA1_CORRECTED"]
    assert r_obj2.fingerprint != fp_before


def test_round3_p1_synchrony_41ms_3shaft_and_compare_runs_car_subset(tmp_path: Path) -> None:
    from brainlens.analytics import annotate_event_synchrony
    from brainlens.models import ChannelSpec

    # 1) Exact ±30 ms synchrony check: 41 ms gap MUST NOT be grouped into ±30 ms window!
    chs_5 = [
        ChannelSpec(id=f"c{i}", name=f"{sh}1", original_label=f"{sh}1", kind="seeg", shaft=sh, contact_index=1)
        for i, sh in enumerate(["LA", "LB", "LC", "LD", "LE"])
    ]
    # Target on LA & LB at [1000, 1050] ms; LC/LD/LE at [1091, 1140] ms -> 41 ms gap > 30 ms!
    evs = [
        {"id": "e0", "channel_id": "c0", "channel_name": "LA1", "shaft": "LA", "start_sample": 1000, "end_sample": 1050},
        {"id": "e1", "channel_id": "c1", "channel_name": "LB1", "shaft": "LB", "start_sample": 1005, "end_sample": 1050},
        {"id": "e2", "channel_id": "c2", "channel_name": "LC1", "shaft": "LC", "start_sample": 1091, "end_sample": 1140},
        {"id": "e3", "channel_id": "c3", "channel_name": "LD1", "shaft": "LD", "start_sample": 1091, "end_sample": 1140},
        {"id": "e4", "channel_id": "c4", "channel_name": "LE1", "shaft": "LE", "start_sample": 1091, "end_sample": 1140},
    ]
    ann, bursts = annotate_event_synchrony(evs, chs_5, sfreq=1000.0, window_ms=30.0, min_burst_shafts=5)
    by_id = {e["id"]: e for e in ann}
    assert by_id["e0"]["coactive_shaft_count"] == 2
    assert by_id["e0"]["synchrony_class"] == "focal"
    assert bursts == []

    # 2) 3-shaft recording with min_burst_shafts=5 MUST stay "regional", NOT "widespread_burst"!
    chs_3 = chs_5[:3]
    evs_3 = [
        {"id": "a", "channel_id": "c0", "channel_name": "LA1", "shaft": "LA", "start_sample": 1000, "end_sample": 1050},
        {"id": "b", "channel_id": "c1", "channel_name": "LB1", "shaft": "LB", "start_sample": 1000, "end_sample": 1050},
        {"id": "c", "channel_id": "c2", "channel_name": "LC1", "shaft": "LC", "start_sample": 1000, "end_sample": 1050},
    ]
    ann3, bursts3 = annotate_event_synchrony(evs_3, chs_3, sfreq=1000.0, window_ms=30.0, min_burst_shafts=5)
    assert all(e["synchrony_class"] == "regional" for e in ann3)
    assert bursts3 == []

    # 3) Auto-CAR pool inequality -> directly_comparable_rates=False; subset-channel compare -> NO KeyError!
    ws = tmp_path / "ws_cmp"
    f_sub = _write_fif(tmp_path / "sub_raw.fif", ["LA1", "LA2", "LB1"], 1000.0, 4000, sidecar={"reference": "common"})
    analyze_case(str(f_sub), workspace_dir=ws, run_name="car_3ch", overrides={"detector_enabled": False, "reference_mode": "average"})
    analyze_case(str(f_sub), workspace_dir=ws, run_name="car_2ch", overrides={"detector_enabled": False, "channels": ["LA1", "LA2"], "reference_mode": "average"})
    cmp_car = compare_runs(WorkbenchStore(ws), "sub_raw", "sub_raw", "car_3ch", "car_2ch")
    assert cmp_car["directly_comparable_rates"] is False

    # Same original reference, 3ch vs 1ch subset with synthetic active ranks -> NO KeyError!
    analyze_case(str(f_sub), workspace_dir=ws, run_name="orig_3ch", overrides={"detector_enabled": False})
    analyze_case(str(f_sub), workspace_dir=ws, run_name="orig_1ch", overrides={"detector_enabled": False, "channels": ["LB1"]})
    store = WorkbenchStore(ws)
    for rn, active_ch in (("orig_3ch", "LA1"), ("orig_1ch", "LB1")):
        sp = store.run_generation_dir("sub_raw", "sub_raw", rn) / "summary.json"
        sd = json.loads(sp.read_text(encoding="utf-8"))
        for row in sd["channels"]:
            if row["channel_name"] == active_ch:
                row["status"] = "measured"
                row["focal_rank"] = 1
                row["focal_count"] = 2
                row["focal_rate_per_min"] = 30.0
        sp.write_text(json.dumps(sd), encoding="utf-8")
    cmp_sub = compare_runs(store, "sub_raw", "sub_raw", "orig_3ch", "orig_1ch")
    assert cmp_sub["directly_comparable_rates"] is True
    row_names = {r["channel"] for r in cmp_sub["channels"]}
    assert {"LA1", "LB1"}.issubset(row_names)


def test_round3_p1_roles_legacy_sheet_bids_long_run_and_cli_workbench(tmp_path: Path) -> None:
    from brainlens.cli import build_parser

    ws = tmp_path / "ws_r3"
    f = _write_fif(tmp_path / "r3_raw.fif", ["LA1", "LA2"], 1000.0, 4000)
    analyze_case(str(f), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    store = WorkbenchStore(ws)
    rdir = store.run_generation_dir("r3_raw", "r3_raw", "baseline")
    meta_r3 = read_json(rdir / "run_meta.json")
    rec_r3 = store.load_recording("r3_raw", "r3_raw")
    ev = {
        "id": "ev_r3_001",
        "channel_id": rec_r3.channels[0].id,
        "channel_name": "LA1",
        "shaft": "LA",
        "start_sample": 1000,
        "end_sample": 1060,
        "start_seconds": 1.0,
        "end_seconds": 1.06,
        "scores": {},
        "labels": {},
    }
    det_r3 = read_json(Path(meta_r3["detection_json_path"]))
    det_r3["status"] = "complete"
    det_r3["valid_intervals"] = {c.id: [[0, 4000]] for c in rec_r3.channels}
    det_r3["events"] = [ev]
    atomic_write_json(Path(meta_r3["detection_json_path"]), det_r3)
    atomic_write_json(Path(meta_r3["classification_json_path"]), {**det_r3, "classifier_status": "skipped", "classification": {"requested": [], "stages": {}}})
    from brainlens.pipeline import refresh_verdict_tallies
    refresh_verdict_tallies(store, "r3_raw", "r3_raw", "baseline")

    # 1) Researcher confirm MUST NOT overwrite Clinician artifact; report-action rehearsal MUST NOT count as clinician!
    srv = WorkbenchHTTPServer(workspace_dir=ws, port=0)
    srv._save_judgment({"case": "r3_raw", "recording": "r3_raw", "run_name": "baseline", "channel": "LA1", "event_id": "ev_r3_001", "verdict": "artifact", "author": "Dr. Clin", "role": "clinician"})
    srv._save_judgment({"case": "r3_raw", "recording": "r3_raw", "run_name": "baseline", "channel": "LA1", "event_id": "ev_r3_001", "verdict": "confirm", "author": "Res. Bob", "role": "researcher"})
    srv._record_report_action({
        "case": "r3_raw",
        "recording": "r3_raw",
        "run_name": "baseline",
        "channel": "LA1",
        "event_id": "ev_r3_001",
        "action": "verdict",
        "verdict": "confirm",
        "role": "engineering_rehearsal",
        "author": "QA",
    })
    bundle = store.load_run("r3_raw", "r3_raw", "baseline")
    ev_after = bundle["events"][0]
    assert ev_after["clinician_verdict"] == "artifact"
    assert ev_after["researcher_verdict"] == "confirm"
    assert ev_after["rehearsal_verdict"] == "confirm"
    assert bundle["summary"]["counts"]["clinician_artifact"] == 1
    assert bundle["summary"]["counts"]["clinician_confirmed"] == 0
    assert bundle["summary"]["counts"]["researcher_confirmed"] == 1
    assert bundle["summary"]["counts"]["rehearsal_confirmed"] == 1

    # 2) Legacy <case>_case_sheet.html belonging to onnx_clf MUST NOT be served when active_run == baseline!
    exp_dir = store.case_dir("r3_raw") / "exports"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "r3_raw_case_sheet.html").write_text(
        '<html><head></head><body><script>window.__BRAINLENS_REPORT__={"case":"r3_raw","primary_run":"onnx_clf"};</script></body></html>',
        encoding="utf-8",
    )
    boot_base = srv._bootstrap({"case": "r3_raw", "recording": "r3_raw", "run_name": "baseline", "lite": "1"})
    assert boot_base["active"]["sheet_url"] is None

    # 3) Long 69-char run_name BIDS exports MUST produce distinct files ending in _confirmed_focal & _confirmed_all!
    long_run = "r" * 69
    analyze_case(str(f), workspace_dir=ws, run_name=long_run, overrides={"detector_enabled": False})
    b_cf = export_bids_events("r3_raw", recording="r3_raw", run_name=long_run, only_confirmed=True, exclude_widespread_bursts=True, workspace_dir=ws)
    b_ca = export_bids_events("r3_raw", recording="r3_raw", run_name=long_run, only_confirmed=True, exclude_widespread_bursts=False, workspace_dir=ws)
    assert b_cf["tsv_path"] != b_ca["tsv_path"]
    assert b_cf["tsv_path"].endswith("_confirmed_focal_events.tsv")
    assert b_ca["tsv_path"].endswith("_confirmed_all_events.tsv")

    # 4) P2: brainlens-workbench --workspace DIR --no-open flag order lifting
    parser = build_parser()
    # Emulate workbench_main lifting --workspace before subcommand
    raw_argv = ["--workspace", str(ws), "--no-open", "--port", "9999"]
    global_a, sub_a = [], []
    i = 0
    while i < len(raw_argv):
        if raw_argv[i] == "--workspace" and i + 1 < len(raw_argv):
            global_a.extend([raw_argv[i], raw_argv[i + 1]])
            i += 2
        else:
            sub_a.append(raw_argv[i])
            i += 1
    parsed = parser.parse_args([*global_a, "workbench", *sub_a])
    assert parsed.command == "workbench"
    assert parsed.workspace == str(ws)
    assert parsed.no_open is True
    assert parsed.port == 9999


# ---------------------------------------------------------------------------
# Round 4 Adversarial Counterexample Regression Gates (P0 / P1)
# ---------------------------------------------------------------------------


def test_round4_p0_source_recording_modified_rejected_on_reader_and_triad(tmp_path: Path) -> None:
    ws = tmp_path / "ws_src_mut"
    t4 = np.arange(4 * 512, dtype=np.float64) / 512.0
    sig10 = np.vstack([5e-6 * np.sin(2 * np.pi * 10.0 * t4), 5e-6 * np.cos(2 * np.pi * 10.0 * t4)])
    info4 = mne.create_info(["LA1", "LA2"], sfreq=512.0, ch_types=["seeg", "seeg"])
    f_mut = tmp_path / "patient_raw.fif"
    mne.io.RawArray(sig10, info4, verbose="ERROR").save(f_mut, overwrite=True, verbose="ERROR")

    analyze_case(str(f_mut), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    store = WorkbenchStore(ws)
    # Initial read succeeds cleanly (~10 uV P2P)
    mc_ok = compute_multichannel_batch(store, "patient_raw", "patient_raw", run_name="baseline", start_seconds=0.0, duration_seconds=2.0)
    vals_ok = [v for v in mc_ok["traces"][0]["values_uv"] if v is not None]
    assert max(vals_ok) - min(vals_ok) < 20.0

    # Replace same path with 8s, 100 uV recording -> analyze raises run_conflict AND Reader/Triad MUST refuse!
    t8 = np.arange(8 * 512, dtype=np.float64) / 512.0
    sig100 = np.vstack([50e-6 * np.sin(2 * np.pi * 10.0 * t8), 50e-6 * np.cos(2 * np.pi * 10.0 * t8)])
    mne.io.RawArray(sig100, info4, verbose="ERROR").save(f_mut, overwrite=True, verbose="ERROR")
    with pytest.raises(ToolkitError) as exc_conf:
        analyze_case(str(f_mut), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    assert exc_conf.value.code == "run_conflict"

    with pytest.raises(ToolkitError) as exc_mc:
        compute_multichannel_batch(store, "patient_raw", "patient_raw", run_name="baseline", start_seconds=0.0, duration_seconds=2.0)
    assert exc_mc.value.code == "source_recording_modified"

    with pytest.raises(ToolkitError) as exc_tr:
        compute_triad_view("patient_raw/patient_raw", workspace_dir=ws, recording="patient_raw", run_name="baseline", channel="LA1", center_seconds=1.0, event_id=None)
    assert exc_tr.value.code == "source_recording_modified"

    # Also test same-size, same-duration (4s) voltage replacement (10 uV -> 100 uV) -> MUST STILL BE REJECTED!
    sig100_4s = np.vstack([50e-6 * np.sin(2 * np.pi * 10.0 * t4), 50e-6 * np.cos(2 * np.pi * 10.0 * t4)])
    mne.io.RawArray(sig100_4s, info4, verbose="ERROR").save(f_mut, overwrite=True, verbose="ERROR")
    with pytest.raises(ToolkitError) as exc_same_sz:
        compute_multichannel_batch(store, "patient_raw", "patient_raw", run_name="baseline", start_seconds=0.0, duration_seconds=2.0)
    assert exc_same_sz.value.code == "source_recording_modified"


def test_round4_p1_batch_recording_slug_no_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "batch_case"
    _write_fif(root / "a" / "visit_raw.fif", ["LA1", "LA2"], 512.0, 2048)
    _write_fif(root / "b" / "visit_raw.fif", ["LB1", "LB2"], 512.0, 2048)
    _write_fif(root / "z" / "a_visit_raw.fif", ["LC1", "LC2"], 512.0, 2048)
    ws = tmp_path / "ws_batch"
    insp = inspect_case(str(root), workspace_dir=ws, persist=True)
    assert insp["recording_count"] == 3
    recs = WorkbenchStore(ws).list_recordings(insp["case"])
    assert len(recs) == 3
    assert len(set(recs)) == 3


def test_round4_p1_mvcc_torn_read_and_concurrent_conflict(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    ws = tmp_path / "ws_mvcc"
    f = _write_fif(tmp_path / "mvcc_raw.fif", ["LA1", "LA2"], 512.0, 5120)  # 10s
    store = WorkbenchStore(ws)

    # 1) Concurrent overwrite=False with DIFFERENT configs for the SAME run_name:
    #    EXACTLY ONE succeeds and the other MUST raise run_conflict!
    inspect_case(str(f), workspace_dir=ws, persist=True)

    barrier = threading.Barrier(2)

    def _run_cfg(win: tuple[float, float]) -> str:
        barrier.wait(timeout=10.0)
        try:
            analyze_case(
                "mvcc_raw/mvcc_raw",
                workspace_dir=ws,
                run_name="race_run",
                overwrite=False,
                overrides={"detector_enabled": False, "start_seconds": win[0], "end_seconds": win[1]},
            )
            return "ok"
        except ToolkitError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [
            pool.submit(_run_cfg, (0.0, 4.0)),
            pool.submit(_run_cfg, (6.0, 10.0)),
        ]
        outcomes = sorted(fut.result() for fut in futs)
    assert outcomes == ["ok", "run_conflict"], outcomes

    # 2) Interleaved concurrent load_run() during repeated overwrite=True MUST NEVER tear generations!
    errors: list[str] = []
    stop_flag = [False]

    def _reader() -> None:
        for _ in range(40):
            if stop_flag[0]:
                break
            try:
                b = store.load_run("mvcc_raw", "mvcc_raw", "race_run")
                g_m = b["meta"].get("generation_id")
                g_c = b["config"].get("generation_id")
                g_s = b["summary"].get("generation_id")
                if not (g_m and g_m == g_c == g_s):
                    errors.append(f"torn gen: {g_m} vs {g_c} vs {g_s}")
                c_start = b["config"]["analysis_config"]["start_seconds"]
                s_start = b["summary"]["selection"]["start_seconds"]
                if c_start != s_start:
                    errors.append(f"torn window: cfg={c_start} vs sum={s_start}")
            except Exception as exc:
                errors.append(f"reader_exc: {exc}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        rfut = pool.submit(_reader)
        for win in ((0.0, 4.0), (6.0, 10.0), (0.0, 4.0), (6.0, 10.0)):
            analyze_case(
                "mvcc_raw/mvcc_raw",
                workspace_dir=ws,
                run_name="race_run",
                overwrite=True,
                overrides={"detector_enabled": False, "start_seconds": win[0], "end_seconds": win[1]},
            )
        stop_flag[0] = True
        rfut.result()
    assert errors == [], errors


def test_round4_p1_run_lowpass_and_mixed_rate_channel_fr_guard(tmp_path: Path) -> None:
    import edfio

    ws = tmp_path / "ws_fr_r4"
    # 1) Run analysis lowpass = 100 Hz on 1024 Hz recording -> FR MUST be unavailable in Reader & Triad!
    f1024 = _write_fif(tmp_path / "fs1024_raw.fif", ["LA1", "LA2"], 1024.0, 4096)
    analyze_case(
        str(f1024),
        workspace_dir=ws,
        run_name="lp100_run",
        overrides={"detector_enabled": False, "filters": {"lowpass": 100.0}},
    )
    store = WorkbenchStore(ws)
    mc_lp = compute_multichannel_batch(
        store, "fs1024_raw", "fs1024_raw", run_name="lp100_run", start_seconds=0.0, duration_seconds=2.0, display_filter="fast_ripple_250_450"
    )
    assert mc_lp["display_filter_available"] is False
    tr_lp = compute_triad_view(
        "fs1024_raw/fs1024_raw", workspace_dir=ws, recording="fs1024_raw", run_name="lp100_run", channel="LA1", center_seconds=1.5, event_id=None
    )
    assert tr_lp["bands"]["fast_ripple"]["available"] is False

    # 2) Mixed-rate EDF (LA1 @ 1024 Hz, LB1 @ 256 Hz): selecting LB1 alone MUST report FR unavailable!
    rng = np.random.default_rng(19)
    edf_p = tmp_path / "mixed_rate.edf"
    s1 = edfio.EdfSignal(rng.normal(0.0, 20.0, 4 * 1024), sampling_frequency=1024.0, label="LA1", physical_dimension="uV")
    s2 = edfio.EdfSignal(rng.normal(0.0, 20.0, 4 * 256), sampling_frequency=256.0, label="LB1", physical_dimension="uV")
    edfio.Edf([s1, s2]).write(edf_p)
    inspect_case(str(edf_p), workspace_dir=ws, persist=True)
    mc_lb = compute_multichannel_batch(
        store, "mixed_rate", "mixed_rate", shaft="LB", start_seconds=0.0, duration_seconds=2.0, display_filter="fast_ripple_250_450"
    )
    assert mc_lb["display_filter_available"] is False
    tr_lb = compute_triad_view(
        "mixed_rate/mixed_rate", workspace_dir=ws, recording="mixed_rate", channel="LB1", center_seconds=1.5, event_id=None
    )
    assert tr_lb["bands"]["fast_ripple"]["available"] is False


def _write_sine_fif(path: Path, amp_uv: float, *, sfreq: float = 512.0, n_samples: int = 2048, split_size: str | int | None = None) -> Path:
    t = np.arange(n_samples, dtype=np.float64) / sfreq
    sig_v = (amp_uv * 1e-6) * np.sin(2.0 * np.pi * 12.0 * t)
    data = np.vstack([sig_v, sig_v * 0.5])
    info = mne.create_info(["LA1", "LA2"], sfreq=sfreq, ch_types=["seeg", "seeg"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    kw: dict[str, Any] = {"overwrite": True, "verbose": "ERROR"}
    if split_size is not None:
        kw["split_size"] = split_size
    raw.save(str(path), **kw)
    return path


def test_round5_p0_run_recording_binding_and_registered_branch_integrity(tmp_path: Path) -> None:
    ws = tmp_path / "ws_r5_bind"
    store = WorkbenchStore(ws)

    # (A) Replace FIF in place (10 uV -> 100 uV), analyze new run_name -> OLD baseline MUST be rejected!
    fif_a = _write_sine_fif(tmp_path / "patientA_raw.fif", 10.0)
    analyze_case(str(fif_a), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    mc0 = compute_multichannel_batch(store, "patientA_raw", "patientA_raw", run_name="baseline", start_seconds=0.0, duration_seconds=2.0)
    peak0 = max(abs(v) for v in mc0["traces"][0]["values_uv"] if v is not None)
    assert 6.0 <= peak0 <= 15.0, peak0

    _write_sine_fif(fif_a, 100.0)
    analyze_case(str(fif_a), workspace_dir=ws, run_name="reanalyzed", overrides={"detector_enabled": False})

    with pytest.raises(ToolkitError) as e_mc:
        compute_multichannel_batch(store, "patientA_raw", "patientA_raw", run_name="baseline", start_seconds=0.0, duration_seconds=2.0)
    assert e_mc.value.code == "run_recording_mismatch"

    with pytest.raises(ToolkitError) as e_tr:
        compute_triad_view("patientA_raw/patientA_raw", workspace_dir=ws, recording="patientA_raw", run_name="baseline", channel="LA1", center_seconds=1.0, event_id=None)
    assert e_tr.value.code == "run_recording_mismatch"

    mc_new = compute_multichannel_batch(store, "patientA_raw", "patientA_raw", run_name="reanalyzed", start_seconds=0.0, duration_seconds=2.0)
    peak_new = max(abs(v) for v in mc_new["traces"][0]["values_uv"] if v is not None)
    assert 70.0 <= peak_new <= 130.0, peak_new

    # (B) Replace FIF on disk, then branch using REGISTERED target 'case/rec' -> MUST refuse with source_recording_modified!
    fif_b = _write_sine_fif(tmp_path / "patientB_raw.fif", 10.0)
    analyze_case(str(fif_b), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    _write_sine_fif(fif_b, 100.0)
    with pytest.raises(ToolkitError) as e_br:
        analyze_case("patientB_raw/patientB_raw", workspace_dir=ws, run_name="branch_after_mutate", overrides={"detector_enabled": False, "start_seconds": 0.5})
    assert e_br.value.code == "source_recording_modified"


def test_round5_p0_split_fif_large_brainvision_and_legacy_integrity(tmp_path: Path) -> None:
    ws = tmp_path / "ws_r5_integ"
    store = WorkbenchStore(ws)

    # (A) Split FIF: mutating ONLY the 2nd split part (*-1.fif) MUST be rejected!
    sp_dir = tmp_path / "split_case"
    sp_dir.mkdir()
    sp_base = sp_dir / "part_raw.fif"
    _write_sine_fif(sp_base, 10.0, sfreq=512.0, n_samples=512 * 80, split_size="1.2MB")
    part2 = sp_dir / "part_raw-1.fif"
    assert part2.is_file(), f"Expected MNE split FIF continuation file at {part2}"
    analyze_case(str(sp_base), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    rec_sp = store.load_recording("part_raw", "part_raw")
    assert str(part2.resolve()) in (rec_sp.companion_files or {}), rec_sp.companion_files

    # Build replacement 100 uV split FIF in another dir and swap ONLY part-1.fif!
    alt_dir = tmp_path / "split_alt"
    alt_dir.mkdir()
    _write_sine_fif(alt_dir / "part_raw.fif", 100.0, sfreq=512.0, n_samples=512 * 80, split_size="1.2MB")
    shutil.copyfile(alt_dir / "part_raw-1.fif", part2)
    with pytest.raises(ToolkitError) as e_sp:
        compute_multichannel_batch(store, "part_raw", "part_raw", run_name="baseline", start_seconds=60.0, duration_seconds=2.0)
    assert e_sp.value.code == "source_recording_modified"

    # (B) >32 MiB BrainVision .eeg mutated with IDENTICAL size & IDENTICAL mtime MUST be rejected!
    bv_dir = tmp_path / "bv_large"
    bv_dir.mkdir()
    vhdr = bv_dir / "big_ieeg.vhdr"
    vmrk = bv_dir / "big_ieeg.vmrk"
    eeg = bv_dir / "big_ieeg.eeg"
    n_pts = 4_194_816  # 2 channels * float32 (4B) = 33,558,528 bytes (> 32 MiB!)
    arr = np.zeros((n_pts, 2), dtype="<f4")
    arr[:, 0] = 10.0  # 10 uV
    arr.tofile(eeg)
    assert eeg.stat().st_size == 33_558_528
    vmrk.write_text("Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\nDataFile=big_ieeg.eeg\n[Marker Infos]\n", encoding="utf-8")
    vhdr.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\nDataFile=big_ieeg.eeg\nMarkerFile=big_ieeg.vmrk\nDataFormat=BINARY\n"
        "DataOrientation=MULTIPLEXED\nNumberOfChannels=2\nSamplingInterval=1000\n"
        "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n"
        "[Channel Infos]\nCh1=LA1,,1,µV\nCh2=LA2,,1,µV\n",
        encoding="utf-8",
    )
    inspect_case(str(vhdr), workspace_dir=ws, persist=True)
    orig_st = eeg.stat()
    # Mutate .eeg bytes in-place preserving exact size & exact atime_ns/mtime_ns!
    with eeg.open("r+b") as fh:
        fh.seek(0)
        fh.write(np.full(1024, 100.0, dtype="<f4").tobytes())
    os.utime(eeg, ns=(orig_st.st_atime_ns, orig_st.st_mtime_ns))
    assert eeg.stat().st_size == orig_st.st_size
    assert eeg.stat().st_mtime_ns == orig_st.st_mtime_ns
    with pytest.raises(ToolkitError) as e_bv:
        compute_multichannel_batch(store, "big_ieeg", "big_ieeg", start_seconds=0.0, duration_seconds=2.0)
    assert e_bv.value.code == "source_recording_modified"

    # (C) Legacy recording.json lacking source_sha256/source_file_size/source_file_mtime_ns/companion_files:
    fif_leg = _write_sine_fif(tmp_path / "legacy_rec_raw.fif", 10.0)
    analyze_case(str(fif_leg), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    rpath = store.recording_path("legacy_rec_raw", "legacy_rec_raw")
    rdoc = read_json(rpath)
    for k in ("source_sha256", "source_file_size", "source_file_mtime_ns", "companion_files"):
        rdoc["spec"].pop(k, None)
    atomic_write_json(rpath, rdoc)
    # Unchanged legacy file still works!
    mc_leg_ok = compute_multichannel_batch(store, "legacy_rec_raw", "legacy_rec_raw", run_name="baseline", start_seconds=0.0, duration_seconds=1.0)
    assert len(mc_leg_ok["traces"]) == 2
    # Mutate same-size legacy file preserving mtime -> MUST raise source_recording_modified!
    leg_st = fif_leg.stat()
    _write_sine_fif(fif_leg, 100.0)
    os.utime(fif_leg, ns=(leg_st.st_atime_ns, leg_st.st_mtime_ns))
    with pytest.raises(ToolkitError) as e_leg:
        compute_multichannel_batch(store, "legacy_rec_raw", "legacy_rec_raw", run_name="baseline", start_seconds=0.0, duration_seconds=1.0)
    assert e_leg.value.code == "source_recording_modified"


def test_round5_p0_incremental_reimport_preserves_existing_recording_slugs(tmp_path: Path) -> None:
    ws = tmp_path / "ws_r5_slug"
    folder = tmp_path / "inc_case"
    (folder / "a").mkdir(parents=True)
    (folder / "b").mkdir(parents=True)
    f_a = _write_sine_fif(folder / "a" / "visit_raw.fif", 10.0)
    f_b = _write_sine_fif(folder / "b" / "visit_raw.fif", 25.0)

    analyze_case(str(folder), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    store = WorkbenchStore(ws)
    assert set(store.list_recordings("inc_case")) == {"a_visit_raw", "b_visit_raw"}
    assert Path(store.load_recording("inc_case", "a_visit_raw").path).resolve() == f_a.resolve()

    # Now add z/a_visit_raw.fif (100 uV) and incrementally re-import folder!
    (folder / "z").mkdir(parents=True)
    f_z = _write_sine_fif(folder / "z" / "a_visit_raw.fif", 100.0)
    inspect_case(str(folder), workspace_dir=ws, persist=True)

    recs_after = set(store.list_recordings("inc_case"))
    assert len(recs_after) == 3, recs_after
    # Existing `a_visit_raw` MUST still bind to a/visit_raw.fif (10 uV), NEVER stolen by z/a_visit_raw.fif!
    assert Path(store.load_recording("inc_case", "a_visit_raw").path).resolve() == f_a.resolve()
    assert Path(store.load_recording("inc_case", "b_visit_raw").path).resolve() == f_b.resolve()
    z_slugs = [r for r in recs_after if Path(store.load_recording("inc_case", r).path).resolve() == f_z.resolve()]
    assert len(z_slugs) == 1 and z_slugs[0] != "a_visit_raw", z_slugs
    mc_a = compute_multichannel_batch(store, "inc_case", "a_visit_raw", run_name="baseline", start_seconds=0.0, duration_seconds=1.0)
    peak_a = max(abs(v) for v in mc_a["traces"][0]["values_uv"] if v is not None)
    assert 6.0 <= peak_a <= 15.0, peak_a


def test_round5_p1_mvcc_gc_safety_verdict_sync_and_atomic_top_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import brainlens.pipeline as pipe_mod

    ws = tmp_path / "ws_r5_mvcc"
    store = WorkbenchStore(ws)
    analyze_case(str(SYNTHETIC_EDF), workspace_dir=ws, run_name="baseline")
    b0 = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")
    assert len(b0["events"]) > 0
    ev0 = b0["events"][0]

    # (A) Reader pins gdir; 6 rapid verdict refreshes run -> gdir MUST NOT be deleted (grace-period GC)!
    pinned_gdir = Path(b0["generation_dir"])
    for idx in range(6):
        store.record_judgment("synthetic_seeg", {
            "kind": "event_verdict",
            "recording": "synthetic_seeg",
            "run_name": "baseline",
            "channel": ev0["channel_name"],
            "event_id": ev0["id"],
            "verdict": "confirm" if idx % 2 == 0 else "uncertain",
            "author": "Res. A",
            "role": "researcher",
            "note": f"refresh {idx}",
        })
        refresh_verdict_tallies(store, "synthetic_seeg", "synthetic_seeg", "baseline")
    assert pinned_gdir.is_dir() and (pinned_gdir / "summary.json").is_file()

    # (B) Atomic top-level symlinks point through single `current` symlink (zero lag between CURRENT.json and events.csv)
    rdir = store.run_dir("synthetic_seeg", "synthetic_seeg", "baseline")
    assert (rdir / "current").is_symlink()
    for top_name in ("CURRENT.json", "channel_rates.csv", "epoch_rates.csv", "events.csv"):
        assert (rdir / top_name).is_symlink(), top_name
        assert os.readlink(rdir / top_name) == f"current/{top_name}"

    # (C) Verdict accepted during an overwrite=True AFTER summarize_run MUST NOT be lost on commit!
    # Clear judgments first on a fresh branch run
    analyze_case("synthetic_seeg/synthetic_seeg", workspace_dir=ws, run_name="sync_ovw")
    b_ovw = store.load_run("synthetic_seeg", "synthetic_seeg", "sync_ovw")
    ev_ovw = b_ovw["events"][0]

    orig_summarize = pipe_mod.summarize_run
    injected = {"done": False}

    def _hooked_summarize(*args: Any, **kwargs: Any):
        res = orig_summarize(*args, **kwargs)
        if not injected["done"]:
            injected["done"] = True
            # Simulate a concurrent researcher verdict accepted right after initial summarize_run!
            store.record_judgment("synthetic_seeg", {
                "kind": "event_verdict",
                "recording": "synthetic_seeg",
                "run_name": "sync_ovw",
                "channel": ev_ovw["channel_name"],
                "event_id": ev_ovw["id"],
                "verdict": "confirm",
                "author": "Res. Concurrent",
                "role": "researcher",
                "note": "arrived during overwrite",
            })
        return res

    monkeypatch.setattr(pipe_mod, "summarize_run", _hooked_summarize)
    analyze_case("synthetic_seeg/synthetic_seeg", workspace_dir=ws, run_name="sync_ovw", overwrite=True)
    b_after = store.load_run("synthetic_seeg", "synthetic_seeg", "sync_ovw")
    assert b_after["summary"]["counts"]["researcher_confirmed"] == 1, b_after["summary"]["counts"]
    assert b_after["meta"]["counts"]["researcher_confirmed"] == 1


def test_round5_p1_overwrite_invalidates_mdt_sheet_in_bootstrap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trust_dir = tmp_path / "trust_r5_ovw"
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(trust_dir))
    ws = tmp_path / "ws_r5_sheet_ovw"
    analyze_case(str(SYNTHETIC_EDF), workspace_dir=ws, run_name="baseline")
    export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline")

    srv = WorkbenchHTTPServer(ws, host="127.0.0.1", port=0)
    url = srv.start()
    try:
        with urllib.request.urlopen(f"{url}/api/bootstrap?case=synthetic_seeg&run_name=baseline&lite=1", timeout=5) as r1:
            b1 = json.loads(r1.read().decode("utf-8"))
        assert b1["active"]["sheet_url"] is not None

        # Overwrite baseline to 0 events -> /api/bootstrap MUST invalidate the stale 6-event sheet!
        analyze_case("synthetic_seeg/synthetic_seeg", workspace_dir=ws, run_name="baseline", overwrite=True, overrides={"detector_enabled": False})
        rdir = WorkbenchStore(ws).run_dir("synthetic_seeg", "synthetic_seeg", "baseline")
        # Top-level events.csv MUST immediately reflect 0 rows (header only)!
        csv_lines = [ln for ln in (rdir / "events.csv").read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(csv_lines) == 1, csv_lines

        with urllib.request.urlopen(f"{url}/api/bootstrap?case=synthetic_seeg&run_name=baseline&lite=1", timeout=5) as r2:
            b2 = json.loads(r2.read().decode("utf-8"))
        assert b2["active"]["sheet_url"] is None, f"Stale sheet not invalidated: {b2['active']['sheet_url']}"

        # Rebuilding the sheet restores a verified fresh 0-event sheet!
        export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline")
        with urllib.request.urlopen(f"{url}/api/bootstrap?case=synthetic_seeg&run_name=baseline&lite=1", timeout=5) as r3:
            b3 = json.loads(r3.read().decode("utf-8"))
        assert b3["active"]["sheet_url"] is not None
    finally:
        srv.stop()


def test_round5_p1_triad_lowpass_20hz_caps_tfr_and_omits_hfo_peak(tmp_path: Path) -> None:
    ws = tmp_path / "ws_r5_lp20"
    f = _write_sine_fif(tmp_path / "lp20_raw.fif", 20.0, sfreq=512.0, n_samples=2048)
    analyze_case(str(f), workspace_dir=ws, run_name="lp20_run", overrides={"detector_enabled": False, "lowpass": 20.0})
    tr = compute_triad_view("lp20_raw/lp20_raw", workspace_dir=ws, recording="lp20_raw", run_name="lp20_run", channel="LA1", center_seconds=1.5, event_id=None)
    assert tr["bands"]["ripple"]["available"] is False
    assert tr["bands"]["fast_ripple"]["available"] is False
    assert max(tr["tfr"]["frequencies_hz"]) <= 20.05, tr["tfr"]["frequencies_hz"][-3:]
    assert tr["peak_hfo_frequency_hz"] is None
    assert tr["peak_island_db"] is None
    # Rendering a PNG card at lowpass=20 Hz must also succeed cleanly!
    card = render_evidence_card("lp20_raw/lp20_raw", workspace_dir=ws, recording="lp20_raw", run_name="lp20_run", channel="LA1", center_seconds=1.5, event_id=None)
    assert Path(card["image_path"]).is_file() and card["peak_hfo_frequency_hz"] is None


def test_round5_p1_documented_flat_detector_overrides_and_unknown_key_rejection(tmp_path: Path) -> None:
    ws = tmp_path / "ws_r5_ov"
    analyze_case(str(SYNTHETIC_EDF), workspace_dir=ws, run_name="baseline")
    store = WorkbenchStore(ws)
    base_meta = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")["meta"]

    # Flat documented detector parameter `min_gap` MUST be saved into detector.parameters & re-run detection!
    res = analyze_case(
        "synthetic_seeg/synthetic_seeg",
        workspace_dir=ws,
        run_name="gap_5s",
        parent_run="baseline",
        overrides={"min_gap": 5.0, "min_win": 0.008},
    )
    assert "detect" not in (res["runs"][0].get("reused_stages") or {})
    b_gap = store.load_run("synthetic_seeg", "synthetic_seeg", "gap_5s")
    assert b_gap["config"]["analysis_config"]["detector"]["parameters"]["min_gap"] == 5.0
    assert b_gap["config"]["analysis_config"]["detector"]["parameters"]["min_window"] == 0.008
    assert b_gap["meta"]["stage_fingerprints"]["detect"] != base_meta["stage_fingerprints"]["detect"]

    # Unknown top-level override key MUST raise unknown_override_key (never silently ignored!)
    with pytest.raises(ToolkitError) as e_unk:
        analyze_case("synthetic_seeg/synthetic_seeg", workspace_dir=ws, run_name="bad_ov", overrides={"not_a_real_param": 123})
    assert e_unk.value.code == "unknown_override_key"

    # Detector parameter belonging to another detector (sd_threshold is HIL-only, not STE) MUST raise invalid_detector_parameter!
    with pytest.raises(ToolkitError) as e_det:
        analyze_case("synthetic_seeg/synthetic_seeg", workspace_dir=ws, run_name="bad_det_p", overrides={"detector": "STE", "sd_threshold": 5.0})
    assert e_det.value.code == "invalid_detector_parameter"


def test_round5_p2_failed_engine_probe_not_cached_and_researcher_sheet_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brainlens.pyhfo_runner import _probe_worker_imports

    # (A) Failed engine probe MUST NOT be cached: fixing dependency in place immediately returns ok=True!
    fake_src = tmp_path / "fake_pyhfo"
    (fake_src / "src").mkdir(parents=True)
    app_py = fake_src / "src" / "hfo_app.py"
    app_py.write_text("# stub\n", encoding="utf-8")
    flag_file = tmp_path / "dep_installed.flag"
    fake_py = tmp_path / "fake_python.sh"
    fake_py.write_text(
        "#!/bin/sh\n"
        f'if [ -f "{flag_file}" ]; then\n'
        '  echo \'BRAINLENS_PROBE:{"ok": true, "missing": []}\'\n'
        "else\n"
        '  echo \'BRAINLENS_PROBE:{"ok": false, "missing": ["p_tqdm: missing"]}\'\n'
        "fi\n",
        encoding="utf-8",
    )
    fake_py.chmod(0o755)
    ok1, miss1, _ = _probe_worker_imports(fake_src, fake_py, None)
    assert ok1 is False and any("p_tqdm" in m for m in miss1)
    # Fix missing dependency WITHOUT touching fake_py or app_py mtimes!
    flag_file.write_text("installed", encoding="utf-8")
    ok2, miss2, _ = _probe_worker_imports(fake_src, fake_py, None)
    assert ok2 is True and miss2 == []

    # (B) Researcher-only verdict on MDT Case Sheet displays a separate unreviewed clinician + Researcher tally!
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r5_p2"))
    ws = tmp_path / "ws_r5_p2"
    analyze_case(str(SYNTHETIC_EDF), workspace_dir=ws, run_name="baseline")
    store = WorkbenchStore(ws)
    ev0 = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")["events"][0]
    store.record_judgment("synthetic_seeg", {
        "kind": "event_verdict",
        "recording": "synthetic_seeg",
        "run_name": "baseline",
        "channel": ev0["channel_name"],
        "event_id": ev0["id"],
        "verdict": "confirm",
        "author": "Res. Bob",
        "role": "researcher",
        "note": "researcher review",
    })
    refresh_verdict_tallies(store, "synthetic_seeg", "synthetic_seeg", "baseline")
    exp = export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline")
    html_txt = Path(exp["report_path"]).read_text(encoding="utf-8")
    assert 'Clinician: <span class="vtag">unreviewed</span> · Researcher: confirm · Agent: unreviewed' in html_txt
    assert "Researcher: 1 (1 confirmed)" in html_txt


def test_round6_p0_concurrent_stage_cache_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import brainlens.pipeline as pipe_mod

    ws = tmp_path / "ws_r6_stage"
    store = WorkbenchStore(ws)
    inspect_case(str(SYNTHETIC_EDF), workspace_dir=ws, persist=True)

    orig_prepare = pipe_mod.prepare_signals
    barrier = threading.Barrier(2)
    seen_dirs: list[Path] = []
    lock = threading.Lock()
    a_done = threading.Event()

    def _hooked_prepare(recording, cfg, qc, out_dir: Path):
        with lock:
            idx = len(seen_dirs)
            seen_dirs.append(out_dir)
        barrier.wait(timeout=15.0)
        if idx == 0:
            res = orig_prepare(recording, cfg, qc, out_dir)
            a_done.set()
            return res
        # Writer B waits until A has written prepared.json, then writes/truncates in its OWN dir and crashes!
        a_done.wait(timeout=30.0)
        np.lib.format.open_memmap(out_dir / "data.npy", mode="w+", dtype=np.float64, shape=(6, 10))
        np.lib.format.open_memmap(out_dir / "valid.npy", mode="w+", dtype=bool, shape=(6, 10))
        raise RuntimeError("simulated writer B crash")

    monkeypatch.setattr(pipe_mod, "prepare_signals", _hooked_prepare)
    results: dict[str, Any] = {}

    def _run(rname: str) -> None:
        try:
            results[rname] = analyze_case("synthetic_seeg/synthetic_seeg", workspace_dir=ws, run_name=rname)
        except Exception as exc:
            results[rname] = exc

    t1 = threading.Thread(target=_run, args=("run_a",))
    t2 = threading.Thread(target=_run, args=("run_b",))
    t1.start(); t2.start()
    t1.join(timeout=120); t2.join(timeout=120)

    assert len(seen_dirs) == 2 and seen_dirs[0] != seen_dirs[1], seen_dirs
    ok_name = "run_a" if isinstance(results["run_a"], dict) else "run_b"
    err_name = "run_b" if ok_name == "run_a" else "run_a"
    assert isinstance(results[ok_name], dict) and isinstance(results[err_name], RuntimeError)

    b_ok = store.load_run("synthetic_seeg", "synthetic_seeg", ok_name)
    prep_doc = read_json(Path(b_ok["meta"]["prepared_json_path"]))
    valid_arr = np.load(prep_doc["valid_path"], allow_pickle=False)
    assert int(valid_arr.sum()) > 100_000, f"Committed prepare cache corrupted: valid={int(valid_arr.sum())}"
    p2p_before = b_ok["events"][0]["peak_to_peak_uv"]
    assert p2p_before and p2p_before > 10.0
    refresh_verdict_tallies(store, "synthetic_seeg", "synthetic_seeg", ok_name)
    b_after = store.load_run("synthetic_seeg", "synthetic_seeg", ok_name)
    assert abs(b_after["events"][0]["peak_to_peak_uv"] - p2p_before) < 1e-6


def test_round6_p0_toctou_swap_between_verify_and_read_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import brainlens.signal_prep as sp_mod
    import brainlens.triad_views as tv_mod

    ws = tmp_path / "ws_r6_toctou"
    store = WorkbenchStore(ws)
    fif_p = _write_sine_fif(tmp_path / "toctou_raw.fif", 10.0)
    fif_100 = _write_sine_fif(tmp_path / "swap_100_raw.fif", 100.0)
    analyze_case(str(fif_p), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})

    # (A) Reader TOCTOU: swap 10 uV -> 100 uV AFTER pre-read SHA verify and right BEFORE read_scaled_volts!
    orig_tv_read = tv_mod.read_scaled_volts
    swapped = {"done": False}

    def _swap_on_tv_read(*args: Any, **kwargs: Any):
        if not swapped["done"]:
            swapped["done"] = True
            shutil.copyfile(fif_100, tmp_path / "swap_tmp.fif")
            os.replace(tmp_path / "swap_tmp.fif", fif_p)
        return orig_tv_read(*args, **kwargs)

    monkeypatch.setattr(tv_mod, "read_scaled_volts", _swap_on_tv_read)
    with pytest.raises(ToolkitError) as e_r:
        compute_multichannel_batch(store, "toctou_raw", "toctou_raw", run_name="baseline", start_seconds=0.0, duration_seconds=2.0)
    assert e_r.value.code == "source_recording_modified"
    monkeypatch.setattr(tv_mod, "read_scaled_volts", orig_tv_read)

    # Restore original 10 uV file for (B) signal_prep TOCTOU
    fif_p2 = _write_sine_fif(tmp_path / "toctou2_raw.fif", 10.0)
    analyze_case(str(fif_p2), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})

    orig_sp_read = sp_mod.read_scaled_volts
    swapped2 = {"done": False}

    def _swap_on_sp_read(*args: Any, **kwargs: Any):
        if not swapped2["done"]:
            swapped2["done"] = True
            shutil.copyfile(fif_100, tmp_path / "swap_tmp2.fif")
            os.replace(tmp_path / "swap_tmp2.fif", fif_p2)
        return orig_sp_read(*args, **kwargs)

    monkeypatch.setattr(sp_mod, "read_scaled_volts", _swap_on_sp_read)
    with pytest.raises(ToolkitError) as e_sp:
        analyze_case("toctou2_raw/toctou2_raw", workspace_dir=ws, run_name="branch_swap", overrides={"detector_enabled": False, "start_seconds": 0.2})
    assert e_sp.value.code == "source_recording_modified"
    assert not store.has_run("toctou2_raw", "toctou2_raw", "branch_swap")


def test_round6_p1_mvcc_active_writer_and_reader_pin_survive_zero_grace_gc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import brainlens.pipeline as pipe_mod
    import brainlens.storage as st_mod

    monkeypatch.setenv("BRAINLENS_MVCC_GC_GRACE_SECONDS", "0")
    ws = tmp_path / "ws_r6_gc"
    store = WorkbenchStore(ws)
    analyze_case(str(SYNTHETIC_EDF), workspace_dir=ws, run_name="baseline")
    ev0 = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")["events"][0]

    def _do_refreshes(n: int) -> None:
        for i in range(n):
            store.record_judgment("synthetic_seeg", {
                "kind": "event_verdict",
                "recording": "synthetic_seeg",
                "run_name": "baseline",
                "channel": ev0["channel_name"],
                "event_id": ev0["id"],
                "verdict": "confirm" if i % 2 == 0 else "uncertain",
                "author": "Res. GC",
                "role": "researcher",
                "note": f"gc_{i}",
            })
            refresh_verdict_tallies(store, "synthetic_seeg", "synthetic_seeg", "baseline")

    # (A) Active writer paused mid-prepare with grace=0 while 10 refreshes run -> staging dir survives WITH qc_summary.json!
    orig_prepare = pipe_mod.prepare_signals
    hook_fired = {"done": False}

    def _pause_writer_for_gc(*args: Any, **kwargs: Any):
        if not hook_fired["done"]:
            hook_fired["done"] = True
            _do_refreshes(10)
        return orig_prepare(*args, **kwargs)

    monkeypatch.setattr(pipe_mod, "prepare_signals", _pause_writer_for_gc)
    analyze_case("synthetic_seeg/synthetic_seeg", workspace_dir=ws, run_name="baseline", overwrite=True)
    monkeypatch.setattr(pipe_mod, "prepare_signals", orig_prepare)
    b_w = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")
    assert b_w["qc"] and "per_channel" in b_w["qc"], "Active writer lost qc_summary.json during zero-grace GC!"

    # (B) Active reader paused mid-load_run (after reading summary.json) while 12 refreshes run with grace=0 ->
    #     reader MUST get complete matching events (6) and qc (never summary=6 + events=[])!
    orig_read_json = st_mod.read_json
    paused_once = {"done": False}

    def _pause_reader_mid_load(p: Any):
        res = orig_read_json(p)
        if not paused_once["done"] and Path(p).name == "summary.json" and Path(p).parent.parent.name == "generations":
            paused_once["done"] = True
            # Run 12 refreshes in another thread so our shared LOCK_SH pin is genuinely exercised!
            t = threading.Thread(target=_do_refreshes, args=(12,))
            t.start()
            t.join(timeout=30.0)
        return res

    monkeypatch.setattr(st_mod, "read_json", _pause_reader_mid_load)
    b_r = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")
    monkeypatch.setattr(st_mod, "read_json", orig_read_json)
    assert paused_once["done"] is True
    assert b_r["summary"]["counts"]["total_raw_events"] == len(b_r["events"]) == 6
    assert b_r["qc"] and "per_channel" in b_r["qc"]


def test_round6_p1_legacy_flat_run_upgrade_atomic_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import brainlens.pipeline as pipe_mod

    ws = tmp_path / "ws_r6_flat"
    store = WorkbenchStore(ws)
    analyze_case(str(SYNTHETIC_EDF), workspace_dir=ws, run_name="baseline")
    rdir = store.run_dir("synthetic_seeg", "synthetic_seeg", "baseline")
    gdir0 = store.run_generation_dir("synthetic_seeg", "synthetic_seeg", "baseline")

    # Convert to authentic pre-MVCC flat run layout (no `current`, no `CURRENT.json`, regular top-level files)
    flat_tmp = tmp_path / "flat_copy"
    shutil.copytree(gdir0, flat_tmp)
    (flat_tmp / "CURRENT.json").unlink(missing_ok=True)
    shutil.rmtree(rdir)
    shutil.copytree(flat_tmp, rdir)
    assert not (rdir / "current").exists() and not (rdir / "CURRENT.json").exists()
    assert (rdir / "events.csv").is_file() and not (rdir / "events.csv").is_symlink()
    old_rows = len([ln for ln in (rdir / "events.csv").read_text(encoding="utf-8").splitlines() if ln.strip()])
    assert old_rows == 7  # header + 6 events

    # Observe exact state immediately BEFORE the final `current` flip to new_gen:
    # top-level events.csv MUST ALREADY be symlinked to `current/events.csv` (pointing at old 6-event legacy gen)!
    orig_sym = pipe_mod._atomic_symlink
    observed_pre_final: list[tuple[bool, int]] = []

    def _spy_sym(target_rel: str, dest: Path) -> None:
        if dest == (rdir / "current") and (rdir / "events.csv").is_symlink():
            rows_now = len([ln for ln in (rdir / "events.csv").read_text(encoding="utf-8").splitlines() if ln.strip()])
            observed_pre_final.append((True, rows_now))
        orig_sym(target_rel, dest)

    monkeypatch.setattr(pipe_mod, "_atomic_symlink", _spy_sym)
    analyze_case("synthetic_seeg/synthetic_seeg", workspace_dir=ws, run_name="baseline", overwrite=True, overrides={"detector_enabled": False})
    assert observed_pre_final == [(True, 7)], f"Legacy upgrade did not symlink CSVs before atomic flip: {observed_pre_final}"
    new_rows = len([ln for ln in (rdir / "events.csv").read_text(encoding="utf-8").splitlines() if ln.strip()])
    assert new_rows == 1  # header only (0 events)


def test_round6_p1_mdt_sheet_snapshot_signature_and_comparison_branch_invalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import brainlens.case_sheet as cs_mod

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r6_sheet"))
    ws = tmp_path / "ws_r6_sheet"
    analyze_case(str(SYNTHETIC_EDF), workspace_dir=ws, run_name="baseline")
    analyze_case("synthetic_seeg/synthetic_seeg", workspace_dir=ws, run_name="cmp_branch", overrides={"detector_enabled": False})

    srv = WorkbenchHTTPServer(ws, host="127.0.0.1", port=0)
    url = srv.start()
    try:
        # (A) Comparison branch overwrite (0 -> 6 events) MUST immediately invalidate stale MDT sheet in /api/bootstrap!
        export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline", comparison_run="cmp_branch")
        with urllib.request.urlopen(f"{url}/api/bootstrap?case=synthetic_seeg&run_name=baseline&lite=1", timeout=5) as r1:
            assert json.loads(r1.read().decode("utf-8"))["active"]["sheet_url"] is not None

        analyze_case("synthetic_seeg/synthetic_seeg", workspace_dir=ws, run_name="cmp_branch", overwrite=True)
        with urllib.request.urlopen(f"{url}/api/bootstrap?case=synthetic_seeg&run_name=baseline&lite=1", timeout=5) as r2:
            b2 = json.loads(r2.read().decode("utf-8"))
        assert b2["active"]["sheet_url"] is None, f"Stale comparison sheet not invalidated: {b2['active']['sheet_url']}"

        # (B) Concurrent export-vs-overwrite race: overwrite baseline (6 -> 0) AFTER rendering 6-event HTML but
        #     BEFORE record_verified_sheet -> MUST sign with rendered 6-event sig so /api/bootstrap rejects it!
        orig_rec = cs_mod.record_verified_sheet

        def _race_overwrite(*args: Any, **kwargs: Any):
            analyze_case("synthetic_seeg/synthetic_seeg", workspace_dir=ws, run_name="baseline", overwrite=True, overrides={"detector_enabled": False})
            return orig_rec(*args, **kwargs)

        monkeypatch.setattr(cs_mod, "record_verified_sheet", _race_overwrite)
        export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline", comparison_run="cmp_branch")
        monkeypatch.setattr(cs_mod, "record_verified_sheet", orig_rec)

        with urllib.request.urlopen(f"{url}/api/bootstrap?case=synthetic_seeg&run_name=baseline&lite=1", timeout=5) as r3:
            b3 = json.loads(r3.read().decode("utf-8"))
        assert b3["active"]["sheet_url"] is None, f"Raced stale 6-event sheet accepted after 6->0 overwrite: {b3['active']['sheet_url']}"
    finally:
        srv.stop()


def test_round6_p1_channel_subset_branch_mdt_sheet_no_500(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r6_subset"))
    ws = tmp_path / "ws_r6_subset"
    analyze_case(str(SYNTHETIC_EDF), workspace_dir=ws, run_name="baseline")
    analyze_case("synthetic_seeg/synthetic_seeg", workspace_dir=ws, run_name="only_a2", overrides={"channels": ["SYN_A2"]})

    # Default export auto-selects `only_a2` as comparison -> MUST succeed without formatting NoneType error!
    out = export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline")
    html_txt = Path(out["report_path"]).read_text(encoding="utf-8")
    assert "only_a2" in html_txt and "unassessed" in html_txt

    srv = WorkbenchHTTPServer(ws, host="127.0.0.1", port=0)
    url = srv.start()
    try:
        req = urllib.request.Request(
            f"{url}/api/export-sheet",
            data=json.dumps({"case": "synthetic_seeg", "primary_run": "baseline"}).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-BrainLens-Token": srv.auth_token},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            assert r.status == 200
            body = json.loads(r.read().decode("utf-8"))
            assert body["sheet_url"].endswith("synthetic_seeg_baseline_case_sheet.html")
    finally:
        srv.stop()


def test_round6_p1_published_bipolar_pairs_mutation_rejected(tmp_path: Path) -> None:
    ws = tmp_path / "ws_r6_bp"
    rec_dir = tmp_path / "bp_case"
    rec_dir.mkdir()
    sfreq = 512.0
    t = np.arange(2048, dtype=np.float64) / sfreq
    data = np.vstack([
        (20e-6) * np.sin(2 * np.pi * 10 * t),   # LA1
        (10e-6) * np.sin(2 * np.pi * 10 * t),   # LA2 -> LA1-LA2 = 10 uV
        (-80e-6) * np.sin(2 * np.pi * 10 * t),  # LA3 -> LA1-LA3 = 100 uV
    ])
    info = mne.create_info(["LA1", "LA2", "LA3"], sfreq=sfreq, ch_types="seeg")
    fif = rec_dir / "bp_raw.fif"
    mne.io.RawArray(data, info, verbose="ERROR").save(fif, overwrite=True, verbose="ERROR")
    (rec_dir / "bp_raw.json").write_text(json.dumps({"reference": "common", "line_freq": 60.0}), encoding="utf-8")
    bp_json = rec_dir / "published_bipolar_pairs.json"
    bp_json.write_text(json.dumps({"pairs": [{"anode": "LA1", "cathode": "LA2", "name": "LA1-LA2"}]}), encoding="utf-8")

    analyze_case(str(fif), workspace_dir=ws, run_name="bp_base", overrides={"reference_mode": "bipolar", "detector_enabled": False})

    # Mutate sidecar cathode LA2 -> LA3 while preserving exact byte length & mtime!
    orig_st = bp_json.stat()
    new_txt = json.dumps({"pairs": [{"anode": "LA1", "cathode": "LA3", "name": "LA1-LA2"}]})
    assert len(new_txt) == orig_st.st_size
    bp_json.write_text(new_txt, encoding="utf-8")
    os.utime(bp_json, ns=(orig_st.st_atime_ns, orig_st.st_mtime_ns))

    with pytest.raises(ToolkitError) as exc:
        analyze_case("bp_raw/bp_raw", workspace_dir=ws, run_name="bp_mut", overrides={"reference_mode": "bipolar", "detector_enabled": False, "start_seconds": 0.2})
    assert exc.value.code == "source_recording_modified"


def test_round6_p1_fiff_header_split_bids_and_independent_dash1_and_p2s(tmp_path: Path) -> None:
    ws = tmp_path / "ws_r6_fiff"

    # (A) MNE BIDS split naming -> MUST register as EXACTLY ONE full-duration recording (no duplicate split-02)!
    bids_dir = tmp_path / "bids_split_case"
    bids_dir.mkdir()
    sfreq = 512.0
    n_samps = int(sfreq * 80)
    data = np.zeros((2, n_samps), dtype=np.float64)
    info = mne.create_info(["LA1", "LA2"], sfreq=sfreq, ch_types="seeg")
    mne.io.RawArray(data, info, verbose="ERROR").save(
        bids_dir / "sub-01_ieeg.fif", split_size="1.2MB", split_naming="bids", overwrite=True, verbose="ERROR"
    )
    fifs_written = sorted(p.name for p in bids_dir.glob("*.fif"))
    assert len(fifs_written) >= 2 and any("split-02" in n for n in fifs_written), fifs_written
    insp_bids = inspect_case(str(bids_dir), workspace_dir=ws, persist=True)
    assert insp_bids["recording_count"] == 1, insp_bids["recordings"]
    assert abs(insp_bids["recordings"][0]["duration_seconds"] - 80.0) < 0.2

    # (B) Two independent valid FIFs named study_raw.fif and study_raw-1.fif -> BOTH MUST be registered!
    ind_dir = tmp_path / "indep_fifs"
    ind_dir.mkdir()
    _write_sine_fif(ind_dir / "study_raw.fif", 10.0)
    _write_sine_fif(ind_dir / "study_raw-1.fif", 20.0)
    insp_ind = inspect_case(str(ind_dir), workspace_dir=ws, persist=True)
    assert insp_ind["recording_count"] == 2, [r["recording"] for r in insp_ind["recordings"]]

    # (C) P2 #1: detector_params passed as list MUST raise invalid_detector_parameter (never silently ignored!)
    with pytest.raises(ToolkitError) as e_dp:
        analyze_case("indep_fifs/study_raw", workspace_dir=ws, run_name="bad_dp", overrides={"detector_params": ["min_gap", 5]})
    assert e_dp.value.code == "invalid_detector_parameter"

    # (D) P2 #2: lowpass=2.0 Hz caps Triad TFR strictly at <= 2.05 Hz with peak_hfo_frequency_hz=None!
    analyze_case("indep_fifs/study_raw", workspace_dir=ws, run_name="lp2_run", overrides={"detector_enabled": False, "lowpass": 2.0})
    tr2 = compute_triad_view("indep_fifs/study_raw", workspace_dir=ws, recording="study_raw", run_name="lp2_run", channel="LA1", center_seconds=1.5, event_id=None)
    assert max(tr2["tfr"]["frequencies_hz"]) <= 2.05, tr2["tfr"]["frequencies_hz"][-3:]
    assert tr2["peak_hfo_frequency_hz"] is None and tr2["peak_island_db"] is None


def test_round7_p0_initial_inspect_swap_and_parent_dir_swap_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import brainlens.io_readers as io_mod
    import brainlens.signal_prep as sp_mod
    import brainlens.triad_views as tv_mod

    ws = tmp_path / "ws_r7_p0"
    store = WorkbenchStore(ws)

    # (A) Initial registration swap: replace exam_raw.fif with reversed-channel file mid-inspect ->
    #     MUST reject (never save old channel table paired with new file's SHA-256!).
    sfreq = 512.0
    t = np.arange(2048, dtype=np.float64) / sfreq
    s10 = (10e-6) * np.sin(2.0 * np.pi * 10.0 * t)
    s100 = (100e-6) * np.sin(2.0 * np.pi * 10.0 * t)

    exam_p = tmp_path / "exam_raw.fif"
    alt_p = tmp_path / "exam_swapped_raw.fif"
    mne.io.RawArray(np.vstack([s10, s100]), mne.create_info(["LA1", "LB1"], sfreq, "seeg"), verbose="ERROR").save(
        exam_p, overwrite=True, verbose="ERROR"
    )
    mne.io.RawArray(np.vstack([s100, s10]), mne.create_info(["LB1", "LA1"], sfreq, "seeg"), verbose="ERROR").save(
        alt_p, overwrite=True, verbose="ERROR"
    )

    orig_open = io_mod.open_mne_raw
    swapped_insp = {"done": False}

    def _swap_during_inspect(path: Path, preload: bool = False):
        raw = orig_open(path, preload=preload)
        if not swapped_insp["done"] and Path(path).name == "exam_raw.fif":
            swapped_insp["done"] = True
            tmp_rep = tmp_path / "exam_rep.fif"
            shutil.copyfile(alt_p, tmp_rep)
            os.replace(tmp_rep, exam_p)
        return raw

    monkeypatch.setattr(io_mod, "open_mne_raw", _swap_during_inspect)
    with pytest.raises(ToolkitError) as e_insp:
        inspect_case(str(exam_p), workspace_dir=ws, persist=True)
    assert e_insp.value.code == "source_recording_modified"
    monkeypatch.setattr(io_mod, "open_mne_raw", orig_open)

    # Now inspect the stable swapped file on disk: LB1 MUST be ~100 uV and LA1 ~10 uV (NEVER inverted!).
    analyze_case(str(exam_p), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    mc_ex = compute_multichannel_batch(store, "exam_raw", "exam_raw", run_name="baseline", start_seconds=0.0, duration_seconds=2.0)
    by_name = {tr["channel_name"]: max(abs(v) for v in tr["values_uv"] if v is not None) for tr in mc_ex["traces"]}
    assert 70.0 <= by_name["LB1"] <= 130.0 and 6.0 <= by_name["LA1"] <= 15.0, by_name

    # (B) Parent-directory swap-and-restore during read_scaled_volts in BOTH Reader and prepare_signals:
    #     Because reads come strictly from SHA-256-verified private 0700 snapshots, BOTH Reader and prepare_signals
    #     MUST return/save ONLY the verified 10 uV waveform (NEVER 100 uV under 10 uV fingerprint!).
    acq_dir = tmp_path / "acq_dir"
    alt_dir = tmp_path / "alt_dir"
    bak_dir = tmp_path / "bak_dir"
    acq_dir.mkdir()
    alt_dir.mkdir()
    _write_sine_fif(acq_dir / "swapdir_raw.fif", 10.0)
    _write_sine_fif(alt_dir / "swapdir_raw.fif", 100.0)
    analyze_case(str(acq_dir / "swapdir_raw.fif"), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})

    orig_tv_read = tv_mod.read_scaled_volts
    tv_hook_ran = {"done": False}

    def _dir_swap_restore_tv(*args: Any, **kwargs: Any):
        if not tv_hook_ran["done"]:
            tv_hook_ran["done"] = True
            os.rename(acq_dir, bak_dir)
            os.rename(alt_dir, acq_dir)
            try:
                return orig_tv_read(*args, **kwargs)
            finally:
                os.rename(acq_dir, alt_dir)
                os.rename(bak_dir, acq_dir)
        return orig_tv_read(*args, **kwargs)

    monkeypatch.setattr(tv_mod, "read_scaled_volts", _dir_swap_restore_tv)
    mc_sr = compute_multichannel_batch(store, "swapdir_raw", "swapdir_raw", run_name="baseline", start_seconds=0.0, duration_seconds=2.0)
    monkeypatch.setattr(tv_mod, "read_scaled_volts", orig_tv_read)
    assert tv_hook_ran["done"] is True
    peak_tv = max(abs(v) for v in mc_sr["traces"][0]["values_uv"] if v is not None)
    assert 6.0 <= peak_tv <= 15.0, f"Reader leaked swapped dir waveform: {peak_tv} uV"

    orig_sp_read = sp_mod.read_scaled_volts
    sp_hook_ran = {"done": False}

    def _dir_swap_restore_sp(*args: Any, **kwargs: Any):
        if not sp_hook_ran["done"]:
            sp_hook_ran["done"] = True
            os.rename(acq_dir, bak_dir)
            os.rename(alt_dir, acq_dir)
            try:
                return orig_sp_read(*args, **kwargs)
            finally:
                os.rename(acq_dir, alt_dir)
                os.rename(bak_dir, acq_dir)
        return orig_sp_read(*args, **kwargs)

    monkeypatch.setattr(sp_mod, "read_scaled_volts", _dir_swap_restore_sp)
    analyze_case(
        "swapdir_raw/swapdir_raw",
        workspace_dir=ws,
        run_name="dirswap_branch",
        overrides={"detector_enabled": False, "start_seconds": 0.2},
    )
    monkeypatch.setattr(sp_mod, "read_scaled_volts", orig_sp_read)
    assert sp_hook_ran["done"] is True
    b_br = store.load_run("swapdir_raw", "swapdir_raw", "dirswap_branch")
    prep_doc = read_json(Path(b_br["meta"]["prepared_json_path"]))
    prep_arr = np.load(prep_doc["data_path"], allow_pickle=False)
    peak_prep_uv = float(np.max(np.abs(prep_arr[0]))) * 1e6
    assert 6.0 <= peak_prep_uv <= 15.0, f"prepare_signals stored swapped dir waveform: {peak_prep_uv} uV"


def test_round7_p1_mvcc_pre_pin_gc_eio_fail_and_legacy_verdict_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno
    import brainlens.pipeline as pipe_mod
    import brainlens.storage as st_mod

    monkeypatch.setenv("BRAINLENS_MVCC_GC_GRACE_SECONDS", "0")
    ws = tmp_path / "ws_r7_mvcc"
    store = WorkbenchStore(ws)
    analyze_case(str(SYNTHETIC_EDF), workspace_dir=ws, run_name="baseline")
    analyze_case(
        "synthetic_seeg/synthetic_seeg",
        workspace_dir=ws,
        run_name="lp100",
        overrides={"detector_enabled": False, "filters": {"lowpass": 100.0}},
    )
    ev0 = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")["events"][0]

    def _churn_run(rname: str, n: int = 12) -> None:
        for i in range(n):
            store.record_judgment("synthetic_seeg", {
                "kind": "event_verdict",
                "recording": "synthetic_seeg",
                "run_name": rname,
                "channel": ev0["channel_name"],
                "event_id": ev0["id"],
                "verdict": "confirm" if i % 2 == 0 else "uncertain",
                "author": "Res. PrePin",
                "role": "researcher",
                "note": f"prepin_{rname}_{i}",
            })
            refresh_verdict_tallies(store, "synthetic_seeg", "synthetic_seeg", rname)

    # (A) Pre-pin GC race: churn 12 refreshes BEFORE pin_generation_read acquires lock ->
    #     load_run, Reader (keeping 100 Hz lowpass FR disabled!), and branch donor reuse MUST ALL SUCCEED!
    orig_pin = st_mod.pin_generation_read
    fired = {"count": 0}

    from contextlib import contextmanager

    @contextmanager
    def _pre_pin_churn(gdir: Path):
        if fired["count"] == 0 and gdir.parent.name == "generations" and gdir.parent.parent.name == "lp100":
            fired["count"] += 1
            t = threading.Thread(target=_churn_run, args=("lp100", 12))
            t.start()
            t.join(timeout=30.0)
        with orig_pin(gdir):
            yield

    monkeypatch.setattr(st_mod, "pin_generation_read", _pre_pin_churn)
    mc_lp = compute_multichannel_batch(
        store,
        "synthetic_seeg",
        "synthetic_seeg",
        run_name="lp100",
        start_seconds=0.0,
        duration_seconds=2.0,
        display_filter="fast_ripple_250_450",
    )
    assert fired["count"] == 1
    assert mc_lp["display_filter_available"] is False, "Pre-pin GC dropped 100 Hz lowpass FR guard!"

    fired["count"] = 0
    res_child = analyze_case(
        "synthetic_seeg/synthetic_seeg",
        workspace_dir=ws,
        run_name="lp100_child",
        parent_run="lp100",
        overrides={"detector_enabled": False, "start_seconds": 1.0},
    )
    assert res_child["runs"][0]["run_name"] == "lp100_child"
    monkeypatch.setattr(st_mod, "pin_generation_read", orig_pin)

    # (B) POSIX EIO on `current` symlink swap MUST raise OSError (NEVER fall back or overwrite top-level CSV!)
    rdir_base = store.run_dir("synthetic_seeg", "synthetic_seeg", "baseline")
    orig_sym = pipe_mod._atomic_symlink

    def _fail_eio(target_rel: str, dest: Path) -> None:
        if dest == (rdir_base / "current"):
            raise OSError(errno.EIO, "Simulated POSIX I/O error on current commit")
        orig_sym(target_rel, dest)

    monkeypatch.setattr(pipe_mod, "_atomic_symlink", _fail_eio)
    with pytest.raises(OSError) as e_io:
        analyze_case(
            "synthetic_seeg/synthetic_seeg",
            workspace_dir=ws,
            run_name="baseline",
            overwrite=True,
            overrides={"detector_enabled": False},
        )
    assert e_io.value.errno == errno.EIO
    monkeypatch.setattr(pipe_mod, "_atomic_symlink", orig_sym)

    b_after_eio = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")
    assert b_after_eio["summary"]["counts"]["total_raw_events"] == 6
    csv_lines = [ln for ln in (rdir_base / "events.csv").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(csv_lines) == 7, f"Top-level events.csv torn after EIO: {len(csv_lines)} lines"

    # (C) Historical pre-MVCC flat run refresh_verdict_tallies MUST NEVER mutate flat files in place!
    gdir_cur = store.run_generation_dir("synthetic_seeg", "synthetic_seeg", "baseline")
    flat_tmp = tmp_path / "flat_leg"
    shutil.copytree(gdir_cur, flat_tmp)
    (flat_tmp / "CURRENT.json").unlink(missing_ok=True)
    shutil.rmtree(rdir_base)
    shutil.copytree(flat_tmp, rdir_base)
    assert not (rdir_base / "current").exists() and not (rdir_base / "CURRENT.json").exists()

    store.record_judgment("synthetic_seeg", {
        "kind": "event_verdict",
        "recording": "synthetic_seeg",
        "run_name": "baseline",
        "channel": ev0["channel_name"],
        "event_id": ev0["id"],
        "verdict": "confirm",
        "author": "Dr. Flat",
        "role": "clinician",
        "note": "atomic legacy vote",
    })

    # Simulate crash right before atomic MVCC commit -> original flat files MUST stay 100% untouched!
    orig_pub = pipe_mod._publish_mvcc_generation

    def _crash_before_pub(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Simulated crash before MVCC publish")

    monkeypatch.setattr(pipe_mod, "_publish_mvcc_generation", _crash_before_pub)
    with pytest.raises(RuntimeError):
        refresh_verdict_tallies(store, "synthetic_seeg", "synthetic_seeg", "baseline")
    monkeypatch.setattr(pipe_mod, "_publish_mvcc_generation", orig_pub)
    assert read_json(rdir_base / "summary.json")["counts"]["clinician_confirmed"] == 0
    assert read_json(rdir_base / "run_meta.json")["counts"]["clinician_confirmed"] == 0

    # Normal refresh atomically upgrades flat run to MVCC & commits clinician_confirmed=1 everywhere!
    refresh_verdict_tallies(store, "synthetic_seeg", "synthetic_seeg", "baseline")
    assert (rdir_base / "current").is_symlink()
    b_leg = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")
    assert b_leg["summary"]["counts"]["clinician_confirmed"] == 1
    assert b_leg["meta"]["counts"]["clinician_confirmed"] == 1
    assert any(e.get("clinician_verdict") == "confirm" for e in b_leg["events"])


def test_round7_p1_mdt_export_post_race_bookmark_stale_third_branch_and_kpi_tally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import brainlens.case_sheet as cs_mod

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r7_mdt"))
    ws = tmp_path / "ws_r7_mdt"
    store = WorkbenchStore(ws)
    analyze_case(str(SYNTHETIC_EDF), workspace_dir=ws, run_name="baseline")
    analyze_case("synthetic_seeg/synthetic_seeg", workspace_dir=ws, run_name="cmp_b", overrides={"detector_enabled": False})
    analyze_case("synthetic_seeg/synthetic_seeg", workspace_dir=ws, run_name="extra")

    srv = WorkbenchHTTPServer(ws, host="127.0.0.1", port=0)
    url = srv.start()
    try:
        # (A) POST /api/export-sheet racing baseline overwrite (6 -> 0) MUST auto-rerender and return fresh 0-event sheet!
        orig_rec = cs_mod.record_verified_sheet
        raced_once = {"done": False}

        def _race_once(*args: Any, **kwargs: Any):
            if not raced_once["done"]:
                raced_once["done"] = True
                analyze_case(
                    "synthetic_seeg/synthetic_seeg",
                    workspace_dir=ws,
                    run_name="baseline",
                    overwrite=True,
                    overrides={"detector_enabled": False},
                )
            return orig_rec(*args, **kwargs)

        monkeypatch.setattr(cs_mod, "record_verified_sheet", _race_once)
        req_exp = urllib.request.Request(
            f"{url}/api/export-sheet",
            data=json.dumps({"case": "synthetic_seeg", "primary_run": "baseline", "comparison_run": "cmp_b"}).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-BrainLens-Token": srv.auth_token},
            method="POST",
        )
        with urllib.request.urlopen(req_exp, timeout=20) as r_exp:
            exp_body = json.loads(r_exp.read().decode("utf-8"))
        monkeypatch.setattr(cs_mod, "record_verified_sheet", orig_rec)
        assert raced_once["done"] is True
        sheet_url = exp_body["sheet_url"]
        with urllib.request.urlopen(f"{url}{sheet_url}", timeout=10) as r_html:
            html_fresh = r_html.read().decode("utf-8")
        assert '<meta name="brainlens-token"' in html_fresh
        assert '<meta name="brainlens-sheet-stale" content="1"' not in html_fresh
        assert "<span class=\"big accent\">0</span>" in html_fresh

        # (B) Direct GET /exports/... bookmark to stale HMAC-valid sheet MUST NOT inject btk-token & MUST mark btk-sheet-stale!
        analyze_case("synthetic_seeg/synthetic_seeg", workspace_dir=ws, run_name="baseline", overwrite=True)
        with urllib.request.urlopen(f"{url}{sheet_url}", timeout=10) as r_stale:
            html_stale = r_stale.read().decode("utf-8")
        assert '<meta name="brainlens-sheet-stale" content="1"' in html_stale
        assert '<meta name="brainlens-token"' not in html_stale

        # (C) 3rd-branch (`extra`) clipped card: overwriting ONLY `extra` (6 -> 0) MUST invalidate sheet on ALL paths!
        ev_extra = store.load_run("synthetic_seeg", "synthetic_seeg", "extra")["events"][0]
        store.record_judgment("synthetic_seeg", {
            "kind": "event_clip",
            "recording": "synthetic_seeg",
            "run_name": "extra",
            "channel": ev_extra["channel_name"],
            "event_id": ev_extra["id"],
            "clipped": True,
            "author": "Dr. Clip",
            "role": "clinician",
            "note": "clip from 3rd branch extra",
        })
        export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline", comparison_run="cmp_b")
        with urllib.request.urlopen(f"{url}/api/bootstrap?case=synthetic_seeg&run_name=baseline&lite=1", timeout=5) as r_b1:
            assert json.loads(r_b1.read().decode("utf-8"))["active"]["sheet_url"] is not None
        with urllib.request.urlopen(f"{url}{sheet_url}", timeout=10) as r_ex1:
            html_ex1 = r_ex1.read().decode("utf-8")
        assert 'data-run-name="extra"' in html_ex1 and '<meta name="brainlens-token"' in html_ex1

        # Overwrite ONLY `extra` to 0 events!
        analyze_case(
            "synthetic_seeg/synthetic_seeg",
            workspace_dir=ws,
            run_name="extra",
            overwrite=True,
            overrides={"detector_enabled": False},
        )
        with urllib.request.urlopen(f"{url}/api/bootstrap?case=synthetic_seeg&run_name=baseline&lite=1", timeout=5) as r_b2:
            assert json.loads(r_b2.read().decode("utf-8"))["active"]["sheet_url"] is None
        with urllib.request.urlopen(f"{url}{sheet_url}", timeout=10) as r_ex2:
            html_ex2 = r_ex2.read().decode("utf-8")
        assert '<meta name="brainlens-sheet-stale" content="1"' in html_ex2 and '<meta name="brainlens-token"' not in html_ex2

        # (D) MDT clinical review tally: POST /api/report-action verdict -> /api/case-verdicts review_tallies == 1
        #     AND /api/bootstrap invalidates pre-verdict sheet HTML so re-nav rebuilds with 1 confirmed!
        out_d = export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline", comparison_run="cmp_b")
        ev_base = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")["events"][0]
        req_v = urllib.request.Request(
            f"{url}/api/report-action",
            data=json.dumps({
                "case": "synthetic_seeg",
                "workspace_id": store.workspace_id,
                "source": "case_sheet",
                "primary_run": "baseline",
                "export_filename": out_d["export_filename"],
                "export_hash": out_d["export_hash"],
                "action": "verdict",
                "recording": "synthetic_seeg",
                "run_name": "baseline",
                "channel": ev_base["channel_name"],
                "event_id": ev_base["id"],
                "verdict": "confirm",
                "author": "Dr. MDT",
                "role": "clinician",
                "comment": "Clinician confirmed",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-BrainLens-Token": srv.auth_token},
            method="POST",
        )
        with urllib.request.urlopen(req_v, timeout=10) as r_v:
            assert r_v.status == 200
        with urllib.request.urlopen(f"{url}/api/case-verdicts?case=synthetic_seeg", timeout=5) as r_proj:
            proj = json.loads(r_proj.read().decode("utf-8"))
        rt = proj["review_tallies"]["synthetic_seeg|baseline"]
        assert rt["clinician_confirmed"] == 1 and rt["reviewed_focal_events"] == 1, rt
        # Pre-verdict sheet HTML is invalidated in bootstrap so re-navigation rebuilds with live 1 confirmed!
        with urllib.request.urlopen(f"{url}/api/bootstrap?case=synthetic_seeg&run_name=baseline&lite=1", timeout=5) as r_b3:
            assert json.loads(r_b3.read().decode("utf-8"))["active"]["sheet_url"] is None
        out_re = export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline", comparison_run="cmp_b")
        re_html = Path(out_re["report_path"]).read_text(encoding="utf-8")
        assert "kpi-review" in re_html and "1 confirmed" in re_html
    finally:
        srv.stop()


def test_round7_p1_r6_bipolar_upgrade_mixed_depth_dir_and_lp02(tmp_path: Path) -> None:
    from brainlens.io_readers import _discover_folder_recordings

    ws = tmp_path / "ws_r7_misc"
    store = WorkbenchStore(ws)

    # (A) Round 6 upgraded bipolar recording (has source_sha256 & published_bipolar_pairs_path, empty companion_files={}):
    #     mutating cathode LA2 -> LA3 preserving exact size & mtime MUST raise source_recording_modified!
    rec_dir = tmp_path / "r6_bp_case"
    rec_dir.mkdir()
    sfreq = 512.0
    t = np.arange(2048, dtype=np.float64) / sfreq
    data = np.vstack([
        (20e-6) * np.sin(2 * np.pi * 10 * t),
        (10e-6) * np.sin(2 * np.pi * 10 * t),
        (-80e-6) * np.sin(2 * np.pi * 10 * t),
    ])
    fif = rec_dir / "r6bp_raw.fif"
    mne.io.RawArray(data, mne.create_info(["LA1", "LA2", "LA3"], sfreq, "seeg"), verbose="ERROR").save(
        fif, overwrite=True, verbose="ERROR"
    )
    (rec_dir / "r6bp_raw.json").write_text(json.dumps({"reference": "common", "line_freq": 60.0}), encoding="utf-8")
    bp_json = rec_dir / "published_bipolar_pairs.json"
    bp_json.write_text(json.dumps({"pairs": [{"anode": "LA1", "cathode": "LA2", "name": "LA1-LA2"}]}), encoding="utf-8")

    inspect_case(str(fif), workspace_dir=ws, persist=True)
    # Strip companion_files to reproduce exact Round 6 recording.json schema!
    rpath = store.recording_path("r6bp_raw", "r6bp_raw")
    rdoc = read_json(rpath)
    rdoc["spec"]["companion_files"] = {}
    atomic_write_json(rpath, rdoc)

    # Unchanged file still passes cleanly!
    analyze_case("r6bp_raw/r6bp_raw", workspace_dir=ws, run_name="bp_ok", overrides={"reference_mode": "bipolar", "detector_enabled": False})

    # Mutate LA2 -> LA3 preserving byte size & mtime!
    orig_st = bp_json.stat()
    mut_txt = json.dumps({"pairs": [{"anode": "LA1", "cathode": "LA3", "name": "LA1-LA2"}]})
    assert len(mut_txt) == orig_st.st_size
    bp_json.write_text(mut_txt, encoding="utf-8")
    os.utime(bp_json, ns=(orig_st.st_atime_ns, orig_st.st_mtime_ns))

    with pytest.raises(ToolkitError) as e_r6bp:
        analyze_case("r6bp_raw/r6bp_raw", workspace_dir=ws, run_name="bp_bad", overrides={"reference_mode": "bipolar", "detector_enabled": False, "start_seconds": 0.2})
    assert e_r6bp.value.code == "source_recording_modified"

    # (B) Mixed-depth directory (root baseline + subdir follow-up): discovery, inspect, AND analyze ALL find 2 recordings!
    mix_dir = tmp_path / "mixed_depth_case"
    (mix_dir / "sub").mkdir(parents=True)
    _write_sine_fif(mix_dir / "baseline_raw.fif", 10.0)
    _write_sine_fif(mix_dir / "sub" / "followup_raw.fif", 20.0)
    assert len(_discover_folder_recordings(mix_dir)) == 2
    insp_mix = inspect_case(str(mix_dir), workspace_dir=ws, persist=True)
    assert insp_mix["recording_count"] == 2, [r["recording"] for r in insp_mix["recordings"]]
    ana_mix = analyze_case(str(mix_dir), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    assert ana_mix["recordings_analyzed"] == 2

    # (C) P2: lowpass=0.2 Hz caps Triad TFR strictly <= 0.21 Hz with peak_hfo_frequency_hz=None!
    analyze_case("mixed_depth_case/baseline_raw", workspace_dir=ws, run_name="lp02", overrides={"detector_enabled": False, "lowpass": 0.2})
    tr02 = compute_triad_view("mixed_depth_case/baseline_raw", workspace_dir=ws, recording="baseline_raw", run_name="lp02", channel="LA1", center_seconds=1.5, event_id=None)
    assert max(tr02["tfr"]["frequencies_hz"]) <= 0.21, tr02["tfr"]["frequencies_hz"][-3:]
    assert tr02["peak_hfo_frequency_hz"] is None and tr02["peak_island_db"] is None


def test_round8_p0_recursive_subject_coords_and_sidecar_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import brainlens.io_readers as io_mod

    ws = tmp_path / "ws_r8_p0"
    store = WorkbenchStore(ws)

    # (A1) Recursive directory with conflicting subject PROVENANCE.json & distinct coordinates ->
    #      MUST resolve followup accurately alone AND refuse to merge distinct subjects from parent!
    ds = tmp_path / "multi_subj_ds"
    sub = ds / "followup"
    sub.mkdir(parents=True)
    _write_sine_fif(ds / "base_raw.fif", 10.0)
    _write_sine_fif(sub / "followup_raw.fif", 50.0)
    (ds / "PROVENANCE.json").write_text(
        json.dumps({"same_subject_association": {"name": "Research", "participant_identity": {"participant_id": "subject-A"}}}),
        encoding="utf-8",
    )
    (ds / "coords_a.json").write_text(
        json.dumps({"spatial": {"coordinate_system": "native", "units": "mm", "source": "test_a", "positions": {"LA1": [15.0, 20.0, 45.0]}}}),
        encoding="utf-8",
    )
    (sub / "PROVENANCE.json").write_text(
        json.dumps({"same_subject_association": {"name": "Research", "participant_identity": {"participant_id": "subject-B"}}}),
        encoding="utf-8",
    )
    (sub / "coords_b.json").write_text(
        json.dumps({"spatial": {"coordinate_system": "native", "units": "mm", "source": "test_b", "positions": {"LA1": [-30.0, -55.0, 70.0]}}}),
        encoding="utf-8",
    )

    insp_sub = inspect_case(str(sub), workspace_dir=ws, persist=True)
    rec_sub = store.load_recording(insp_sub["case"], "followup_raw")
    assert rec_sub.provenance.participant_id == "subject-B"
    assert rec_sub.spatial is not None and tuple(rec_sub.spatial.positions["LA1"]) == (-30.0, -55.0, 70.0)

    with pytest.raises(ToolkitError) as e_subj:
        inspect_case(str(ds), workspace_dir=ws, persist=True)
    assert e_subj.value.code == "conflicting_subjects"

    # (A2) Same subject across root + subfolder with distinct per-visit coordinates ->
    #      BOTH recordings preserve their OWN closest spatial coordinates regardless of import root!
    ds_same = tmp_path / "same_subj_ds"
    sub_same = ds_same / "followup"
    sub_same.mkdir(parents=True)
    _write_sine_fif(ds_same / "base_raw.fif", 10.0)
    _write_sine_fif(sub_same / "followup_raw.fif", 50.0)
    (ds_same / "PROVENANCE.json").write_text(
        json.dumps({"same_subject_association": {"name": "Research", "participant_identity": {"participant_id": "subject-A"}}}),
        encoding="utf-8",
    )
    (ds_same / "coords_v1.json").write_text(
        json.dumps({"spatial": {"coordinate_system": "native", "units": "mm", "source": "test_v1", "positions": {"LA1": [15.0, 20.0, 45.0]}}}),
        encoding="utf-8",
    )
    (sub_same / "coords_v2.json").write_text(
        json.dumps({"spatial": {"coordinate_system": "native", "units": "mm", "source": "test_v2", "positions": {"LA1": [-30.0, -55.0, 70.0]}}}),
        encoding="utf-8",
    )
    insp_same = inspect_case(str(ds_same), workspace_dir=ws, persist=True)
    r_base = store.load_recording(insp_same["case"], "base_raw")
    r_foll = store.load_recording(insp_same["case"], "followup_raw")
    assert r_base.spatial is not None and tuple(r_base.spatial.positions["LA1"]) == (15.0, 20.0, 45.0)
    assert r_foll.spatial is not None and tuple(r_foll.spatial.positions["LA1"]) == (-30.0, -55.0, 70.0)

    # (B) Sidecar-to-signal swap during inspect_single_recording (swap dir A -> dir B right after _discover_recording_sidecars):
    #     MUST raise source_recording_modified (NEVER mix A's common/non-synthetic/LA1-BAD sidecars with B's 100 uV signal!).
    live_dir = tmp_path / "meta_live"
    stage_dir = tmp_path / "meta_stage"
    old_dir = tmp_path / "meta_old"
    live_dir.mkdir()
    stage_dir.mkdir()
    _write_sine_fif(live_dir / "rec_raw.fif", 10.0)
    (live_dir / "rec_raw.json").write_text(json.dumps({"reference": "common", "synthetic": False}), encoding="utf-8")
    (live_dir / "rec_raw_channels.tsv").write_text("name\ttype\tstatus\nLA1\tSEEG\tbad\nLA2\tSEEG\tgood\n", encoding="utf-8")
    _write_sine_fif(stage_dir / "rec_raw.fif", 100.0)
    (stage_dir / "rec_raw.json").write_text(json.dumps({"reference": "bipolar", "synthetic": True}), encoding="utf-8")
    (stage_dir / "rec_raw_channels.tsv").write_text("name\ttype\tstatus\nLA1\tSEEG\tgood\nLA2\tSEEG\tbad\n", encoding="utf-8")

    orig_disc = io_mod._discover_recording_sidecars
    swapped_meta = {"done": False}

    def _swap_after_sidecars(p: Path):
        res = orig_disc(p)
        if not swapped_meta["done"] and p.name == "rec_raw.fif":
            swapped_meta["done"] = True
            os.rename(live_dir, old_dir)
            os.rename(stage_dir, live_dir)
        return res

    monkeypatch.setattr(io_mod, "_discover_recording_sidecars", _swap_after_sidecars)
    with pytest.raises(ToolkitError) as e_meta:
        inspect_case(str(live_dir / "rec_raw.fif"), workspace_dir=ws, persist=True)
    assert e_meta.value.code == "source_recording_modified"
    assert swapped_meta["done"] is True


def _analyze_synthetic_run(
    ws: Path,
    run_name: str = "baseline",
    *,
    target: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run real PyHFO on SYNTHETIC_EDF when configured, or seed 6 deterministic focal events when running PyHFO-free."""
    tgt = target or str(SYNTHETIC_EDF)
    if _has_pyhfo_engine():
        return analyze_case(tgt, workspace_dir=ws, run_name=run_name, overwrite=overwrite)
    out = analyze_case(
        tgt,
        workspace_dir=ws,
        run_name=run_name,
        overrides={"detector_enabled": False},
        overwrite=overwrite,
    )
    store = WorkbenchStore(ws)
    rdir = store.run_generation_dir("synthetic_seeg", "synthetic_seeg", run_name)
    meta = read_json(rdir / "run_meta.json")
    summ = read_json(rdir / "summary.json")
    rec = store.load_recording("synthetic_seeg", "synthetic_seeg")
    sfreq = float(rec.sfreq)
    n_samps = int(rec.n_samples)
    ch0 = next(c for c in summ["channels"] if c["channel_name"] == "SYN_A1")
    ch1 = next(c for c in summ["channels"] if c["channel_name"] == "SYN_A2")
    seed_dir = store.run_dir("synthetic_seeg", "synthetic_seeg", run_name) / "_seeded"
    seed_dir.mkdir(parents=True, exist_ok=True)
    det = read_json(Path(meta["detection_json_path"]))
    det["status"] = "complete"
    det["valid_intervals"] = {c["channel_id"]: [[0, n_samps]] for c in summ["channels"]}
    det["events"] = [
        {
            "id": f"ev_syn_{run_name}_{i:03d}",
            "channel_id": (ch0 if i < 4 else ch1)["channel_id"],
            "channel_name": (ch0 if i < 4 else ch1)["channel_name"],
            "shaft": "SYN_A",
            "start_sample": int((2.0 + i * 3.0) * sfreq),
            "end_sample": int((2.08 + i * 3.0) * sfreq),
            "start_seconds": 2.0 + i * 3.0,
            "end_seconds": 2.08 + i * 3.0,
            "scores": {},
            "labels": {},
        }
        for i in range(6)
    ]
    det_seed_p = seed_dir / "detection.json"
    atomic_write_json(det_seed_p, det)
    clf = read_json(Path(meta["classification_json_path"]))
    clf["events"] = det["events"]
    clf_seed_p = seed_dir / "classification.json"
    atomic_write_json(clf_seed_p, clf)
    meta["detection_json_path"] = str(det_seed_p)
    meta["classification_json_path"] = str(clf_seed_p)
    meta["stage_fingerprints"] = {**(meta.get("stage_fingerprints") or {}), "detect": f"seeded_{run_name}"}
    atomic_write_json(rdir / "run_meta.json", meta)
    refresh_verdict_tallies(store, "synthetic_seeg", "synthetic_seeg", run_name)
    return out


def test_round8_p1_list_runs_gc_crash_recovery_role_clips_overlay_and_bv_ansi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r8_p1"))
    monkeypatch.setenv("BRAINLENS_MVCC_GC_GRACE_SECONDS", "0")
    ws = tmp_path / "ws_r8_p1"
    store = WorkbenchStore(ws)
    _analyze_synthetic_run(ws, run_name="baseline")
    analyze_case("synthetic_seeg/synthetic_seeg", workspace_dir=ws, run_name="quiet", overrides={"detector_enabled": False})
    _analyze_synthetic_run(ws, run_name="third", target="synthetic_seeg/synthetic_seeg")

    srv = WorkbenchHTTPServer(ws, host="127.0.0.1", port=0)
    url = srv.start()
    try:
        # (A) list_runs() GC gap: churn 12 refreshes of `quiet` while bootstrap lists runs -> MUST keep `quiet` (0 events), NEVER fall back to baseline (6)!
        orig_res = srv.store.resolve_run_gen_dir
        paused_once = {"done": False}

        def _hold_in_list(c: str, r: str, rn: str):
            val = orig_res(c, r, rn)
            if rn == "quiet" and not paused_once["done"]:
                # Check caller stack for list_runs -> load_run
                f = sys._getframe(1)
                names = []
                while f is not None and len(names) < 5:
                    names.append(f.f_code.co_name)
                    f = f.f_back
                if "list_runs" in names:
                    paused_once["done"] = True
                    t = threading.Thread(target=lambda: [refresh_verdict_tallies(store, c, r, "quiet") for _ in range(12)])
                    t.start()
                    t.join(timeout=30.0)
            return val

        monkeypatch.setattr(srv.store, "resolve_run_gen_dir", _hold_in_list)
        with urllib.request.urlopen(f"{url}/api/bootstrap?case=synthetic_seeg&recording=synthetic_seeg&run_name=quiet&lite=1", timeout=15) as r_boot:
            act_q = json.loads(r_boot.read().decode("utf-8"))["active"]
        monkeypatch.setattr(srv.store, "resolve_run_gen_dir", orig_res)
        assert paused_once["done"] is True
        assert act_q["active_run"] == "quiet" and act_q["run_summary"]["counts"]["total_raw_events"] == 0, act_q["active_run"]

        # (B) Crash between judgments.jsonl append and refresh_verdict_tallies -> load_run, projection, & bootstrap MUST all agree (researcher_confirmed == 1)!
        ev0 = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")["events"][0]
        store.record_judgment("synthetic_seeg", {
            "kind": "event_verdict",
            "recording": "synthetic_seeg",
            "run_name": "baseline",
            "channel": ev0["channel_name"],
            "event_id": ev0["id"],
            "verdict": "confirm",
            "author": "Res. Crash",
            "role": "researcher",
            "note": "journaled before crash",
        })
        # Notice we intentionally did NOT call refresh_verdict_tallies!
        proj = store.case_verdicts_projection("synthetic_seeg")
        assert proj["verdicts"][f"synthetic_seeg|baseline|{ev0['id']}"]["researcher_verdict"] == "confirm"
        assert proj["review_tallies"]["synthetic_seeg|baseline"]["researcher_confirmed"] == 1
        with urllib.request.urlopen(f"{url}/api/bootstrap?case=synthetic_seeg&recording=synthetic_seeg&run_name=baseline&lite=1", timeout=10) as r_cr:
            act_cr = json.loads(r_cr.read().decode("utf-8"))["active"]
        assert act_cr["run_summary"]["counts"]["researcher_confirmed"] == 1

        # (C) Per-role clip independence: clinician + researcher both clip 3rd-branch `third` event;
        #     researcher unclips -> clinician's clipped card MUST remain on exported MDT sheet!
        ev3 = store.load_run("synthetic_seeg", "synthetic_seeg", "third")["events"][0]
        for role_v, flag_v in [("clinician", True), ("researcher", True), ("researcher", False)]:
            srv._save_judgment({
                "case": "synthetic_seeg",
                "recording": "synthetic_seeg",
                "run_name": "third",
                "event_id": ev3["id"],
                "channel": ev3["channel_name"],
                "role": role_v,
                "author": role_v,
                "clipped": flag_v,
            })
        out_clip = export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline", comparison_run="quiet")
        html_clip = Path(out_clip["report_path"]).read_text(encoding="utf-8")
        assert 'data-run-name="third"' in html_clip and f'data-event-id="{ev3["id"]}"' in html_clip

        # Now clinician ALSO unclips -> card removed AND old bookmark MUST become stale (no btk-token)!
        sheet_url = f"/exports/synthetic_seeg/{Path(out_clip['report_path']).name}"
        srv._save_judgment({
            "case": "synthetic_seeg",
            "recording": "synthetic_seeg",
            "run_name": "third",
            "event_id": ev3["id"],
            "channel": ev3["channel_name"],
            "role": "clinician",
            "author": "clinician",
            "clipped": False,
        })
        with urllib.request.urlopen(f"{url}{sheet_url}", timeout=10) as r_unclip_bm:
            bm_txt = r_unclip_bm.read().decode("utf-8")
        assert '<meta name="brainlens-sheet-stale" content="1"' in bm_txt and '<meta name="brainlens-token"' not in bm_txt

        # (D) Clinical overlay update via POST /api/overlay MUST invalidate MDT sheet on BOTH /api/bootstrap and /exports/!
        def _post_overlay(soz_list: list[str], src: str) -> None:
            req = urllib.request.Request(
                f"{url}/api/overlay",
                data=json.dumps({"case": "synthetic_seeg", "source": src, "soz_channels": soz_list}).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-BrainLens-Token": srv.auth_token},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                assert resp.status == 200

        _post_overlay(["SYN_A1"], "overlay rev 1")
        export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline", comparison_run="quiet")
        with urllib.request.urlopen(f"{url}/api/bootstrap?case=synthetic_seeg&run_name=baseline&lite=1", timeout=5) as r_ov1:
            assert json.loads(r_ov1.read().decode("utf-8"))["active"]["sheet_url"] is not None

        _post_overlay(["SYN_A2"], "overlay rev 2")
        with urllib.request.urlopen(f"{url}/api/bootstrap?case=synthetic_seeg&run_name=baseline&lite=1", timeout=5) as r_ov2:
            assert json.loads(r_ov2.read().decode("utf-8"))["active"]["sheet_url"] is None
        with urllib.request.urlopen(f"{url}{sheet_url}", timeout=10) as r_ov_bm:
            ov_bm_txt = r_ov_bm.read().decode("utf-8")
        assert '<meta name="brainlens-sheet-stale" content="1"' in ov_bm_txt and '<meta name="brainlens-token"' not in ov_bm_txt
    finally:
        srv.stop()

    # (E) BrainVision Codepage=ANSI (cp1252) with raw 0xB5 ('µV') & 0xE9 ('paté.eeg') MUST decode cleanly & measure channel!
    bv_dir = tmp_path / "bv_ansi_case"
    bv_dir.mkdir()
    eeg_p = bv_dir / "paté.eeg"
    vmrk_p = bv_dir / "pat.vmrk"
    vhdr_p = bv_dir / "paté.vhdr"
    t = np.arange(2048, dtype=np.float64) / 512.0
    sig = np.round(25.0 * np.sin(2.0 * np.pi * 10.0 * t)).astype("<i2")
    eeg_p.write_bytes(sig.tobytes())
    vmrk_p.write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\r\n"
        "[Common Infos]\r\n"
        "Codepage=UTF-8\r\n"
        "DataFile=paté.eeg\r\n"
        "[Marker Infos]\r\n",
        encoding="utf-8",
    )
    vhdr_p.write_bytes(
        (
            "Brain Vision Data Exchange Header File Version 1.0\r\n"
            "[Common Infos]\r\n"
            "Codepage=ANSI\r\n"
            "DataFile=paté.eeg\r\n"
            "MarkerFile=pat.vmrk\r\n"
            "DataFormat=BINARY\r\n"
            "DataOrientation=MULTIPLEXED\r\n"
            "NumberOfChannels=1\r\n"
            "SamplingInterval=1953.125\r\n"
            "[Binary Infos]\r\n"
            "BinaryFormat=INT_16\r\n"
            "[Channel Infos]\r\n"
            "Ch1=LA1,,1,µV\r\n"
        ).encode("cp1252")
    )
    insp_ansi = inspect_case(str(vhdr_p), workspace_dir=ws, persist=True)
    rec_ansi = store.load_recording(insp_ansi["case"], insp_ansi["recordings"][0]["recording"])
    assert rec_ansi.channels[0].unit == "V" and rec_ansi.channels[0].source_unit == "µV", rec_ansi.channels[0]
    res_ansi = analyze_case(str(vhdr_p), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    b_ansi = store.load_run(insp_ansi["case"], insp_ansi["recordings"][0]["recording"], "baseline")
    assert b_ansi["qc"]["invalid_intervals"]["ch0000"] == []
    assert b_ansi["summary"]["channels"][0]["unit"] == "V"
    mc_ansi = compute_multichannel_batch(
        store, insp_ansi["case"], insp_ansi["recordings"][0]["recording"], run_name="baseline", start_seconds=0.0, duration_seconds=2.0
    )
    peak_ansi = max(abs(v) for v in mc_ansi["traces"][0]["values_uv"] if v is not None)
    assert 20.0 <= peak_ansi <= 30.0, peak_ansi


# ---------------------------------------------------------------------------
# Round 9 Adversarial Regression Gates (2 P0 + 5 P1)
# ---------------------------------------------------------------------------


def _write_fif_with_subject(
    path: Path,
    ch_names: list[str],
    sfreq: float,
    dur_s: float,
    his_id: str | None = None,
    seed: int = 11,
) -> None:
    rng = np.random.default_rng(seed)
    n = int(round(sfreq * dur_s))
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="seeg")
    if his_id is not None:
        info["subject_info"] = {"his_id": his_id}
    raw = mne.io.RawArray(rng.normal(0.0, 20e-6, size=(len(ch_names), n)), info, verbose="ERROR")
    raw.save(path, overwrite=True, verbose="ERROR")


def test_round9_p0_bids_fiff_header_malformed_boundary_incremental_and_sidecar_swap_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import brainlens.io_readers as io_mod

    ws = tmp_path / "ws_r9_p0"
    store = WorkbenchStore(ws)

    # (A) BIDS sub-A + sub-B under a parent PROVENANCE.json declaring sub-A + parent coords:
    #     1) Importing the shared parent dir MUST reject with conflicting_subjects!
    #     2) Importing sub-B alone MUST NOT inherit sub-A's provenance or sub-A's coordinates!
    bids_root = tmp_path / "bids_shared_parent"
    sub_a_dir = bids_root / "sub-A" / "ses-01" / "ieeg"
    sub_b_dir = bids_root / "sub-B" / "ses-01" / "ieeg"
    sub_a_dir.mkdir(parents=True)
    sub_b_dir.mkdir(parents=True)
    (bids_root / "PROVENANCE.json").write_text(
        json.dumps({"participant_id": "sub-A", "dataset_name": "SharedBIDS", "origin_kind": "real_public_dataset"}),
        encoding="utf-8",
    )
    (bids_root / "coords.json").write_text(
        json.dumps({"spatial": {"coordinate_system": "native", "units": "mm", "source": "sub-A-parent", "positions": {"LA1": [10.0, 20.0, 30.0]}}}),
        encoding="utf-8",
    )
    _write_fif_with_subject(sub_a_dir / "sub-A_ieeg_raw.fif", ["LA1", "LA2"], 512.0, 4.0, his_id="sub-A", seed=1)
    _write_fif_with_subject(sub_b_dir / "sub-B_ieeg_raw.fif", ["LA1", "LA2"], 512.0, 4.0, his_id="sub-B", seed=2)

    with pytest.raises(ToolkitError) as exc_bids:
        inspect_case(str(bids_root), workspace_dir=ws, persist=True)
    assert exc_bids.value.code == "conflicting_subjects", exc_bids.value

    insp_b = inspect_case(str(sub_b_dir), workspace_dir=ws, persist=True)
    rec_b = store.load_recording(insp_b["case"], insp_b["recordings"][0]["recording"])
    assert rec_b.provenance.participant_id == "sub-B", rec_b.provenance
    assert rec_b.spatial is None, f"sub-B leaked parent sub-A coordinates: {rec_b.spatial}"

    # (B) Two FIF files with distinct native subject_info.his_id in the SAME folder MUST reject!
    fif_mix = tmp_path / "fif_native_mix"
    fif_mix.mkdir()
    _write_fif_with_subject(fif_mix / "visit1_raw.fif", ["LA1", "LA2"], 512.0, 4.0, his_id="patient-A", seed=3)
    _write_fif_with_subject(fif_mix / "visit2_raw.fif", ["LA1", "LA2"], 512.0, 4.0, his_id="patient-B", seed=4)
    with pytest.raises(ToolkitError) as exc_fif:
        inspect_case(str(fif_mix), workspace_dir=ws, persist=True)
    assert exc_fif.value.code == "conflicting_subjects", exc_fif.value

    # (C) Malformed child PROVENANCE.json MUST fail closed (invalid_provenance) and NEVER inherit parent coords!
    mal_root = tmp_path / "malformed_boundary"
    mal_child = mal_root / "child_visit"
    mal_child.mkdir(parents=True)
    (mal_root / "PROVENANCE.json").write_text(
        json.dumps({"participant_id": "patient-A", "dataset_name": "ParentCase", "origin_kind": "real_public_dataset"}),
        encoding="utf-8",
    )
    (mal_root / "electrodes.json").write_text(
        json.dumps({"spatial": {"coordinate_system": "native", "units": "mm", "source": "parent-A", "positions": {"LA1": [10.0, 20.0, 40.0]}}}),
        encoding="utf-8",
    )
    (mal_child / "PROVENANCE.json").write_text("{broken json", encoding="utf-8")
    _write_fif_with_subject(mal_child / "visit_raw.fif", ["LA1", "LA2"], 512.0, 4.0, his_id="patient-B", seed=5)
    with pytest.raises(ToolkitError) as exc_mal:
        inspect_case(str(mal_child), workspace_dir=ws, persist=True)
    assert exc_mal.value.code == "invalid_provenance", exc_mal.value

    # (D) Incremental re-import into the same case folder with a different patient MUST reject & preserve patient-A!
    inc_dir = tmp_path / "clinic_inbox"
    inc_dir.mkdir()
    _write_fif_with_subject(inc_dir / "visit1_raw.fif", ["LA1", "LA2"], 512.0, 4.0, his_id="patient-A", seed=6)
    insp_inc1 = inspect_case(str(inc_dir), workspace_dir=ws, persist=True)
    assert store.list_recordings(insp_inc1["case"]) == ["visit1_raw"]
    (inc_dir / "visit1_raw.fif").unlink()
    _write_fif_with_subject(inc_dir / "visit2_raw.fif", ["LA1", "LA2"], 512.0, 4.0, his_id="patient-B", seed=7)
    with pytest.raises(ToolkitError) as exc_inc:
        inspect_case(str(inc_dir), workspace_dir=ws, persist=True)
    assert exc_inc.value.code == "conflicting_subjects", exc_inc.value
    assert store.list_recordings(insp_inc1["case"]) == ["visit1_raw"]

    # (E) Sidecar swap-and-restore (A -> B -> A) around sidecar parsing MUST NOT mix B's metadata with A's signal!
    sc_dir = tmp_path / "sidecar_swap_case"
    sc_dir.mkdir()
    sc_fif = sc_dir / "rec_raw.fif"
    sc_json = sc_dir / "rec_raw.json"
    _write_fif_with_subject(sc_fif, ["LA1", "LB1"], 512.0, 4.0, his_id="patient-SC", seed=8)
    doc_a = {
        "reference": "common",
        "synthetic": False,
        "line_freq": 60.0,
        "markers": [{"start": 0.0, "end": 4.0, "label": "BAD_channel", "source": "channel_status", "channels": ["LA1"]}],
    }
    doc_b = {
        "reference": "bipolar",
        "synthetic": True,
        "line_freq": 50.0,
        "markers": [{"start": 0.0, "end": 4.0, "label": "BAD_channel", "source": "channel_status", "channels": ["LB1"]}],
    }
    sc_json.write_text(json.dumps(doc_a), encoding="utf-8")

    # Prepare an alternate Version-B directory for true whole-directory swap-and-restore (leaves A's file inodes/mtimes/ctimes intact!)
    sc_dir_b = tmp_path / "sidecar_swap_case_b"
    sc_dir_b.mkdir()
    shutil.copy2(sc_fif, sc_dir_b / "rec_raw.fif")
    (sc_dir_b / "rec_raw.json").write_text(json.dumps(doc_b), encoding="utf-8")
    sc_dir_bak = tmp_path / "sidecar_swap_case_bak"

    orig_discover = io_mod._discover_recording_sidecars
    swapped = {"mode": "overwrite", "count": 0}

    def _swap_and_restore_on_discover(rec_path: Path):
        # At the moment _discover_recording_sidecars runs, mutate the ORIGINAL live sidecar A -> B -> A!
        # Mode 1 ("overwrite"): mtime/ctime changes -> final stability check rejects with source_recording_modified!
        # Mode 2 ("dir_swap_restore"): whole-directory A -> B -> A swap restores exact file inode/mtime/ctime ->
        #   because inspect_single_recording parses strictly from snap_dir, B MUST NEVER leak into A!
        swapped["count"] += 1
        if swapped["mode"] == "overwrite":
            sc_json.write_text(json.dumps(doc_b), encoding="utf-8")
            try:
                return orig_discover(rec_path)
            finally:
                sc_json.write_text(json.dumps(doc_a), encoding="utf-8")
        elif swapped["mode"] == "dir_swap_restore":
            os.rename(sc_dir, sc_dir_bak)
            os.rename(sc_dir_b, sc_dir)
            try:
                return orig_discover(rec_path)
            finally:
                os.rename(sc_dir, sc_dir_b)
                os.rename(sc_dir_bak, sc_dir)
        return orig_discover(rec_path)

    monkeypatch.setattr(io_mod, "_discover_recording_sidecars", _swap_and_restore_on_discover)
    # E1: overwrite A -> B -> A (changes mtime/ctime) -> MUST reject as source_recording_modified!
    with pytest.raises(ToolkitError) as exc_sc1:
        inspect_case(str(sc_fif), workspace_dir=ws, persist=True)
    assert exc_sc1.value.code == "source_recording_modified", exc_sc1.value
    # E2: whole-directory swap-and-restore A -> B -> A -> MUST persist pure A (NEVER mix B's bipolar/synthetic/LB1)!
    swapped["mode"] = "dir_swap_restore"
    insp_sc = inspect_case(str(sc_fif), workspace_dir=ws, persist=True)
    rec_sc = store.load_recording(insp_sc["case"], insp_sc["recordings"][0]["recording"])
    assert swapped["count"] >= 2
    assert rec_sc.reference == "common", rec_sc.reference
    assert rec_sc.provenance.is_synthetic is False, rec_sc.provenance
    bad_chs = {c for m in rec_sc.markers if "BAD" in m.label.upper() for c in m.channels}
    assert bad_chs == {"LA1"}, f"Sidecar swap leaked B's BAD channel into A: {bad_chs}"


def test_round9_p1_aba_overlay_stale_action_409_interval_clips_questions_and_crash_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import brainlens.case_sheet as cs_mod

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r9_p1"))
    ws = tmp_path / "ws_r9_p1"
    store = WorkbenchStore(ws)
    _analyze_synthetic_run(ws, run_name="baseline")
    analyze_case("synthetic_seeg", workspace_dir=ws, run_name="alternate", overrides={"detector_enabled": False})

    # (A) ABA transient overlay (None -> TEMP -> None) mid-export MUST NOT leak into concordance text OR spatial figure!
    # Baseline reference spatial image with NO overlay
    clean_sheet = export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline")
    clean_html = Path(clean_sheet["report_path"]).read_text(encoding="utf-8")
    assert "External Clinical Annotation Reference Overlay" not in clean_html

    orig_conc = cs_mod.compute_overlay_concordance
    orig_sp = cs_mod.render_spatial_figure
    sp_has_overlay: list[bool] = []

    def _conc_with_aba(*args, **kwargs):
        store.save_clinical_overlay("synthetic_seeg", {"source": "transient_ABA", "soz_channels": ["SYN_A1", "SYN_A2"]})
        try:
            return orig_conc(*args, **kwargs)
        finally:
            store.delete_clinical_overlay("synthetic_seeg")

    def _sp_with_aba(*args, **kwargs):
        store.save_clinical_overlay("synthetic_seeg", {"source": "transient_ABA", "soz_channels": ["SYN_A1", "SYN_A2"]})
        try:
            res = orig_sp(*args, **kwargs)
            sp_has_overlay.append(bool(res.get("has_external_overlay")))
            return res
        finally:
            store.delete_clinical_overlay("synthetic_seeg")

    monkeypatch.setattr(cs_mod, "compute_overlay_concordance", _conc_with_aba)
    monkeypatch.setattr(cs_mod, "render_spatial_figure", _sp_with_aba)
    aba_sheet = export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline")
    aba_html = Path(aba_sheet["report_path"]).read_text(encoding="utf-8")
    assert "External Clinical Annotation Reference Overlay" not in aba_html
    assert "transient_ABA" not in aba_html
    assert sp_has_overlay == [False], f"Transient ABA overlay leaked into spatial figure: {sp_has_overlay}"
    monkeypatch.undo()
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r9_p1"))

    # (B) Stale sheet POST /api/report-action verdict MUST return HTTP 409 stale_sheet and write ZERO verdicts!
    srv = WorkbenchHTTPServer(workspace_dir=ws, host="127.0.0.1", port=0)
    url = srv.start()
    try:
        store.save_clinical_overlay("synthetic_seeg", {"source": "SOZ_v1", "soz_channels": ["SYN_A1"]})
        sh_v1 = export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline")
        ev0 = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")["events"][0]
        # Mutate SOZ v1 -> v2 (invalidating sh_v1)
        req_ov = urllib.request.Request(
            f"{url}/api/overlay",
            data=json.dumps({"case": "synthetic_seeg", "source": "SOZ_v2", "soz_channels": ["SYN_A2"]}).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-BrainLens-Token": srv.auth_token},
            method="POST",
        )
        with urllib.request.urlopen(req_ov, timeout=5) as r_ov:
            assert r_ov.status == 200

        req_stale = urllib.request.Request(
            f"{url}/api/report-action",
            data=json.dumps({
                "case": "synthetic_seeg",
                "workspace_id": store.workspace_id,
                "source": "case_sheet",
                "primary_run": "baseline",
                "export_filename": sh_v1["export_filename"],
                "export_hash": sh_v1["export_hash"],
                "action": "verdict",
                "recording": "synthetic_seeg",
                "run_name": "baseline",
                "channel": ev0["channel_name"],
                "event_id": ev0["id"],
                "verdict": "confirm",
                "author": "Dr. Late",
                "role": "clinician",
                "comment": "Delayed click on stale sheet",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-BrainLens-Token": srv.auth_token},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_409:
            urllib.request.urlopen(req_stale, timeout=5)
        assert exc_409.value.code == 409
        err_body = json.loads(exc_409.value.read().decode("utf-8"))
        assert err_body["error"]["code"] == "stale_sheet", err_body
        assert store.load_judgments("synthetic_seeg") == []

        # (C) Overlapping interval clips [10, 14] and [11, 13] MUST stay 2 distinct cards with exact boundaries!
        for op_i, iv in enumerate(([10.0, 14.0], [11.0, 13.0])):
            req_iv = urllib.request.Request(
                f"{url}/api/judgment",
                data=json.dumps({
                    "case": "synthetic_seeg",
                    "operation_id": f"iv_clip_{op_i}",
                    "recording": "synthetic_seeg",
                    "run_name": "baseline",
                    "channel": "SYN_A1",
                    "center_seconds": 12.0,
                    "interval_seconds": iv,
                    "clipped": True,
                    "role": "clinician",
                    "author": "Dr. Clip",
                    "note": f"Interval {iv}",
                }).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-BrainLens-Token": srv.auth_token},
                method="POST",
            )
            with urllib.request.urlopen(req_iv, timeout=5) as r_iv:
                assert r_iv.status == 200
        clips = store.active_clipped_targets("synthetic_seeg")
        assert len(clips) == 2, clips
        assert [c["interval_seconds"] for c in clips] == [[10.0, 14.0], [11.0, 13.0]], clips
        sh_iv = export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline", max_cards=6)
        iv_html = Path(sh_iv["report_path"]).read_text(encoding="utf-8")
        assert 'data-start-seconds="10.00000" data-end-seconds="14.00000"' in iv_html
        assert 'data-start-seconds="11.00000" data-end-seconds="13.00000"' in iv_html
        assert "[10.00–14.00s]" in iv_html and "[11.00–13.00s]" in iv_html

        # (D) Concurrent clinical questions MUST NOT overwrite each other, MUST record in WAL, AND invalidate baseline sheet!
        sh_pre_q = export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline")
        barrier = threading.Barrier(2)
        q_codes: list[int] = []

        def _post_q(q_txt: str, op: str) -> None:
            barrier.wait()
            rq = urllib.request.Request(
                f"{url}/api/judgment",
                data=json.dumps({
                    "case": "synthetic_seeg",
                    "operation_id": op,
                    "recording": "synthetic_seeg",
                    "run_name": "alternate",
                    "clinical_question": q_txt,
                    "role": "clinician",
                    "author": "Dr. Q",
                }).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-BrainLens-Token": srv.auth_token},
                method="POST",
            )
            with urllib.request.urlopen(rq, timeout=5) as resp_q:
                q_codes.append(resp_q.status)

        t1 = threading.Thread(target=_post_q, args=("Check SYN_A1 bipolar attenuation?", "q_op_1"))
        t2 = threading.Thread(target=_post_q, args=("Compare sleep state A vs B?", "q_op_2"))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        assert q_codes == [200, 200]
        saved_qs = store.load_case("synthetic_seeg").get("open_questions", [])
        assert "Check SYN_A1 bipolar attenuation?" in saved_qs and "Compare sleep state A vs B?" in saved_qs, saved_qs
        wal_qs = {j.get("clinical_question") for j in store.load_judgments("synthetic_seeg") if j.get("clinical_question")}
        assert wal_qs >= {"Check SYN_A1 bipolar attenuation?", "Compare sleep state A vs B?"}, wal_qs
        # Baseline sheet exported BEFORE alternate-branch clinical question MUST now be stale!
        with urllib.request.urlopen(
            f"{url}/api/sheet-freshness?case=synthetic_seeg&file={sh_pre_q['export_filename']}&export_hash={sh_pre_q['export_hash']}",
            timeout=5,
        ) as r_fr:
            assert json.loads(r_fr.read().decode("utf-8"))["fresh"] is False
        with urllib.request.urlopen(f"{url}/api/bootstrap?case=synthetic_seeg&run_name=baseline&lite=1", timeout=5) as r_bs:
            assert json.loads(r_bs.read().decode("utf-8"))["active"]["sheet_url"] is None
    finally:
        srv.stop()

    # (E) Crash recovery after judgments.jsonl WAL append MUST heal leading_focal_channels, leading_raw_channels,
    #     events.csv, AND channel_rates.csv (both on query_evidence AND immediately on server restart)!
    b_pre = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")
    ev_crash1 = b_pre["events"][0]
    store.record_judgment(
        "synthetic_seeg",
        {
            "kind": "event_verdict",
            "operation_id": "crash_op_1",
            "recording": "synthetic_seeg",
            "run_name": "baseline",
            "channel": ev_crash1["channel_name"],
            "event_id": ev_crash1["id"],
            "verdict": "confirm",
            "author": "Dr. Crash",
            "role": "clinician",
            "note": "Committed right before crash",
        },
    )
    # Intentionally DO NOT call refresh_verdict_tallies! Verify query_evidence heals everything!
    q_res = query_case("synthetic_seeg/synthetic_seeg", mode="recording_summary", workspace_dir=ws, run_a="baseline")
    assert q_res["counts"]["clinician_confirmed"] == 1
    lf_hit = next(r for r in q_res["leading_focal_channels"] if r["channel_id"] == ev_crash1["channel_id"])
    lr_hit = next(r for r in q_res["leading_raw_channels"] if r["channel_id"] == ev_crash1["channel_id"])
    assert lf_hit["clinician_confirmed"] == 1, lf_hit
    assert lr_hit["clinician_confirmed"] == 1, lr_hit
    rdir = store.run_dir("synthetic_seeg", "synthetic_seeg", "baseline")
    with (rdir / "events.csv").open("r", encoding="utf-8") as fh:
        ev_rows = {row["id"]: row for row in csv.DictReader(fh)}
    assert ev_rows[ev_crash1["id"]]["clinician_verdict"] == "confirm", ev_rows[ev_crash1["id"]]
    with (rdir / "channel_rates.csv").open("r", encoding="utf-8") as fh:
        ch_rows = {row["channel_id"]: row for row in csv.DictReader(fh)}
    assert int(ch_rows[ev_crash1["channel_id"]]["clinician_confirmed"]) == 1, ch_rows[ev_crash1["channel_id"]]

    # Now simulate a SECOND crash and verify WorkbenchHTTPServer.start() heals CSVs BEFORE any query!
    ev_crash2 = b_pre["events"][1]
    store.record_judgment(
        "synthetic_seeg",
        {
            "kind": "event_verdict",
            "operation_id": "crash_op_2",
            "recording": "synthetic_seeg",
            "run_name": "baseline",
            "channel": ev_crash2["channel_name"],
            "event_id": ev_crash2["id"],
            "verdict": "artifact",
            "author": "Dr. Crash2",
            "role": "clinician",
            "note": "Crash before restart",
        },
    )
    srv2 = WorkbenchHTTPServer(workspace_dir=ws, host="127.0.0.1", port=0)
    srv2.start()
    try:
        with (rdir / "events.csv").open("r", encoding="utf-8") as fh:
            ev_rows2 = {row["id"]: row for row in csv.DictReader(fh)}
        assert ev_rows2[ev_crash2["id"]]["clinician_verdict"] == "artifact", ev_rows2[ev_crash2["id"]]
        with (rdir / "channel_rates.csv").open("r", encoding="utf-8") as fh:
            ch_rows2 = {row["channel_id"]: row for row in csv.DictReader(fh)}
        assert int(ch_rows2[ev_crash2["channel_id"]]["clinician_artifact"]) == 1, ch_rows2[ev_crash2["channel_id"]]
    finally:
        srv2.stop()


# ---------------------------------------------------------------------------
# Round 10 Adversarial Regression Gates (3 P0 + 5 P1 + 1 P2)
# ---------------------------------------------------------------------------


def test_round10_p0_paired_sidecar_coord_bleed_top_level_prov_arrival_and_symlink_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import brainlens.io_readers as io_mod

    ws = tmp_path / "ws_r10_p0"
    store = WorkbenchStore(ws)

    # (A) Paired sidecar coordinate bleed: sub-A_ieeg.json (x=+10) MUST NEVER override sub-B's own sidecar (x=-10)!
    shared = tmp_path / "shared_two_subjects"
    shared.mkdir()
    _write_fif_with_subject(shared / "sub-A_ieeg_raw.fif", ["LA1", "LA2"], 512.0, 4.0, his_id="sub-A", seed=101)
    (shared / "sub-A_ieeg.json").write_text(
        json.dumps({
            "reference": "common",
            "spatial": {"coordinate_system": "native", "units": "mm", "source": "sub-A", "positions": {"LA1": [10.0, 0.0, 0.0]}},
        }),
        encoding="utf-8",
    )
    _write_fif_with_subject(shared / "sub-B_ieeg_raw.fif", ["LA1", "LA2"], 512.0, 4.0, his_id="sub-B", seed=102)
    (shared / "sub-B_ieeg_raw.json").write_text(
        json.dumps({
            "reference": "common",
            "spatial": {"coordinate_system": "native", "units": "mm", "source": "sub-B", "positions": {"LA1": [-10.0, 0.0, 0.0]}},
        }),
        encoding="utf-8",
    )
    insp_b = inspect_case(str(shared / "sub-B_ieeg_raw.fif"), workspace_dir=ws, persist=True)
    rec_b = store.load_recording(insp_b["case"], insp_b["recordings"][0]["recording"])
    assert rec_b.provenance.participant_id == "sub-B", rec_b.provenance
    assert rec_b.spatial is not None and rec_b.spatial.source == "sub-B", rec_b.spatial
    assert tuple(rec_b.spatial.positions["LA1"]) == (-10.0, 0.0, 0.0), rec_b.spatial

    # (B) Top-level participant_id in valid child PROVENANCE.json MUST persist & MUST reject multi-subject recursive import!
    prov_root = tmp_path / "top_level_prov_mix"
    pa_dir = prov_root / "visit_a"
    pb_dir = prov_root / "visit_b"
    pa_dir.mkdir(parents=True)
    pb_dir.mkdir(parents=True)
    (pa_dir / "PROVENANCE.json").write_text(json.dumps({"participant_id": "P-A"}), encoding="utf-8")
    (pb_dir / "PROVENANCE.json").write_text(json.dumps({"participant_id": "P-B"}), encoding="utf-8")
    _write_fif_with_subject(pa_dir / "rec1_raw.fif", ["LA1", "LA2"], 512.0, 4.0, his_id=None, seed=103)
    _write_fif_with_subject(pb_dir / "rec2_raw.fif", ["LA1", "LA2"], 512.0, 4.0, his_id=None, seed=104)
    with pytest.raises(ToolkitError) as exc_top_prov:
        inspect_case(str(prov_root), workspace_dir=ws, persist=True)
    assert exc_top_prov.value.code == "conflicting_subjects", exc_top_prov.value
    insp_pa = inspect_case(str(pa_dir), workspace_dir=ws, persist=True)
    rec_pa = store.load_recording(insp_pa["case"], insp_pa["recordings"][0]["recording"])
    assert rec_pa.provenance.participant_id == "P-A", rec_pa.provenance

    # (C) Late-arriving child PROVENANCE.json + child coords arriving mid-registration MUST reject (never persist stale parent A)!
    arr_root = tmp_path / "arrival_race"
    arr_child = arr_root / "child_b"
    arr_child.mkdir(parents=True)
    (arr_root / "PROVENANCE.json").write_text(json.dumps({"participant_id": "P-A"}), encoding="utf-8")
    (arr_root / "parent_coords.json").write_text(
        json.dumps({"spatial": {"coordinate_system": "native", "units": "mm", "source": "P-A", "positions": {"LA1": [10.0, 0.0, 0.0]}}}),
        encoding="utf-8",
    )
    arr_fif = arr_child / "rec_raw.fif"
    _write_fif_with_subject(arr_fif, ["LA1", "LA2"], 512.0, 4.0, his_id=None, seed=105)

    orig_copy = io_mod._copy_and_hash_file
    injected = {"done": False}

    def _arrive_child_metadata(src: Path, dest: Path):
        if not injected["done"] and src.name == "rec_raw.fif":
            injected["done"] = True
            (arr_child / "PROVENANCE.json").write_text(json.dumps({"participant_id": "P-B"}), encoding="utf-8")
            (arr_child / "child_coords.json").write_text(
                json.dumps({"spatial": {"coordinate_system": "native", "units": "mm", "source": "P-B", "positions": {"LA1": [-10.0, 0.0, 0.0]}}}),
                encoding="utf-8",
            )
        return orig_copy(src, dest)

    monkeypatch.setattr(io_mod, "_copy_and_hash_file", _arrive_child_metadata)
    with pytest.raises(ToolkitError) as exc_arr:
        inspect_case(str(arr_root), workspace_dir=ws, persist=True)
    assert exc_arr.value.code == "source_recording_modified", exc_arr.value
    monkeypatch.undo()
    # Immediately retrying on the now-stable directory MUST persist P-B and x=-10!
    insp_arr = inspect_case(str(arr_root), workspace_dir=ws, persist=True)
    rec_arr = store.load_recording(insp_arr["case"], insp_arr["recordings"][0]["recording"])
    assert rec_arr.provenance.participant_id == "P-B", rec_arr.provenance
    assert rec_arr.spatial is not None and tuple(rec_arr.spatial.positions["LA1"]) == (-10.0, 0.0, 0.0), rec_arr.spatial

    # (D) Symlinked BTK JSON + BIDS channels.tsv pointing into metadata/ MUST preserve reference & BAD channel exclusion!
    sym_dir = tmp_path / "symlink_sidecar_case"
    meta_dir = sym_dir / "metadata"
    meta_dir.mkdir(parents=True)
    sym_fif = sym_dir / "sub-S1_task-rest_ieeg_raw.fif"
    _write_fif_with_subject(sym_fif, ["LA1", "LA2"], 512.0, 8.0, his_id="sub-S1", seed=106)
    real_json = meta_dir / "btk_meta_v2.json"
    real_json.write_text(json.dumps({"reference": "bipolar", "line_freq": 60.0}), encoding="utf-8")
    real_tsv = meta_dir / "chan_status_v2.tsv"
    real_tsv.write_text("name\ttype\tstatus\nLA1\tSEEG\tbad\nLA2\tSEEG\tgood\n", encoding="utf-8")
    os.symlink("metadata/btk_meta_v2.json", sym_dir / "sub-S1_task-rest_ieeg_raw.json")
    os.symlink("metadata/chan_status_v2.tsv", sym_dir / "sub-S1_task-rest_channels.tsv")

    analyze_case(str(sym_fif), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    sym_case = "sub-S1_task-rest_ieeg_raw"
    rec_sym = store.load_recording(sym_case, sym_case)
    assert rec_sym.reference == "bipolar", rec_sym.reference
    b_sym = store.load_run(sym_case, sym_case, "baseline")
    assert b_sym["qc"]["invalid_intervals"]["ch0000"] == [[0, 4096]], b_sym["qc"]["invalid_intervals"]
    assert b_sym["qc"]["invalid_intervals"]["ch0001"] == [], b_sym["qc"]["invalid_intervals"]


def test_round10_p1_control_full_window_note_amend_409_spatial_refresh_relocated_wal_and_short_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r10_p1"))
    ws = tmp_path / "ws_r10_p1"
    store = WorkbenchStore(ws)
    _analyze_synthetic_run(ws, run_name="baseline")

    # (A) Quiet Control MUST validate the FULL plotted window (not only central 4s)!
    #     Seed/verify events on SYN_A1 around ~8s and ~18s (or find two events within 12s) and request a 12s control centered between them!
    b_base = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")
    syn_a1_evs = sorted([e for e in b_base["events"] if e["channel_name"] == "SYN_A1"], key=lambda e: e["start_seconds"])
    assert len(syn_a1_evs) >= 2, syn_a1_evs
    mid_between = 0.5 * (syn_a1_evs[0]["end_seconds"] + syn_a1_evs[-1]["start_seconds"])
    wide_span = max(6.0, (syn_a1_evs[-1]["end_seconds"] - syn_a1_evs[0]["start_seconds"]) + 1.0)
    with pytest.raises(ToolkitError) as exc_ctrl:
        compute_triad_view(
            target="synthetic_seeg/synthetic_seeg",
            workspace_dir=ws,
            recording="synthetic_seeg",
            run_name="baseline",
            channel="SYN_A1",
            event_id="control",
            center_seconds=mid_between,
            window_seconds=wide_span,
        )
    assert exc_ctrl.value.code == "control_window_not_clean", exc_ctrl.value

    # (B) Clinical note retraction invalidates MDT sheet bookmark & rejects delayed Confirm with HTTP 409!
    srv = WorkbenchHTTPServer(workspace_dir=ws, host="127.0.0.1", port=0)
    url = srv.start()
    try:
        ev0 = b_base["events"][0]
        srv._save_judgment({
            "case": "synthetic_seeg",
            "operation_id": "r10_note_init",
            "recording": "synthetic_seeg",
            "run_name": "baseline",
            "channel": "SYN_A1",
            "note": "Initial right-sided lateralization hypothesis",
            "role": "clinician",
            "author": "Dr. A",
        })
        sh1 = export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline")
        with urllib.request.urlopen(f"{url}/exports/synthetic_seeg/{sh1['export_filename']}", timeout=5) as r_ok:
            ok_html = r_ok.read().decode("utf-8")
        assert '<meta name="brainlens-token"' in ok_html and '<meta name="brainlens-sheet-stale"' not in ok_html

        # Retract clinical note -> bookmark MUST immediately go stale AND delayed Confirm MUST return 409!
        srv._save_judgment({
            "case": "synthetic_seeg",
            "operation_id": "r10_note_withdraw",
            "recording": "synthetic_seeg",
            "run_name": "baseline",
            "channel": "SYN_A1",
            "note": "WITHDRAW earlier lateralization — bilateral onset possible",
            "role": "clinician",
            "author": "Dr. A",
        })
        with urllib.request.urlopen(f"{url}/exports/synthetic_seeg/{sh1['export_filename']}", timeout=5) as r_st:
            st_html = r_st.read().decode("utf-8")
        assert '<meta name="brainlens-sheet-stale" content="1"' in st_html and '<meta name="brainlens-token"' not in st_html

        req_late = urllib.request.Request(
            f"{url}/api/report-action",
            data=json.dumps({
                "case": "synthetic_seeg",
                "workspace_id": store.workspace_id,
                "source": "case_sheet",
                "primary_run": "baseline",
                "export_filename": sh1["export_filename"],
                "export_hash": sh1["export_hash"],
                "action": "verdict",
                "recording": "synthetic_seeg",
                "run_name": "baseline",
                "channel": ev0["channel_name"],
                "event_id": ev0["id"],
                "verdict": "confirm",
                "author": "Dr. Late",
                "role": "clinician",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-BrainLens-Token": srv.auth_token},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_409:
            urllib.request.urlopen(req_late, timeout=5)
        assert exc_409.value.code == 409
        assert json.loads(exc_409.value.read().decode("utf-8"))["error"]["code"] == "stale_sheet"

        # (C) Updating electrode coordinates MUST immediately refresh committed run spatial & invalidate old MDT sheet!
        sh2 = export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="baseline")
        new_sp = tmp_path / "corrected_coords.json"
        new_sp.write_text(
            json.dumps({
                "spatial": {
                    "coordinate_system": "synthetic Cartesian",
                    "units": "mm",
                    "source": "post-op CT correction",
                    "template": False,
                    "positions": {
                        "SYN_A1": [45.0, 0.0, 0.0],
                        "SYN_A2": [50.0, 0.0, 0.0],
                        "SYN_A3": [55.0, 0.0, 0.0],
                        "SYN_A4": [60.0, 0.0, 0.0],
                        "SYN_A5": [65.0, 0.0, 0.0],
                        "SYN_A6": [70.0, 0.0, 0.0],
                    },
                }
            }),
            encoding="utf-8",
        )
        inspect_case("synthetic_seeg", workspace_dir=ws, spatial_sidecar=new_sp)
        b_sp = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")
        c_a1 = next(c for c in b_sp["summary"]["spatial"]["contacts"] if c["channel_name"] == "SYN_A1")
        assert c_a1["xyz"][0] == 45.0, c_a1
        with urllib.request.urlopen(
            f"{url}/api/sheet-freshness?case=synthetic_seeg&file={sh2['export_filename']}&export_hash={sh2['export_hash']}",
            timeout=5,
        ) as r_fr:
            assert json.loads(r_fr.read().decode("utf-8"))["fresh"] is False
    finally:
        srv.stop()

    # (D) P2: Short clipped interval [10.00, 10.10] draws distinct highlight markers on the 0.6s context PNG!
    card_plain = render_evidence_card(
        "synthetic_seeg/synthetic_seeg",
        workspace_dir=ws,
        recording="synthetic_seeg",
        run_name="baseline",
        channel="SYN_A1",
        event_id=None,
        center_seconds=10.05,
        window_seconds=0.6,
        return_base64=True,
    )
    card_hl = render_evidence_card(
        "synthetic_seeg/synthetic_seeg",
        workspace_dir=ws,
        recording="synthetic_seeg",
        run_name="baseline",
        channel="SYN_A1",
        event_id=None,
        center_seconds=10.05,
        window_seconds=0.6,
        return_base64=True,
        highlight_interval=[10.0, 10.1],
    )
    assert card_hl["png_base64"] != card_plain["png_base64"]

    # (E) Relocated workspace crash recovery: append WAL verdict -> move entire ws to a NEW path & delete old ws ->
    #     reconcile_pending_judgments() MUST return 1 (and 0 on repeat), healing events.csv & channel_rates.csv!
    ev_reloc = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")["events"][0]
    store.record_judgment(
        "synthetic_seeg",
        {
            "kind": "event_verdict",
            "operation_id": "reloc_crash_op",
            "recording": "synthetic_seeg",
            "run_name": "baseline",
            "channel": ev_reloc["channel_name"],
            "event_id": ev_reloc["id"],
            "verdict": "confirm",
            "author": "Dr. Reloc",
            "role": "clinician",
            "note": "Committed right before moving workspace",
        },
    )
    ws_moved = tmp_path / "ws_r10_relocated_new_dir"
    shutil.copytree(ws, ws_moved, symlinks=True)
    shutil.rmtree(ws)

    moved_store = WorkbenchStore(ws_moved)
    assert moved_store.reconcile_pending_judgments() == 1
    assert moved_store.reconcile_pending_judgments() == 0
    rdir_m = moved_store.run_dir("synthetic_seeg", "synthetic_seeg", "baseline")
    with (rdir_m / "events.csv").open("r", encoding="utf-8") as fh:
        ev_rows_m = {row["id"]: row for row in csv.DictReader(fh)}
    assert ev_rows_m[ev_reloc["id"]]["clinician_verdict"] == "confirm", ev_rows_m[ev_reloc["id"]]
    with (rdir_m / "channel_rates.csv").open("r", encoding="utf-8") as fh:
        ch_rows_m = {row["channel_id"]: row for row in csv.DictReader(fh)}
    assert int(ch_rows_m[ev_reloc["channel_id"]]["clinician_confirmed"]) == 1, ch_rows_m[ev_reloc["channel_id"]]


def test_round11_p0_btk_source_file_overlay_prov_mismatch_main_symlink_fif_and_bv_companion(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "ws_r11_p0"
    store = WorkbenchStore(ws)

    # (A) Patient A's *_braintoolkit.json (with provenance.source_file=patientA_raw.fif) MUST NOT
    #     leak LA1.x=+45 onto Patient B's native-ID recording in the same folder!
    shared_dir = tmp_path / "shared_clinic"
    shared_dir.mkdir()
    (shared_dir / "patientA_braintoolkit.json").write_text(
        json.dumps({
            "provenance": {"source_file": "patientA_raw.fif", "participant_id": "P-A"},
            "spatial": {
                "coordinate_system": "MNI",
                "units": "mm",
                "source": "Patient A CT",
                "positions": {"LA1": [45.0, 0.0, 0.0], "LA2": [50.0, 0.0, 0.0]},
            },
        }),
        encoding="utf-8",
    )
    info_b = mne.create_info(["LA1", "LA2"], sfreq=1024.0, ch_types="seeg")
    info_b["subject_info"] = {"his_id": "P-B"}
    rng = np.random.default_rng(1101)
    raw_b = mne.io.RawArray(rng.normal(0, 20e-6, size=(2, 4096)), info_b, verbose="ERROR")
    f_b = shared_dir / "patientB_raw.fif"
    raw_b.save(f_b, overwrite=True, verbose="ERROR")

    res_b = inspect_case(str(f_b), workspace_dir=ws, persist=True)
    rec_b = store.load_recording(res_b["case"], res_b["recordings"][0]["recording"])
    assert rec_b.spatial is None, f"Patient A _braintoolkit.json leaked onto Patient B: {rec_b.spatial}"

    # (B) Static folder with PROVENANCE.json (P-B) + spatial overlay tagged P-A MUST drop P-A overlay!
    static_dir = tmp_path / "static_mismatch"
    static_dir.mkdir()
    (static_dir / "PROVENANCE.json").write_text(
        json.dumps({"participant_id": "P-B", "dataset": "local_study"}),
        encoding="utf-8",
    )
    (static_dir / "electrodes_map.json").write_text(
        json.dumps({
            "participant_id": "P-A",
            "spatial": {
                "coordinate_system": "MNI",
                "units": "mm",
                "source": "Patient A overlay",
                "positions": {"LA1": [45.0, 0.0, 0.0], "LA2": [50.0, 0.0, 0.0]},
            },
        }),
        encoding="utf-8",
    )
    f_stat = _write_fif(static_dir / "scan_raw.fif", ["LA1", "LA2"], 1024.0, 4096, seed=1102)
    res_stat = inspect_case(str(f_stat), workspace_dir=ws, persist=True)
    rec_stat = store.load_recording(res_stat["case"], res_stat["recordings"][0]["recording"])
    assert rec_stat.spatial is None, f"P-A overlay leaked onto PROVENANCE P-B recording: {rec_stat.spatial}"

    # (C) Main recording is a symlink sub-S1_ieeg_raw.fif -> objects/datahash_raw.fif with regular
    #     paired JSON (bipolar) + BIDS channels.tsv (LA1 bad) -> MUST preserve bipolar & exclude ALL 4,096 bad samples!
    sym_dir = tmp_path / "main_symlink_case"
    obj_dir = sym_dir / "objects"
    obj_dir.mkdir(parents=True)
    target_fif = _write_fif(obj_dir / "datahash_raw.fif", ["LA1", "LA2"], 1024.0, 4096, seed=1103)
    main_link = sym_dir / "sub-S1_ieeg_raw.fif"
    os.symlink(target_fif, main_link)
    (sym_dir / "sub-S1_ieeg_raw.json").write_text(
        json.dumps({"reference": "bipolar", "line_freq": 60.0}),
        encoding="utf-8",
    )
    (sym_dir / "sub-S1_channels.tsv").write_text(
        "name\ttype\tstatus\nLA1\tSEEG\tbad\nLA2\tSEEG\tgood\n",
        encoding="utf-8",
    )
    res_sym = analyze_case(str(main_link), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    c_sym = res_sym["case"]
    r_sym = res_sym["runs"][0]["recording"]
    rec_sym = store.load_recording(c_sym, r_sym)
    assert rec_sym.reference == "bipolar", f"Main symlink lost paired JSON reference: {rec_sym.reference}"
    b_sym = store.load_run(c_sym, r_sym, "baseline")
    assert b_sym["qc"]["invalid_intervals"]["ch0000"] == [[0, 4096]], b_sym["qc"]["invalid_intervals"]
    assert b_sym["qc"]["invalid_intervals"]["ch0001"] == [], b_sym["qc"]["invalid_intervals"]

    # (D) BrainVision .vhdr whose companion .vmrk & .eeg are symlinks to objects/* ->
    #     MUST both register AND analyze cleanly (no FileNotFoundError) with ~50 uV P2P!
    bv_dir = tmp_path / "bv_symlink_comp"
    bv_obj = bv_dir / "objects"
    bv_obj.mkdir(parents=True)
    sfreq = 1000.0
    n_pts = 2000
    t = np.arange(n_pts, dtype=np.float64) / sfreq
    arr = np.zeros((n_pts, 2), dtype="<f4")
    arr[:, 0] = (25.0 * np.sin(2.0 * np.pi * 10.0 * t)).astype("<f4")  # 50 uV peak-to-peak
    arr[:, 1] = (10.0 * np.cos(2.0 * np.pi * 10.0 * t)).astype("<f4")
    real_eeg = bv_obj / "raw_v1.eeg"
    arr.tofile(real_eeg)
    real_vmrk = bv_obj / "marker_v1.vmrk"
    real_vmrk.write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\nDataFile=wave.eeg\n[Marker Infos]\n",
        encoding="utf-8",
    )
    os.symlink(real_eeg, bv_dir / "wave.eeg")
    os.symlink(real_vmrk, bv_dir / "wave.vmrk")
    vhdr = bv_dir / "wave.vhdr"
    vhdr.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\nDataFile=wave.eeg\nMarkerFile=wave.vmrk\nDataFormat=BINARY\n"
        "DataOrientation=MULTIPLEXED\nNumberOfChannels=2\nSamplingInterval=1000\n"
        "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n"
        "[Channel Infos]\nCh1=LA1,,1,µV\nCh2=LA2,,1,µV\n",
        encoding="utf-8",
    )
    res_bv = analyze_case(str(vhdr), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    mc_bv = compute_multichannel_batch(
        store,
        res_bv["case"],
        res_bv["runs"][0]["recording"],
        run_name="baseline",
        start_seconds=0.0,
        duration_seconds=2.0,
    )
    vals0 = [v for v in mc_bv["traces"][0]["values_uv"] if v is not None]
    p2p = max(vals0) - min(vals0)
    assert 45.0 <= p2p <= 55.0, f"Unexpected BrainVision symlink P2P: {p2p}"


def test_round11_p1_relocated_bipolar_views_compare_fspath_coords_and_flipflop_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r11_p1"))
    ws = tmp_path / "ws_r11_orig"
    store = WorkbenchStore(ws)

    # (A) Relocated workspace MUST preserve bipolar Reader waveforms, Triad P2P, and compare_runs coverage attribution!
    analyze_case(
        str(SYNTHETIC_EDF),
        workspace_dir=ws,
        run_name="bip_a",
        overrides={"reference_mode": "bipolar", "detector_enabled": False},
    )
    analyze_case(
        str(SYNTHETIC_EDF),
        workspace_dir=ws,
        run_name="bip_b",
        overrides={"reference_mode": "bipolar", "detector_enabled": False},
    )
    # Seed 2 events on SYN_A1-SYN_A2 in bip_a and 0 events (with identical full valid coverage) in bip_b
    for rn, seed_evs in (("bip_a", True), ("bip_b", False)):
        gdir = store.run_generation_dir("synthetic_seeg", "synthetic_seeg", rn)
        meta = read_json(gdir / "run_meta.json")
        summ = read_json(gdir / "summary.json")
        rec_m = store.load_recording("synthetic_seeg", "synthetic_seeg")
        n_samps = int(round(rec_m.sfreq * rec_m.duration))
        ch0 = summ["channels"][0]
        assert ch0["channel_name"] == "SYN_A1-SYN_A2", ch0
        ev_list = (
            [
                {
                    "id": f"ev_bip_{i}",
                    "channel_id": ch0["channel_id"],
                    "channel_name": ch0["channel_name"],
                    "shaft": ch0["shaft"],
                    "start_sample": int((4.0 + i * 3.0) * rec_m.sfreq),
                    "end_sample": int((4.08 + i * 3.0) * rec_m.sfreq),
                    "start_seconds": 4.0 + i * 3.0,
                    "end_seconds": 4.08 + i * 3.0,
                    "scores": {},
                    "labels": {},
                }
                for i in range(2)
            ]
            if seed_evs
            else []
        )
        det_doc = {
            "status": "complete",
            "valid_intervals": {c["channel_id"]: [[0, n_samps]] for c in summ["channels"]},
            "events": ev_list,
        }
        iso_dir = gdir / "_seeded_r11"
        iso_dir.mkdir(exist_ok=True)
        det_p = iso_dir / "detection.json"
        clf_p = iso_dir / "classification.json"
        atomic_write_json(det_p, det_doc)
        atomic_write_json(clf_p, {**det_doc, "classifier_status": "skipped", "classification": {"requested": [], "stages": {}}})
        meta["detection_json_path"] = str(det_p)
        meta["classification_json_path"] = str(clf_p)
        atomic_write_json(gdir / "run_meta.json", meta)
        from brainlens.pipeline import refresh_verdict_tallies

        refresh_verdict_tallies(store, "synthetic_seeg", "synthetic_seeg", rn)

    mc_before = compute_multichannel_batch(
        store, "synthetic_seeg", "synthetic_seeg", run_name="bip_a", start_seconds=3.0, duration_seconds=2.0
    )
    assert len(mc_before["traces"]) == 5 and mc_before["traces"][0]["channel_name"] == "SYN_A1-SYN_A2"
    tr_before = compute_triad_view(
        "synthetic_seeg/synthetic_seeg",
        workspace_dir=ws,
        recording="synthetic_seeg",
        run_name="bip_a",
        channel="SYN_A1-SYN_A2",
        event_id="ev_bip_0",
    )
    bb_before = [v for v in tr_before["broadband_uv"] if v is not None]
    p2p_before = max(bb_before) - min(bb_before)
    cmp_before = compare_runs(store, "synthetic_seeg", "synthetic_seeg", "bip_a", "bip_b")
    assert cmp_before["event_fate"]["disappeared_within_analyzed_coverage"] == 2
    assert cmp_before["event_fate"]["excluded_by_changed_time_or_channel_mask"] == 0

    # Relocate the entire workspace directory and DELETE the original path!
    ws_moved = tmp_path / "ws_r11_moved_elsewhere"
    shutil.copytree(ws, ws_moved, symlinks=True)
    shutil.rmtree(ws)
    moved_store = WorkbenchStore(ws_moved)

    mc_after = compute_multichannel_batch(
        moved_store, "synthetic_seeg", "synthetic_seeg", run_name="bip_a", start_seconds=3.0, duration_seconds=2.0
    )
    assert len(mc_after["traces"]) == 5 and mc_after["traces"][0]["channel_name"] == "SYN_A1-SYN_A2", [
        t["channel_name"] for t in mc_after["traces"]
    ]
    tr_after = compute_triad_view(
        "synthetic_seeg/synthetic_seeg",
        workspace_dir=ws_moved,
        recording="synthetic_seeg",
        run_name="bip_a",
        channel="SYN_A1-SYN_A2",
        event_id="ev_bip_0",
    )
    bb_after = [v for v in tr_after["broadband_uv"] if v is not None]
    p2p_after = max(bb_after) - min(bb_after)
    assert abs(p2p_after - p2p_before) < 1e-4, f"Relocated bipolar P2P drifted: {p2p_before} vs {p2p_after}"
    cmp_after = compare_runs(moved_store, "synthetic_seeg", "synthetic_seeg", "bip_a", "bip_b")
    assert cmp_after["event_fate"]["disappeared_within_analyzed_coverage"] == 2, cmp_after["event_fate"]
    assert cmp_after["event_fate"]["excluded_by_changed_time_or_channel_mask"] == 0, cmp_after["event_fate"]

    # (B) Updating electrode coordinates via FILESYSTEM PATH inspect_case() MUST refresh committed runs & spatial PNG!
    ws_coord = tmp_path / "ws_r11_fscoord"
    coord_store = WorkbenchStore(ws_coord)
    src_copy_dir = tmp_path / "synth_fs_copy"
    src_copy_dir.mkdir()
    edf_copy = src_copy_dir / "synthetic_seeg.edf"
    json_copy = src_copy_dir / "synthetic_seeg.json"
    shutil.copy2(SYNTHETIC_EDF, edf_copy)
    shutil.copy2(SYNTHETIC_EDF.with_suffix(".json"), json_copy)

    res_fs0 = analyze_case(str(edf_copy), workspace_dir=ws_coord, run_name="baseline", overrides={"detector_enabled": False})
    c_fs = res_fs0["case"]
    r_fs = res_fs0["runs"][0]["recording"]
    sh_fs1 = export_case_sheet(c_fs, workspace_dir=ws_coord, primary_run="baseline")
    html_fs1 = Path(sh_fs1["report_path"]).read_text(encoding="utf-8")

    new_sp = tmp_path / "fs_corrected_coords.json"
    new_sp.write_text(
        json.dumps({
            "spatial": {
                "coordinate_system": "synthetic Cartesian",
                "units": "mm",
                "source": "CT revision via filesystem inspect",
                "template": False,
                "positions": {
                    "SYN_A1": [45.0, 0.0, 0.0],
                    "SYN_A2": [50.0, 0.0, 0.0],
                    "SYN_A3": [55.0, 0.0, 0.0],
                    "SYN_A4": [60.0, 0.0, 0.0],
                    "SYN_A5": [65.0, 0.0, 0.0],
                    "SYN_A6": [70.0, 0.0, 0.0],
                },
            }
        }),
        encoding="utf-8",
    )
    # Re-inspect using the FILESYSTEM PATH (not registered case slug)!
    inspect_case(str(edf_copy), workspace_dir=ws_coord, spatial_sidecar=new_sp)
    b_fs_after = coord_store.load_run(c_fs, r_fs, "baseline")
    c_a1_after = next(c for c in b_fs_after["summary"]["spatial"]["contacts"] if c["channel_name"] == "SYN_A1")
    assert c_a1_after["xyz"][0] == 45.0, f"Filesystem inspect_case did not refresh run spatial: {c_a1_after}"
    sh_fs2 = export_case_sheet(c_fs, workspace_dir=ws_coord, primary_run="baseline")
    html_fs2 = Path(sh_fs2["report_path"]).read_text(encoding="utf-8")
    assert html_fs2 != html_fs1, "Updated coordinates via filesystem path did not change exported MDT sheet spatial PNG"

    # (C) Consecutive-only clinical note dedup MUST retain A -> Withdraw -> A in full chronological order!
    for idx, txt in enumerate([
        "Right temporal onset confirmed on stereo-EEG",
        "WITHDRAW earlier lateralization pending review",
        "Right temporal onset confirmed on stereo-EEG",
    ]):
        coord_store.record_judgment(
            c_fs,
            {
                "kind": "event_note",
                "operation_id": f"r11_flipflop_{idx}",
                "recording": r_fs,
                "run_name": "baseline",
                "channel": "SYN_A1",
                "author": "Dr. Flip",
                "role": "clinician",
                "note": txt,
            },
        )
    sh_ff = export_case_sheet(c_fs, workspace_dir=ws_coord, primary_run="baseline")
    html_ff = Path(sh_ff["report_path"]).read_text(encoding="utf-8")
    first_pos = html_ff.find("Right temporal onset confirmed on stereo-EEG")
    withdraw_pos = html_ff.find("WITHDRAW earlier lateralization pending review")
    second_pos = html_ff.find("Right temporal onset confirmed on stereo-EEG", first_pos + 1)
    assert first_pos > 0 and withdraw_pos > first_pos and second_pos > withdraw_pos, (
        f"Flip-flop final re-confirmation was silently dropped: first={first_pos}, withdraw={withdraw_pos}, second={second_pos}"
    )


def test_round12_p0_mixed_rate_edf_bids_conflict_bipolar_isolation_and_folder_symlink_dedup(
    tmp_path: Path,
) -> None:
    import edfio

    ws = tmp_path / "ws_r12_p0"
    store = WorkbenchStore(ws)

    # (A) P0 #1: Mixed-sampling-rate EDF (1024 Hz + 256 Hz) MUST preserve exact native sample voltage
    #     and identical values across wide/narrow Reader windows AND distinct analysis time branches!
    edf_dir = tmp_path / "mixed_edf_case"
    edf_dir.mkdir()
    edf_p = edf_dir / "mixed_case.edf"
    dur = 7
    t_la = np.arange(dur * 1024, dtype=np.float64) / 1024.0
    t_lb = np.arange(dur * 256, dtype=np.float64) / 256.0
    sig_la = 100.0 * np.cos(2.0 * np.pi * 10.5 * t_la + 0.11)
    sig_lb = 100.0 * np.cos(2.0 * np.pi * 12.75 * t_lb + 0.11)
    edfio.Edf(
        [
            edfio.EdfSignal(sig_la, sampling_frequency=1024, label="LA1", physical_dimension="uV"),
            edfio.EdfSignal(sig_lb, sampling_frequency=256, label="LB1", physical_dimension="uV"),
        ],
        data_record_duration=1,
    ).write(edf_p)
    native_lb = edfio.read_edf(edf_p).signals[1].data
    k_nat = int(round(2.1171875 * 256))
    expected_uv = float(native_lb[k_nat])
    assert expected_uv > 99.0, expected_uv

    reg_m = inspect_case(str(edf_p), workspace_dir=ws, persist=True)
    c_m = reg_m["case"]
    r_m = reg_m["recordings"][0]["recording"]
    mc_wide = compute_multichannel_batch(store, c_m, r_m, shaft="ALL", start_seconds=1.375, duration_seconds=1.75)
    mc_narrow = compute_multichannel_batch(store, c_m, r_m, shaft="ALL", start_seconds=2.0, duration_seconds=0.5)
    tr_w = next(t for t in mc_wide["traces"] if t["channel_name"] == "LB1")
    tr_n = next(t for t in mc_narrow["traces"] if t["channel_name"] == "LB1")
    w_map = dict(zip(tr_w["times"], tr_w["values_uv"]))
    overlap_diffs = [abs(w_map[t] - v) for t, v in zip(tr_n["times"], tr_n["values_uv"]) if t in w_map]
    assert len(overlap_diffs) == 512 and max(overlap_diffs) < 1e-6, max(overlap_diffs)
    idx_n = min(range(len(tr_n["times"])), key=lambda i: abs(tr_n["times"][i] - 2.1171875))
    assert abs(tr_n["values_uv"][idx_n] - expected_uv) < 1e-4, (tr_n["values_uv"][idx_n], expected_uv)

    # Also verify two real pipeline time branches write identical voltage into data.npy!
    analyze_case(
        str(edf_p),
        workspace_dir=ws,
        run_name="br_wide",
        overrides={"start_seconds": 1.375, "end_seconds": 3.125, "highpass": None, "lowpass": None, "notch": [], "detector_enabled": False},
    )
    analyze_case(
        str(edf_p),
        workspace_dir=ws,
        run_name="br_narrow",
        overrides={"start_seconds": 2.0, "end_seconds": 2.5, "highpass": None, "lowpass": None, "notch": [], "detector_enabled": False},
    )
    p_w = read_json(Path(store.load_run(c_m, r_m, "br_wide")["meta"]["prepared_json_path"]))
    p_n = read_json(Path(store.load_run(c_m, r_m, "br_narrow")["meta"]["prepared_json_path"]))
    d_w = np.load(p_w["data_path"], mmap_mode="r")[1, int(round((2.1171875 - 1.375) * 1024))] * 1e6
    d_n = np.load(p_n["data_path"], mmap_mode="r")[1, int(round((2.1171875 - 2.0) * 1024))] * 1e6
    assert abs(float(d_w) - expected_uv) < 1e-4 and abs(float(d_n) - expected_uv) < 1e-4, (d_w, d_n, expected_uv)

    # (B) P0 #3: BIDS directory sub-A conflicting with filename/native sub-B MUST be rejected!
    bids_root = tmp_path / "bids_conflict"
    sub_a_ieeg = bids_root / "sub-A" / "ieeg"
    sub_a_ieeg.mkdir(parents=True)
    (bids_root / "sub-A" / "electrodes.json").write_text(
        json.dumps({"spatial": {"coordinate_system": "MNI", "units": "mm", "positions": {"LA1": [10.0, 20.0, 40.0]}}}),
        encoding="utf-8",
    )
    info_cb = mne.create_info(["LA1", "LA2"], sfreq=1024.0, ch_types="seeg")
    info_cb["subject_info"] = {"his_id": "sub-B"}
    f_cb = sub_a_ieeg / "sub-B_task-rest_ieeg.fif"
    mne.io.RawArray(np.random.default_rng(1201).normal(0, 20e-6, size=(2, 2048)), info_cb, verbose="ERROR").save(
        f_cb, overwrite=True, verbose="ERROR"
    )
    with pytest.raises(ToolkitError) as exc_bids:
        inspect_case(str(f_cb), workspace_dir=ws, persist=True)
    assert exc_bids.value.code == "conflicting_subjects", exc_bids.value

    # And case-level same_subject_association MUST NOT copy parent PROVENANCE.json for Patient A onto Patient B!
    parent_ab = tmp_path / "parent_prov_a"
    child_b = parent_ab / "sub-B"
    child_b.mkdir(parents=True)
    (parent_ab / "PROVENANCE.json").write_text(
        json.dumps({
            "participant_id": "P-A",
            "same_subject_association": {"participant_identity": {"participant_id": "P-A"}},
        }),
        encoding="utf-8",
    )
    f_child_b = _write_fif(child_b / "sub-B_ieeg_raw.fif", ["LA1", "LA2"], 1024.0, 2048, seed=1202)
    res_child_b = inspect_case(str(f_child_b), workspace_dir=ws, persist=True)
    assert res_child_b.get("same_subject_association") is None, res_child_b

    # (C) P0 #4: Patient B MUST NOT adopt Patient A's *_bipolar_pairs.json in the same directory!
    bp_dir = tmp_path / "shared_bp_dir"
    bp_dir.mkdir()
    (bp_dir / "patientA_bipolar_pairs.json").write_text(
        json.dumps({
            "participant_id": "P-A",
            "source_file": "patientA_raw.fif",
            "bipolar_pairs": [{"name": "LA1-LA3", "anode": "LA1", "cathode": "LA3"}],
        }),
        encoding="utf-8",
    )
    info_b3 = mne.create_info(["LA1", "LA2", "LA3"], sfreq=1024.0, ch_types="seeg")
    info_b3["subject_info"] = {"his_id": "P-B"}
    f_b3 = bp_dir / "patientB_raw.fif"
    mne.io.RawArray(np.random.default_rng(1203).normal(0, 20e-6, size=(3, 4096)), info_b3, verbose="ERROR").save(
        f_b3, overwrite=True, verbose="ERROR"
    )
    (bp_dir / "patientB_raw.json").write_text(json.dumps({"reference": "common", "line_freq": 60.0}), encoding="utf-8")
    res_bp = analyze_case(str(f_b3), workspace_dir=ws, run_name="bip", overrides={"reference_mode": "bipolar", "detector_enabled": False})
    b_bp = store.load_run(res_bp["case"], res_bp["runs"][0]["recording"], "bip")
    bp_chs = [c["channel_name"] for c in b_bp["summary"]["channels"]]
    assert bp_chs == ["LA1-LA2", "LA2-LA3"], f"Patient B adopted Patient A's bipolar pairs: {bp_chs}"

    # (D) P0 #5: Folder import MUST prefer logical BIDS symlink over shallower objects/a_raw.fif backing blob,
    #     preserving sub-07, bipolar reference, and LA1 BAD [0, 4096] — and refuse conflicting subject aliases!
    ds_root = tmp_path / "bids_annex_ds"
    (ds_root / "objects").mkdir(parents=True)
    sub07_dir = ds_root / "sub-07" / "ieeg"
    sub07_dir.mkdir(parents=True)
    blob_fif = _write_fif(ds_root / "objects" / "a_raw.fif", ["LA1", "LA2"], 1024.0, 4096, seed=1204)
    os.symlink(blob_fif, sub07_dir / "sub-07_ieeg_raw.fif")
    (sub07_dir / "sub-07_ieeg_raw.json").write_text(json.dumps({"reference": "bipolar", "line_freq": 60.0}), encoding="utf-8")
    (sub07_dir / "sub-07_channels.tsv").write_text("name\ttype\tstatus\nLA1\tSEEG\tbad\nLA2\tSEEG\tgood\n", encoding="utf-8")

    res_ds = analyze_case(str(ds_root), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    c_ds = res_ds["case"]
    r_ds = res_ds["runs"][0]["recording"]
    rec_ds = store.load_recording(c_ds, r_ds)
    assert rec_ds.provenance.participant_id == "sub-07", rec_ds.provenance
    assert rec_ds.reference == "bipolar", rec_ds.reference
    b_ds = store.load_run(c_ds, r_ds, "baseline")
    assert b_ds["qc"]["invalid_intervals"]["ch0000"] == [[0, 4096]], b_ds["qc"]["invalid_intervals"]

    # Adding a conflicting sub-08 symlink to the same backing object MUST raise conflicting_subjects!
    sub08_dir = ds_root / "sub-08" / "ieeg"
    sub08_dir.mkdir(parents=True)
    os.symlink(blob_fif, sub08_dir / "sub-08_ieeg_raw.fif")
    with pytest.raises(ToolkitError) as exc_alias:
        inspect_case(str(ds_root), workspace_dir=ws, persist=False)
    assert exc_alias.value.code == "conflicting_subjects", exc_alias.value


def test_round12_p1_bv_repoint_symlink_reimport_case_named_cases_coord_crash_and_two_session_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r12_p1"))
    ws = tmp_path / "ws_r12_p1"
    store = WorkbenchStore(ws)

    # (A) P1 #1: Repointing BrainVision wave.eeg symlink to a NEW 150 uV file while old 50 uV target still exists
    #     MUST be rejected with source_recording_modified!
    bv_dir = tmp_path / "bv_repoint"
    bv_obj = bv_dir / "objects"
    bv_obj.mkdir(parents=True)
    t = np.arange(2000, dtype=np.float64) / 1000.0
    a50 = np.zeros((2000, 2), dtype="<f4")
    a50[:, 0] = (25.0 * np.sin(2.0 * np.pi * 10.0 * t)).astype("<f4")
    a150 = np.zeros((2000, 2), dtype="<f4")
    a150[:, 0] = (75.0 * np.sin(2.0 * np.pi * 10.0 * t)).astype("<f4")
    e_v1 = bv_obj / "raw_v1.eeg"
    e_v2 = bv_obj / "raw_v2.eeg"
    a50.tofile(e_v1)
    a150.tofile(e_v2)
    m_v1 = bv_obj / "mark_v1.vmrk"
    m_v1.write_text("Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\nDataFile=wave.eeg\n[Marker Infos]\n", encoding="utf-8")
    os.symlink(e_v1, bv_dir / "wave.eeg")
    os.symlink(m_v1, bv_dir / "wave.vmrk")
    vhdr = bv_dir / "wave.vhdr"
    vhdr.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\nDataFile=wave.eeg\nMarkerFile=wave.vmrk\nDataFormat=BINARY\n"
        "DataOrientation=MULTIPLEXED\nNumberOfChannels=2\nSamplingInterval=1000\n"
        "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n"
        "[Channel Infos]\nCh1=LA1,,1,µV\nCh2=LA2,,1,µV\n",
        encoding="utf-8",
    )
    res_bv = analyze_case(str(vhdr), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    # Repoint wave.eeg -> raw_v2.eeg (150 uV), leaving raw_v1.eeg untouched on disk!
    (bv_dir / "wave.eeg").unlink()
    os.symlink(e_v2, bv_dir / "wave.eeg")
    with pytest.raises(ToolkitError) as exc_rep:
        compute_multichannel_batch(store, res_bv["case"], res_bv["runs"][0]["recording"], run_name="baseline", start_seconds=0.0, duration_seconds=1.0)
    assert exc_rep.value.code == "source_recording_modified", exc_rep.value

    # (B) P1 #2: Re-importing exact same main FIF symlink MUST reuse the existing case & preserve open questions!
    sym_dir = tmp_path / "reimport_sym"
    (sym_dir / "objects").mkdir(parents=True)
    tgt_fif = _write_fif(sym_dir / "objects" / "blob_raw.fif", ["LA1", "LA2"], 1024.0, 2048, seed=1205)
    link_fif = sym_dir / "sub-R12_ieeg_raw.fif"
    os.symlink(tgt_fif, link_fif)
    r1 = inspect_case(str(link_fif), workspace_dir=ws, persist=True)
    store.add_open_question(r1["case"], "Does sub-R12 need extra insular sampling?")
    r2 = inspect_case(str(link_fif), workspace_dir=ws, persist=True)
    r3 = inspect_case(str(link_fif), workspace_dir=ws, case_name=r1["case"], persist=True)
    assert r2["case"] == r1["case"] and r3["case"] == r1["case"], (r1["case"], r2["case"], r3["case"])
    assert "Does sub-R12 need extra insular sampling?" in store.load_case(r1["case"])["open_questions"]

    # (C) P1 #3 & #4: Relocated workspace containing a case legitimately named 'cases' MUST resolve prepared.json
    #     AND cached segment session.pybrain paths cleanly!
    ws_cases = tmp_path / "ws_cases_orig"
    inspect_case(str(SYNTHETIC_EDF), workspace_dir=ws_cases, case_name="cases", persist=True)
    analyze_case(
        "cases",
        workspace_dir=ws_cases,
        run_name="bip_cases",
        overrides={"reference_mode": "bipolar", "detector_enabled": False},
    )
    # Also place a dummy cached session.pybrain under cases/cases/.../detect/seg_0/ and verify remap!
    st_cases = WorkbenchStore(ws_cases)
    det_p_old = Path(st_cases.load_run("cases", "synthetic_seeg", "bip_cases")["meta"]["detection_json_path"])
    seg0_dir = det_p_old.parent / "seg_0"
    seg0_dir.mkdir(exist_ok=True)
    fake_sess_old = seg0_dir / "session.pybrain"
    fake_sess_old.write_bytes(b"pybrain_stub")

    ws_cases_moved = tmp_path / "ws_cases_relocated"
    shutil.copytree(ws_cases, ws_cases_moved, symlinks=True)
    shutil.rmtree(ws_cases)
    moved_cases_store = WorkbenchStore(ws_cases_moved)
    mc_cases = compute_multichannel_batch(
        moved_cases_store, "cases", "synthetic_seeg", run_name="bip_cases", start_seconds=2.0, duration_seconds=2.0
    )
    assert len(mc_cases["traces"]) == 5 and mc_cases["traces"][0]["channel_name"] == "SYN_A1-SYN_A2", [
        t["channel_name"] for t in mc_cases["traces"]
    ]
    resolved_sess = moved_cases_store.resolve_workspace_artifact_path(str(fake_sess_old))
    assert resolved_sess is not None and resolved_sess.is_file() and resolved_sess.is_relative_to(ws_cases_moved.resolve()), resolved_sess

    # (D) P1 #5: Interrupted coordinate update (crash after save_recording before run refresh) MUST heal on retry AND on export_case_sheet!
    ws_crash = tmp_path / "ws_coord_crash"
    st_crash = WorkbenchStore(ws_crash)
    src_c = tmp_path / "coord_crash_src"
    src_c.mkdir()
    edf_c = src_c / "synthetic_seeg.edf"
    shutil.copy2(SYNTHETIC_EDF, edf_c)
    shutil.copy2(SYNTHETIC_EDF.with_suffix(".json"), src_c / "synthetic_seeg.json")
    res_c0 = analyze_case(str(edf_c), workspace_dir=ws_crash, run_name="baseline", overrides={"detector_enabled": False})
    c_c, r_c = res_c0["case"], res_c0["runs"][0]["recording"]
    sh_c0 = export_case_sheet(c_c, workspace_dir=ws_crash, primary_run="baseline")
    html_c0 = Path(sh_c0["report_path"]).read_text(encoding="utf-8")

    sp45 = tmp_path / "sp45.json"
    sp45.write_text(
        json.dumps({
            "spatial": {
                "coordinate_system": "synthetic Cartesian",
                "units": "mm",
                "source": "CT 45",
                "template": False,
                "positions": {f"SYN_A{i}": [40.0 + i * 5.0, 0.0, 0.0] for i in range(1, 7)},
            }
        }),
        encoding="utf-8",
    )
    # Simulate process crash right after save_recording(x=45), BEFORE refresh_verdict_tallies ran!
    rm_cur = st_crash.load_recording(c_c, r_c)
    rm_cur.spatial = rm_cur.spatial.model_copy(
        update={"positions": {f"SYN_A{i}": (40.0 + i * 5.0, 0.0, 0.0) for i in range(1, 7)}, "source": "CT 45"}
    )
    st_crash.save_recording(c_c, r_c, rm_cur)
    # Verify run still has stale x=0 before retry
    assert next(c for c in st_crash.load_run(c_c, r_c, "baseline")["summary"]["spatial"]["contacts"] if c["channel_name"] == "SYN_A1")["xyz"][0] == 0.0
    # Retry inspect_case with the SAME filesystem path & sidecar -> MUST detect lagged run and heal to x=45!
    inspect_case(str(edf_c), workspace_dir=ws_crash, spatial_sidecar=sp45)
    assert next(c for c in st_crash.load_run(c_c, r_c, "baseline")["summary"]["spatial"]["contacts"] if c["channel_name"] == "SYN_A1")["xyz"][0] == 45.0
    sh_c1 = export_case_sheet(c_c, workspace_dir=ws_crash, primary_run="baseline")
    html_c1 = Path(sh_c1["report_path"]).read_text(encoding="utf-8")
    assert html_c1 != html_c0

    # Also test export_case_sheet healing an interrupted crash directly (without retrying inspect_case)!
    rm_cur2 = st_crash.load_recording(c_c, r_c)
    rm_cur2.spatial = rm_cur2.spatial.model_copy(
        update={"positions": {f"SYN_A{i}": (70.0 + i * 5.0, 0.0, 0.0) for i in range(1, 7)}, "source": "CT 75"}
    )
    st_crash.save_recording(c_c, r_c, rm_cur2)
    sh_c2 = export_case_sheet(c_c, workspace_dir=ws_crash, primary_run="baseline")
    assert next(c for c in st_crash.load_run(c_c, r_c, "baseline")["summary"]["spatial"]["contacts"] if c["channel_name"] == "SYN_A1")["xyz"][0] == 75.0
    assert Path(sh_c2["report_path"]).read_text(encoding="utf-8") != html_c1

    # (E) P1 #7: Identical clinical note across TWO recordings in the same case MUST BOTH render in MDT sheet!
    ws_two = tmp_path / "ws_two_rec"
    st_two = WorkbenchStore(ws_two)
    two_dir = tmp_path / "two_sessions"
    _write_fif(two_dir / "session1_raw.fif", ["RA1", "RA2"], 1024.0, 2048, seed=1206)
    _write_fif(two_dir / "session2_raw.fif", ["RA1", "RA2"], 1024.0, 2048, seed=1207)
    res_two = analyze_case(str(two_dir), workspace_dir=ws_two, run_name="baseline", overrides={"detector_enabled": False})
    c_two = res_two["case"]
    for idx, rname in enumerate(["session1_raw", "session2_raw"]):
        st_two.record_judgment(
            c_two,
            {
                "kind": "event_note",
                "operation_id": f"r12_two_rec_{idx}",
                "recording": rname,
                "run_name": "baseline",
                "channel": "RA1",
                "author": "Dr. Multi",
                "role": "clinician",
                "note": "Reproducible mesial temporal ripple pattern",
            },
        )
    sh_two = export_case_sheet(c_two, workspace_dir=ws_two, primary_run="baseline")
    html_two = Path(sh_two["report_path"]).read_text(encoding="utf-8")
    assert "session1_raw / RA1" in html_two and "session2_raw / RA1" in html_two, (
        "Identical clinical note on second recording was dropped by MDT sheet deduplication!"
    )


def test_round13_p0_sidecar_source_file_ext_nested_spatial_and_replaced_recording_mdt_409(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edfio

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r13_p0"))
    ws = tmp_path / "ws_r13_p0"
    store = WorkbenchStore(ws)

    # (A) P0 #1a: Patient B exact sidecar (common, x=-35) + fuzzy *_braintoolkit.json with TOP-LEVEL source_file=patientA
    #     MUST preserve B's common reference and x=-35 (NEVER adopt A's bipolar / x=+35)!
    sc_dir = tmp_path / "fuzzy_sc_dir"
    sc_dir.mkdir()
    info_b = mne.create_info(["LA1", "LA2"], sfreq=1024.0, ch_types="seeg")
    info_b["subject_info"] = {"his_id": "P-B"}
    f_b = sc_dir / "patientB_raw.fif"
    mne.io.RawArray(np.random.default_rng(1301).normal(0, 20e-6, size=(2, 2048)), info_b, verbose="ERROR").save(
        f_b, overwrite=True, verbose="ERROR"
    )
    (sc_dir / "patientB_raw.json").write_text(
        json.dumps({
            "SamplingFrequency": 1024.0,
            "reference": "common",
            "spatial": {"coordinate_system": "MNI", "units": "mm", "source": "B CT", "positions": {"LA1": [-35.0, 0.0, 0.0]}},
        }),
        encoding="utf-8",
    )
    (sc_dir / "raw_braintoolkit.json").write_text(
        json.dumps({
            "source_file": "patientA_raw.fif",
            "reference": "bipolar",
            "spatial": {"coordinate_system": "MNI", "units": "mm", "source": "A CT", "positions": {"LA1": [35.0, 0.0, 0.0]}},
        }),
        encoding="utf-8",
    )
    res_b = inspect_case(str(f_b), workspace_dir=ws, persist=True)
    rec_b = store.load_recording(res_b["case"], res_b["recordings"][0]["recording"])
    assert rec_b.reference == "common", rec_b.reference
    assert rec_b.spatial is not None and tuple(rec_b.spatial.positions["LA1"]) == (-35.0, 0.0, 0.0), rec_b.spatial

    # (B) P0 #1b: Sidecar top-level declares P-B, nested spatial.participant_id declares P-A -> MUST raise conflicting_subjects!
    nest_dir = tmp_path / "nested_sp_conflict"
    nest_dir.mkdir()
    f_n = _write_fif(nest_dir / "subjB_raw.fif", ["LA1", "LA2"], 1024.0, 2048, seed=1302)
    (nest_dir / "subjB_raw.json").write_text(
        json.dumps({
            "participant_id": "P-B",
            "reference": "common",
            "spatial": {
                "participant_id": "P-A",
                "coordinate_system": "MNI",
                "units": "mm",
                "source": "A CT",
                "positions": {"LA1": [35.0, 0.0, 0.0]},
            },
        }),
        encoding="utf-8",
    )
    with pytest.raises(ToolkitError) as exc_nest:
        inspect_case(str(f_n), workspace_dir=ws, persist=True)
    assert exc_nest.value.code == "conflicting_subjects", exc_nest.value

    # (C) P0 #2: Same stem, different format (visit_raw.edf vs visit_raw.fif) with bipolar pairs declaring visit_raw.edf
    #     MUST NOT match visit_raw.fif!
    ext_dir = tmp_path / "same_stem_diff_ext"
    ext_dir.mkdir()
    edfio.Edf(
        [edfio.EdfSignal(np.zeros(2048), sampling_frequency=1024, label=ch, physical_dimension="uV") for ch in ("LA1", "LA2", "LA3")],
        data_record_duration=1,
    ).write(ext_dir / "visit_raw.edf")
    info_vf = mne.create_info(["LA1", "LA2", "LA3"], sfreq=1024.0, ch_types="seeg")
    info_vf["subject_info"] = {"his_id": "P-B"}
    f_vf = ext_dir / "visit_raw.fif"
    mne.io.RawArray(np.random.default_rng(1303).normal(0, 20e-6, size=(3, 2048)), info_vf, verbose="ERROR").save(
        f_vf, overwrite=True, verbose="ERROR"
    )
    (ext_dir / "visit_raw.json").write_text(json.dumps({"reference": "common", "line_freq": 60.0}), encoding="utf-8")
    (ext_dir / "bipolar_pairs.json").write_text(
        json.dumps({
            "source_file": "visit_raw.edf",
            "pairs": [{"name": "LA1-LA3", "anode": "LA1", "cathode": "LA3"}],
        }),
        encoding="utf-8",
    )
    res_vf = analyze_case(str(f_vf), workspace_dir=ws, run_name="bip", overrides={"reference_mode": "bipolar", "detector_enabled": False})
    b_vf = store.load_run(res_vf["case"], res_vf["runs"][0]["recording"], "bip")
    assert [c["channel_name"] for c in b_vf["summary"]["channels"]] == ["LA1-LA2", "LA2-LA3"]

    # (D) P0 #4: Replacing & re-inspecting recording MUST immediately mark old MDT sheet stale, strip token,
    #     reject delayed Confirm with HTTP 409 stale_sheet, AND refuse exporting a new sheet on the mismatched old run!
    ws_rep = tmp_path / "ws_r13_replace"
    rep_src = tmp_path / "rep_src"
    rep_src.mkdir()
    edf_rep = rep_src / "synthetic_seeg.edf"
    shutil.copy2(SYNTHETIC_EDF, edf_rep)
    shutil.copy2(SYNTHETIC_EDF.with_suffix(".json"), rep_src / "synthetic_seeg.json")
    analyze_case(str(edf_rep), workspace_dir=ws_rep, run_name="baseline", overrides={"detector_enabled": False})
    st_rep = WorkbenchStore(ws_rep)
    gdir_rep = st_rep.run_generation_dir("synthetic_seeg", "synthetic_seeg", "baseline")
    m_rep = read_json(gdir_rep / "run_meta.json")
    s_rep = read_json(gdir_rep / "summary.json")
    rec_rep = st_rep.load_recording("synthetic_seeg", "synthetic_seeg")
    n_rep = int(round(rec_rep.sfreq * rec_rep.duration))
    det_rep = {
        "status": "complete",
        "valid_intervals": {c["channel_id"]: [[0, n_rep]] for c in s_rep["channels"]},
        "events": [{
            "id": "ev_r13_old_001",
            "channel_id": s_rep["channels"][0]["channel_id"],
            "channel_name": s_rep["channels"][0]["channel_name"],
            "shaft": s_rep["channels"][0]["shaft"],
            "start_sample": int(4.0 * rec_rep.sfreq),
            "end_sample": int(4.08 * rec_rep.sfreq),
            "start_seconds": 4.0,
            "end_seconds": 4.08,
            "scores": {},
            "labels": {},
        }],
    }
    iso_rep = gdir_rep / "_seeded_r13"
    iso_rep.mkdir(exist_ok=True)
    atomic_write_json(iso_rep / "detection.json", det_rep)
    atomic_write_json(iso_rep / "classification.json", {**det_rep, "classifier_status": "skipped", "classification": {"requested": [], "stages": {}}})
    m_rep["detection_json_path"] = str(iso_rep / "detection.json")
    m_rep["classification_json_path"] = str(iso_rep / "classification.json")
    atomic_write_json(gdir_rep / "run_meta.json", m_rep)
    from brainlens.pipeline import refresh_verdict_tallies

    refresh_verdict_tallies(st_rep, "synthetic_seeg", "synthetic_seeg", "baseline")
    sh_old = export_case_sheet("synthetic_seeg", workspace_dir=ws_rep, primary_run="baseline")
    ev_old = st_rep.load_run("synthetic_seeg", "synthetic_seeg", "baseline")["events"][0]

    # Replace synthetic_seeg.edf with quiet zero-event EDF and re-inspect WITHOUT creating a new run yet!
    edfio.Edf(
        [edfio.EdfSignal(np.random.default_rng(1304).normal(0, 5.0, 2000 * 6), sampling_frequency=2000, label=f"SYN_A{i}", physical_dimension="uV") for i in range(1, 7)],
        data_record_duration=1,
    ).write(edf_rep)
    inspect_case(str(edf_rep), workspace_dir=ws_rep, persist=True)

    # Re-exporting old run MUST raise run_recording_mismatch!
    with pytest.raises(ToolkitError) as exc_mm:
        export_case_sheet("synthetic_seeg", workspace_dir=ws_rep, primary_run="baseline")
    assert exc_mm.value.code == "run_recording_mismatch", exc_mm.value

    srv = WorkbenchHTTPServer(workspace_dir=ws_rep, host="127.0.0.1", port=0)
    url = srv.start()
    try:
        with urllib.request.urlopen(
            f"{url}/api/sheet-freshness?case=synthetic_seeg&file={sh_old['export_filename']}&export_hash={sh_old['export_hash']}",
            timeout=5,
        ) as r_fr:
            assert json.loads(r_fr.read().decode("utf-8"))["fresh"] is False
        with urllib.request.urlopen(f"{url}/exports/synthetic_seeg/{sh_old['export_filename']}", timeout=5) as r_html:
            html_stale = r_html.read().decode("utf-8")
        assert '<meta name="brainlens-sheet-stale" content="1"' in html_stale and '<meta name="brainlens-token"' not in html_stale
        req_409 = urllib.request.Request(
            f"{url}/api/report-action",
            data=json.dumps({
                "case": "synthetic_seeg",
                "workspace_id": st_rep.workspace_id,
                "source": "case_sheet",
                "primary_run": "baseline",
                "export_filename": sh_old["export_filename"],
                "export_hash": sh_old["export_hash"],
                "action": "verdict",
                "recording": "synthetic_seeg",
                "run_name": "baseline",
                "channel": ev_old["channel_name"],
                "event_id": ev_old["id"],
                "verdict": "confirm",
                "author": "Dr. Old",
                "role": "engineering_rehearsal",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-BrainLens-Token": srv.auth_token},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_409:
            urllib.request.urlopen(req_409, timeout=5)
        assert exc_409.value.code == 409
        assert st_rep.load_judgments("synthetic_seeg") == []
    finally:
        srv.stop()


def test_round13_p1_nonvoltage_spo2_partial_restore_multi_aux_aliases_timestamp_notes_and_mixed_mem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edfio
    import tracemalloc

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r13_p1"))
    ws = tmp_path / "ws_r13_p1"
    store = WorkbenchStore(ws)

    # (A) P1 #1: Non-voltage auxiliary channel SpO2 (%) MUST be rejected by Triad & render_evidence_card!
    sleep_edf = tmp_path / "sleep_spo2.edf"
    edfio.Edf(
        [
            edfio.EdfSignal(np.random.default_rng(1310).normal(0, 20.0, 2048), sampling_frequency=1024, label="LA1", physical_dimension="uV"),
            edfio.EdfSignal(np.full(2048, 94.0), sampling_frequency=1024, label="SpO2", physical_dimension="%"),
        ],
        data_record_duration=1,
    ).write(sleep_edf)
    reg_sl = inspect_case(str(sleep_edf), workspace_dir=ws, persist=True)
    with pytest.raises(ToolkitError) as exc_nv:
        render_evidence_card(
            f"{reg_sl['case']}/{reg_sl['recordings'][0]['recording']}",
            workspace_dir=ws,
            recording=reg_sl["recordings"][0]["recording"],
            channel="SpO2",
            center_seconds=1.0,
            return_base64=True,
        )
    assert exc_nv.value.code == "non_voltage_channel", exc_nv.value

    # (B) P1 #2: Partially restored workspace missing prepared.json on bipolar run MUST fail explicitly
    #     in Reader/Triad AND in export_case_sheet after coordinate update!
    ws_part = tmp_path / "ws_partial_orig"
    src_p = tmp_path / "part_src"
    src_p.mkdir()
    edf_p = src_p / "synthetic_seeg.edf"
    shutil.copy2(SYNTHETIC_EDF, edf_p)
    shutil.copy2(SYNTHETIC_EDF.with_suffix(".json"), src_p / "synthetic_seeg.json")
    res_p = analyze_case(str(edf_p), workspace_dir=ws_part, run_name="bip", overrides={"reference_mode": "bipolar", "detector_enabled": False})
    c_p, r_p = res_p["case"], res_p["runs"][0]["recording"]
    st_part = WorkbenchStore(ws_part)
    prep_json = Path(st_part.load_run(c_p, r_p, "bip")["meta"]["prepared_json_path"])
    prep_json.unlink()

    with pytest.raises(ToolkitError) as exc_inc_mc:
        compute_multichannel_batch(st_part, c_p, r_p, run_name="bip", start_seconds=2.0, duration_seconds=2.0)
    assert exc_inc_mc.value.code == "incomplete_run_artifacts", exc_inc_mc.value

    # Updating coordinates when prepared.json is missing MUST refuse exporting stale x=0 PNG as fresh!
    rm_p = st_part.load_recording(c_p, r_p)
    rm_p.spatial = rm_p.spatial.model_copy(
        update={"positions": {f"SYN_A{i}": (40.0 + i * 5.0, 0.0, 0.0) for i in range(1, 7)}}
    )
    st_part.save_recording(c_p, r_p, rm_p)
    with pytest.raises(ToolkitError) as exc_inc_sh:
        export_case_sheet(c_p, workspace_dir=ws_part, primary_run="bip")
    assert exc_inc_sh.value.code == "incomplete_run_artifacts", exc_inc_sh.value

    # (C) P1 #3: Multi-patient folder with BOTH A & B bipolar_pairs AND BOTH A & B coords MUST resolve B cleanly,
    #     and two non-BIDS symlink aliases whose sidecars declare P-A vs P-B MUST raise conflicting_subjects!
    multi_dir = tmp_path / "multi_aux_ab"
    multi_dir.mkdir()
    (multi_dir / "patientA_bipolar_pairs.json").write_text(
        json.dumps({"participant_id": "P-A", "pairs": [{"name": "LA1-LA2", "anode": "LA1", "cathode": "LA2"}]}),
        encoding="utf-8",
    )
    (multi_dir / "patientB_bipolar_pairs.json").write_text(
        json.dumps({"participant_id": "P-B", "pairs": [{"name": "LA1-LA3", "anode": "LA1", "cathode": "LA3"}]}),
        encoding="utf-8",
    )
    (multi_dir / "coords_A.json").write_text(
        json.dumps({"participant_id": "P-A", "spatial": {"coordinate_system": "MNI", "units": "mm", "source": "A CT", "positions": {"LA1": [-10.0, 0.0, 0.0]}}}),
        encoding="utf-8",
    )
    (multi_dir / "coords_B.json").write_text(
        json.dumps({"participant_id": "P-B", "spatial": {"coordinate_system": "MNI", "units": "mm", "source": "B CT", "positions": {"LA1": [42.0, 0.0, 0.0]}}}),
        encoding="utf-8",
    )
    info_mb = mne.create_info(["LA1", "LA2", "LA3"], sfreq=1024.0, ch_types="seeg")
    info_mb["subject_info"] = {"his_id": "P-B"}
    f_mb = multi_dir / "scanB_raw.fif"
    mne.io.RawArray(np.random.default_rng(1311).normal(0, 20e-6, size=(3, 2048)), info_mb, verbose="ERROR").save(
        f_mb, overwrite=True, verbose="ERROR"
    )
    (multi_dir / "scanB_raw.json").write_text(json.dumps({"reference": "common", "line_freq": 60.0}), encoding="utf-8")
    res_mb = analyze_case(str(f_mb), workspace_dir=ws, run_name="bip", overrides={"reference_mode": "bipolar", "detector_enabled": False})
    rec_mb = store.load_recording(res_mb["case"], res_mb["runs"][0]["recording"])
    b_mb = store.load_run(res_mb["case"], res_mb["runs"][0]["recording"], "bip")
    assert [c["channel_name"] for c in b_mb["summary"]["channels"]] == ["LA1-LA3"], b_mb["summary"]["channels"]
    assert rec_mb.spatial is not None and tuple(rec_mb.spatial.positions["LA1"]) == (42.0, 0.0, 0.0), rec_mb.spatial

    alias_dir = tmp_path / "nonbids_alias_conflict"
    (alias_dir / "objects").mkdir(parents=True)
    blob = _write_fif(alias_dir / "objects" / "blob_raw.fif", ["LA1", "LA2"], 1024.0, 2048, seed=1312)
    os.symlink(blob, alias_dir / "alias_a_raw.fif")
    os.symlink(blob, alias_dir / "alias_b_raw.fif")
    (alias_dir / "alias_a_raw.json").write_text(json.dumps({"participant_id": "P-A", "reference": "common"}), encoding="utf-8")
    (alias_dir / "alias_b_raw.json").write_text(json.dumps({"participant_id": "P-B", "reference": "common"}), encoding="utf-8")
    with pytest.raises(ToolkitError) as exc_nb:
        inspect_case(str(alias_dir), workspace_dir=ws, persist=False)
    assert exc_nb.value.code == "conflicting_subjects", exc_nb.value

    # (D) P1 #7: Same-channel identical observations at 4.15s and 10.25s MUST BOTH render with timestamps in MDT!
    ws_ts = tmp_path / "ws_ts_notes"
    st_ts = WorkbenchStore(ws_ts)
    analyze_case(str(SYNTHETIC_EDF), workspace_dir=ws_ts, run_name="baseline", overrides={"detector_enabled": False})
    for idx, ts in enumerate((4.15, 10.25)):
        st_ts.record_judgment(
            "synthetic_seeg",
            {
                "kind": "event_note",
                "operation_id": f"r13_ts_{idx}",
                "recording": "synthetic_seeg",
                "run_name": "baseline",
                "channel": "SYN_A1",
                "center_seconds": ts,
                "author": "Dr. Time",
                "role": "clinician",
                "note": "Brief fast ripple burst",
            },
        )
    sh_ts = export_case_sheet("synthetic_seeg", workspace_dir=ws_ts, primary_run="baseline")
    html_ts = Path(sh_ts["report_path"]).read_text(encoding="utf-8")
    assert "@4.15s" in html_ts and "@10.25s" in html_ts, "Timestamped notes on same channel were merged or lost!"

    # (E) P1 #8: 64-ch 30s mixed-rate EDF MUST stay O(window) RAM (< 6 MiB, NOT 15.7 MiB full preload)
    #     during BOTH inspect AND single-channel 0.25s Triad read!
    big64_edf = tmp_path / "big64_mixed.edf"
    sigs64 = [
        edfio.EdfSignal(
            np.zeros(30 * (1024 if i < 32 else 256), dtype=np.float64),
            sampling_frequency=(1024 if i < 32 else 256),
            label=f"C{i:02d}",
            physical_dimension="uV",
        )
        for i in range(64)
    ]
    edfio.Edf(sigs64, data_record_duration=1).write(big64_edf)
    tracemalloc.start()
    reg64 = inspect_case(str(big64_edf), workspace_dir=ws, persist=True)
    _, peak_insp = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak_insp < 8 * 1024 * 1024, f"Mixed-rate inspect allocated full array: {peak_insp:,} bytes"

    tracemalloc.start()
    tr64 = compute_triad_view(
        f"{reg64['case']}/{reg64['recordings'][0]['recording']}",
        workspace_dir=ws,
        recording=reg64["recordings"][0]["recording"],
        channel="C40",
        center_seconds=10.0,
        window_seconds=0.25,
    )
    _, peak_read = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert tr64["channel_name"] == "C40" and peak_read < 8 * 1024 * 1024, f"Mixed-rate 0.25s read allocated full array: {peak_read:,} bytes"


def test_round14_p0_discontinuous_edfd_exact_json_full_path_anon_map_and_replaced_recording_wal_clip_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edfio
    import mne
    import urllib.error
    import urllib.request

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r14_p0"))
    ws = tmp_path / "ws_r14_p0"
    store = WorkbenchStore(ws)

    # (A) Discontinuous EDF+D and BDF+D must expose the TAL clock, including post-gap annotations.
    def _write_discontinuous_edf_or_bdf(dest: Path, *, is_bdf: bool) -> None:
        bps = 3 if is_bdf else 2
        ns = 2
        n_rec = 4
        spr_sig = 100
        spr_tal = 20
        hdr_bytes = 256 + ns * 256
        fixed = bytearray(b" " * 256)
        if is_bdf:
            fixed[0:8] = b"\xffBIOSEMI"
            fixed[192:197] = b"BDF+D"
        else:
            fixed[0:8] = b"0       "
            fixed[192:197] = b"EDF+D"
        fixed[168:176] = b"01.01.20"
        fixed[176:184] = b"00.00.00"
        fixed[184:192] = f"{hdr_bytes:<8d}".encode("ascii")
        fixed[236:244] = f"{n_rec:<8d}".encode("ascii")
        fixed[244:252] = f"{1:<8d}".encode("ascii")
        fixed[252:256] = f"{ns:<4d}".encode("ascii")
        ann_lbl = "BDF Annotations" if is_bdf else "EDF Annotations"
        shdr = bytearray()
        for vals, w in [
            (["LA1", ann_lbl], 16),
            (["", ""], 80),
            (["uV", ""], 8),
            (["-1000", "-1"], 8),
            (["1000", "1"], 8),
            (["-8388608" if is_bdf else "-32768", "-8388608" if is_bdf else "-32768"], 8),
            (["8388607" if is_bdf else "32767", "8388607" if is_bdf else "32767"], 8),
            (["", ""], 80),
            ([str(spr_sig), str(spr_tal)], 8),
            (["", ""], 32),
        ]:
            for v in vals:
                shdr.extend(f"{v:<{w}s}"[:w].encode("ascii"))
        onsets = [b"+0\x14\x14", b"+6\x14\x14\x00+6.5\x150.125\x14pulse\x14", b"+7\x14\x14", b"+8\x14\x14"]
        body = bytearray()
        for tal_b in onsets:
            body.extend(b"\x00" * (spr_sig * bps))
            padded = tal_b + b"\x00" * (spr_tal * bps - len(tal_b))
            body.extend(padded[: spr_tal * bps])
        dest.write_bytes(bytes(fixed) + bytes(shdr) + bytes(body))

    for fname, is_b in (("disc.edf", False), ("disc.bdf", True)):
        p_disc = tmp_path / fname
        _write_discontinuous_edf_or_bdf(p_disc, is_bdf=is_b)
        inspected = inspect_case(str(p_disc), workspace_dir=ws, persist=False)
        rec = next(iter(inspected["_recording_specs"].values()))
        assert rec.duration == 9 and rec.gaps() == [(1, 6)]
        assert any(m.label == "pulse" and m.start == 6.5 and m.end == 6.625 for m in rec.markers)

    # (B) P0 #2: Exact-stem visit_raw.json declaring source_file=visit_raw.edf MUST NOT hijack visit_raw.fif!
    ex_dir = tmp_path / "exact_json_ext"
    ex_dir.mkdir()
    edfio.Edf([edfio.EdfSignal(np.zeros(2048), sampling_frequency=1024, label="LA1", physical_dimension="uV")]).write(
        ex_dir / "visit_raw.edf"
    )
    info_b = mne.create_info(["LA1", "LA2"], sfreq=1024.0, ch_types="seeg")
    info_b["subject_info"] = {"his_id": "P-B"}
    f_b = ex_dir / "visit_raw.fif"
    mne.io.RawArray(np.random.default_rng(1401).normal(0, 20e-6, size=(2, 2048)), info_b, verbose="ERROR").save(
        f_b, overwrite=True, verbose="ERROR"
    )
    (ex_dir / "visit_raw.json").write_text(
        json.dumps({
            "source_file": "visit_raw.edf",
            "participant_id": "P-A",
            "reference": "bipolar",
            "spatial": {"coordinate_system": "MNI", "units": "mm", "source": "A EDF map", "positions": {"LA1": [42.0, 20.0, 30.0]}},
        }),
        encoding="utf-8",
    )
    (ex_dir / "visit_raw_fif_braintoolkit.json").write_text(
        json.dumps({
            "source_file": "visit_raw.fif",
            "participant_id": "P-B",
            "reference": "common",
            "spatial": {"coordinate_system": "MNI", "units": "mm", "source": "B native map", "positions": {"LA1": [-42.0, 20.0, 30.0]}},
        }),
        encoding="utf-8",
    )
    res_exb = inspect_case(str(f_b), workspace_dir=ws, persist=True)
    rec_exb = store.load_recording(res_exb["case"], res_exb["recordings"][0]["recording"])
    assert rec_exb.reference == "common", rec_exb.reference
    assert rec_exb.spatial is not None and tuple(rec_exb.spatial.positions["LA1"]) == (-42.0, 20.0, 30.0), rec_exb.spatial

    # (C) P0 #3: Absolute path source_file pointing to dirA/visit_raw.fif MUST NOT bind to dirB/visit_raw.fif!
    dir_a = tmp_path / "path_dirA"
    dir_b = tmp_path / "path_dirB"
    dir_a.mkdir()
    dir_b.mkdir()
    fa = _write_fif(dir_a / "visit_raw.fif", ["LA1", "LA2", "LA3"], 1024.0, 2048, seed=1402)
    fb = _write_fif(dir_b / "visit_raw.fif", ["LA1", "LA2", "LA3"], 1024.0, 2048, seed=1403)
    (dir_b / "published_bipolar_pairs.json").write_text(
        json.dumps({"source_file": str(fa.resolve()), "pairs": [{"name": "LA1-LA3", "anode": "LA1", "cathode": "LA3"}]}),
        encoding="utf-8",
    )
    res_pb = inspect_case(str(fb), workspace_dir=ws, persist=True)
    rec_pb = store.load_recording(res_pb["case"], res_pb["recordings"][0]["recording"])
    assert rec_pb.published_bipolar_pairs_path is None, rec_pb.published_bipolar_pairs_path

    # (D) P0 #4: Anonymous recording MUST NOT inherit subject-tagged spatial map coords_A / coords_B!
    anon_dir = tmp_path / "anon_maps"
    anon_dir.mkdir()
    f_anon = _write_fif(anon_dir / "unlabeled_raw.fif", ["LA1", "LA2"], 1024.0, 2048, seed=1404)
    (anon_dir / "map_A.json").write_text(
        json.dumps({"participant_id": "P-A", "spatial": {"coordinate_system": "MNI", "units": "mm", "source": "A", "positions": {"LA1": [42.0, 0.0, 0.0]}}}),
        encoding="utf-8",
    )
    (anon_dir / "map_B.json").write_text(
        json.dumps({"participant_id": "P-B", "spatial": {"coordinate_system": "MNI", "units": "mm", "source": "B", "positions": {"LA1": [-42.0, 0.0, 0.0]}}}),
        encoding="utf-8",
    )
    res_an = inspect_case(str(f_anon), workspace_dir=ws, persist=True)
    rec_an = store.load_recording(res_an["case"], res_an["recordings"][0]["recording"])
    assert rec_an.spatial is None, f"Anonymous recording inherited subject-tagged map: {rec_an.spatial}"

    # (E) P0 #6 & P0 #7: Replacing source recording MUST reject stale Reader judgments with HTTP 409 run_recording_mismatch,
    #     AND overwriting baseline on the new recording MUST NOT rebind old clipped interval/note onto the new waveform!
    ws_rep = tmp_path / "ws_r14_rep"
    st_rep = WorkbenchStore(ws_rep)
    src_rep = tmp_path / "src_r14_rep"
    src_rep.mkdir()
    edf_rep = src_rep / "synthetic_seeg.edf"
    shutil.copy2(SYNTHETIC_EDF, edf_rep)
    shutil.copy2(SYNTHETIC_EDF.with_suffix(".json"), src_rep / "synthetic_seeg.json")
    r_v1 = analyze_case(str(edf_rep), workspace_dir=ws_rep, run_name="baseline", overrides={"detector_enabled": False})
    c_rep, rec_rep = r_v1["case"], r_v1["runs"][0]["recording"]
    fp_v1 = st_rep.load_recording(c_rep, rec_rep).fingerprint
    st_rep.record_judgment(
        c_rep,
        {
            "kind": "interval_note",
            "operation_id": "clip_v1_800",
            "recording": rec_rep,
            "run_name": "baseline",
            "channel": "SYN_A1",
            "center_seconds": 8.10,
            "interval_seconds": [8.0, 8.2],
            "clipped": True,
            "author": "Engineer",
            "role": "engineering_rehearsal",
            "note": "Old v1 high-amplitude 207 uV burst",
        },
    )
    assert len(st_rep.load_judgments(c_rep)) == 1
    assert st_rep.load_judgments(c_rep)[0]["recording_fingerprint"] == fp_v1

    # Replace EDF bytes with a quiet 10 uV sine wave and re-inspect (without re-analyzing yet!)
    t_q = np.arange(30000, dtype=np.float64) / 2000.0
    sigs_q = [
        edfio.EdfSignal(10.0 * np.sin(2.0 * np.pi * 8.0 * t_q), sampling_frequency=2000, label=f"SYN_A{i}", physical_dimension="uV")
        for i in range(1, 7)
    ]
    edfio.Edf(sigs_q).write(edf_rep)
    inspect_case(str(edf_rep), workspace_dir=ws_rep, persist=True)
    fp_v2 = st_rep.load_recording(c_rep, rec_rep).fingerprint
    assert fp_v2 != fp_v1

    # Stale Reader POST /api/judgment MUST return HTTP 409 run_recording_mismatch with ZERO new WAL rows!
    srv = WorkbenchHTTPServer(workspace_dir=ws_rep, host="127.0.0.1", port=0)
    url = srv.start().rstrip("/")
    try:
        req = urllib.request.Request(
            f"{url}/api/judgment",
            data=json.dumps({
                "case": c_rep,
                "recording": rec_rep,
                "run_name": "baseline",
                "channel": "SYN_A1",
                "center_seconds": 8.10,
                "note": "Stale reader attempt",
                "author": "Engineer",
                "role": "engineering_rehearsal",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-BrainLens-Token": srv.auth_token},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as http_err:
            urllib.request.urlopen(req, timeout=5)
        assert http_err.value.code == 409, http_err.value.code
        err_body = json.loads(http_err.value.read().decode("utf-8"))
        assert err_body["error"]["code"] == "run_recording_mismatch", err_body
        assert len(st_rep.load_judgments(c_rep)) == 1
    finally:
        srv.stop()

    # Overwrite baseline on v2 and export fresh MDT -> old v1 clip & observation MUST NOT appear on v2's sheet!
    analyze_case(str(edf_rep), workspace_dir=ws_rep, run_name="baseline", overwrite=True, overrides={"detector_enabled": False})
    sh_v2 = export_case_sheet(c_rep, workspace_dir=ws_rep, primary_run="baseline")
    html_v2 = Path(sh_v2["report_path"]).read_text(encoding="utf-8")
    assert "Rehearsal-clipped evidence" not in html_v2, "Old v1 clip was rebound onto replaced v2 waveform!"
    assert "Old v1 high-amplitude 207 uV burst" not in html_v2, "Old v1 note leaked onto replaced v2 sheet!"


def test_round14_p1_p2_cross_fp_compare_bipolar_unipolar_refusal_missing_prep_17_maps_alias_prov_point_marks_and_sub10ms_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edfio
    import mne

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r14_p1"))
    ws = tmp_path / "ws_r14_p1"
    store = WorkbenchStore(ws)

    # (A) P1 #1: Runs from DIFFERENT recording_fingerprints MUST NOT be directly_comparable_rates=True or exported together!
    src_cmp = tmp_path / "src_r14_cmp"
    src_cmp.mkdir()
    edf_cmp = src_cmp / "synthetic_seeg.edf"
    shutil.copy2(SYNTHETIC_EDF, edf_cmp)
    shutil.copy2(SYNTHETIC_EDF.with_suffix(".json"), src_cmp / "synthetic_seeg.json")
    rc1 = analyze_case(str(edf_cmp), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    c_cmp, r_cmp = rc1["case"], rc1["runs"][0]["recording"]
    t_q = np.arange(30000, dtype=np.float64) / 2000.0
    edfio.Edf([
        edfio.EdfSignal(8.0 * np.sin(2.0 * np.pi * 6.0 * t_q), sampling_frequency=2000, label=f"SYN_A{i}", physical_dimension="uV")
        for i in range(1, 7)
    ]).write(edf_cmp)
    analyze_case(str(edf_cmp), workspace_dir=ws, run_name="corrected", overrides={"detector_enabled": False})
    cmp_out = compare_runs(store, c_cmp, r_cmp, "corrected", "baseline")
    assert cmp_out["directly_comparable_rates"] is False, cmp_out
    assert "different recording snapshots" in cmp_out["reason"], cmp_out
    assert store.resolve_default_comparison_run(c_cmp, "corrected") is None
    with pytest.raises(ToolkitError) as exc_cmp_sh:
        export_case_sheet(c_cmp, workspace_dir=ws, primary_run="corrected", comparison_run="baseline")
    assert exc_cmp_sh.value.code == "run_recording_mismatch", exc_cmp_sh.value

    # (B) P1 #2 & P1 #3: Intact bipolar run MUST reject unipolar 'SYN_A1' query with channel_not_in_run,
    #     AND missing prepared.json (with NO coordinate change!) MUST make export_case_sheet fail with incomplete_run_artifacts!
    ws_bp = tmp_path / "ws_r14_bp"
    st_bp = WorkbenchStore(ws_bp)
    r_bp = analyze_case(str(SYNTHETIC_EDF), workspace_dir=ws_bp, run_name="bip", overrides={"reference_mode": "bipolar", "detector_enabled": False})
    c_bp, rec_bp = r_bp["case"], r_bp["runs"][0]["recording"]
    with pytest.raises(ToolkitError) as exc_uni:
        compute_triad_view(f"{c_bp}/{rec_bp}", workspace_dir=ws_bp, recording=rec_bp, run_name="bip", channel="SYN_A1", center_seconds=5.0)
    assert exc_uni.value.code == "channel_not_in_run", exc_uni.value
    # True bipolar pair succeeds
    tr_ok = compute_triad_view(f"{c_bp}/{rec_bp}", workspace_dir=ws_bp, recording=rec_bp, run_name="bip", channel="SYN_A1-SYN_A2", center_seconds=5.0)
    assert tr_ok["channel_name"] == "SYN_A1-SYN_A2"
    # Delete prepared.json WITHOUT changing coordinates -> export_case_sheet MUST still raise incomplete_run_artifacts!
    prep_p = Path(st_bp.load_run(c_bp, rec_bp, "bip")["meta"]["prepared_json_path"])
    prep_p.unlink()
    with pytest.raises(ToolkitError) as exc_no_prep:
        export_case_sheet(c_bp, workspace_dir=ws_bp, primary_run="bip")
    assert exc_no_prep.value.code == "incomplete_run_artifacts", exc_no_prep.value

    # (C) P1 #5: 17th spatial map for P-B MUST be discovered & accurately attributed in recording.json;
    #     subdir PROVENANCE.json alias conflict MUST raise conflicting_subjects;
    #     fuzzy P-A sidecar MUST NOT shadow P-B's exact sidecar!
    m17_dir = tmp_path / "maps_17"
    m17_dir.mkdir()
    for i in range(1, 17):
        (m17_dir / f"map_{i:02d}_A.json").write_text(
            json.dumps({"participant_id": "P-A", "spatial": {"coordinate_system": "MNI", "units": "mm", "source": f"A{i}", "positions": {"LA1": [float(i), 0.0, 0.0]}}}),
            encoding="utf-8",
        )
    (m17_dir / "map_17_B.json").write_text(
        json.dumps({"participant_id": "P-B", "spatial": {"coordinate_system": "MNI", "units": "mm", "source": "B 17th", "positions": {"LA1": [-42.0, 10.0, 20.0]}}}),
        encoding="utf-8",
    )
    info_17 = mne.create_info(["LA1", "LA2"], sfreq=1024.0, ch_types="seeg")
    info_17["subject_info"] = {"his_id": "P-B"}
    f17 = m17_dir / "b_native_raw.fif"
    mne.io.RawArray(np.random.default_rng(1405).normal(0, 20e-6, size=(2, 2048)), info_17, verbose="ERROR").save(f17, overwrite=True, verbose="ERROR")
    res_17 = inspect_case(str(f17), workspace_dir=ws, persist=True)
    rec17_doc = store.load_recording_doc(res_17["case"], res_17["recordings"][0]["recording"])
    assert tuple(rec17_doc["spec"]["spatial"]["positions"]["LA1"]) == (-42.0, 10.0, 20.0), rec17_doc["spec"]["spatial"]
    assert Path(rec17_doc["spatial_overlay_path"]).name == "map_17_B.json", rec17_doc["spatial_overlay_path"]

    prov_alias_dir = tmp_path / "alias_prov_conflict"
    (prov_alias_dir / "objects").mkdir(parents=True)
    (prov_alias_dir / "dir_a").mkdir()
    (prov_alias_dir / "dir_b").mkdir()
    blob_pa = _write_fif(prov_alias_dir / "objects" / "shared_raw.fif", ["LA1", "LA2"], 1024.0, 2048, seed=1406)
    os.symlink(blob_pa, prov_alias_dir / "dir_a" / "visit_raw.fif")
    os.symlink(blob_pa, prov_alias_dir / "dir_b" / "visit_raw.fif")
    (prov_alias_dir / "dir_a" / "PROVENANCE.json").write_text(json.dumps({"participant_id": "P-A"}), encoding="utf-8")
    (prov_alias_dir / "dir_b" / "PROVENANCE.json").write_text(json.dumps({"participant_id": "P-B"}), encoding="utf-8")
    with pytest.raises(ToolkitError) as exc_ap:
        inspect_case(str(prov_alias_dir), workspace_dir=ws, persist=False)
    assert exc_ap.value.code == "conflicting_subjects", exc_ap.value

    fuz_dir = tmp_path / "fuzzy_precedence"
    fuz_dir.mkdir()
    info_fb = mne.create_info(["LA1", "LA2"], sfreq=1024.0, ch_types="seeg")
    info_fb["subject_info"] = {"his_id": "P-B"}
    f_fuz_b = fuz_dir / "visit_beta_raw.fif"
    mne.io.RawArray(np.random.default_rng(1407).normal(0, 20e-6, size=(2, 2048)), info_fb, verbose="ERROR").save(f_fuz_b, overwrite=True, verbose="ERROR")
    (fuz_dir / "visit_braintoolkit.json").write_text(
        json.dumps({"participant_id": "P-A", "reference": "bipolar"}), encoding="utf-8"
    )
    (fuz_dir / "z_exact_b_braintoolkit.json").write_text(
        json.dumps({"source_file": "visit_beta_raw.fif", "participant_id": "P-B", "reference": "common"}), encoding="utf-8"
    )
    res_fuz_b = inspect_case(str(f_fuz_b), workspace_dir=ws, persist=True)
    rec_fuz_b = store.load_recording(res_fuz_b["case"], res_fuz_b["recordings"][0]["recording"])
    assert rec_fuz_b.reference == "common" and rec_fuz_b.provenance.participant_id == "P-B"

    # (D) P1 #6: Zero-duration point annotations MUST be preserved in continuous EDF & FIF!
    pt_dir = tmp_path / "point_markers"
    pt_dir.mkdir()
    edf_pt = pt_dir / "pt.edf"
    edfio.Edf(
        [edfio.EdfSignal(np.zeros(6000), sampling_frequency=1000, label="LA1", physical_dimension="uV")],
        annotations=[edfio.EdfAnnotation(2.0, None, "Push button"), edfio.EdfAnnotation(4.0, 0.2, "State interval")],
    ).write(edf_pt)
    fif_pt = pt_dir / "pt_raw.fif"
    raw_pt = mne.io.RawArray(np.zeros((2, 6000)), mne.create_info(["LA1", "LA2"], 1000.0, ch_types="seeg"), verbose="ERROR")
    raw_pt.set_annotations(mne.Annotations(onset=[2.0, 4.0], duration=[0.0, 0.2], description=["Push button", "State interval"]))
    raw_pt.save(fif_pt, overwrite=True, verbose="ERROR")
    for p_f in (edf_pt, fif_pt):
        r_pt = inspect_case(str(p_f), workspace_dir=ws, persist=True)
        rec_pt = store.load_recording(r_pt["case"], r_pt["recordings"][0]["recording"])
        lbls = {m.label: (round(m.start, 3), round(m.end, 3)) for m in rec_pt.markers}
        assert "Push button" in lbls and lbls["Push button"] == (2.0, 2.0), (p_f.name, lbls)
        assert "State interval" in lbls and lbls["State interval"] == (4.0, 4.2), (p_f.name, lbls)

    # (E) P2 #1: Sub-10ms distinct observations (4.151s & 4.154s) MUST BOTH render in MDT with distinct timestamps!
    ws_sub = tmp_path / "ws_r14_sub10ms"
    st_sub = WorkbenchStore(ws_sub)
    analyze_case(str(SYNTHETIC_EDF), workspace_dir=ws_sub, run_name="baseline", overrides={"detector_enabled": False})
    for idx, ts in enumerate((4.151, 4.154)):
        st_sub.record_judgment(
            "synthetic_seeg",
            {
                "kind": "interval_note",
                "operation_id": f"sub10_{idx}",
                "recording": "synthetic_seeg",
                "run_name": "baseline",
                "channel": "SYN_A1",
                "center_seconds": ts,
                "author": "Engineer",
                "role": "engineering_rehearsal",
                "note": "Same waveform morphology here",
            },
        )
    sh_sub = export_case_sheet("synthetic_seeg", workspace_dir=ws_sub, primary_run="baseline")
    html_sub = Path(sh_sub["report_path"]).read_text(encoding="utf-8")
    assert "@4.151s" in html_sub and "@4.154s" in html_sub, "Sub-10ms distinct notes were collapsed in MDT!"


def test_round15_p0_bipolar_fuzzy_leak_source_path_nested_spatial_edf_continuity_legacy_upgrade_and_bids_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edfio
    import mne

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r15_p0"))
    ws = tmp_path / "ws_r15_p0"
    store = WorkbenchStore(ws)

    # (A) P0 #1: Deferred bipolar resolution & explicit pointer validation
    # Patient A fuzzy sidecar + A leadmap MUST NOT hijack Patient B's bipolar leadmap (LA2-LA3 vs LA1-LA3)!
    fuz_dir = tmp_path / "r15_fuzzy_bp"
    fuz_dir.mkdir()
    rng = np.random.default_rng(1501)
    raw_mat = rng.normal(0.0, 20e-6, size=(3, 4096))
    info_a = mne.create_info(["LA1", "LA2", "LA3"], sfreq=1024.0, ch_types="seeg")
    info_a["subject_info"] = {"his_id": "P-A"}
    fa = fuz_dir / "rest_alpha_raw.fif"
    mne.io.RawArray(raw_mat, info_a, verbose="ERROR").save(fa, overwrite=True, verbose="ERROR")
    info_b = mne.create_info(["LA1", "LA2", "LA3"], sfreq=1024.0, ch_types="seeg")
    info_b["subject_info"] = {"his_id": "P-B"}
    fb = fuz_dir / "rest_visit_zulu_raw.fif"
    mne.io.RawArray(raw_mat, info_b, verbose="ERROR").save(fb, overwrite=True, verbose="ERROR")
    (fuz_dir / "P-A_leadmap.json").write_text(
        json.dumps({"participant_id": "P-A", "pairs": [{"name": "LA1-LA3", "anode": "LA1", "cathode": "LA3"}]}),
        encoding="utf-8",
    )
    (fuz_dir / "P-B_leadmap.json").write_text(
        json.dumps({"participant_id": "P-B", "pairs": [{"name": "LA2-LA3", "anode": "LA2", "cathode": "LA3"}]}),
        encoding="utf-8",
    )
    (fuz_dir / "rest_braintoolkit.json").write_text(
        json.dumps({
            "participant_id": "P-A",
            "reference": "common",
            "provenance": {"participant_id": "P-A", "published_bipolar_pairs_file": "P-A_leadmap.json"},
        }),
        encoding="utf-8",
    )
    (fuz_dir / "rest_visit_zulu_raw.json").write_text(
        json.dumps({
            "participant_id": "P-B",
            "reference": "common",
            "provenance": {"participant_id": "P-B", "published_bipolar_pairs_file": "P-B_leadmap.json"},
        }),
        encoding="utf-8",
    )
    res_fb = analyze_case(str(fb), workspace_dir=ws, run_name="bip", overrides={"reference_mode": "bipolar", "detector_enabled": False})
    b_fb = store.load_run(res_fb["case"], res_fb["runs"][0]["recording"], "bip")
    fb_chs = [c["channel_name"] for c in b_fb["summary"]["channels"]]
    assert fb_chs == ["LA2-LA3"], f"Patient B hijacked by Patient A fuzzy leadmap: {fb_chs}"

    # Explicit published_bipolar_pairs_file pointing to Patient A's leadmap on Patient B MUST raise conflicting_subjects!
    ptr_dir = tmp_path / "r15_bad_ptr"
    ptr_dir.mkdir()
    fb_ptr = ptr_dir / "visit_zulu_raw.fif"
    mne.io.RawArray(raw_mat, info_b, verbose="ERROR").save(fb_ptr, overwrite=True, verbose="ERROR")
    (ptr_dir / "P-A_leadmap.json").write_text(
        json.dumps({"participant_id": "P-A", "pairs": [{"name": "LA1-LA3", "anode": "LA1", "cathode": "LA3"}]}),
        encoding="utf-8",
    )
    (ptr_dir / "visit_zulu_raw.json").write_text(
        json.dumps({
            "participant_id": "P-B",
            "reference": "common",
            "provenance": {"participant_id": "P-B", "published_bipolar_pairs_file": "P-A_leadmap.json"},
        }),
        encoding="utf-8",
    )
    with pytest.raises(ToolkitError) as exc_ptr:
        inspect_case(str(fb_ptr), workspace_dir=ws, persist=False)
    assert exc_ptr.value.code == "conflicting_subjects", exc_ptr.value

    # (B) P0 #2: Relative source_file pointing to an existing recording for A MUST NOT suffix-match B,
    #     AND symlink aliases to the same backing object MUST NOT cross-match A's absolute/relative source_file to B!
    rel_root = tmp_path / "r15_rel_path"
    dir_ra = rel_root / "dir_a"
    dir_rb = rel_root / "dir_b"
    dir_ra.mkdir(parents=True)
    dir_rb.mkdir(parents=True)
    _write_fif(dir_ra / "visit_raw.fif", ["LA1", "LA2", "LA3"], 1024.0, 2048, seed=1502)
    fb_rel = _write_fif(dir_rb / "visit_raw.fif", ["LA1", "LA2", "LA3"], 1024.0, 2048, seed=1503)
    (dir_rb / "shared_bipolar_pairs.json").write_text(
        json.dumps({"source_file": "../dir_a/visit_raw.fif", "pairs": [{"name": "LA1-LA3", "anode": "LA1", "cathode": "LA3"}]}),
        encoding="utf-8",
    )
    res_rel = inspect_case(str(fb_rel), workspace_dir=ws, persist=True)
    rec_rel = store.load_recording(res_rel["case"], res_rel["recordings"][0]["recording"])
    assert rec_rel.published_bipolar_pairs_path is None, rec_rel.published_bipolar_pairs_path

    alias_root = tmp_path / "r15_alias_src"
    (alias_root / "objects").mkdir(parents=True)
    (alias_root / "site_a").mkdir()
    (alias_root / "site_b").mkdir()
    shared_blob = _write_fif(alias_root / "objects" / "shared_raw.fif", ["LA1", "LA2", "LA3"], 1024.0, 2048, seed=1504)
    os.symlink(shared_blob, alias_root / "site_a" / "visit_raw.fif")
    os.symlink(shared_blob, alias_root / "site_b" / "visit_raw.fif")
    (alias_root / "site_a" / "PROVENANCE.json").write_text(json.dumps({"participant_id": "P-A"}), encoding="utf-8")
    (alias_root / "site_b" / "PROVENANCE.json").write_text(json.dumps({"participant_id": "P-B"}), encoding="utf-8")
    for sf_val in (str((alias_root / "site_a" / "visit_raw.fif").absolute()), "../site_a/visit_raw.fif"):
        (alias_root / "site_b" / "shared_bipolar_pairs.json").write_text(
            json.dumps({"source_file": sf_val, "pairs": [{"name": "LA1-LA3", "anode": "LA1", "cathode": "LA3"}]}),
            encoding="utf-8",
        )
        res_al = inspect_case(str(alias_root / "site_b" / "visit_raw.fif"), workspace_dir=ws, persist=False)
        rec_al = res_al["_recording_specs"][res_al["recordings"][0]["recording"]]
        assert rec_al.published_bipolar_pairs_path is None, (sf_val, rec_al.published_bipolar_pairs_path)

    # (C) P0 #3: Self-contradictory nested spatial map (outer P-B, inner spatial.participant_id=P-A) MUST be rejected!
    nest_dir = tmp_path / "r15_nested_sp"
    nest_dir.mkdir()
    f_nb = nest_dir / "visit_raw.fif"
    mne.io.RawArray(raw_mat, info_b, verbose="ERROR").save(f_nb, overwrite=True, verbose="ERROR")
    bad_map = nest_dir / "a_contradictory_coords.json"
    bad_map.write_text(
        json.dumps({
            "participant_id": "P-B",
            "spatial": {
                "participant_id": "P-A",
                "coordinate_system": "MNI",
                "units": "mm",
                "source": "P-A coords",
                "positions": {"LA1": [42.0, 20.0, 30.0]},
            },
        }),
        encoding="utf-8",
    )
    (nest_dir / "z_consistent_b_coords.json").write_text(
        json.dumps({
            "participant_id": "P-B",
            "spatial": {
                "participant_id": "P-B",
                "coordinate_system": "MNI",
                "units": "mm",
                "source": "P-B coords",
                "positions": {"LA1": [-42.0, 20.0, 30.0]},
            },
        }),
        encoding="utf-8",
    )
    res_nb = inspect_case(str(f_nb), workspace_dir=ws, persist=True)
    rec_nb = store.load_recording(res_nb["case"], res_nb["recordings"][0]["recording"])
    assert rec_nb.spatial is not None and tuple(rec_nb.spatial.positions["LA1"]) == (-42.0, 20.0, 30.0), rec_nb.spatial
    with pytest.raises(ToolkitError) as exc_nest:
        inspect_case(str(f_nb), workspace_dir=ws, spatial_sidecar=bad_map, persist=False)
    assert exc_nest.value.code == "conflicting_subjects", exc_nest.value

    # (D) P0 #4: Tight TAL continuity check on ALL records for EDF+C & BDF+C (30s 1s gap, 1s 40ms gap/overlap)
    def _write_tal_edf_or_bdf(dest: Path, *, is_bdf: bool, rec_dur: float, onsets: list[float], spr_sig: int = 256) -> None:
        bps = 3 if is_bdf else 2
        ns = 2
        n_rec = len(onsets)
        spr_tal = 24
        hdr_bytes = 256 + ns * 256
        fixed = bytearray(b" " * 256)
        if is_bdf:
            fixed[0:8] = b"\xffBIOSEMI"
            fixed[192:197] = b"BDF+C"
        else:
            fixed[0:8] = b"0       "
            fixed[192:197] = b"EDF+C"
        fixed[168:176] = b"01.01.20"
        fixed[176:184] = b"00.00.00"
        fixed[184:192] = f"{hdr_bytes:<8d}".encode("ascii")
        fixed[236:244] = f"{n_rec:<8d}".encode("ascii")
        fixed[244:252] = f"{rec_dur:<8g}".encode("ascii")[:8].ljust(8, b" ")
        fixed[252:256] = f"{ns:<4d}".encode("ascii")
        ann_lbl = "BDF Annotations" if is_bdf else "EDF Annotations"
        shdr = bytearray()
        for vals, w in [
            (["LA1", ann_lbl], 16),
            (["", ""], 80),
            (["uV", ""], 8),
            (["-1000", "-1"], 8),
            (["1000", "1"], 8),
            (["-8388608" if is_bdf else "-32768", "-8388608" if is_bdf else "-32768"], 8),
            (["8388607" if is_bdf else "32767", "8388607" if is_bdf else "32767"], 8),
            (["", ""], 80),
            ([str(spr_sig), str(spr_tal)], 8),
            (["", ""], 32),
        ]:
            for v in vals:
                shdr.extend(f"{v:<{w}s}"[:w].encode("ascii"))
        body = bytearray()
        for t0_rec in onsets:
            body.extend(b"\x00" * (spr_sig * bps))
            tal_b = f"+{t0_rec:g}\x14\x14".encode("ascii")
            body.extend((tal_b + b"\x00" * (spr_tal * bps - len(tal_b)))[: spr_tal * bps])
        dest.write_bytes(bytes(fixed) + bytes(shdr) + bytes(body))

    for fname, is_b, rdur, onsets_bad in (
        ("gap30.edf", False, 30.0, [0.125, 31.125, 61.125]),
        ("gap30.bdf", True, 30.0, [0.125, 31.125, 61.125]),
        ("gap40ms.edf", False, 1.0, [0.0, 1.04, 2.04]),
        ("ovlp40ms.edf", False, 1.0, [0.0, 0.96, 1.96]),
    ):
        p_bad = tmp_path / fname
        _write_tal_edf_or_bdf(p_bad, is_bdf=is_b, rec_dur=rdur, onsets=onsets_bad)
        if fname.startswith("ovlp"):
            with pytest.raises(ToolkitError) as exc_tal:
                inspect_case(str(p_bad), workspace_dir=ws, persist=False)
            assert exc_tal.value.code == "discontinuous_recording_unsupported"
        else:
            inspected = inspect_case(str(p_bad), workspace_dir=ws, persist=False)
            rec = next(iter(inspected["_recording_specs"].values()))
            assert np.allclose(rec.gaps(), [(rdur, onsets_bad[1] - onsets_bad[0])])
            assert rec.duration == pytest.approx(onsets_bad[-1] - onsets_bad[0] + rdur)

    p_cont = tmp_path / "cont30.edf"
    _write_tal_edf_or_bdf(p_cont, is_bdf=False, rec_dur=30.0, onsets=[0.125, 30.125, 60.125])
    res_cont = inspect_case(str(p_cont), workspace_dir=ws, persist=False)
    assert res_cont["recordings"][0]["duration_seconds"] == pytest.approx(90.0, abs=0.1)

    # (E) P0 #5: Legacy WAL upgrade (no recording_fingerprint) MUST preserve clip when unchanged,
    #     AND exclude it after recording replacement + recompute while keeping historical WAL intact!
    ws_leg = tmp_path / "ws_r15_legacy_wal"
    st_leg = WorkbenchStore(ws_leg)
    src_leg = tmp_path / "src_r15_leg"
    src_leg.mkdir()
    edf_leg = src_leg / "synthetic_seeg.edf"
    shutil.copy2(SYNTHETIC_EDF, edf_leg)
    shutil.copy2(SYNTHETIC_EDF.with_suffix(".json"), src_leg / "synthetic_seeg.json")
    r_leg = analyze_case(str(edf_leg), workspace_dir=ws_leg, run_name="baseline", overrides={"detector_enabled": False})
    c_leg, rec_leg = r_leg["case"], r_leg["runs"][0]["recording"]
    # Simulate a legacy WAL row written by an older wheel (lacking recording_fingerprint & analysis_signature)
    leg_wal_row = {
        "id": "jdg_0001",
        "case": c_leg,
        "created_at": "2026-01-01T00:00:00+00:00",
        "kind": "interval_note",
        "operation_id": "legacy_wheel_clip_1",
        "recording": rec_leg,
        "run_name": "baseline",
        "channel": "SYN_A1",
        "event_id": None,
        "center_seconds": 8.1,
        "interval_seconds": [8.0, 8.2],
        "verdict": None,
        "clipped": True,
        "author": "Engineer",
        "role": "engineering_rehearsal",
        "note": "Legacy v14 burst observation",
    }
    st_leg.judgments_path(c_leg).write_text(json.dumps(leg_wal_row) + "\n", encoding="utf-8")

    # Normal upgrade (unchanged source): legacy clip & note MUST be preserved!
    assert len(st_leg.active_clipped_targets(c_leg, default_run="baseline")) == 1
    sh_leg1 = export_case_sheet(c_leg, workspace_dir=ws_leg, primary_run="baseline")
    html_leg1 = Path(sh_leg1["report_path"]).read_text(encoding="utf-8")
    assert "Legacy v14 burst observation" in html_leg1 and "Rehearsal-clipped evidence" in html_leg1

    # Replace source EDF (40s -> 5s quiet sine) and re-inspect!
    t_5s = np.arange(5 * 2000, dtype=np.float64) / 2000.0
    edfio.Edf([
        edfio.EdfSignal(10.0 * np.sin(2.0 * np.pi * 8.0 * t_5s), sampling_frequency=2000, label=f"SYN_A{i}", physical_dimension="uV")
        for i in range(1, 7)
    ]).write(edf_leg)
    inspect_case(str(edf_leg), workspace_dir=ws_leg, persist=True)
    rdoc_leg = st_leg.load_recording_doc(c_leg, rec_leg)
    assert int(rdoc_leg.get("replaced_at_wal_index") or 0) == 1, rdoc_leg

    # (F) P0 #6: export_bids_events on stale 40s baseline run after 5s replacement MUST raise run_recording_mismatch!
    with pytest.raises(ToolkitError) as exc_bids_stale:
        export_bids_events(c_leg, recording=rec_leg, run_name="baseline", workspace_dir=ws_leg)
    assert exc_bids_stale.value.code == "run_recording_mismatch", exc_bids_stale.value

    # Overwrite baseline on 5s recording: legacy clip & note MUST drop from active clips & fresh MDT, while WAL keeps row!
    analyze_case(str(edf_leg), workspace_dir=ws_leg, run_name="baseline", overwrite=True, overrides={"detector_enabled": False})
    assert len(st_leg.load_judgments(c_leg)) == 1
    assert st_leg.active_clipped_targets(c_leg, default_run="baseline") == []
    sh_leg2 = export_case_sheet(c_leg, workspace_dir=ws_leg, primary_run="baseline")
    html_leg2 = Path(sh_leg2["report_path"]).read_text(encoding="utf-8")
    assert "Legacy v14 burst observation" not in html_leg2
    assert "Rehearsal-clipped evidence" not in html_leg2

    # Matching export_bids_events succeeds & records RecordingFingerprint + AnalysisGenerationId in sidecar JSON!
    bids_ok = export_bids_events(c_leg, recording=rec_leg, run_name="baseline", workspace_dir=ws_leg)
    bids_json = read_json(Path(bids_ok["json_path"]))
    assert bids_json.get("RecordingFingerprint") == st_leg.load_recording(c_leg, rec_leg).fingerprint
    assert bids_json.get("AnalysisGenerationId")


def test_round15_p1_overwrite_clip_rebind_409_mcp_atomic_mixed_bipolar_pagination_and_bids_symlink_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edfio
    import mne
    import brainlens.mcp_server as mcp_mod

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust_r15_p1"))
    ws = tmp_path / "ws_r15_p1"
    store = WorkbenchStore(ws)

    # (A) P1 #1: Same-recording original -> CAR overwrite MUST drop prior clip from active clips & MDT
    #     AND reject delayed HTTP POST carrying old analysis_generation_id with HTTP 409 stale_run_generation!
    r_orig = analyze_case(str(SYNTHETIC_EDF), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    c_ov, rec_ov = r_orig["case"], r_orig["runs"][0]["recording"]
    b_orig = store.load_run(c_ov, rec_ov, "baseline")
    old_asig = b_orig["meta"]["analysis_signature"]
    old_agen = b_orig["meta"]["analysis_generation_id"]
    old_fp = store.load_recording(c_ov, rec_ov).fingerprint

    store.record_judgment(
        c_ov,
        {
            "kind": "interval_note",
            "operation_id": "r15_before_car",
            "recording": rec_ov,
            "run_name": "baseline",
            "channel": "SYN_A1",
            "center_seconds": 8.1,
            "interval_seconds": [8.0, 8.2],
            "clipped": True,
            "author": "Engineer",
            "role": "engineering_rehearsal",
            "note": "Original montage 207 uV burst",
        },
    )
    assert len(store.active_clipped_targets(c_ov, default_run="baseline")) == 1

    # Overwrite SAME baseline run on SAME recording with CAR (average) montage!
    analyze_case(
        f"{c_ov}/{rec_ov}",
        workspace_dir=ws,
        run_name="baseline",
        overwrite=True,
        overrides={"reference_mode": "average", "detector_enabled": False},
    )
    assert store.active_clipped_targets(c_ov, default_run="baseline") == []
    sh_car = export_case_sheet(c_ov, workspace_dir=ws, primary_run="baseline")
    html_car = Path(sh_car["report_path"]).read_text(encoding="utf-8")
    assert "Original montage 207 uV burst" not in html_car
    assert "Rehearsal-clipped evidence" not in html_car

    srv = WorkbenchHTTPServer(workspace_dir=ws, host="127.0.0.1", port=0)
    url = srv.start().rstrip("/")
    try:
        req = urllib.request.Request(
            f"{url}/api/judgment",
            data=json.dumps({
                "case": c_ov,
                "recording": rec_ov,
                "recording_fingerprint": old_fp,
                "analysis_signature": old_asig,
                "analysis_generation_id": old_agen,
                "run_name": "baseline",
                "channel": "SYN_A1",
                "center_seconds": 8.1,
                "interval_seconds": [8.0, 8.2],
                "clipped": True,
                "author": "Engineer",
                "role": "engineering_rehearsal",
                "note": "Delayed original-tab clip",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-BrainLens-Token": srv.auth_token},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as http_e:
            urllib.request.urlopen(req, timeout=5)
        assert http_e.value.code == 409, http_e.value.code
        err_j = json.loads(http_e.value.read().decode("utf-8"))
        assert err_j["error"]["code"] == "stale_run_generation", err_j
        assert len(store.load_judgments(c_ov)) == 1
    finally:
        srv.stop()

    # (B) P1 #2: Composite MCP record_judgment MUST be all-or-nothing (zero partial question or SOZ overlay on error)!
    ws_mcp = tmp_path / "ws_r15_mcp_atomic"
    st_mcp = WorkbenchStore(ws_mcp)
    monkeypatch.setattr(mcp_mod, "_WORKSPACE_ROOT", ws_mcp.resolve())
    src_m = tmp_path / "src_r15_mcp"
    src_m.mkdir()
    edf_m = src_m / "synthetic_seeg.edf"
    shutil.copy2(SYNTHETIC_EDF, edf_m)
    shutil.copy2(SYNTHETIC_EDF.with_suffix(".json"), src_m / "synthetic_seeg.json")
    r_m = analyze_case(str(edf_m), workspace_dir=ws_mcp, run_name="baseline", overrides={"detector_enabled": False})
    cm, recm = r_m["case"], r_m["runs"][0]["recording"]
    # Seed 1 event into baseline so event_id is real before replacement
    b_m = st_mcp.load_run(cm, recm, "baseline", reconcile=False)
    gdir_m = Path(b_m["generation_dir"])
    ev_seed = {
        "id": "ev_r15_001",
        "channel_id": "ch0000",
        "channel_name": "SYN_A1",
        "shaft": "SYN_A",
        "start_sample": 16000,
        "end_sample": 16160,
        "start_seconds": 8.0,
        "end_seconds": 8.08,
        "duration_ms": 80.0,
        "peak_to_peak_uv": 120.0,
        "peak_freq_hz": 180.0,
        "dominant_band": "ripple",
        "synchrony_class": "focal",
        "coactive_shaft_count": 1,
        "coactive_channel_count": 1,
    }
    det_p_m = Path(b_m["meta"]["detection_json_path"])
    det_doc_m = read_json(det_p_m)
    det_doc_m["events"] = [ev_seed]
    atomic_write_json(det_p_m, det_doc_m)
    clf_p_m = Path(b_m["meta"]["classification_json_path"])
    atomic_write_json(clf_p_m, {**det_doc_m, "classifier_status": "skipped", "classification": {"requested": [], "stages": {}}})
    refresh_verdict_tallies(st_mcp, cm, recm, "baseline")
    ev_id_m = st_mcp.load_run(cm, recm, "baseline")["events"][0]["id"]

    # First check invalid overlay (missing source) combined with valid verdict -> ZERO WAL write!
    res_bad_ov = mcp_mod.record_judgment(
        case=cm,
        recording=recm,
        run_name="baseline",
        event_id=ev_id_m,
        verdict="confirm",
        clinical_question="Should not persist on invalid overlay?",
        clinical_overlay={"soz_channels": ["SYN_A1"]},
        author="round15",
    )
    assert res_bad_ov.get("error") is True and res_bad_ov.get("code") == "missing_overlay_source", res_bad_ov
    assert st_mcp.load_judgments(cm) == []
    assert st_mcp.load_case(cm).get("open_questions", []) == []
    assert st_mcp.load_clinical_overlay(cm) is None

    # Now replace source recording & re-inspect -> composite MCP call with valid overlay + question + stale event MUST fail atomically!
    t_rep = np.arange(5 * 2000, dtype=np.float64) / 2000.0
    edfio.Edf([
        edfio.EdfSignal(6.0 * np.sin(2.0 * np.pi * 6.0 * t_rep), sampling_frequency=2000, label=f"SYN_A{i}", physical_dimension="uV")
        for i in range(1, 7)
    ]).write(edf_m)
    inspect_case(str(edf_m), workspace_dir=ws_mcp, persist=True)
    res_stale_mcp = mcp_mod.record_judgment(
        case=cm,
        recording=recm,
        run_name="baseline",
        event_id=ev_id_m,
        verdict="confirm",
        clinical_question="Re-assess original discharge?",
        clinical_overlay={"source": "Pre-replacement review", "soz_channels": ["SYN_A1"]},
        author="round15",
    )
    assert res_stale_mcp.get("error") is True and res_stale_mcp.get("code") == "run_recording_mismatch", res_stale_mcp
    assert st_mcp.load_judgments(cm) == [], "WAL mutated on failed MCP call!"
    assert st_mcp.load_case(cm).get("open_questions", []) == [], f"open_questions mutated on failed MCP call: {st_mcp.load_case(cm)}"
    assert st_mcp.load_clinical_overlay(cm) is None, f"clinical_overlay mutated on failed MCP call: {st_mcp.load_clinical_overlay(cm)}"

    # (C) P1 #3: Per-trace & adjacent-bipolar Fast Ripple bandwidth MUST be 100% pagination-invariant!
    edf_mx = tmp_path / "r15_mixed_17.edf"
    t1024 = np.arange(4 * 1024, dtype=np.float64) / 1024.0
    t256 = np.arange(4 * 256, dtype=np.float64) / 256.0
    sigs_17 = [
        edfio.EdfSignal(
            40.0 * np.sin(2.0 * np.pi * 320.0 * t1024) + 30.0 * np.sin(2.0 * np.pi * 12.0 * t1024),
            sampling_frequency=1024,
            label=f"LA{i}",
            physical_dimension="uV",
        )
        for i in range(1, 17)
    ] + [
        edfio.EdfSignal(
            80.0 * np.sin(2.0 * np.pi * 10.0 * t256),
            sampling_frequency=256,
            label="LA17",
            physical_dimension="uV",
        )
    ]
    edfio.Edf(sigs_17).write(edf_mx)
    reg_mx = inspect_case(str(edf_mx), workspace_dir=ws, persist=True)
    cmx, rmx = reg_mx["case"], reg_mx["recordings"][0]["recording"]
    mc_p16 = compute_multichannel_batch(
        store, cmx, rmx, start_seconds=0.5, duration_seconds=2.0, page=0, page_size=16, montage="adjacent_bipolar", display_filter="fast_ripple_250_450"
    )
    mc_all = compute_multichannel_batch(
        store, cmx, rmx, start_seconds=0.5, duration_seconds=2.0, shaft="ALL", montage="adjacent_bipolar", display_filter="fast_ripple_250_450"
    )
    t16_p16 = next(t for t in mc_p16["traces"] if t["display_label"] == "LA16-LA17")
    t16_all = next(t for t in mc_all["traces"] if t["display_label"] == "LA16-LA17")
    assert t16_p16["display_filter"] == "broadband" and t16_p16["display_filter_available"] is False
    assert t16_all["display_filter"] == "broadband" and t16_all["display_filter_available"] is False
    assert t16_p16["values_uv"] == t16_all["values_uv"], "LA16-LA17 waveform changed between default page and ALL!"
    t1_p16 = next(t for t in mc_p16["traces"] if t["display_label"] == "LA1-LA2")
    t1_all = next(t for t in mc_all["traces"] if t["display_label"] == "LA1-LA2")
    assert t1_p16["display_filter"] == "fast_ripple_250_450" and t1_p16["display_filter_available"] is True
    assert t1_p16["values_uv"] == t1_all["values_uv"]

    # Also test acquired montage pagination invariance (page_size=1 vs page_size=2 with LA1=1024Hz & LA2=256Hz)
    edf_p2 = tmp_path / "r15_mixed_2.edf"
    edfio.Edf([
        edfio.EdfSignal(40.0 * np.sin(2.0 * np.pi * 320.0 * t1024), sampling_frequency=1024, label="LA1", physical_dimension="uV"),
        edfio.EdfSignal(40.0 * np.sin(2.0 * np.pi * 10.0 * t256), sampling_frequency=256, label="LA2", physical_dimension="uV"),
    ]).write(edf_p2)
    reg_p2 = inspect_case(str(edf_p2), workspace_dir=ws, persist=True)
    cp2, rp2 = reg_p2["case"], reg_p2["recordings"][0]["recording"]
    mc_sz1 = compute_multichannel_batch(store, cp2, rp2, start_seconds=0.5, duration_seconds=2.0, page=0, page_size=1, display_filter="fast_ripple_250_450")
    mc_sz2 = compute_multichannel_batch(store, cp2, rp2, start_seconds=0.5, duration_seconds=2.0, page=0, page_size=2, display_filter="fast_ripple_250_450")
    assert mc_sz1["traces"][0]["display_filter"] == "fast_ripple_250_450" and mc_sz1["traces"][0]["display_filter_available"] is True
    assert mc_sz2["traces"][0]["display_filter"] == "fast_ripple_250_450" and mc_sz2["traces"][0]["display_filter_available"] is True
    assert mc_sz2["traces"][1]["display_filter"] == "broadband" and mc_sz2["traces"][1]["display_filter_available"] is False
    assert mc_sz1["traces"][0]["values_uv"] == mc_sz2["traces"][0]["values_uv"]

    # (D) P1 #5: Directory symlink sub-B/ieeg -> sub-A/ieeg MUST raise conflicting_subjects,
    #     AND exact sidecar with archival provenance.source_file declaring sub-A MUST NOT bind to sub-B!
    bids_sym = tmp_path / "r15_bids_dir_sym"
    real_a_ieeg = bids_sym / "sub-A" / "ieeg"
    real_a_ieeg.mkdir(parents=True)
    info_sa = mne.create_info(["LA1", "LA2"], sfreq=1024.0, ch_types="seeg")
    info_sa["subject_info"] = {"his_id": "sub-A"}
    mne.io.RawArray(np.zeros((2, 2048)), info_sa, verbose="ERROR").save(
        real_a_ieeg / "sub-A_task-rest_ieeg.fif", overwrite=True, verbose="ERROR"
    )
    (bids_sym / "sub-B").mkdir()
    os.symlink(real_a_ieeg, bids_sym / "sub-B" / "ieeg")
    with pytest.raises(ToolkitError) as exc_dirsym:
        inspect_case(str(bids_sym / "sub-B" / "ieeg" / "sub-A_task-rest_ieeg.fif"), workspace_dir=ws, persist=False)
    assert exc_dirsym.value.code == "conflicting_subjects", exc_dirsym.value
    with pytest.raises(ToolkitError) as exc_dirsym_an:
        analyze_case(str(bids_sym / "sub-B" / "ieeg" / "sub-A_task-rest_ieeg.fif"), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    assert exc_dirsym_an.value.code == "conflicting_subjects", exc_dirsym_an.value
    with pytest.raises(ToolkitError) as exc_dirsym_dir:
        inspect_case(str(bids_sym / "sub-B" / "ieeg"), workspace_dir=ws, persist=False)
    assert exc_dirsym_dir.value.code == "conflicting_subjects", exc_dirsym_dir.value

    arch_dir = tmp_path / "r15_archival_sub"
    arch_dir.mkdir()
    info_sb = mne.create_info(["LA1", "LA2"], sfreq=1024.0, ch_types="seeg")
    info_sb["subject_info"] = {"his_id": "sub-B"}
    f_sb = arch_dir / "sub-B_task-local_ieeg.fif"
    mne.io.RawArray(np.zeros((2, 2048)), info_sb, verbose="ERROR").save(f_sb, overwrite=True, verbose="ERROR")
    (arch_dir / "sub-B_task-local_ieeg.json").write_text(
        json.dumps({
            "provenance": {"source_file": "sub-A_ses-01_task-interictal_ieeg.fif"},
            "reference": "common",
            "spatial": {"coordinate_system": "MNI", "units": "mm", "source": "sub-A archive", "positions": {"LA1": [42.0, 20.0, 30.0]}},
        }),
        encoding="utf-8",
    )
    res_arch = inspect_case(str(f_sb), workspace_dir=ws, persist=True)
    rec_arch = store.load_recording(res_arch["case"], res_arch["recordings"][0]["recording"])
    assert rec_arch.spatial is None, f"Archival sub-A sidecar polluted sub-B recording: {rec_arch.spatial}"


def test_round16_cleanup_live_projection_registered_preview_and_mcp_idempotency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify live verdict projection floor, post-overwrite re-confirmation, registered persist=False preview, and MCP overlay atomicity/idempotency."""
    import mne
    import brainlens.mcp_server as mcp_mod

    ws = tmp_path / "ws_r16_clean"
    store = WorkbenchStore(ws)
    monkeypatch.setattr(mcp_mod, "_WORKSPACE_ROOT", ws.resolve())

    # (1) Online MDT projection & re-confirmation after analysis threshold overwrite
    analyze_case(str(SYNTHETIC_EDF), workspace_dir=ws, run_name="baseline")
    b0 = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline")
    ev0 = b0["events"][0]
    ev1 = b0["events"][1]

    r1 = mcp_mod.record_judgment(
        case="synthetic_seeg",
        recording="synthetic_seeg",
        run_name="baseline",
        channel=ev0["channel_name"],
        event_id=ev0["id"],
        verdict="confirm",
        note="Verified focal ripple",
        author="dr_a",
        role="clinician",
    )
    assert not r1.get("error"), r1
    j1_id = r1["saved_judgment"]["id"]
    proj1 = store.case_verdicts_projection("synthetic_seeg")
    assert proj1["verdicts"][f"synthetic_seeg|baseline|{ev0['id']}"]["verdict"] == "confirm"
    assert proj1["review_tallies"]["synthetic_seeg|baseline"]["clinician_confirmed"] == 1

    # Overwrite baseline with modified artifact threshold (same event IDs, new analysis_signature & WAL floor)
    analyze_case(
        "synthetic_seeg/synthetic_seeg",
        workspace_dir=ws,
        run_name="baseline",
        overrides={"classifier_thresholds": {"artifact": 0.65}},
        overwrite=True,
    )
    proj2 = store.case_verdicts_projection("synthetic_seeg")
    assert f"synthetic_seeg|baseline|{ev0['id']}" not in proj2["verdicts"], (
        f"Superseded verdict leaked into live projection: {proj2['verdicts']}"
    )
    assert proj2["review_tallies"]["synthetic_seeg|baseline"]["clinician_confirmed"] == 0

    # Re-confirming same event with same text and NO operation_id MUST append a new judgment (never dedup against invalidated j1)
    r2 = mcp_mod.record_judgment(
        case="synthetic_seeg",
        recording="synthetic_seeg",
        run_name="baseline",
        channel=ev0["channel_name"],
        event_id=ev0["id"],
        verdict="confirm",
        note="Verified focal ripple",
        author="dr_a",
        role="clinician",
    )
    assert not r2.get("error"), r2
    assert r2["saved_judgment"]["id"] != j1_id
    proj3 = store.case_verdicts_projection("synthetic_seeg")
    assert proj3["verdicts"][f"synthetic_seeg|baseline|{ev0['id']}"]["verdict"] == "confirm"
    assert proj3["review_tallies"]["synthetic_seeg|baseline"]["clinician_confirmed"] == 1

    # (2) MCP combined invalid overlay field MUST reject upfront with zero WAL/question/overlay side effects,
    #     and replaying an earlier operation_id MUST NOT revert a newer SOZ overlay!
    wal_before_err = len(store.load_judgments("synthetic_seeg"))
    r_bad = mcp_mod.record_judgment(
        case="synthetic_seeg",
        recording="synthetic_seeg",
        run_name="baseline",
        channel=ev1["channel_name"],
        event_id=ev1["id"],
        verdict="artifact",
        clinical_question="Should never persist on invalid overlay",
        clinical_overlay={"source": "MDT", "soz_channels": 12345},
        author="dr_a",
        role="clinician",
    )
    assert r_bad.get("error") is True and r_bad.get("code") == "invalid_overlay", r_bad
    assert len(store.load_judgments("synthetic_seeg")) == wal_before_err
    assert "Should never persist on invalid overlay" not in (store.load_case("synthetic_seeg").get("open_questions") or [])
    assert store.load_clinical_overlay("synthetic_seeg") is None

    r_ok1 = mcp_mod.record_judgment(
        case="synthetic_seeg",
        recording="synthetic_seeg",
        run_name="baseline",
        channel=ev1["channel_name"],
        event_id=ev1["id"],
        verdict="artifact",
        clinical_overlay={"source": "MDT", "soz_channels": ["SYN_A1"]},
        operation_id="op_r16_combined_1",
        author="dr_a",
        role="clinician",
    )
    assert not r_ok1.get("error"), r_ok1
    assert store.load_clinical_overlay("synthetic_seeg")["soz_channels"] == ["SYN_A1"]

    r_ok2 = mcp_mod.record_judgment(
        case="synthetic_seeg",
        clinical_overlay={"source": "MDT revised", "soz_channels": ["SYN_A5"]},
        operation_id="op_r16_overlay_2",
        author="dr_a",
        role="clinician",
    )
    assert not r_ok2.get("error"), r_ok2
    assert store.load_clinical_overlay("synthetic_seeg")["soz_channels"] == ["SYN_A5"]

    wal_before_retry = len(store.load_judgments("synthetic_seeg"))
    r_retry = mcp_mod.record_judgment(
        case="synthetic_seeg",
        recording="synthetic_seeg",
        run_name="baseline",
        channel=ev1["channel_name"],
        event_id=ev1["id"],
        verdict="artifact",
        clinical_overlay={"source": "MDT", "soz_channels": ["SYN_A1"]},
        operation_id="op_r16_combined_1",
        author="dr_a",
        role="clinician",
    )
    assert not r_retry.get("error"), r_retry
    assert len(store.load_judgments("synthetic_seeg")) == wal_before_retry
    assert store.load_clinical_overlay("synthetic_seeg")["soz_channels"] == ["SYN_A5"]

    # (3) Registered case/recording preview with spatial_sidecar and persist=False MUST NOT mutate disk or bump generation!
    sp_dir = tmp_path / "r16_reg_preview"
    sp_dir.mkdir()
    fif_p = sp_dir / "visit_raw.fif"
    info_b = mne.create_info(["LA1", "LA2"], sfreq=512.0, ch_types="seeg")
    info_b["subject_info"] = {"his_id": "P-B"}
    mne.io.RawArray(np.zeros((2, 1024)), info_b, verbose="ERROR").save(fif_p, overwrite=True, verbose="ERROR")
    sp_orig = sp_dir / "orig_coords.json"
    sp_orig.write_text(
        json.dumps({"spatial": {"coordinate_system": "MNI", "units": "mm", "source": "B original", "positions": {"LA1": [-42.0, 20.0, 30.0]}}}),
        encoding="utf-8",
    )
    sp_rev = tmp_path / "rev_coords.json"
    sp_rev.write_text(
        json.dumps({"spatial": {"coordinate_system": "MNI", "units": "mm", "source": "B revised", "positions": {"LA1": [-34.0, 20.0, 30.0]}}}),
        encoding="utf-8",
    )
    reg_b = inspect_case(str(fif_p), workspace_dir=ws, spatial_sidecar=sp_orig, persist=True)
    cb, rb = reg_b["case"], reg_b["recordings"][0]["recording"]
    analyze_case(f"{cb}/{rb}", workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})
    gen_before = store.load_run(cb, rb, "baseline")["meta"]["generation_id"]

    for target_ref in (cb, f"{cb}/{rb}"):
        prev_out = inspect_case(target_ref, workspace_dir=ws, spatial_sidecar=sp_rev, persist=False)
        assert prev_out["recordings"][0]["spatial"]["source"] == "B revised"
        assert list(prev_out["_recording_specs"][rb].spatial.positions["LA1"]) == [-34.0, 20.0, 30.0]
        on_disk_rec = store.load_recording(cb, rb)
        assert on_disk_rec.spatial.source == "B original"
        assert list(on_disk_rec.spatial.positions["LA1"]) == [-42.0, 20.0, 30.0]
        assert store.load_run(cb, rb, "baseline")["meta"]["generation_id"] == gen_before


def test_claude_review_role_failclosed_torn_jsonl_replay_and_unmatched_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify P1-1 fail-closed role buckets, F3 partial-failure replay healing, JSONL torn-tail safety, read-only projection, and unmatched SOZ disclosure."""
    import brainlens.mcp_server as mcp_mod
    from brainlens.analytics import compute_overlay_concordance
    from brainlens.storage import classify_judgment_role

    # (1) P1-1: Unknown / LLM role strings MUST fail closed to 'agent' (NEVER 'clinician')!
    for unknown_role in ("llm", "claude", "AI agent", "gpt", "copilot", "", "   "):
        assert classify_judgment_role(unknown_role) == "agent", unknown_role
    assert classify_judgment_role("clinician") == "clinician"
    assert classify_judgment_role("physician") == "clinician"
    assert classify_judgment_role("researcher") == "researcher"
    assert classify_judgment_role("engineering_rehearsal") == "rehearsal"

    ws = tmp_path / "ws_claude_fixes"
    store = WorkbenchStore(ws)
    monkeypatch.setattr(mcp_mod, "_WORKSPACE_ROOT", ws.resolve())
    analyze_case(str(SYNTHETIC_EDF), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})

    # MCP rejects arbitrary non-allowlisted role strings upfront
    r_bad_role = mcp_mod.record_judgment(
        case="synthetic_seeg",
        note="Should not count as clinician",
        role="claude",  # type: ignore[arg-type]
    )
    assert r_bad_role.get("error") is True and r_bad_role.get("code") == "invalid_role", r_bad_role
    assert store.load_judgments("synthetic_seeg") == []

    # (2) F3 partial-failure healing: if WAL recorded op_id before overlay committed, retry with SAME op_id MUST commit overlay!
    store.record_judgment(
        "synthetic_seeg",
        {
            "kind": "case_note",
            "operation_id": "op_partial_heal_1",
            "recording": "synthetic_seeg",
            "run_name": "baseline",
            "author": "dr_a",
            "role": "clinician",
            "note": "Initial note before crash",
        },
    )
    assert store.load_clinical_overlay("synthetic_seeg") is None
    r_heal = mcp_mod.record_judgment(
        case="synthetic_seeg",
        recording="synthetic_seeg",
        run_name="baseline",
        note="Initial note before crash",
        clinical_overlay={"source": "MDT", "soz_channels": ["SYN_A1", "TYPO_CH99"]},
        operation_id="op_partial_heal_1",
        author="dr_a",
        role="clinician",
    )
    assert not r_heal.get("error"), r_heal
    assert store.load_clinical_overlay("synthetic_seeg")["soz_channels"] == ["SYN_A1", "TYPO_CH99"]
    conc = compute_overlay_concordance(store, "synthetic_seeg", rec_slug="synthetic_seeg", run_name="baseline")
    assert conc["unmatched_overlay_channels"]["soz"] == ["TYPO_CH99"]

    # (3) JSONL torn-tail safety + monotonic jdg_NNNN IDs (both judgments.jsonl and report_actions.jsonl)
    jpath = store.judgments_path("synthetic_seeg")
    with jpath.open("ab") as fh:
        fh.write(b'{"id": "jdg_corrupt_unterminated"')
    j_after = store.record_judgment(
        "synthetic_seeg",
        {"kind": "case_note", "author": "dr_a", "role": "clinician", "note": "Survived torn tail"},
    )
    loaded_j = store.load_judgments("synthetic_seeg")
    assert any(r.get("note") == "Survived torn tail" for r in loaded_j)
    assert len({r["id"] for r in loaded_j}) == len(loaded_j), loaded_j
    assert j_after["id"] == "jdg_0002"

    apath = store.report_actions_path()
    store.append_report_action({"case": "synthetic_seeg", "action": "annotate", "comment": "first"})
    with apath.open("ab") as fh:
        fh.write(b'{"seq": 99, "broken_tail"')
    act_after = store.append_report_action({"case": "synthetic_seeg", "action": "annotate", "comment": "after torn tail"})
    all_acts = store.load_report_actions("synthetic_seeg")["actions"]
    assert any(a.get("comment") == "after torn tail" for a in all_acts) and act_after["seq"] >= 2

    # (4) Read-only GET case_verdicts_projection MUST NOT rewrite MVCC generation on disk!
    gen_before = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline", reconcile=False)["meta"]["generation_id"]
    proj = store.case_verdicts_projection("synthetic_seeg")
    assert proj["case"] == "synthetic_seeg"
    gen_after = store.load_run("synthetic_seeg", "synthetic_seeg", "baseline", reconcile=False)["meta"]["generation_id"]
    assert gen_after == gen_before


def test_brainlens_rename_and_legacy_compat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify BrainLens branding, *_brainlens.json priority, and full legacy BTK/BrainToolkit compatibility."""
    import urllib.request
    import brainlens.models as models_mod
    import brainlens.pyhfo_runner as pr

    # 1) *_brainlens.json discovered AND takes priority over legacy *_braintoolkit.json
    d_both = tmp_path / "both_sc"
    d_both.mkdir()
    rec_f = _write_fif(d_both / "sub-09_run-01_ieeg_raw.fif", ["LA1", "LA2"], 512.0, 2048)
    (d_both / "sub-09_run-01_ieeg_raw_braintoolkit.json").write_text(
        json.dumps({"reference": "common", "line_freq": 50.0}),
        encoding="utf-8",
    )
    (d_both / "sub-09_run-01_ieeg_raw_brainlens.json").write_text(
        json.dumps({"reference": "bipolar", "line_freq": 60.0}),
        encoding="utf-8",
    )
    ws = tmp_path / "ws_compat"
    insp_both = inspect_case(str(rec_f), workspace_dir=ws, persist=True)
    rec0 = WorkbenchStore(ws).load_recording(insp_both["case"], insp_both["recordings"][0]["recording"])
    assert rec0.reference == "bipolar" and rec0.line_freq == 60.0

    # 2) Legacy *_braintoolkit.json alone still pairs cleanly
    d_leg = tmp_path / "legacy_sc"
    d_leg.mkdir()
    rec_leg = _write_fif(d_leg / "sub-10_run-01_ieeg_raw.fif", ["RA1", "RA2"], 512.0, 2048)
    (d_leg / "sub-10_run-01_ieeg_raw_braintoolkit.json").write_text(
        json.dumps({"reference": "average", "line_freq": 50.0}),
        encoding="utf-8",
    )
    insp_leg = inspect_case(str(rec_leg), workspace_dir=ws, persist=True)
    rec1 = WorkbenchStore(ws).load_recording(insp_leg["case"], insp_leg["recordings"][0]["recording"])
    assert rec1.reference == "average" and rec1.line_freq == 50.0

    # 3) Legacy BTK_* / BRAINTOOLKIT_* env fallbacks & ~/.braintoolkit/engine.json fallback
    for env_k in (
        "BRAINLENS_WORKSPACE", "BRAINLENS_WORKBENCH_DIR", "BRAINTOOLKIT_WORKSPACE", "BRAINTOOLKIT_WORKBENCH_DIR",
        "BRAINLENS_TRUST_DIR", "BTK_TRUST_DIR",
        "BRAINLENS_PYHFO_SOURCE", "BRAINLENS_PYHFO_PYTHON", "BTK_PYHFO_SOURCE", "BTK_PYHFO_PYTHON",
        "BRAINLENS_ALLOW_WORKSPACE_ENGINE", "BTK_ALLOW_WORKSPACE_ENGINE",
    ):
        monkeypatch.delenv(env_k, raising=False)
    monkeypatch.setenv("BRAINTOOLKIT_WORKSPACE", str(tmp_path / "legacy_ws"))
    assert models_mod.default_workspace_dir() == (tmp_path / "legacy_ws").resolve()
    monkeypatch.setenv("BRAINLENS_WORKSPACE", str(tmp_path / "new_ws"))
    assert models_mod.default_workspace_dir() == (tmp_path / "new_ws").resolve()

    monkeypatch.setenv("BTK_TRUST_DIR", str(tmp_path / "leg_trust"))
    assert models_mod.trust_store_dir() == (tmp_path / "leg_trust").resolve()
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "new_trust"))
    assert models_mod.trust_store_dir() == (tmp_path / "new_trust").resolve()

    leg_eng = tmp_path / "legacy_user_engine.json"
    atomic_write_json(leg_eng, {"source_root": "/tmp/pyhfo_leg", "python_executable": "/tmp/py_leg"})
    monkeypatch.setattr(pr, "USER_ENGINE_JSON", tmp_path / "missing_new_engine.json")
    monkeypatch.setattr(pr, "LEGACY_USER_ENGINE_JSON", leg_eng)
    monkeypatch.setattr(pr, "IS_SOURCE_CHECKOUT", False)
    assert resolve_engine_settings().source_root == "/tmp/pyhfo_leg"

    # 4) HTTP server reports service='brainlens-workbench' and accepts both X-BrainLens-Token and legacy X-BTK-Token
    srv = WorkbenchHTTPServer(workspace_dir=ws, host="127.0.0.1", port=0)
    base_url = srv.start().rstrip("/")
    try:
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=3.0) as resp:
            h = json.loads(resp.read().decode("utf-8"))
        assert h["ok"] is True and h["service"] == "brainlens-workbench"

        for hdr_name in ("X-BrainLens-Token", "X-BTK-Token"):
            req = urllib.request.Request(
                f"{base_url}/api/ui-state",
                data=json.dumps({"view": "overview", "updated_by": "compat_test"}).encode("utf-8"),
                headers={"Content-Type": "application/json", hdr_name: srv.auth_token},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                assert resp.status == 200
    finally:
        srv.stop()
















