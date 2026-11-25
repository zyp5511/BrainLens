"""Comprehensive end-to-end verification suite for BrainLens Neurophysiology Workbench (v1.0 + Pillars I-III)."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "workbench" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import brainlens
from brainlens.analytics import compute_overlay_concordance, query_case, select_control_windows
from brainlens.bids_export import export_bids_events
from brainlens.case_sheet import export_case_sheet
from brainlens.cohort import run_cohort
from brainlens.io_readers import inspect_case
from brainlens.models import ToolkitError
from brainlens.pipeline import analyze_case, refresh_verdict_tallies
from brainlens.protocols import list_protocols, save_protocol
from brainlens.storage import WorkbenchStore
from brainlens.triad_views import render_evidence_card, render_spatial_figure
from brainlens.workbench_http import WorkbenchHTTPServer


def assert_no_legacy_imports() -> None:
    leaked = [
        m for m in sys.modules
        if m in {"braintoolkit", "btk"} or m.startswith(("braintoolkit.", "btk."))
    ]
    assert not leaked, f"Legacy braintoolkit/btk package leaked into sys.modules: {leaked}"


def run_synthetic_and_protocol_checks(ws: Path) -> None:
    print("[1/7] Testing synthetic_seeg (2000 Hz): baseline, stage reuse, 3-stage ONNX, state slice, bipolar, overlay, protocols...")
    insp = inspect_case("inputs/synthetic_seeg.edf", workspace_dir=ws)
    assert insp["case"] == "synthetic_seeg"
    assert insp["recordings"][0]["provenance"]["is_synthetic"] is True
    assert insp["recordings"][0]["provenance"]["origin_kind"] == "synthetic_engineering_data"
    assert insp["recordings"][0]["spatial"]["available"] is True

    r_base = analyze_case("synthetic_seeg", workspace_dir=ws, run_name="baseline")
    assert r_base["runs"][0]["counts"]["total_raw_events"] > 0
    assert r_base["runs"][0]["reused_stages"] == {}

    # State marker slicing check (synthetic_state_A = [0, 20)s)
    r_state = analyze_case(
        "synthetic_seeg",
        workspace_dir=ws,
        run_name="state_a_slice",
        overrides={"state": "synthetic_state_A"},
        reason="Verify marker-based sleep/state slicing",
    )
    store = WorkbenchStore(ws)
    st_sum = store.load_run("synthetic_seeg", "synthetic_seeg", "state_a_slice")["summary"]
    assert st_sum["selection"]["state"] == "synthetic_state_A"
    assert st_sum["counts"]["total_raw_events"] <= r_base["runs"][0]["counts"]["total_raw_events"]

    # Branch 1: detector threshold override -> must reuse qc + prepare
    r_rms = analyze_case(
        "synthetic_seeg",
        workspace_dir=ws,
        run_name="ste_rms_6",
        overrides={"rms_thres": 6.0},
        reason="Stricter STE RMS threshold sensitivity test",
    )
    assert set(r_rms["runs"][0]["reused_stages"].keys()) == {"qc", "prepare"}, r_rms["runs"][0]["reused_stages"]

    # Branch 2: ONNX classifier at 2000 Hz -> must reuse qc + prepare + detect, and all 3 ONNX models must succeed
    r_onnx = analyze_case(
        "synthetic_seeg",
        workspace_dir=ws,
        run_name="onnx_clf",
        overrides={"classifier_enabled": True, "classifier_stages": ["artifact", "spike", "ehfo"]},
        reason="Full 3-model PyHFO ONNX neural classifier check at 2000 Hz",
    )
    assert set(r_onnx["runs"][0]["reused_stages"].keys()) == {"qc", "prepare", "detect"}, r_onnx["runs"][0]["reused_stages"]
    onnx_bundle = store.load_run("synthetic_seeg", "synthetic_seeg", "onnx_clf")
    stages = onnx_bundle["summary"]["classification"]["stages"]
    for stage_name in ("artifact", "spike", "ehfo"):
        assert stages[stage_name]["status"] == "complete", f"Expected {stage_name} complete at 2000 Hz, got {stages[stage_name]}"
    assert r_onnx["runs"][0]["counts"]["ehfo_events"] is not None
    # Verify spectral annotations on events
    assert all("dominant_band" in e and "peak_to_peak_uv" in e for e in onnx_bundle["events"])

    # Branch 3: Bipolar reference + patient-specific exclusion -> save protocol and verify exclusions are stripped
    r_bip = analyze_case(
        "synthetic_seeg",
        workspace_dir=ws,
        run_name="bipolar_excl",
        overrides={
            "reference_mode": "bipolar",
            "classifier_enabled": True,
            "exclude_intervals": [{"start": 0.0, "end": 3.0, "reason": "Patient-specific lead-in"}],
        },
    )
    assert r_bip["runs"][0]["effective_reference"]["mode"] == "bipolar"

    proto_res = save_protocol(
        name="SEEG Bipolar ONNX",
        case_slug="synthetic_seeg",
        run_name="bipolar_excl",
        description="Reusable bipolar STE + ONNX screening protocol",
        workspace_dir=ws,
    )
    proto_doc = json.loads(Path(proto_res["path"]).read_text("utf-8"))
    assert "exclusions" not in proto_doc and "channels" not in proto_doc
    assert proto_doc["reference_mode"] == "bipolar"
    assert proto_doc["classifier"]["enabled"] is True
    assert len(list_protocols(ws)) >= 1

    # External clinical overlay check using documented synthetic truth (mandatory source enforced!)
    source_missing_rejected = False
    try:
        store.save_clinical_overlay("synthetic_seeg", {"soz_channels": ["SYN_A1"]})
    except ToolkitError as exc:
        source_missing_rejected = exc.code == "missing_overlay_source"
    assert source_missing_rejected, "Overlay without provenance source must be rejected!"

    store.save_clinical_overlay("synthetic_seeg", {
        "source": "inputs/synthetic_seeg_truth.json (injected 140 Hz synthetic oscillations — engineering test, not clinical SOZ)",
        "soz_channels": ["SYN_A1", "SYN_A3"],
        "resected_channels": ["SYN_A1", "SYN_A2", "SYN_A3"],
        "notes": "Engineering test of external reference overlay concordance math.",
    })
    conc = compute_overlay_concordance(store, "synthetic_seeg", run_name="baseline")
    assert conc["available"] is True and conc["metrics"]["soz_precision_at_top_k"] is not None
    syn_sheet = export_case_sheet("synthetic_seeg", workspace_dir=ws, primary_run="onnx_clf", comparison_run="bipolar_excl")
    syn_html = Path(syn_sheet["report_path"]).read_text("utf-8")
    assert "EXTERNAL REFERENCE" in syn_html and "PyHFO ONNX" in syn_html and syn_sheet["has_spatial_figure"] is True


def run_hup134_clinical_checks(ws: Path) -> None:
    print("[2/7] Testing HUP134 (1024 Hz, run-01 & run-02): synchrony, ranking, sensitivity, ONNX contract, MDT report...")
    insp = inspect_case("inputs/HUP134", workspace_dir=ws)
    assert insp["case"] == "HUP134"
    for rec_info in insp["recordings"]:
        assert rec_info["provenance"]["is_synthetic"] is False
        assert rec_info["provenance"]["origin_kind"] == "public_dataset"
        assert rec_info["spatial"]["available"] is True
        assert rec_info["spatial"]["template"] is True
        assert rec_info["spatial"]["coordinate_system"] == "fsaverage"

    base = analyze_case("HUP134", workspace_dir=ws, run_name="baseline")
    assert base["recordings_analyzed"] == 2

    store = WorkbenchStore(ws)
    s01 = store.load_run("HUP134", "run-01", "baseline")["summary"]
    s02 = store.load_run("HUP134", "run-02", "baseline")["summary"]

    # 1. Verify EKG1/EKG2 are unassessed (never ranked) & focal-quiet vs raw-quiet separated
    assert set(s01["unassessed_channel_names"]) >= {"EKG1", "EKG2"}
    assert set(s02["unassessed_channel_names"]) >= {"EKG1", "EKG2"}
    assert s01["counts"]["focal_quiet_measured_channels"] == 80
    assert s02["counts"]["focal_quiet_measured_channels"] == 82
    assert "invalid_channel_fraction" in s01["timeline_bins"][0]
    assert "qc_overview" in s01

    # 2. Verify ZERO 0-event channels ever receive a numerical rank!
    for s in (s01, s02):
        for ch_row in s["channels"]:
            if (ch_row["focal_count"] or 0) == 0:
                assert ch_row["focal_rank"] is None, f"0-focal-event channel got ranked: {ch_row}"
            if (ch_row["raw_count"] or 0) == 0:
                assert ch_row["raw_rank"] is None, f"0-raw-event channel got ranked: {ch_row}"
        for lead in s["leading_focal_channels"]:
            assert lead["focal_count"] > 0 and lead["focal_rank"] is not None

    # 3. Verify HUP134/run-01 widespread burst discovery & LB10 cross-run focal concordance!
    raw_01 = s01["counts"]["total_raw_events"]
    burst_01 = s01["counts"]["widespread_burst_events"]
    non_burst_01 = s01["counts"]["non_burst_events"]
    assert raw_01 > 1000
    assert burst_01 / raw_01 > 0.95
    assert 0 < non_burst_01 < 50
    assert s01["leading_focal_channels"][0]["channel_name"] == "LB10"
    assert s02["leading_focal_channels"][0]["channel_name"] == "LB10"
    assert len(s02["leading_focal_channels"]) == 1

    # 4. Cross-recording concordance (unpooled)
    q_cross = query_case("HUP134", workspace_dir=ws, mode="compare_recordings")
    assert "LB10" in q_cross["cross_recording"]["pairwise"][0]["focal_top5_shared"]

    # 5. Sensitivity branch on HUP134/run-01: exclude 0-60s and verify symmetric union & event fate conservation
    r_excl = analyze_case(
        "HUP134/run-01",
        workspace_dir=ws,
        run_name="excl_0s_60s",
        overrides={"exclude_intervals": [{"start": 0.0, "end": 60.0, "reason": "Exclude early segment with 50.5s burst"}]},
        reason="Test whether LB10 focal lead survives excluding first 60s",
    )
    assert r_excl["runs"][0]["counts"]["total_raw_events"] < raw_01
    cmp_runs = query_case("HUP134", workspace_dir=ws, mode="compare_runs", recording="run-01", run_a="baseline", run_b="excl_0s_60s")
    fate = cmp_runs["event_fate"]
    assert (
        fate["unchanged_exact_id"]
        + fate["disappeared_within_analyzed_coverage"]
        + fate["excluded_by_changed_time_or_channel_mask"]
        == cmp_runs["summary_metrics"]["raw_events_a"]
    ), fate
    s_excl = store.load_run("HUP134", "run-01", "excl_0s_60s")["summary"]
    top_b_names = {r["channel_name"] for r in s_excl["leading_focal_channels"] if r["focal_rank"] == 1}
    row_names = {r["channel"] for r in cmp_runs["channels"]}
    assert top_b_names <= row_names

    # 6. ONNX bandwidth contract on 1024 Hz HUP134/run-02: artifact complete, ehfo honestly unavailable
    r_hup_onnx = analyze_case(
        "HUP134/run-02",
        workspace_dir=ws,
        run_name="onnx_clf",
        overrides={"classifier_enabled": True, "classifier_stages": ["artifact", "spike", "ehfo"]},
        reason="Check 1024 Hz ONNX model bandwidth contract on LB10 candidates",
    )
    assert set(r_hup_onnx["runs"][0]["reused_stages"].keys()) == {"qc", "prepare", "detect"}
    hup_stages = store.load_run("HUP134", "run-02", "onnx_clf")["summary"]["classification"]["stages"]
    assert hup_stages["artifact"]["status"] == "complete", hup_stages
    assert hup_stages["ehfo"]["status"] != "complete", hup_stages

    # 7. Candidate-free control windows & Triad/Spatial evidence cards
    ctrls = select_control_windows(store, "HUP134", "run-02", run_name="baseline", channel="LB10")
    assert len(ctrls) >= 1
    ev02 = store.load_run("HUP134", "run-02", "baseline")["events"]
    for cw in ctrls:
        for ev in ev02:
            if ev["channel_name"] == "LB10":
                assert cw["end_seconds"] <= ev["start_seconds"] or cw["start_seconds"] >= ev["end_seconds"]

    c_focal_02 = render_evidence_card("HUP134", workspace_dir=ws, recording="run-02", channel="LB10", event_id="top_focal")
    c_focal_01 = render_evidence_card("HUP134", workspace_dir=ws, recording="run-01", channel="LB10", event_id="top_focal")
    c_burst_01 = render_evidence_card("HUP134", workspace_dir=ws, recording="run-01", channel="LB10", event_id="top_burst")
    c_ctrl_02 = render_evidence_card("HUP134", workspace_dir=ws, recording="run-02", channel="LB10", event_id="control")
    c_spatial = render_evidence_card("HUP134", workspace_dir=ws, recording="run-01", event_id="spatial")
    # Control card must NOT print a fake peak HFO frequency!
    assert c_ctrl_02["peak_hfo_frequency_hz"] is None
    for card in (c_focal_02, c_focal_01, c_burst_01, c_ctrl_02, c_spatial):
        assert Path(card["image_path"]).is_file() and Path(card["image_path"]).stat().st_size > 15_000

    ev_id = c_focal_02["focus_event"]["id"]
    j1 = store.record_judgment("HUP134", {
        "kind": "event_verdict",
        "recording": "run-02",
        "run_name": "baseline",
        "channel": "LB10",
        "event_id": ev_id,
        "verdict": "confirm",
        "clipped": True,
        "author": "epileptologist",
        "role": "clinician",
        "note": "Clear ~230 Hz ripple riding on sharp transient in LB10, localized on LB10-LB9 difference.",
    })
    n_after_j1 = len(store.load_judgments("HUP134"))
    j2 = store.record_judgment("HUP134", {
        "kind": "event_verdict",
        "recording": "run-02",
        "run_name": "baseline",
        "channel": "LB10",
        "event_id": ev_id,
        "verdict": "confirm",
        "clipped": True,
        "author": "epileptologist",
        "role": "clinician",
        "note": "Clear ~230 Hz ripple riding on sharp transient in LB10, localized on LB10-LB9 difference.",
    })
    assert j1 == j2 and len(store.load_judgments("HUP134")) == n_after_j1 and n_after_j1 >= 1

    # P0 Flip-Flop Fidelity Test: Confirm -> Artifact -> Confirm MUST preserve the final 'confirm'!
    import uuid
    op_art_id = f"op_flip_art_{uuid.uuid4().hex[:8]}"
    j_art = store.record_judgment("HUP134", {
        "kind": "event_verdict",
        "operation_id": op_art_id,
        "recording": "run-02",
        "run_name": "baseline",
        "channel": "LB10",
        "event_id": ev_id,
        "verdict": "artifact",
        "clipped": True,
        "author": "epileptologist",
        "role": "clinician",
        "note": "Re-check artifact hypothesis.",
    })
    assert j_art["verdict"] == "artifact"
    # Same operation_id must be strictly idempotent
    j_art_dup = store.record_judgment("HUP134", {
        "kind": "event_verdict",
        "operation_id": op_art_id,
        "recording": "run-02",
        "run_name": "baseline",
        "channel": "LB10",
        "event_id": ev_id,
        "verdict": "artifact",
        "clipped": True,
        "author": "epileptologist",
        "role": "clinician",
        "note": "Re-check artifact hypothesis.",
    })
    assert j_art_dup["id"] == j_art["id"]
    # Flip back to 'confirm' with identical original note (NO operation_id -> legacy path must ONLY compare against latest row!)
    j_back = store.record_judgment("HUP134", {
        "kind": "event_verdict",
        "recording": "run-02",
        "run_name": "baseline",
        "channel": "LB10",
        "event_id": ev_id,
        "verdict": "confirm",
        "clipped": True,
        "author": "epileptologist",
        "role": "clinician",
        "note": "Clear ~230 Hz ripple riding on sharp transient in LB10, localized on LB10-LB9 difference.",
    })
    assert j_back["verdict"] == "confirm" and j_back["id"] != j1["id"]
    proj = store.case_verdicts_projection("HUP134")
    assert proj["verdicts"][f"run-02|baseline|{ev_id}"]["verdict"] == "confirm"
    refresh_verdict_tallies(store, "HUP134", "run-02", "baseline")

    sheet = export_case_sheet(
        case_slug="HUP134",
        workspace_dir=ws,
        primary_run="baseline",
        comparison_run="excl_0s_60s",
        clinical_question="Does focal HFO activity on lateral temporal contact LB10 persist across both interictal runs once global multi-shaft bursts are separated?",
    )
    html_text = Path(sheet["report_path"]).read_text("utf-8")
    assert "do not establish SOZ or resection targets" in html_text
    assert "80 focal-quiet" in html_text
    assert "SEEG Depth-Shaft Spatial Distribution" in html_text
    assert sheet["has_spatial_figure"] is True
    assert sheet["embedded_cards"] >= 3


def run_zurich_and_reference_guard_checks(ws: Path) -> None:
    print("[3/7] Testing ZurichHFO BrainVision (.vhdr + published bipolar + ONNX + MDT cross-ref sheet + BIDS export)...")
    z_insp = inspect_case("inputs/ZurichHFO", workspace_dir=ws)
    assert z_insp["recordings"][0]["sfreq_hz"] == 2000.0
    assert z_insp["recordings"][0]["provenance"]["is_synthetic"] is False
    assert z_insp["recordings"][0]["published_bipolar_pairs_available"] is True
    assert z_insp["recordings"][0]["spatial"]["available"] is False

    z_base = analyze_case(
        "ZurichHFO",
        workspace_dir=ws,
        run_name="baseline",
        overrides={"end_seconds": 60.0},
        reason="First 60s SWS common-reference screening on Zurich sub-01",
    )
    assert z_base["runs"][0]["counts"]["total_raw_events"] > 0
    z_bip = analyze_case(
        "ZurichHFO",
        workspace_dir=ws,
        run_name="published_bipolar_onnx",
        overrides={
            "end_seconds": 60.0,
            "reference_mode": "bipolar",
            "classifier_enabled": True,
            "classifier_stages": ["artifact", "spike", "ehfo"],
        },
        reason="Published bipolar montage + full 3-stage ONNX scoring at 2000 Hz",
    )
    assert z_bip["runs"][0]["effective_reference"]["mode"] == "bipolar"
    z_store = WorkbenchStore(ws)
    z_rec = z_store.list_recordings("ZurichHFO")[0]
    z_sum = z_store.load_run("ZurichHFO", z_rec, "published_bipolar_onnx")["summary"]
    assert "bipolar" in z_sum["effective_reference_label"]
    for st_name in ("artifact", "spike", "ehfo"):
        assert z_sum["classification"]["stages"][st_name]["status"] == "complete"

    z_card = render_evidence_card("ZurichHFO", workspace_dir=ws, run_name="published_bipolar_onnx", event_id="top_focal")
    assert "-" in z_card["channel"] and Path(z_card["image_path"]).is_file()

    # Export ZurichHFO MDT Case Sheet comparing published_bipolar_onnx vs baseline (L7 cross-derivation side-by-side table!)
    z_sheet = export_case_sheet(
        case_slug="ZurichHFO",
        workspace_dir=ws,
        primary_run="published_bipolar_onnx",
        comparison_run="baseline",
        clinical_question="Which bipolar parahippocampal/amygdala contacts carry focal SWS HFOs confirmed by ONNX eHFO scoring?",
    )
    z_html = Path(z_sheet["report_path"]).read_text("utf-8")
    assert "Analyzed: <code>bipolar" in z_html
    assert "Cross-Derivation Comparison" in z_html
    assert "eHFO (/min)" in z_html

    # Standard BIDS *_events.tsv + *_events.json export check
    bids = export_bids_events("ZurichHFO", run_name="published_bipolar_onnx", workspace_dir=ws)
    assert bids["events_exported"] > 0
    tsv_path = Path(bids["tsv_path"])
    assert tsv_path.is_file() and Path(bids["json_path"]).is_file()
    with tsv_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) == bids["events_exported"]
    assert {"onset", "duration", "trial_type", "channel", "ehfo_score"} <= set(rows[0].keys())

    # Test already-bipolar re-referencing preflight guard in an isolated temp workspace
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy2(PROJECT_ROOT / "inputs/synthetic_seeg.edf", tmp_path / "already_bip.edf")
        (tmp_path / "already_bip.json").write_text(json.dumps({"reference": "bipolar", "synthetic": True}), encoding="utf-8")
        tmp_ws = tmp_path / "ws"
        raised = False
        try:
            analyze_case(str(tmp_path / "already_bip.edf"), workspace_dir=tmp_ws, run_name="bad_avg", overrides={"reference_mode": "average"})
        except ToolkitError as exc:
            raised = exc.code == "reference_incompatible"
        assert raised
        assert not (tmp_ws / "cases" / "already_bip" / "recordings" / "already_bip" / "runs" / "bad_avg").exists()


def run_cohort_checks(ws: Path) -> None:
    print("[4/7] Testing Multi-Patient Cohort Batch Runner (Group Protocol + graceful L7 skip)...")
    coh = run_cohort(
        targets=["HUP134", "ZurichHFO", "synthetic_seeg"],
        protocol="SEEG_Bipolar_ONNX",
        cohort_name="bipolar_onnx_cohort",
        overrides={"end_seconds": 40.0},
        workspace_dir=ws,
    )
    assert coh["total_recordings"] == 4  # HUP134 run-01, run-02 + ZurichHFO + synthetic_seeg
    assert coh["completed_recordings"] == 2  # ZurichHFO + synthetic_seeg (documented common reference)
    assert coh["skipped_recordings"] == 2  # HUP134 run-01 & run-02 (unknown reference -> safely skipped!)
    assert Path(coh["summary_csv_path"]).is_file()
    assert Path(coh["channels_csv_path"]).is_file()
    with Path(coh["summary_csv_path"]).open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert any(r["case"] == "HUP134" and "skipped_reference_incompatible" in r["status"] for r in rows)
    assert any(r["case"] == "ZurichHFO" and r["status"] == "complete" for r in rows)


def run_http_server_checks(ws: Path) -> None:
    print("[5/7] Testing WorkbenchHTTPServer security, pagination, SSE, per-case lifecycle & async jobs...")
    srv_json_path = ws / "workbench_server.json"
    saved_srv_json = srv_json_path.read_bytes() if srv_json_path.is_file() else None
    srv = WorkbenchHTTPServer(workspace_dir=ws, port=0)
    base_url = srv.start().rstrip("/")
    tok = srv.auth_token
    ws_id = srv.store.workspace_id
    try:
        def _get(path: str, headers: dict[str, str] | None = None) -> dict:
            req = urllib.request.Request(base_url + path, headers=headers or {}, method="GET")
            with urllib.request.urlopen(req, timeout=15) as r:
                assert r.status == 200
                return json.loads(r.read().decode("utf-8"))

        def _post(path: str, payload: dict, *, headers: dict[str, str] | None = None, raw_body: bytes | None = None) -> tuple[int, dict]:
            hdrs = {"Content-Type": "application/json", "X-BrainLens-Token": tok}
            if headers is not None:
                hdrs.update(headers)
            body = raw_body if raw_body is not None else json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(base_url + path, data=body, headers=hdrs, method="POST")
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, json.loads(r.read().decode("utf-8"))

        def _expect_http_error(path: str, method: str, expected_code: int, *, headers: dict[str, str] | None = None, data: bytes | None = None) -> None:
            req = urllib.request.Request(base_url + path, data=data, headers=headers or {}, method=method)
            try:
                urllib.request.urlopen(req, timeout=10)
                raise AssertionError(f"Expected HTTP {expected_code} for {method} {path}, got 200")
            except urllib.error.HTTPError as err:
                assert err.code == expected_code, f"Expected HTTP {expected_code} for {method} {path}, got {err.code}"

        health = _get("/api/health")
        assert health["ok"] is True and health["workspace_id"] == ws_id

        # Security Boundary Hardening Assertions
        # 1) Missing X-BrainLens-Token -> 401
        _expect_http_error(
            "/api/report-action",
            "POST",
            401,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"case": "HUP134", "action": "annotate", "comment": "unauth"}).encode("utf-8"),
        )
        # 2) text/plain simple-request CSRF -> 415
        _expect_http_error(
            "/api/report-action",
            "POST",
            415,
            headers={"Content-Type": "text/plain", "X-BrainLens-Token": tok},
            data=b'{"case":"HUP134"}',
        )
        # 3) Foreign Origin -> 403
        _expect_http_error(
            "/api/report-action",
            "POST",
            403,
            headers={"Content-Type": "application/json", "X-BrainLens-Token": tok, "Origin": "https://evil.example"},
            data=json.dumps({"case": "HUP134"}).encode("utf-8"),
        )
        # 4) Origin: null must NOT be echoed in Access-Control-Allow-Origin
        req_null = urllib.request.Request(base_url + "/api/health", headers={"Origin": "null"}, method="GET")
        with urllib.request.urlopen(req_null, timeout=10) as r_null:
            assert r_null.headers.get("Access-Control-Allow-Origin") is None, r_null.headers.items()
        # 5) GET wait?claim=1 without X-BrainLens-Token -> 401 (prevents GET-CSRF state mutation)
        _expect_http_error("/api/report-actions/wait?case=HUP134&timeout=1&claim=1", "GET", 401)
        # 6) Workspace mismatch -> 409
        _expect_http_error(
            "/api/report-action",
            "POST",
            409,
            headers={"Content-Type": "application/json", "X-BrainLens-Token": tok},
            data=json.dumps({"case": "HUP134", "workspace_id": "wrong_workspace_id", "action": "annotate", "comment": "x"}).encode("utf-8"),
        )

        # 7) Token injected dynamically over HTTP, NEVER written to on-disk HTML!
        with urllib.request.urlopen(base_url + "/", timeout=5) as resp_idx:
            idx_html = resp_idx.read().decode("utf-8")
        assert f'content="{tok}"' in idx_html
        on_disk_sheet = (ws / "cases" / "HUP134" / "exports" / "HUP134_baseline_case_sheet.html").read_text("utf-8")
        assert tok not in on_disk_sheet and 'meta name="brainlens-token"' not in on_disk_sheet

        boot = _get("/api/bootstrap?case=HUP134&recording=run-01&run_name=baseline")
        assert boot["active"]["case"] == "HUP134" and boot["workspace_id"] == ws_id
        assert len(boot["cohorts"]) >= 1
        assert boot["active"]["sheet_mtime_ns"] > 0

        # Clinical Integrity: event_id ownership & channel mismatch rejection on /api/judgment
        hup_ev = WorkbenchStore(ws).load_run("HUP134", "run-02", "baseline")["events"][0]
        _expect_http_error(
            "/api/judgment",
            "POST",
            400,
            headers={"Content-Type": "application/json", "X-BrainLens-Token": tok},
            data=json.dumps({
                "case": "HUP134",
                "recording": "run-02",
                "run_name": "baseline",
                "channel": "LA1",
                "event_id": hup_ev["id"],
                "verdict": "confirm",
            }).encode("utf-8"),
        )
        _expect_http_error(
            "/api/judgment",
            "POST",
            400,
            headers={"Content-Type": "application/json", "X-BrainLens-Token": tok},
            data=json.dumps({
                "case": "HUP134",
                "recording": "run-02",
                "run_name": "baseline",
                "channel": "LB10",
                "event_id": "ev_does_not_exist",
                "verdict": "confirm",
            }).encode("utf-8"),
        )

        # Case verdicts projection endpoint
        cv = _get("/api/case-verdicts?case=HUP134")
        assert cv["case"] == "HUP134" and cv["judgment_revision"] >= 1

        mc_notch = _get("/api/multichannel?case=HUP134&recording=run-02&run_name=baseline&shaft=LB&start_seconds=176&duration_seconds=4&montage=acquired&display_filter=broadband&display_notch=1")
        assert mc_notch["display_notch"] is True and "notch" in (mc_notch["montage_notice"] or "").lower()
        mc_multi = _get("/api/multichannel?case=HUP134&recording=run-02&run_name=baseline&shaft=LB,LC&start_seconds=176&duration_seconds=4&montage=acquired&display_filter=fast_ripple_250_450")
        multi_shafts = {tr["shaft"] for tr in mc_multi["traces"]}
        assert multi_shafts == {"LB", "LC"}, multi_shafts
        assert "fast ripple" in (mc_multi["montage_notice"] or "").lower()

        # Queue pagination verification (offset + limit + has_more)
        z_rec = WorkbenchStore(ws).list_recordings("ZurichHFO")[0]
        q_p1 = _get(f"/api/queue?case=ZurichHFO&recording={z_rec}&run_name=published_bipolar_onnx&mode=ehfo_fr&sort=ehfo&offset=0&limit=5")
        assert q_p1["total"] > 5 and len(q_p1["items"]) == 5 and q_p1["has_more"] is True
        q_p2 = _get(f"/api/queue?case=ZurichHFO&recording={z_rec}&run_name=published_bipolar_onnx&mode=ehfo_fr&sort=ehfo&offset=5&limit=5")
        assert len(q_p2["items"]) > 0
        assert {x["id"] for x in q_p1["items"]}.isdisjoint({x["id"] for x in q_p2["items"]})

        tri_ctrl = _get("/api/triad?case=HUP134&recording=run-02&run_name=baseline&channel=LB10&event_id=control")
        assert tri_ctrl["peak_hfo_frequency_hz"] is None
        assert tri_ctrl["signal_metrics"]["ripple_rms_uv"] is not None

        # Async Job test (POST /api/jobs/branch -> 202 -> poll until complete)
        code, job = _post("/api/jobs/branch", {
            "target": "synthetic_seeg/synthetic_seeg",
            "run_name": "async_rms_55",
            "overrides": {"rms_thres": 5.5},
            "parent_run": "baseline",
        })
        assert code == 202 and job["job_id"]
        deadline = time.monotonic() + 25.0
        final_job = job
        while time.monotonic() < deadline:
            final_job = _get(f"/api/jobs/{job['job_id']}")
            if final_job["status"] in {"complete", "failed"}:
                break
            time.sleep(0.2)
        assert final_job["status"] == "complete" and final_job["progress"] == 1.0, final_job

        # BIDS TSV route + Cohort CSV download route
        code_b, bids_res = _post("/api/export-bids", {"case": "HUP134", "recording": "run-02", "run_name": "baseline"})
        assert code_b == 200 and bids_res["tsv_url"].startswith("/exports/")
        with urllib.request.urlopen(base_url + bids_res["tsv_url"], timeout=5) as resp:
            assert resp.status == 200 and b"onset\tduration" in resp.read()
        with urllib.request.urlopen(base_url + "/cohorts/bipolar_onnx_cohort/cohort_summary.csv", timeout=5) as resp:
            assert resp.status == 200 and b"ZurichHFO" in resp.read()

        # Tier 1 + Tier 2: Progressive Interactive Report HTML + Per-Case Lifecycle + SSE + Claim Non-Spin
        with urllib.request.urlopen(base_url + "/exports/HUP134/HUP134_baseline_case_sheet.html", timeout=5) as resp:
            rep_html = resp.read().decode("utf-8")
        assert "window.__BRAINLENS_REPORT__" in rep_html and f'content="{tok}"' in rep_html
        assert 'data-anchor-type="figure"' in rep_html and 'data-run-name="' in rep_html
        assert 'data-verdict="uncertain"' in rep_html and "offlineOpenLink" in rep_html

        # Resolve any leftover pending actions on HUP134 & ZurichHFO for clean baseline
        for c_clean in ("HUP134", "ZurichHFO"):
            ex = _get(f"/api/report-actions?case={c_clean}&pending=1&include_claimed=1&limit=200")
            ex_ids = [x["id"] for x in ex["actions"] if x.get("status") != "resolved"]
            if ex_ids:
                _post("/api/report-actions/resolve", {
                    "case": c_clean,
                    "action_ids": ex_ids,
                    "agent_reply": "Cleared prior test fixtures",
                    "resolved_by": "test_init",
                })

        # Verify inspect_waveform is telemetry-only (does NOT inflate inbox pending_count!)
        c_tel, tel_res = _post("/api/report-action", {
            "case": "HUP134",
            "recording": "run-02",
            "run_name": "baseline",
            "action": "inspect_waveform",
            "channel": "LB10",
            "trigger_agent": False,
        })
        assert c_tel == 200 and tel_res["telemetry_only"] is True and tel_res["pending_count"] == 0

        # Start live SSE listener thread to verify real-time event push
        import threading
        sse_events: list[str] = []
        sse_ready = threading.Event()

        def _sse_reader() -> None:
            try:
                with urllib.request.urlopen(base_url + "/api/events", timeout=8) as r_sse:
                    sse_ready.set()
                    for raw_line in r_sse:
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if line.startswith("event:"):
                            sse_events.append(line.split(":", 1)[1].strip())
                            if "report_actions" in sse_events:
                                break
            except Exception:
                sse_ready.set()

        st_thr = threading.Thread(target=_sse_reader, daemon=True)
        st_thr.start()
        sse_ready.wait(timeout=2.0)

        # Post a pending action on ZurichHFO FIRST (lower seq) to prove HUP134 resolve NEVER clobbers ZurichHFO!
        c_z, a_z = _post("/api/report-action", {
            "case": "ZurichHFO",
            "recording": z_rec,
            "run_name": "published_bipolar_onnx",
            "action": "ask_agent",
            "anchor_type": "channel_row",
            "channel": "PHL1-PHL2",
            "comment": "Verify Zurich bipolar parahippocampal eHFO stability.",
            "trigger_agent": True,
        })
        assert c_z == 200 and a_z["ok"] is True
        z_action_id = a_z["action"]["id"]

        before_hup = _get("/api/report-actions?case=HUP134")
        start_seq = int(before_hup["latest_seq"])

        # Launch Tier 2 long-poll waiter on HUP134 with claim=1 + X-BrainLens-Token
        wait_box: dict = {}

        def _waiter() -> None:
            t0 = time.monotonic()
            wait_box["res"] = _get(
                f"/api/report-actions/wait?case=HUP134&since={start_seq}&timeout=8&trigger_only=1&claim=1&consumer=e2e_agent",
                headers={"X-BrainLens-Token": tok},
            )
            wait_box["elapsed"] = time.monotonic() - t0

        wt = threading.Thread(target=_waiter, daemon=True)
        wt.start()
        time.sleep(0.25)

        # Post a quiet annotation (trigger_agent=False) -> waiter must stay waiting to bundle
        c1, a1 = _post("/api/report-action", {
            "case": "HUP134",
            "recording": "run-02",
            "run_name": "baseline",
            "action": "annotate",
            "anchor_type": "text_selection",
            "channel": "LB10",
            "quoted_text": "1147/1150 (99.7%) raw detections",
            "comment": "Verify LB10 focal specificity relative to LC7/LC8.",
            "trigger_agent": False,
        })
        assert c1 == 200 and a1["ok"] is True
        time.sleep(0.2)
        assert wt.is_alive(), "trigger_only=1 waiter should wait for an explicit ask_agent/submit_batch trigger"

        # Now post an explicit ask_agent trigger -> waiter must wake immediately with BOTH HUP134 actions bundled & claimed!
        c2, a2 = _post("/api/report-action", {
            "case": "HUP134",
            "recording": "run-02",
            "run_name": "baseline",
            "action": "ask_agent",
            "anchor_type": "figure",
            "channel": "LB10",
            "quoted_text": "run-02 · LB10 · Clinician-clipped evidence",
            "comment": "Compare LB10 acquired vs LB10-LB9 differential amplitude here.",
            "trigger_agent": True,
        })
        assert c2 == 200 and a2["ok"] is True
        wt.join(timeout=3.0)
        assert not wt.is_alive() and wait_box["res"]["woke"] is True
        assert wait_box["elapsed"] < 2.5
        woke_ids = [x["id"] for x in wait_box["res"]["actions"]]
        assert a1["action"]["id"] in woke_ids and a2["action"]["id"] in woke_ids
        assert z_action_id not in woke_ids

        # Second immediate wait on HUP134 must NOT spin on already-claimed trigger!
        wait2 = _get(
            f"/api/report-actions/wait?case=HUP134&since={start_seq}&timeout=0.6&trigger_only=1&claim=1&consumer=e2e_agent",
            headers={"X-BrainLens-Token": tok},
        )
        assert wait2["woke"] is False and wait2["timed_out"] is True, wait2

        # SSE stream must have received report_actions event
        st_thr.join(timeout=2.0)
        assert "report_actions" in sse_events, f"SSE did not push report_actions: {sse_events}"

        # Resolve HUP134 actions with explicit agent_reply -> MUST NOT clobber ZurichHFO's lower-seq action!
        # First verify unknown/cross-case IDs in resolve are atomically rejected (400) without mutating anything!
        _expect_http_error(
            "/api/report-actions/resolve",
            "POST",
            400,
            headers={"Content-Type": "application/json", "X-BrainLens-Token": tok},
            data=json.dumps({
                "case": "HUP134",
                "action_ids": [a1["action"]["id"], z_action_id],
                "agent_reply": "Should fail atomically",
            }).encode("utf-8"),
        )
        c_res, res_hup = _post("/api/report-actions/resolve", {
            "case": "HUP134",
            "action_ids": [a1["action"]["id"], a2["action"]["id"]],
            "agent_reply": "Confirmed LB10-LB9 differential retains focal ~230 Hz island while LC7/LC8 stay quiet.",
            "resolved_by": "e2e_agent",
        })
        assert c_res == 200 and res_hup["pending_count"] == 0 and res_hup["claimed_count"] == 0
        # Duplicate resolve with empty reply MUST NOT erase existing agent_reply!
        c_res2, _ = _post("/api/report-actions/resolve", {
            "case": "HUP134",
            "action_ids": [a2["action"]["id"]],
            "agent_reply": "",
            "resolved_by": "e2e_agent",
        })
        assert c_res2 == 200
        hup_hist = _get("/api/report-actions?case=HUP134&limit=10")["actions"]
        assert any(a["id"] == a2["action"]["id"] and a["status"] == "resolved" and "LB10-LB9" in a.get("agent_reply", "") for a in hup_hist)

        # ZurichHFO action MUST STILL be pending!
        z_after = _get("/api/report-actions?case=ZurichHFO&pending=1")
        assert z_after["pending_count"] >= 1 and any(a["id"] == z_action_id for a in z_after["actions"])

        # Round-2 Security & Integrity Regression Checks:
        # A) workbench_server.json created with 0600 permissions
        srv_json_mode = (ws / "workbench_server.json").stat().st_mode & 0o777
        assert srv_json_mode == 0o600, f"Expected 0600 on workbench_server.json, got {oct(srv_json_mode)}"
        # B) Universal workspace_id 409 check on /api/ui-state + no workspace_id leak into ui_state.json
        _expect_http_error(
            "/api/ui-state",
            "POST",
            409,
            headers={"Content-Type": "application/json", "X-BrainLens-Token": tok},
            data=json.dumps({"case": "HUP134", "workspace_id": "wrong_ws"}).encode("utf-8"),
        )
        c_ui, ui_saved = _post("/api/ui-state", {"case": "HUP134", "workspace_id": ws_id, "updated_by": "e2e_test"})
        assert c_ui == 200 and "workspace_id" not in ui_saved
        # C) Event judgment with missing recording or unknown run_name MUST fail loudly (400)
        _expect_http_error(
            "/api/judgment",
            "POST",
            400,
            headers={"Content-Type": "application/json", "X-BrainLens-Token": tok},
            data=json.dumps({"case": "HUP134", "event_id": hup_ev["id"], "verdict": "confirm"}).encode("utf-8"),
        )
        _expect_http_error(
            "/api/judgment",
            "POST",
            400,
            headers={"Content-Type": "application/json", "X-BrainLens-Token": tok},
            data=json.dumps({"case": "HUP134", "recording": "run-02", "run_name": "no_such_run", "event_id": hup_ev["id"], "verdict": "confirm"}).encode("utf-8"),
        )
        # D) Rejected report verdict MUST NOT leave an orphaned action in report_actions.jsonl!
        cnt_before_bad = len(WorkbenchStore(ws)._read_all_report_actions())
        _expect_http_error(
            "/api/report-action",
            "POST",
            400,
            headers={"Content-Type": "application/json", "X-BrainLens-Token": tok},
            data=json.dumps({
                "case": "HUP134",
                "recording": "run-02",
                "run_name": "baseline",
                "channel": "LA1",
                "event_id": hup_ev["id"],
                "action": "verdict",
                "verdict": "confirm",
            }).encode("utf-8"),
        )
        assert len(WorkbenchStore(ws)._read_all_report_actions()) == cnt_before_bad
        # E) SSE restart resync check (mismatched epoch in Last-Event-ID -> immediate event: resync)
        req_rs = urllib.request.Request(base_url + "/api/events", headers={"Last-Event-ID": "00000000:999"}, method="GET")
        with urllib.request.urlopen(req_rs, timeout=5) as r_rs:
            got_resync = False
            for _ in range(10):
                line = r_rs.readline().decode("utf-8", errors="ignore").strip()
                if line == "event: resync":
                    got_resync = True
                    break
            assert got_resync, "SSE did not emit resync on mismatched epoch Last-Event-ID!"

        # F) Stress + Lease + Contiguous Cursor + Partial-Line Tail Index in isolated temp workspace:
        with tempfile.TemporaryDirectory() as tmp_trig:
            tmp_root = Path(tmp_trig)
            tsrv = WorkbenchHTTPServer(workspace_dir=tmp_root, port=0)
            turl = tsrv.start().rstrip("/")
            ttok = tsrv.auth_token
            try:
                # 101 ordinary notes + trigger #102 (crosses bundle limit 100!)
                for i in range(101):
                    req_i = urllib.request.Request(
                        turl + "/api/report-action",
                        data=json.dumps({
                            "case": "StressCase",
                            "action": "annotate",
                            "comment": f"Quiet note {i}",
                            "trigger_agent": False,
                        }).encode("utf-8"),
                        headers={"Content-Type": "application/json", "X-BrainLens-Token": ttok},
                        method="POST",
                    )
                    with urllib.request.urlopen(req_i, timeout=5) as r_i:
                        assert r_i.status == 200
                req_t = urllib.request.Request(
                    turl + "/api/report-action",
                    data=json.dumps({
                        "case": "StressCase",
                        "action": "ask_agent",
                        "comment": "Buried trigger #102",
                        "trigger_agent": True,
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json", "X-BrainLens-Token": ttok},
                    method="POST",
                )
                with urllib.request.urlopen(req_t, timeout=5) as r_t:
                    trig_id = json.loads(r_t.read().decode("utf-8"))["action"]["id"]
                req_w = urllib.request.Request(
                    turl + "/api/report-actions/wait?case=StressCase&since=0&timeout=2&trigger_only=1&claim=1&consumer=agentA",
                    headers={"X-BrainLens-Token": ttok},
                    method="GET",
                )
                with urllib.request.urlopen(req_w, timeout=5) as r_w:
                    w_data = json.loads(r_w.read().decode("utf-8"))
                assert w_data["woke"] is True
                assert any(a["id"] == trig_id for a in w_data["actions"]), "Trigger buried behind 101 notes was missed!"
                # Contiguous cursor MUST stay at 100 (NOT jump past unclaimed note #101!)
                cur_a = tsrv.store.load_report_actions(case_slug="StressCase", only_pending=True, include_claimed=False, consumer="agentA")
                assert cur_a["consumer_cursor"] == 100 and cur_a["pending_count"] == 1, cur_a
                # Non-steal lease check: agentB cannot steal agentA's claimed trigger!
                steal = tsrv.store.claim_report_actions("StressCase", [trig_id], consumer="agentB")
                assert steal["claimed_ids"] == [] and trig_id in steal["already_claimed_ids"], steal
                # Release by agentA lets agentB claim it cleanly!
                rel = tsrv.store.release_report_actions(case_slug="StressCase", action_ids=[trig_id], consumer="agentA")
                assert trig_id in rel["released_ids"]
                reclaim = tsrv.store.claim_report_actions("StressCase", [trig_id], consumer="agentB")
                assert trig_id in reclaim["claimed_ids"]
                # Partial-line tail index safety check (split JSONL write without losing line)
                from brainlens.workbench_http import _ActionTailIndex
                tail_store = WorkbenchStore(tmp_root / "tail_sub")
                idx = _ActionTailIndex(tail_store)
                raw_line = json.dumps({"id": "act_0001", "seq": 1, "case": "SplitCase", "action": "annotate", "comment": "ok"}) + "\n"
                mid_pt = len(raw_line) // 2
                with tail_store.report_actions_path().open("wb") as fh:
                    fh.write(raw_line[:mid_pt].encode("utf-8"))
                r_half, _ = idx.snapshot()
                assert len(r_half) == 0
                with tail_store.report_actions_path().open("ab") as fh:
                    fh.write(raw_line[mid_pt:].encode("utf-8"))
                r_full, _ = idx.snapshot()
                assert len(r_full) == 1 and r_full[0]["id"] == "act_0001"
            finally:
                tsrv.stop()
    finally:
        srv.stop()
        if saved_srv_json is not None:
            srv_json_path.write_bytes(saved_srv_json)
            os.chmod(srv_json_path, 0o600)


async def _mcp_wire_check(ws: Path) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=str(PROJECT_ROOT / "workbench" / "bin" / "brainlens-mcp"),
        args=["--workspace", str(ws)],
        env={**os.environ, "BRAINLENS_NO_BROWSER": "1"},
    )
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_res = await session.list_tools()
            tool_names = [t.name for t in tools_res.tools]
            expected = [
                "inspect_case",
                "analyze_case",
                "query_evidence",
                "render_evidence_card",
                "open_workbench",
                "record_judgment",
                "export_or_protocol",
            ]
            assert tool_names == expected, f"Expected exact 7 tools {expected}, got {tool_names}"

            res_q = await session.call_tool("query_evidence", {"target": "bipolar_onnx_cohort", "mode": "cohort_summary"})
            q_payload = json.loads(res_q.content[0].text)
            assert q_payload["completed_recordings"] == 2

            res_b = await session.call_tool("export_or_protocol", {"action": "export_bids_events", "case": "HUP134", "recording": "run-02"})
            b_payload = json.loads(res_b.content[0].text)
            assert b_payload["events_exported"] == 2

            # Verify MCP Agent perceives ZurichHFO pending action (left intact by HUP134 resolve!) and resolves it with agent_reply!
            res_qz = await session.call_tool("query_evidence", {"target": "ZurichHFO", "mode": "auto"})
            qz_payload = json.loads(res_qz.content[0].text)
            assert qz_payload.get("pending_action_count", 0) >= 1
            assert any("parahippocampal" in (a.get("comment") or "") for a in qz_payload.get("pending_report_actions", []))

            # Verify open_workbench(action='status', case='ZurichHFO') is STRICTLY read-only (does NOT advance ui_state seq!)
            seq_before_status = WorkbenchStore(ws).load_ui_state().get("seq", 0)
            res_st = await session.call_tool("open_workbench", {"action": "status", "case": "ZurichHFO"})
            st_payload = json.loads(res_st.content[0].text)
            assert st_payload["pending_action_count"] >= 1
            assert WorkbenchStore(ws).load_ui_state().get("seq", 0) == seq_before_status
            z_ids = [a["id"] for a in st_payload["pending_report_actions"]]

            # Verify MCP record_judgment defaults to author='agent', role='agent' (never misattributed as clinician!)
            res_j = await session.call_tool("record_judgment", {"case": "ZurichHFO", "note": "Agent provenance verification note"})
            j_payload = json.loads(res_j.content[0].text)
            assert j_payload["saved_judgment"]["author"] == "agent" and j_payload["saved_judgment"]["role"] == "agent"

            # Bare resolve_actions without case/action_ids MUST be rejected (prevents cross-case clobber!)
            res_bad = await session.call_tool("open_workbench", {"action": "resolve_actions"})
            bad_payload = json.loads(res_bad.content[0].text)
            assert bad_payload.get("error") is True

            res_ack = await session.call_tool("open_workbench", {
                "action": "resolve_actions",
                "case": "ZurichHFO",
                "action_ids": z_ids,
                "agent_reply": "Verified Zurich bipolar parahippocampal eHFO cluster across 60s SWS window.",
            })
            ack_payload = json.loads(res_ack.content[0].text)
            assert ack_payload["ok"] is True and ack_payload["pending_count"] == 0


def run_mcp_checks(ws: Path) -> None:
    print("[6/7] Testing real MCP JSON-RPC stdio server wire protocol (exact 7 orthogonal tools)...")
    asyncio.run(_mcp_wire_check(ws))


def main() -> None:
    # ALWAYS execute inside an isolated TemporaryDirectory so running E2E NEVER mutates live workbench_results/!
    # With --reuse, copy existing workbench_results/ into the temp workspace first for fast cached runs.
    use_reuse = "--reuse" in sys.argv
    v1_dir = PROJECT_ROOT / "results"
    v1_before = sum(1 for p in v1_dir.rglob("*") if p.is_file()) if v1_dir.is_dir() else 0
    tmp_ctx = tempfile.TemporaryDirectory(prefix="brainlens_e2e_")
    ws = Path(tmp_ctx.name) / "ws"
    live_ws = PROJECT_ROOT / "workbench_results"
    if use_reuse and live_ws.is_dir():
        shutil.copytree(
            live_ws,
            ws,
            ignore=shutil.ignore_patterns("workbench_server.json", "*.lock"),
        )
    else:
        ws.mkdir(parents=True, exist_ok=True)
    try:
        run_synthetic_and_protocol_checks(ws)
        run_hup134_clinical_checks(ws)
        run_zurich_and_reference_guard_checks(ws)
        run_cohort_checks(ws)
        run_http_server_checks(ws)
        run_mcp_checks(ws)

        print("[7/7] Verifying zero legacy braintoolkit imports & historical results/ preservation...")
        assert_no_legacy_imports()
        v1_after = sum(1 for p in v1_dir.rglob("*") if p.is_file()) if v1_dir.is_dir() else 0
        assert v1_after == v1_before, f"Historical results/ count changed: before={v1_before}, after={v1_after}"
        print(f"ALL 7 PILLAR I-III E2E CHECKS PASSED! Isolated workspace: {ws} (results/ untouched: {v1_after} files).")
    finally:
        tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
