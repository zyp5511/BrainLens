"""Regression tests for case-level, externally sourced anatomical labels."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

WORKBENCH_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = WORKBENCH_DIR / "src"
if importlib.util.find_spec("brainlens") is None and SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import test_release_gate as gate
from brainlens import mcp_server, workbench_http
from brainlens.analytics import compare_recordings, compare_runs, query_case
from brainlens.anatomy import (
    AnatomyMatcher,
    anatomy_coverage,
    load_anatomy_labels,
    match_anatomy,
    parse_anatomy_table,
    save_anatomy_labels,
    validate_anatomy_labels,
)
from brainlens.bids_export import export_bids_events
from brainlens.case_sheet import export_case_sheet
from brainlens.cli import main as cli_main
from brainlens.cohort import run_cohort
from brainlens.io_readers import inspect_case
from brainlens.models import AnalysisConfig, ToolkitError, record_verified_sheet
from brainlens.pipeline import _stage_fps, analyze_case
from brainlens.storage import WorkbenchStore, run_analysis_signature
from brainlens.triad_views import compute_multichannel_batch, compute_triad_view
from brainlens.workbench_http import WorkbenchHTTPServer

CASE = "synthetic_seeg"
SOURCE = "Example localization <review 4>"
NON_NATIVE = "Insula (non-patient-native)"


def labels() -> dict:
    return {
        "source": SOURCE, "patient_native": False, "tool": "LeGUI", "atlas": "DKT",
        "labels": {
            "syn_a1": {"region": "Insula", "hemisphere": "left", "tissue": "gray"},
            "SYN_A2": {"region": "Amygdala", "in_lesion": False},
            "ABSENT": {"region": "Other"},
        },
    }


@pytest.fixture
def analyzed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WorkbenchStore:
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    monkeypatch.setattr(gate, "_has_pyhfo_engine", lambda: False)
    ws = tmp_path / "workspace"
    gate._analyze_synthetic_run(ws)
    return WorkbenchStore(ws)


@pytest.mark.parametrize("column", ["anatomy", "region", "label", "roi", "atlas_label", "destrieux", "dkt", "aparc"])
def test_bids_electrodes_columns(column: str) -> None:
    text = (
        f"\ufeffname\tx\ty\tz\t{column}\twm_gm\themisphere\tlesion\n"
        "LA1\t1\t2\t3\tInsula\tGM\tL\tfalse\n"
        "LA2\t4\t5\t6\tn/a\tWM\tR\ttrue\n"
    )
    imported, counts = parse_anatomy_table(text)
    assert imported == {"LA1": {"region": "Insula", "tissue": "gray", "hemisphere": "L", "in_lesion": False}}
    assert counts == {"rows_read": 2, "labels_imported": 1, "rows_skipped": 1}


def test_simple_tables_and_input_validation() -> None:
    imported, counts = parse_anatomy_table(
        'contact,region,tissue,in_lesion,unused\nLA1,"Head, anterior",white,1,x\nLA2, ,gray,,x\n'
    )
    assert imported == {"LA1": {"region": "Head, anterior", "tissue": "white", "in_lesion": True}}
    assert counts["labels_imported"] == 1 and counts["rows_skipped"] == 1
    imported, _ = parse_anatomy_table("contact\tregion\tin_lesion\nLA1\tInsula\tn/a\n")
    assert imported["LA1"]["in_lesion"] is None
    with pytest.raises(ToolkitError):
        parse_anatomy_table("name\tx\ty\tz\nLA1\t0\t1\t2\n")
    for updates in (
        {"patient_native": "false"}, {"labels": {"LA1": {"region": ""}}},
        {"labels": {"LA1": {"region": "Insula", "in_lesion": "yes"}}},
        {"labels": {"LA1": {"region": "Insula", "tissue": "bone"}}},
    ):
        with pytest.raises(ToolkitError):
            validate_anatomy_labels({**labels(), **updates})
    for field, code in [("source", "missing_anatomy_source"), ("patient_native", "invalid_anatomy_labels")]:
        payload = labels()
        del payload[field]
        with pytest.raises(ToolkitError) as exc:
            validate_anatomy_labels(payload)
        assert exc.value.code == code


def test_bipolar_direct_precedence_and_coverage() -> None:
    doc = labels()
    pair = {"channel_name": "SYN_A1-SYN_A2", "contacts": ["SYN_A1", "SYN_A2"]}
    matched = match_anatomy(doc, **pair)
    assert matched["display_label"] == "Insula → Amygdala (non-patient-native)"
    assert "in_lesion" not in matched
    assert match_anatomy(doc, "SYN_A1")["region"] == "Insula"
    assert match_anatomy(doc, "SYN_A10") is None
    assert anatomy_coverage(doc, [pair, {"channel_name": "SYN_A10"}]) == {
        "channels_without_labels": ["SYN_A10"], "unmatched_label_keys": ["ABSENT"],
    }
    doc["labels"]["SYN_A2"] = {"region": "Insula"}
    doc["patient_native"] = True
    assert match_anatomy(doc, **pair)["display_label"] == "Insula"
    doc["labels"]["syn_a1-syn_a2"] = {"region": "Direct pair label"}
    assert match_anatomy(doc, **pair)["display_label"] == "Direct pair label"


def test_cli_and_mcp_import_judgment_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    ws = tmp_path / "workspace"
    inspect_case(str(gate.SYNTHETIC_EDF), workspace_dir=ws)
    store = WorkbenchStore(ws)
    monkeypatch.setattr(mcp_server, "_WORKSPACE_ROOT", ws)
    path = tmp_path / "sub-demo_electrodes.tsv"
    path.write_text("name\tx\troi\nSYN_A1\t1\tInsula\n", encoding="utf-8")
    cli = ["--workspace", str(ws), "anatomy", CASE]
    assert cli_main([*cli, str(path)]) == 2
    assert json.loads(capsys.readouterr().out)["code"] == "missing_anatomy_source"
    assert mcp_server.inspect_case(CASE, anatomy_sidecar=str(path))["code"] == "missing_anatomy_source"
    assert load_anatomy_labels(store, CASE) is None
    imported = mcp_server.inspect_case(CASE, anatomy_sidecar=str(path), anatomy_source=SOURCE)
    assert imported["anatomy_labels"]["patient_native"] is False
    assert imported["anatomy_import"]["labels_imported"] == 1
    assert cli_main([*cli, str(path), "--source", SOURCE, "--patient-native"]) == 0
    assert json.loads(capsys.readouterr().out)["anatomy_labels"]["patient_native"] is True
    assert cli_main([*cli, "--clear"]) == 0
    assert load_anatomy_labels(store, CASE) is None

    # One invalid member of a composite MCP request must leave all case files untouched.
    invalid = mcp_server.record_judgment(
        CASE, clinical_question="New question", clinical_overlay={"source": SOURCE, "soz_channels": ["SYN_A1"]},
        anatomy_labels={**labels(), "source": "  "},
    )
    assert invalid["code"] == "missing_anatomy_source"
    assert store.load_clinical_overlay(CASE) is None and load_anatomy_labels(store, CASE) is None
    assert not store.load_judgments(CASE) and not store.load_case(CASE).get("open_questions")
    first = mcp_server.record_judgment(CASE, anatomy_labels=labels(), author="reviewer", operation_id="op-1")
    saved = first["anatomy_labels"]
    assert saved["author"] == "reviewer" and saved["updated_at"] and saved["external_only_disclaimer"]
    assert mcp_server.record_judgment(CASE, anatomy_labels=labels(), operation_id="op-1")["anatomy_labels"] == saved
    newer = save_anatomy_labels(store, CASE, {**labels(), "source": "Updated"}, operation_id="op-2")
    assert mcp_server.record_judgment(CASE, anatomy_labels=labels(), operation_id="op-1")["anatomy_labels"] == newer


def test_query_and_review_views(analyzed: WorkbenchStore) -> None:
    store = analyzed
    save_anatomy_labels(store, CASE, labels())
    q = query_case(CASE, workspace_dir=store.root, mode="recording_summary")
    for group in ("channels", "leading_focal_channels", "leading_raw_channels"):
        row = next(r for r in q[group] if r["channel_name"] == "SYN_A1")
        assert row["anatomy"]["display_label"] == NON_NATIVE
    assert q["anatomy_coverage"]["unmatched_label_keys"] == ["ABSENT"]
    assert "SYN_A3" in q["anatomy_coverage"]["channels_without_labels"]
    server = WorkbenchHTTPServer(store.root)
    args = {"case": CASE, "recording": CASE, "channel": "SYN_A1"}
    active = server._bootstrap({**args, "lite": "1"})["active"]
    assert active["run_summary"]["leading_focal_channels"][0]["anatomy"]["display_label"] == NON_NATIVE
    spec_row = next(r for r in active["recording_spec"]["channels"] if r["name"] == "SYN_A1")
    assert spec_row["anatomy"]["source"] == SOURCE
    for mode in ("focal", "controls"):
        items = server._queue({**args, "mode": mode})["items"]
        assert items and all(row["anatomy"]["display_label"] == NON_NATIVE for row in items)
    reader = compute_multichannel_batch(store, CASE, CASE, montage="adjacent_bipolar")
    trace = next(t for t in reader["traces"] if t["display_label"] == "SYN_A1-SYN_A2")
    assert trace["anatomy"]["display_label"] == "Insula → Amygdala (non-patient-native)"
    channels = {ch.name: ch for ch in store.load_recording(CASE, CASE).channels}
    alias = channels["SYN_A1"].model_copy(update={"name": "Acquisition alias", "contacts": ["SYN_A1"]})
    assert AnatomyMatcher(labels()).match_pair(alias, channels["SYN_A2"])["display_label"] == trace["anatomy"]["display_label"]
    for target in ({"center_seconds": 7, "event_id": None}, {"event_id": "top_focal"}):
        triad = compute_triad_view(CASE, workspace_dir=store.root, channel="SYN_A1", **target)
        assert triad["channel_name"] == "SYN_A1" and triad["anatomy"]["display_label"] == NON_NATIVE
        if triad["focus_event"]:
            assert triad["focus_event"]["anatomy"] == triad["anatomy"]
    cmp = compare_runs(store, CASE, CASE, "baseline", "baseline")
    assert next(r for r in cmp["channels"] if r["channel"] == "SYN_A1")["anatomy"]["display_label"] == NON_NATIVE
    exported = export_bids_events(CASE, workspace_dir=store.root)
    with Path(exported["tsv_path"]).open(encoding="utf-8") as file:
        assert NON_NATIVE in {row["anatomy"] for row in csv.DictReader(file, delimiter="\t")}
    metadata = json.loads(Path(exported["json_path"]).read_text())["anatomy"]
    assert metadata["Source"] == SOURCE and metadata["PatientNative"] is False


def test_http_anatomy_read_write_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Run the standard HTTP request parser and BrainLens handler on in-memory byte streams.
    class InMemoryServer:
        def __init__(self, address, handler):
            self.server_port = address[1]
            self.RequestHandlerClass = handler

        serve_forever = Mock()
        shutdown = Mock()
        server_close = Mock()

    monkeypatch.setattr(workbench_http, "ThreadingHTTPServer", InMemoryServer)
    inspect_case(str(gate.SYNTHETIC_EDF), workspace_dir=tmp_path)
    server = WorkbenchHTTPServer(tmp_path)
    server.start()

    def request(method, *, body=None, token=None, origin=None):
        payload = json.dumps(body).encode() if body is not None else b""
        headers = [f"{method} /api/anatomy?case={CASE} HTTP/1.1", "Host: localhost",
                   "Content-Type: application/json", f"Content-Length: {len(payload)}"]
        if token:
            headers.append(f"X-BrainLens-Token: {token}")
        if origin:
            headers.append(f"Origin: {origin}")
        sock = Mock()
        sock.makefile.return_value = io.BytesIO(("\r\n".join(headers) + "\r\n\r\n").encode() + payload)
        response = io.BytesIO()
        sock.sendall.side_effect = response.write
        server.httpd.RequestHandlerClass(sock, ("127.0.0.1", 1), server.httpd)
        head, data = response.getvalue().split(b"\r\n\r\n", 1)
        return int(head.split()[1]), json.loads(data)

    try:
        assert request("GET")[1]["anatomy_labels"] is None
        assert request("POST", body=labels())[0] == 401
        assert request("POST", token=server.auth_token, origin="https://example.invalid", body=labels())[0] == 403
        status, saved = request("POST", token=server.auth_token, body={**labels(), "operation_id": "post-1"})
        assert status == 200 and saved["anatomy_labels"]["source"] == SOURCE
        assert request("DELETE")[0] == 401
        status, loaded = request("GET")
        assert status == 200 and loaded["anatomy_labels"] == saved["anatomy_labels"]
        assert loaded["coverage_by_recording"][CASE]["unmatched_label_keys"] == ["ABSENT"]
        assert request("DELETE", token=server.auth_token)[0] == 200
        assert request("GET")[1]["anatomy_labels"] is None
    finally:
        server.stop()


def test_per_recording_comparison_and_cohort_export(analyzed: WorkbenchStore) -> None:
    store = analyzed
    doc = save_anatomy_labels(store, CASE, labels())
    # Two montages can use the same display name for different source contacts.
    snapshots = {}
    for rec, contact in (("r1", "SYN_A1"), ("r2", "SYN_A2")):
        bundle = store.load_run(CASE, CASE, "baseline")
        row = next(r for r in bundle["summary"]["channels"] if r["channel_name"] == "SYN_A1")
        row.update(channel_name="review-pair", contacts=[contact])
        snapshots[rec] = bundle
    comparison = compare_recordings(store, CASE, bundles=snapshots, anatomy_labels=doc)
    pair = next(r for r in comparison["aligned_active_channels"] if r["channel"] == "review-pair")
    assert pair["by_recording"]["r1"]["anatomy"]["region"] == "Insula"
    assert pair["by_recording"]["r2"]["anatomy"]["region"] == "Amygdala"

    cohort = run_cohort([CASE], workspace_dir=store.root, run_name="baseline", overrides={"detector_enabled": False})
    with Path(cohort["channels_csv_path"]).open(encoding="utf-8") as file:
        exported = next(r for r in csv.DictReader(file) if r["channel_name"] == "SYN_A1")
    assert exported["anatomy_source"] == SOURCE and exported["anatomy"] == NON_NATIVE


def test_signed_sheet_freshness_and_analysis_reuse(analyzed: WorkbenchStore) -> None:
    store = analyzed
    server = WorkbenchHTTPServer(store.root)
    generation = store.run_generation_dir(CASE, CASE, "baseline")
    original_summary = (generation / "summary.json").read_bytes()
    recording = store.load_recording(CASE, CASE)
    recording_bytes = store.recording_path(CASE, CASE).read_bytes()
    original_run = store.load_run(CASE, CASE, "baseline")
    analysis_sig = run_analysis_signature(original_run)
    cfg = AnalysisConfig.model_validate(original_run["config"]["analysis_config"])
    stage_fps = _stage_fps(recording, cfg)

    sheet = Path(export_case_sheet(CASE, workspace_dir=store.root, max_cards=0)["report_path"])
    old_signature = store.case_sheet_run_signature(CASE, "baseline")
    del old_signature["anatomy_labels_digest"]
    record_verified_sheet(store.workspace_id, CASE, sheet.name, sheet.read_bytes(), "baseline", run_signature=old_signature)
    assert server._check_verified_sheet_state(CASE, sheet) == "fresh"
    assert server._check_verified_sheet_state(CASE, sheet, check_judgments=False) == "fresh"
    watched = server._scan_case_evidence_sigs()[CASE]
    save_anatomy_labels(store, CASE, labels())
    assert server._scan_case_evidence_sigs()[CASE] != watched
    assert server._check_verified_sheet_state(CASE, sheet) == "stale"
    assert server._check_verified_sheet_state(CASE, sheet, check_judgments=False) == "stale"

    sheet = Path(export_case_sheet(CASE, workspace_dir=store.root, max_cards=0)["report_path"])
    html = sheet.read_text(encoding="utf-8")
    assert "Anatomy labels: external, source=Example localization &lt;review 4&gt;, patient-native=no, atlas=DKT" in html
    assert NON_NATIVE in html
    assert server._check_verified_sheet_state(CASE, sheet) == "fresh"
    save_anatomy_labels(store, CASE, {**labels(), "patient_native": True})
    assert server._check_verified_sheet_state(CASE, sheet) == "stale"
    reused = analyze_case(CASE, workspace_dir=store.root, run_name="baseline", overrides={"detector_enabled": False})
    assert reused["runs"][0]["already_existed"] is True
    assert reused["runs"][0]["leading_focal_channels"][0]["anatomy"]["patient_native"] is True
    current_rec = store.load_recording(CASE, CASE)
    current_run = store.load_run(CASE, CASE, "baseline")
    assert run_analysis_signature(current_run) == analysis_sig
    assert current_rec.fingerprint == recording.fingerprint
    assert _stage_fps(current_rec, cfg) == stage_fps
    assert current_run["meta"]["stage_fingerprints"] == original_run["meta"]["stage_fingerprints"]
    assert store.run_generation_dir(CASE, CASE, "baseline") == generation
    assert (generation / "summary.json").read_bytes() == original_summary
    assert store.recording_path(CASE, CASE).read_bytes() == recording_bytes
