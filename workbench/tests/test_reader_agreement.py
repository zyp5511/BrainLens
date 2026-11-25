"""Reader reliability, frozen sampling, live projections and report integration."""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import io
import json
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlencode

import pytest

WORKBENCH_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = WORKBENCH_DIR / "src"
if importlib.util.find_spec("brainlens") is None and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlens import cli, mcp_server
from brainlens.agreement import agreement_from_snapshot, reader_agreement
from brainlens.case_sheet import export_case_sheet
from brainlens.models import ToolkitError
from brainlens.pipeline import analyze_case
from brainlens.review_sets import create_review_set, list_review_sets, load_review_set, reader_event_order
from brainlens.storage import WorkbenchStore
from brainlens.workbench_http import WorkbenchHTTPServer

CASE = REC = "synthetic_seeg"
C, A, U = "confirm", "artifact", "uncertain"


def _snapshot(tmp_path, ratings, *, readers=None, extra=(), floor=0):
    events, wal = [], []
    for i, row in enumerate(ratings):
        eid = f"e{i}"
        events.append({"id": eid, "channel_name": f"A{i % 2}", "start_seconds": i / 2,
                       "synchrony_class": "regional" if i % 2 else "focal"})
        for author, verdict in zip(("Alice", "Bob", "Chen"), row):
            if verdict:
                wal.append({"recording": REC, "event_id": eid, "author": author, "verdict": verdict,
                            "role": "researcher" if author == "Bob" else "clinician"})
    ctx = {"rec_fps": {REC: "fp"}, "run_asigs": {(REC, "baseline"): "sig"},
           "run_wal_floors": {(REC, "baseline"): floor}}
    bundle = {"meta": {"recording_fingerprint": "fp", "analysis_signature": "sig"}, "events": events}
    return agreement_from_snapshot(WorkbenchStore(tmp_path), CASE, REC, "baseline", bundle, wal + list(extra),
                                   ctx, readers=readers)


def test_textbook_cohen(tmp_path):
    # Classic 50-item example. Each row totals 25, so chance agreement is 1/2.
    ratings = [(C, C)] * 20 + [(C, A)] * 5 + [(A, C)] * 10 + [(A, A)] * 15
    report = _snapshot(tmp_path, ratings)
    pair = report["pairwise"][0]
    assert pair["readers"] == ["Alice", "Bob"]
    assert pair["confusion_matrix"] == [[20, 5, 0], [10, 15, 0], [0, 0, 0]]
    assert pair["percent_agreement"] == 70 and pair["n_shared"] == 50
    assert pair["kappa"]["value"] == pytest.approx(0.4)
    assert pair["binary_kappa"] == pair["kappa"]
    lo, hi = pair["kappa"]["ci95"]
    assert -1 <= lo < 0.4 < hi <= 1
    assert pair["prevalence_index"] == pytest.approx(0.1)
    assert pair["bias_index"] == pytest.approx(0.1)


def test_three_categories_and_disagreements(tmp_path):
    ratings = [(C, C), (C, C), (C, A), (A, C), (A, A), (A, A), (U, A), (U, U), (C, None)]
    report = _snapshot(tmp_path, ratings)
    p = report["pairwise"][0]
    assert report["n_events"] == 9 and p["n_shared"] == 8
    assert p["confusion_matrix"] == [[2, 1, 0], [1, 2, 0], [0, 1, 1]]
    assert p["percent_agreement"] == 62.5
    assert p["kappa"]["value"] == pytest.approx(17 / 41)
    assert p["binary_kappa"]["value"] == pytest.approx(7 / 15)
    assert report["event_matrix"][-1]["verdicts"] == {"Alice": C, "Bob": None}
    assert [r["event_id"] for r in report["disagreements"]] == ["e2", "e3", "e6"]
    assert report["disagreements"][0] == {"event_id": "e2", "channel": "A0", "time_seconds": 1.0,
                                          "synchrony_class": "focal", "verdicts": {"Alice": C, "Bob": A}}
    assert report["by_synchrony_class"]["regional"]["pairwise"][0]["n_shared"] == 4


def test_fleiss_complete_cases_only(tmp_path):
    report = _snapshot(tmp_path, [(C, C, C), (C, A, C), (A, A, A), (U, U, A), (A, U, None)])
    assert len(report["pairwise"]) == 3
    assert report["pairwise"][0]["n_shared"] == 5
    assert report["fleiss"]["n_shared"] == 4
    assert report["fleiss"]["value"] == pytest.approx(7 / 15)
    assert report["fleiss"]["ci95"][0] <= 7 / 15 <= report["fleiss"]["ci95"][1]


@pytest.mark.parametrize("ratings,n,pct", [([(C, C), (C, C)], 2, 100), ([(U, A)], 1, 0),
                                           ([(C, None), (None, A)], 0, None)])
def test_undefined_kappa(tmp_path, ratings, n, pct):
    report = _snapshot(tmp_path, ratings)
    p = report["pairwise"][0]
    assert p["n_shared"] == n and p["percent_agreement"] == pct
    for metric in ("kappa", "binary_kappa"):
        assert p[metric]["value"] is None and p[metric]["ci95"] is None and p[metric]["reason"]
    json.dumps(report, allow_nan=False)


def test_latest_per_reader_and_snapshot_binding(tmp_path):
    row = {"author": "Alice", "role": "clinician", "event_id": "e0", "recording": REC, "verdict": A}
    extra = [row, {**row, "verdict": None, "note": "later note"}, {**row, "verdict": U, "role": "agent"},
             {**row, "verdict": U, "recording": "another"}, {**row, "verdict": U, "run_name": "another"},
             {**row, "verdict": U, "recording_fingerprint": "old"}, {**row, "verdict": U, "analysis_signature": "old"},
             {**row, "author": "Bot", "role": "unrecognized"}, {**row, "author": "Demo", "role": "engineering_rehearsal"}]
    report = _snapshot(tmp_path, [(C, C), (C, A)], extra=extra, floor=1)
    assert report["readers"] == ["Alice", "Bob"]
    assert report["event_matrix"][0]["verdicts"] == {"Alice": A, "Bob": C}
    selected = _snapshot(tmp_path, [(C, C)], readers=["Bot", "Demo"], extra=extra)
    assert selected["reader_roles"] == {"Bot": "agent", "Demo": "engineering_rehearsal"}
    assert selected["pairwise"][0]["n_shared"] == 1
    truncated = _snapshot(tmp_path, [(C, C), (A, U)], floor=1)
    assert truncated["event_matrix"][0]["verdicts"] == {"Alice": None, "Bob": C}


def test_ambiguous_roles_require_distinct_reader_identity(tmp_path):
    other_role = [{"author": "Alice", "role": "researcher", "recording": REC, "event_id": "e0", "verdict": A}]
    ratings = [(C, C), (C, A)]
    report = _snapshot(tmp_path, ratings, extra=other_role)
    assert report["excluded_ambiguous_readers"] == {"Alice": ["clinician", "researcher"]}
    selected = _snapshot(tmp_path, ratings, readers=["researcher:Alice", "Bob"], extra=other_role)
    assert selected["pairwise"][0]["n_shared"] == 1 and selected["pairwise"][0]["percent_agreement"] == 0
    for readers, code in [(["Alice", "Bob"], "ambiguous_reader"),
                          (["clinician:Alice", "researcher:Alice"], "duplicate_reader")]:
        with pytest.raises(ToolkitError) as error:
            _snapshot(tmp_path, ratings, readers=readers, extra=other_role)
        assert error.value.code == code


@pytest.fixture
def stored_run(tmp_path, monkeypatch):
    # Exercise the same six real synthetic EDF windows on machines with or without PyHFO.
    import test_release_gate as gate

    monkeypatch.setattr(gate, "_has_pyhfo_engine", lambda: False)
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    store = WorkbenchStore(tmp_path / "ws")
    gate._analyze_synthetic_run(store.root)
    events = store.load_run(CASE, REC, "baseline")["events"]
    return store, [e["id"] for e in events]


def _vote(store, eid, author, role, verdict, note=""):
    store.record_judgment_ex(CASE, {"recording": REC, "run_name": "baseline", "event_id": eid, "kind": "verdict",
                                    "author": author, "role": role, "verdict": verdict, "note": note},
                             validate_event_ownership=True)


def test_frozen_set_sampling_reader_order_and_staleness(stored_run, monkeypatch):
    store, ids = stored_run
    # A saturated majority stratum must not crowd out rare synchrony/band combinations.
    bundle = store.load_run(CASE, REC, "baseline")
    strata = [("focal", "ripple")] * 3 + [("regional", "ripple"), ("focal", "fast_ripple"), ("regional", "fast_ripple")]
    for ev, (cls, band) in zip(bundle["events"], strata, strict=True):
        ev.update(synchrony_class=cls, dominant_band=band)
    with monkeypatch.context() as m:
        m.setattr(store, "load_run", lambda *a, **kw: bundle)
        sample = create_review_set(store, CASE, REC, "baseline", "trial", 4, 17)
        twin = create_review_set(store, CASE, REC, "baseline", "replicate", 4, 17)
    assert sample["event_ids"] == twin["event_ids"]
    assert {strata[ids.index(eid)] for eid in sample["event_ids"]} == set(strata)
    assert load_review_set(store, CASE, REC, "baseline", "trial") == sample
    full = create_review_set(store, CASE, REC, "baseline", "all")
    assert set(full["event_ids"]) == set(ids)
    alice = reader_event_order(full, "Alice")
    assert alice == reader_event_order(full, "Alice") and alice != reader_event_order(full, "Bob")
    assert set(alice) == set(ids)
    assert {d["name"] for d in list_review_sets(store, CASE, REC, "baseline")} == {"all", "trial", "replicate"}
    for name, size, code in [("trial", None, "review_set_exists"), ("../x", None, "invalid_review_set_name"),
                              ("too_many", len(ids) + 1, "invalid_sample_size")]:
        with pytest.raises(ToolkitError) as error:
            create_review_set(store, CASE, REC, "baseline", name, size)
        assert error.value.code == code
    with pytest.raises(ToolkitError, match="another recording or run"):
        load_review_set(store, CASE, REC, "other", "trial")
    analyze_case(f"{CASE}/{REC}", workspace_dir=store.root, run_name="baseline", overwrite=True,
                 overrides={"detector_enabled": False, "reference_mode": "average"})
    for read in (lambda: load_review_set(store, CASE, REC, "baseline", "trial"),
                 lambda: reader_agreement(store, CASE, REC, review_set="trial")):
        with pytest.raises(ToolkitError) as error:
            read()
        assert error.value.code == "stale_review_set"


def test_mcp_cli_csv_and_signed_sheet(stored_run, monkeypatch, capsys):
    store, ids = stored_run
    monkeypatch.setattr(mcp_server, "_WORKSPACE_ROOT", store.root)
    assert {tool.name for tool in asyncio.run(mcp_server.mcp.list_tools())} == {
        "inspect_case", "analyze_case", "query_evidence", "render_evidence_card",
        "open_workbench", "record_judgment", "export_or_protocol",
    }
    doc = mcp_server.export_or_protocol(action="create_review_set", case=CASE, recording=REC, review_set="study",
                                        seed=7, sample_size=4)
    shared = doc["event_ids"][:2]
    outside = next(eid for eid in ids if eid not in doc["event_ids"])
    for eid in [*shared, outside]:
        _vote(store, eid, "R1", "clinician", C)
        _vote(store, eid, "R2", "researcher", A if eid == shared[0] else C)
    report = mcp_server.query_evidence(target=f"{CASE}/{REC}", mode="reader_agreement", review_set="study")
    assert report["n_events"] == 4 and report["pairwise"][0]["n_shared"] == 2
    assert report["pairwise"][0]["percent_agreement"] == 50
    files = mcp_server.export_or_protocol(action="export_agreement_report", case=CASE, recording=REC, review_set="study")
    for key in ("json_path", "matrix_csv_path", "disagreements_csv_path"):
        assert Path(files[key]).is_relative_to(store.case_dir(CASE) / "exports")
    assert json.loads(Path(files["json_path"]).read_text())["analysis_signature"] == doc["analysis_signature"]
    with Path(files["disagreements_csv_path"]).open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1 and rows[0]["event_id"] == shared[0] and rows[0]["verdict:R2"] == A
    with Path(files["matrix_csv_path"]).open() as fh:
        matrix = list(csv.DictReader(fh))
    assert [r["event_id"] for r in matrix] == doc["event_ids"]
    assert matrix[-1]["verdict:R1"] == ""
    base = ["--workspace", str(store.root)]
    args = [CASE, "--recording", REC]
    assert cli.main([*base, "review-set", "create", *args, "--name", "other", "--seed", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["n_events"] == len(ids)
    assert cli.main([*base, "review-set", "list", *args]) == 0
    assert len(json.loads(capsys.readouterr().out)["review_sets"]) == 2
    assert cli.main([*base, "agreement", *args, "--readers", "R2,R1", "--review-set", "study"]) == 0
    assert json.loads(capsys.readouterr().out)["pairwise"][0]["readers"] == ["R2", "R1"]
    sheet = export_case_sheet(CASE, workspace_dir=store.root)
    path = Path(sheet["report_path"])
    assert "Inter-rater agreement" in path.read_text() and "66.7%" in path.read_text()
    assert WorkbenchHTTPServer(store.root)._is_verified_fresh_sheet(CASE, path, "baseline")


def _in_memory_http(workspace):
    """Drive the production HTTP handler with bytes, also on hosts that forbid bind()."""
    server = WorkbenchHTTPServer(workspace)
    with patch.object(ThreadingHTTPServer, "__init__", side_effect=StopIteration) as init:
        with pytest.raises(StopIteration):
            server.start()
    handler = init.call_args.args[1]
    server.httpd = SimpleNamespace(server_port=8765)

    def request(path, params, body=None):
        payload = json.dumps(body).encode() if body is not None else b""
        method = "POST" if body is not None else "GET"
        head = (f"{method} {path}?{urlencode(params)} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n"
                f"Content-Type: application/json\r\nX-BrainLens-Token: {server.auth_token}\r\n"
                f"Content-Length: {len(payload)}\r\n\r\n").encode()
        response = bytearray()
        conn = SimpleNamespace(makefile=lambda *args: io.BytesIO(head + payload), sendall=response.extend)
        handler(conn, ("127.0.0.1", 1), server.httpd)
        headers, raw = bytes(response).split(b"\r\n\r\n", 1)
        return int(headers.split()[1]), json.loads(raw)

    return request


def test_blind_http_payloads_and_progress(stored_run):
    store, ids = stored_run
    request = _in_memory_http(store.root)
    target = {"case": CASE, "recording": REC, "run_name": "baseline"}
    _vote(store, ids[0], "Alice", "clinician", C, "my-own-note")
    _vote(store, ids[0], "Bob", "clinician", A, "SECRET-same-role")
    _vote(store, ids[1], "Scientist", "researcher", U, "SECRET-researcher")
    _vote(store, ids[0], "Robot", "agent", U, "SECRET-agent")
    status, doc = request("/api/review-sets", {}, {**target, "name": "session"})
    assert status == 200 or status == 201, doc
    assert request("/api/review-sets", target)[1]["review_sets"][0]["name"] == "session"
    blind = {**target, "blind_for": "Alice"}
    q = {**blind, "review_set": "session", "mode": "review_set", "reader": "Bob"}
    status, queue = request("/api/queue", q)
    assert status == 200 and [e["id"] for e in queue["items"]] == reader_event_order(doc, "Alice")
    assert queue["progress"] == {"judged": 1, "total": 6}
    first = next(e for e in queue["items"] if e["id"] == ids[0])
    assert first["clinician_verdict"] == C
    assert request("/api/queue", {**q, "limit": 2, "offset": 1})[1]["items"] == queue["items"][1:3]
    for route, params in [("queue", q), ("bootstrap", {**blind, "lite": 1}),
                          ("triad", {**blind, "event_id": ids[0], "channel": "SYN_A1"}),
                          ("multichannel", {**blind, "start_seconds": 0, "duration_seconds": 4})]:
        status, data = request("/api/" + route, params)
        assert status == 200, data
        raw = json.dumps(data)
        for secret in ("SECRET-", '"agent_verdict"', '"researcher_verdict"', '"clinician_confirmed"', '"Bob"', '"Robot"'):
            assert secret not in raw, (route, secret)
        if route == "bootstrap":
            assert "my-own-note" in raw and data["active"]["sheet_url"] is None
        if route == "triad":
            assert data["focus_event"]["clinician_verdict"] == C
    for route in ("/api/agreement", "/api/events", "/api/case-verdicts", "/api/report-actions", "/exports/mock/sheet.html"):
        assert request(route, blind)[0] == 403
    status, report = request("/api/agreement", {**target, "review_set": "session", "readers": "Alice,Bob"})
    assert status == 200 and report["disagreements"][0]["event_id"] == ids[0]
    # Protect both the user's pending event and the frozen population on submit.
    _, subset = request("/api/review-sets", {}, {**target, "name": "small", "sample_size": 2})
    vote = {**target, "review_set": "small", "blind_for": "Alice", "author": "Alice", "role": "clinician",
            "verdict": A, "event_id": subset["event_ids"][0]}
    assert request("/api/judgment", {}, vote) == (200, {"ok": True})
    outside = next(eid for eid in ids if eid not in subset["event_ids"])
    for ineligible in (outside, None):
        status, error = request("/api/judgment", {}, {**vote, "event_id": ineligible})
        assert status == 400 and error["error"]["code"] == "event_not_in_review_set"
    assert request("/api/queue", {**q, "blind_for": ""})[0] == 400
