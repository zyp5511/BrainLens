"""MDT presentation contents, sharing boundaries, and public export routes."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
import xml.etree.ElementTree as ET
from http.client import HTTPConnection
from pathlib import Path
from zipfile import ZipFile

import pytest
from matplotlib.text import Text
from pptx import Presentation

WORKBENCH_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = WORKBENCH_DIR / "src"
if importlib.util.find_spec("brainlens") is None and SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlens import __version__, evidence_map
from brainlens.anatomy import save_anatomy_labels
from brainlens.mdt_slides import export_mdt_slides
from brainlens.models import ToolkitError, atomic_write_json, read_json
from brainlens.pipeline import refresh_verdict_tallies
from brainlens.workbench_http import WorkbenchHTTPServer

import test_release_gate as gate
from test_anatomy_labels import CASE as SYNTH, NON_NATIVE, SOURCE, analyzed, labels
from test_clinical_workflow import CASE as VISIT, IDENTIFIERS, _analyze, _write_edf, reader_run


def _slide_text(slide):
    # DrawingML text covers both text boxes and table cells in presentation order.
    return "\n".join(slide._element.xpath(".//a:t/text()"))


def _open_result(result):
    path = Path(result["slides_path"])
    presentation = Presentation(path)
    assert presentation.slide_width / presentation.slide_height == pytest.approx(16 / 9, rel=1e-4)
    assert len(presentation.slides) == result["slide_count"]
    with ZipFile(path) as package:
        pictures = [name for name in package.namelist() if name.startswith("ppt/media/")]
        assert all(package.read(name).startswith(b"\x89PNG\r\n\x1a\n") for name in pictures)
        assert len(pictures) == result["embedded_images"]
    for slide in presentation.slides:
        assert slide.has_notes_slide
        assert slide.notes_slide.notes_text_frame.text.strip()
    return presentation


def _all_xml_text(path):
    with ZipFile(path) as package:
        return "\n".join(
            " ".join(ET.fromstring(package.read(name)).itertext()) + "\n" + package.read(name).decode("utf-8")
            for name in package.namelist() if name.endswith((".xml", ".rels"))
        )


@pytest.mark.parametrize("with_overlay", [False, True])
def test_synthetic_deck_and_card_budget(analyzed, with_overlay):
    store = analyzed
    if with_overlay:
        save_anatomy_labels(store, SYNTH, labels())
        store.save_clinical_overlay(SYNTH, {
            "source": "Reference chart", "soz_channels": ["SYN_A1"], "resected_channels": ["SYN_A2"],
        })
        # This clipped example is on the second-ranked channel. With two card slots
        # it should displace the automatic focal example and leave room for a control.
        event = next(e for e in store.load_run(SYNTH, SYNTH, "baseline")["events"] if e["channel_name"] == "SYN_A2")
        anchor = {"recording": SYNTH, "run_name": "baseline", "channel": "SYN_A2", "event_id": event["id"]}
        store.record_judgment(SYNTH, {**anchor, "kind": "event_clip", "role": "clinician", "author": "A", "clipped": True})
        for role, verdict in (("clinician", "confirm"), ("researcher", "uncertain"),
                              ("engineering_rehearsal", "artifact"), ("agent", "confirm")):
            store.record_judgment(SYNTH, {**anchor, "kind": "event_verdict", "role": role,
                                          "author": role, "verdict": verdict})
        refresh_verdict_tallies(store, SYNTH, SYNTH, "baseline")
    question = "Which focal contacts lead in this recording?"
    result = export_mdt_slides(SYNTH, store.root, clinical_question=question, max_cards=2)
    presentation = _open_result(result)
    assert Path(result["slides_path"]) == store.case_dir(SYNTH) / "exports" / f"{SYNTH}_baseline_mdt_slides.pptx"
    assert result["case"] == SYNTH and result["primary_run"] == "baseline" and result["comparison_run"] is None
    assert isinstance(result["warnings"], list)
    assert result["embedded_images"] == 3  # spatial projection, focal example, candidate-free control
    slides = [_slide_text(slide) for slide in presentation.slides]
    title = slides[0].lower()
    assert all(text in title for text in (SYNTH, "baseline", "not a medical device", "not a standalone"))
    assert "resection boundary" in title
    assert question in slides[1]
    assert "control" in " ".join(slides).lower()
    assert "synthetic" in slides[-1].lower() and __version__ in slides[-1]
    leaders = [i for i, slide in enumerate(presentation.slides)
               if any(shape.has_table for shape in slide.shapes) and "SYN_A1" in slides[i]]
    assert leaders
    if with_overlay:
        assert any(all(tag in slides[i] for tag in ("SOZ", "Resected")) for i in leaders)
        assert any("external reference" in slides[i].lower() for i in leaders)
        assert all(NON_NATIVE in slides[i] and SOURCE in slides[i] for i in leaders)
        assert any(NON_NATIVE in text and "control" in text.lower() and SOURCE in text for text in slides)
        clipped = next(text for text, s in zip(slides, presentation.slides) if s._element.xpath(".//p:pic")
                       and "clip" in text.lower() and "SYN_A2" in text)
        assert all(role in clipped.lower() for role in ("clinician", "researcher", "rehearsal", "agent"))
    else:
        assert NON_NATIVE not in "\n".join(slides)
        # A constrained selection must still include the available candidate-free control.
        compact = export_mdt_slides(SYNTH, store.root, max_cards=1)
        compact_deck = _open_result(compact)
        assert compact["embedded_images"] == 2
        assert compact["slide_count"] == result["slide_count"] - 1
        assert any("control" in _slide_text(s).lower() and s._element.xpath(".//p:pic") for s in compact_deck.slides)


def test_private_identity_filtered_in_slides_notes_and_raster_figures(reader_run, monkeypatch):
    store = reader_run
    question = f"Review {VISIT} and " + " / ".join(item.decode() for item in IDENTIFIERS)
    # Put native header terms in separately gathered material and in the persisted readers' names.
    save_anatomy_labels(store, VISIT, {
        "source": "DOe_JANE location report", "patient_native": False, "atlas": "test atlas",
        "labels": {"LA1": {"region": "Anterior MRN123456"}, "LB1": {"region": "Posterior HOSP983"}},
    })
    bundle = store.load_run(VISIT, VISIT, "baseline")
    # Give the fixture's planted events measured coverage so both triad cards and
    # the evidence map include labels drawn from this identifiable recording.
    det_path = Path(bundle["meta"]["detection_json_path"])
    det = read_json(det_path)
    n_samples = store.load_recording(VISIT, VISIT).n_samples
    det["valid_intervals"] = {r["channel_id"]: [[0, n_samples]] for r in bundle["summary"]["channels"]}
    atomic_write_json(det_path, det)
    events = bundle["events"]
    for index, event in enumerate(events[:3]):
        for role, author in (("clinician", "Reader DOE_JANE"), ("researcher", "Reader HOSP983")):
            store.record_judgment(VISIT, {
                "kind": "event_verdict", "recording": VISIT, "run_name": "baseline", "channel": event["channel_name"],
                "event_id": event["id"], "role": role, "author": author,
                "verdict": ("confirm", "artifact", "uncertain")[index],
            })
    refresh_verdict_tallies(store, VISIT, VISIT, "baseline")
    evidence_map.get_evidence_map(store, VISIT, VISIT, run_name="baseline")

    def no_compute(*args, **kwargs):
        pytest.fail("slide export started a new evidence measurement")

    monkeypatch.setattr(evidence_map, "_base_rows", no_compute)
    monkeypatch.setattr(evidence_map, "compute_ei", no_compute)
    internal = export_mdt_slides(VISIT, store.root, clinical_question=question, max_cards=2)
    original = _all_xml_text(internal["slides_path"])
    assert VISIT in original and all(identifier.decode() in original for identifier in IDENTIFIERS)

    # Observe strings actually drawn into the raster assets. Searching compressed PNG bytes
    # would miss the disclosure if a title or channel label still identified the recording.
    drawn = []
    original_draw = Text.draw

    def capture_draw(artist, renderer):
        if artist.get_visible():
            drawn.append(artist.get_text())
        return original_draw(artist, renderer)

    monkeypatch.setattr(Text, "draw", capture_draw)
    shared = export_mdt_slides(VISIT, store.root, deidentify=True, pseudonym="P001", max_cards=2)
    presentation = _open_result(shared)
    xml = _all_xml_text(shared["slides_path"])
    path = Path(shared["slides_path"])
    assert shared["case"] == "P001" and "P001" in path.name and path != Path(internal["slides_path"])
    assert "P001" in xml and "P001" in " ".join(drawn)
    disclosure = (xml + "\n" + "\n".join(drawn) + "\n" + path.name).casefold()
    assert all(value.casefold() not in disclosure for value in [VISIT, *(v.decode() for v in IDENTIFIERS)])
    slides = [_slide_text(s) for s in presentation.slides]
    assert any("descriptive, not a validated predictor" in s.lower() for s in slides)
    assert shared["embedded_images"] == 3  # focal, control, and the cached map
    agreement = next(s for s in presentation.slides if "agreement" in _slide_text(s).lower() and "κ" in _slide_text(s))
    table = next(shape.table for shape in agreement.shapes if shape.has_table)
    assert "95%" in table.cell(0, 4).text
    # All three jointly reviewed events have identical, non-constant labels (Cohen's κ = 1).
    pair = next(row for row in table.rows if row.cells[1].text == "3")
    assert "clinician" in pair.cells[0].text and "researcher" in pair.cells[0].text
    assert float(pair.cells[2].text.rstrip("%")) == 100
    assert float(pair.cells[3].text) == 1
    assert pair.cells[4].text == "1.000 to 1.000"
    with pytest.raises(ToolkitError) as exc:
        export_mdt_slides(VISIT, store.root, deidentify=True, pseudonym=IDENTIFIERS[0].decode(), max_cards=0)
    assert exc.value.code == "invalid_pseudonym"


def test_absent_cache_branch_and_template_coordinates(analyzed, monkeypatch):
    store = analyzed
    gate._analyze_synthetic_run(store.root, run_name="sensitivity")
    rec = store.load_recording(SYNTH, SYNTH)
    rec.spatial.template = True
    rec.spatial.coordinate_system = "fsaverage"
    # Registration owns the spatial frame. Let the normal snapshot reconciliation see the change.
    path = store.recording_path(SYNTH, SYNTH)
    doc = read_json(path)
    doc["spec"]["spatial"] = rec.spatial.model_dump(mode="json")
    atomic_write_json(path, doc)
    monkeypatch.setattr(evidence_map, "_base_rows", lambda *a, **k: pytest.fail("cache miss computed"))
    result = export_mdt_slides(SYNTH, store.root, comparison_run="sensitivity", max_cards=0)
    presentation = _open_result(result)
    assert result["comparison_run"] == "sensitivity"
    slides = [_slide_text(s) for s in presentation.slides]
    assert "sensitivity" in slides[1].lower()
    assert all("evidence map" not in s.lower() and "inter-rater agreement" not in s.lower() for s in slides)
    spatial = next(text for text, s in zip(slides, presentation.slides) if s._element.xpath(".//p:pic"))
    assert "template" in spatial.lower() and "patient-native" in spatial.lower()


def test_multiple_recordings_retain_own_identities_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    source = tmp_path / "TaylorConference"
    source.mkdir()
    for rec in ("DaySample", "NightSample"):
        _write_edf(source / f"{rec}.edf")
    (source / "DaySample.json").write_text(json.dumps({"synthetic": True}))
    (source / "NightSample.json").write_text(json.dumps({"provenance": {"dataset": {"name": "Public test cohort"}}}))
    workspace = tmp_path / "ws"
    _analyze(source, workspace)
    shared = export_mdt_slides(source.name, workspace, deidentify=True, pseudonym="R006", max_cards=0)
    deck = _open_result(shared)
    text = "\n".join(_slide_text(s) for s in deck.slides)
    xml = _all_xml_text(shared["slides_path"]).lower()
    assert all(name.lower() not in xml for name in (source.name, "DaySample", "NightSample"))
    assert "R006-01" in text and "R006-02" in text
    assert "Public test cohort" in text and "synthetic" in text.lower()


def test_stale_recording_rejected_before_export(reader_run):
    meta_path = reader_run.run_generation_dir(VISIT, VISIT, "baseline") / "run_meta.json"
    atomic_write_json(meta_path, {**read_json(meta_path), "recording_fingerprint": "superseded"})
    with pytest.raises(ToolkitError) as raised:
        export_mdt_slides(VISIT, reader_run.root, max_cards=0)
    assert raised.value.code == "run_recording_mismatch"
    assert not list((reader_run.case_dir(VISIT) / "exports").glob("*.pptx"))


def test_case_change_during_export_preserves_published_deck(analyzed, monkeypatch):
    from pptx.presentation import Presentation as PresentationDocument

    store = analyzed
    output = Path(export_mdt_slides(SYNTH, store.root, max_cards=0)["slides_path"])
    original = output.read_bytes()
    previous_save = PresentationDocument.save

    def save_then_change_case(presentation, path):
        previous_save(presentation, path)
        store.add_open_question(SYNTH, "Question added during rendering.")

    monkeypatch.setattr(PresentationDocument, "save", save_then_change_case)
    with pytest.raises(ToolkitError) as raised:
        export_mdt_slides(SYNTH, store.root, max_cards=0)
    assert raised.value.code == "case_evidence_changed"
    assert output.read_bytes() == original
    assert not list(output.parent.glob("*.tmp"))


def test_declared_participant_can_share_a_recording_name(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    source = tmp_path / "PublicLabel.edf"
    _write_edf(source, code=source.stem)
    atomic_write_json(source.with_suffix(".json"), {"participant_id": source.stem})
    workspace = tmp_path / "workspace"
    _analyze(source, workspace)
    result = export_mdt_slides(source.stem, workspace, deidentify=True, pseudonym="P009", max_cards=0)
    slides = [_slide_text(s) for s in _open_result(result).slides]
    assert "In P009," in slides[1]
    assert source.stem not in "\n".join(slides[:-1])
    assert source.stem in slides[-1]  # The sidecar authorized this participant label.


def test_missing_slides_dependency_reports_install_hint(analyzed, monkeypatch):
    monkeypatch.setitem(sys.modules, "pptx", None)
    with pytest.raises(ToolkitError) as raised:
        export_mdt_slides(SYNTH, analyzed.root, max_cards=0)
    assert raised.value.code == "slides_dependency_missing"
    assert "install" in raised.value.message.lower() and "python-pptx" in raised.value.message


def test_cli_and_mcp_export(reader_run, monkeypatch, capsys):
    from mcp.shared.memory import create_connected_server_and_client_session
    from brainlens import mcp_server
    from brainlens.cli import main

    monkeypatch.setattr(mcp_server, "_WORKSPACE_ROOT", reader_run.root)

    async def export():
        async with create_connected_server_and_client_session(mcp_server.mcp) as client:
            listed = await client.list_tools()
            names = {tool.name for tool in listed.tools}
            assert names == {"inspect_case", "analyze_case", "query_evidence", "render_evidence_card",
                             "open_workbench", "record_judgment", "export_or_protocol"}
            tool = next(t for t in listed.tools if t.name == "export_or_protocol")
            assert "export_mdt_slides" in tool.inputSchema["properties"]["action"]["enum"]
            reply = await client.call_tool("export_or_protocol", {
                "action": "export_mdt_slides", "case": VISIT, "primary_run": "baseline",
                "clinical_question": "Question from the conference.", "deidentify": True, "pseudonym": "P011",
            })
            assert not reply.isError
            return json.loads(reply.content[0].text)

    via_mcp = asyncio.run(export())
    assert via_mcp["case"] == "P011"
    capsys.readouterr()
    assert main(["--workspace", str(reader_run.root), "export-slides", VISIT, "--primary-run", "baseline",
                 "--deidentify", "--pseudonym", "P011"]) == 0
    via_cli = json.loads(capsys.readouterr().out)
    assert via_mcp["slide_count"] == via_cli["slide_count"]
    deck = _open_result(via_cli)
    assert "Question from the conference." in _slide_text(deck.slides[1])


def test_http_job_download_and_attachment(reader_run):
    server = WorkbenchHTTPServer(reader_run.root)
    server.start()

    def request(method, path, body=None, *, authorize=True):
        conn = HTTPConnection("127.0.0.1", server.port, timeout=10)
        try:
            headers = {"Content-Type": "application/json"}
            if authorize:
                headers["X-BrainLens-Token"] = server.auth_token
            conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
            reply = conn.getresponse()
            return reply.status, reply.headers, reply.read()
        finally:
            conn.close()

    try:
        args = {"case": VISIT, "deidentify": True, "pseudonym": "P017", "primary_run": "baseline"}
        assert request("POST", "/api/export-slides", args, authorize=False)[0] in (401, 403)
        status, _, body = request("POST", "/api/export-slides", args)
        assert status == 202, body
        job = json.loads(body)
        deadline = time.monotonic() + 30
        while job["status"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.05)
            status, _, body = request("GET", f"/api/jobs/{job['job_id']}")
            assert status == 200
            job = json.loads(body)
        assert job["status"] == "complete", job
        out = job["result"]
        path = Path(out["slides_path"])
        url = next(value for key, value in out.items() if key.endswith("_url") and value.endswith(".pptx"))
        status, headers, payload = request("GET", url)
        assert status == 200 and payload == path.read_bytes()
        assert headers.get_content_disposition() == "attachment"
        assert headers.get_content_type() == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        assert headers.get_filename() == path.name and "P017" in path.name
    finally:
        server.stop()
