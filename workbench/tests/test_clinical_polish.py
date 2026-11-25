"""Regression coverage for clinical display of acquired signals and saved evidence."""

import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from lxml import html
from matplotlib.text import Text
from pptx import Presentation

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if importlib.util.find_spec("brainlens") is None and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlens.analytics import annotate_summary_anatomy
from brainlens.case_sheet import SHEET_DISPLAY_META, export_case_sheet
from brainlens.clinical_display import model_score_labels
from brainlens.io_readers import inspect_case
from brainlens.mdt_slides import export_mdt_slides
from brainlens.models import atomic_write_json, read_json, record_verified_sheet, verify_sheet_trust
from brainlens.pipeline import analyze_case, refresh_verdict_tallies
from brainlens.storage import WorkbenchStore
from brainlens.timeline import admission_timeline
from brainlens.trends import recording_trends
from brainlens.triad_views import compute_multichannel_batch, compute_triad_view
from brainlens.workbench_http import WorkbenchHTTPServer
from test_browser_cdp import _build_two_case_fixture, _reader_wait, reader_cdp
from test_long_recordings import _synthetic_edf
from test_release_gate import _write_fif


@pytest.mark.parametrize(("modality", "kind", "acquired", "expected"), [
    ("scalp", "eeg", ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3"],
     ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3"]),
    ("scalp", "eeg", ["P3", "P1", "C3", "C1"], ["P3", "P1", "C3", "C1"]),
    ("ieeg", "seeg", ["Vertex", "F7", "Midline", "C3"], ["Vertex", "F7", "Midline", "C3"]),
    ("ieeg", "ecog", ["G3", "G1", "G2", "G4"], ["G3", "G1", "G2", "G4"]),
    ("seeg", "seeg", ["LB10", "Vertex", "LA2", "LB2", "LA1"], ["LA1", "Vertex", "LA2", "LB2", "LB10"]),
])
def test_montage_order_in_pages_neighborhood_and_exports(tmp_path, modality, kind, acquired, expected):
    source = _write_fif(tmp_path / "montage_raw.fif", acquired, 512, 2048,
                        ch_types=kind, sidecar={"modality": modality})
    store = WorkbenchStore(tmp_path / "ws")
    inspected = inspect_case(str(source), workspace_dir=store.root)
    case = inspected["case"]
    slug = inspected["recordings"][0]["recording"]
    recording = store.load_recording(case, slug)

    def traces(**selection):
        return compute_multichannel_batch(store, case, slug, duration_seconds=.5, **selection)["traces"]

    assert [t["channel_name"] for t in traces(shaft="ALL", source_only=True)] == expected
    by_page = [t["channel_name"] for i in range((len(expected) + 1) // 2) for t in traces(page=i, page_size=2)]
    assert by_page == expected
    focus = expected[2]
    near = traces(shaft="NEIGHBORHOOD", focus_channel=focus, page_size=3)
    assert [t["channel_name"] for t in near] == expected[1:4]
    triad = compute_triad_view(case, store.root, recording=slug, channel=focus, center_seconds=1, source_only=True)
    if modality == "seeg":
        assert triad["adjacent_bipolar"]["label"] == "LA2-LA1"
    else:
        assert triad["adjacent_bipolar"] is None

    analyze_case(f"{case}/{slug}", run_name="baseline", workspace_dir=store.root, overrides={"detector_enabled": False})
    bundle = store.load_run(case, slug, "baseline")
    boot = WorkbenchHTTPServer(store.root)._bootstrap({"case": case, "recording": slug, "lite": "1"})["active"]
    assert [r["name"] for r in boot["recording_spec"]["channels"]] == expected
    assert [r["channel_name"] for r in boot["run_summary"]["channels"]] == expected
    for measure in ("rms", "hfo_rate"):
        trend = recording_trends(store, case, slug, measure=measure, epoch_seconds=2)
        assert [r["name"] for r in trend["channels"]] == expected
    csv_path = store.run_generation_dir(case, slug, "baseline") / "channel_rates.csv"
    with csv_path.open(newline="") as stream:
        assert [row["channel_name"] for row in csv.DictReader(stream)] == expected

    # Old summaries may be ranked/alphabetized. Reproject them without changing the stored JSON.
    old = dict(bundle["summary"], channels=sorted(bundle["summary"]["channels"], key=lambda r: r["channel_name"]))
    old_order = [row["channel_name"] for row in old["channels"]]
    view = annotate_summary_anatomy(old, None, recording=recording)
    assert [row["channel_name"] for row in view["channels"]] == expected
    assert [row["channel_name"] for row in old["channels"]] == old_order


@pytest.mark.parametrize(("header_date", "next_date"), [("06.11.76", "07.11.76"), ("01.01.85", "02.01.85")])
def test_surrogate_headers_preserve_cross_midnight_offsets(tmp_path, header_date, next_date):
    folder = tmp_path / "overnight"
    folder.mkdir()
    for name, date, time in [("night", header_date, "23.59.50"), ("morning", next_date, "00.00.10")]:
        path = _synthetic_edf(folder / f"{name}.edf", [0, 1], subtype="EDF+C")
        with path.open("r+b") as stream:
            stream.seek(88)
            stream.write(b"X".ljust(80) + (date + time).encode("ascii"))
    store = WorkbenchStore(tmp_path / "ws")
    case = inspect_case(str(folder), workspace_dir=store.root)["case"]
    rows = admission_timeline(store, case, include_absolute=True)["recordings"]
    assert [r["offset_seconds"] for r in rows] == [0, 20]
    assert all(r["absolute_start"] is None and r["header_date_unreliable"] for r in rows)
    agent = admission_timeline(store, case)
    assert [r["offset_seconds"] for r in agent["recordings"]] == [0, 20]
    assert "absolute_start" not in json.dumps(agent)


@pytest.mark.parametrize("day", ["0001-01-01", "1900-01-01", "1970-01-01"])
def test_placeholder_epoch_keeps_only_relative_time(tmp_path, day):
    file = _synthetic_edf(tmp_path / "sample.edf", [0, 1], subtype="EDF+C")
    store = WorkbenchStore(tmp_path / "ws")
    case = inspect_case(str(file), workspace_dir=store.root)["case"]
    registration = store.recording_path(case, store.list_recordings(case)[0])
    doc = read_json(registration)
    doc["header_clock"] = {"start": f"{day}T09:00:00"}
    atomic_write_json(registration, doc)
    row = admission_timeline(store, case, include_absolute=True)["recordings"][0]
    assert (row["header_date_unreliable"], row["absolute_start"], row["offset_seconds"]) == (True, None, 0)


def test_score_display_uses_saved_labels_and_handles_missing_stages():
    assert model_score_labels({"scores": {"artifact": 0}, "labels": {"artifact": 0}}) == [
        "Artifact model: artifact (real-HFO p 0.00)"]
    # A legacy score with no saved label cannot support a classification.
    assert model_score_labels({"scores": {"artifact": .91, "ehfo": None}}) == ["Artifact model: real-HFO p 0.91"]
    assert model_score_labels({}) == []


def test_reader_model_labels_follow_selected_language():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the offline reader catalog check")
    script = """
        global.window = {};
        global.document = {documentElement: {}, querySelectorAll: () => [], dispatchEvent: () => {}};
        Object.defineProperty(globalThis, "navigator", {value: {language: "en"}});
        require(process.argv[1]);
        const {modelScoreLabel, setLanguage} = window.brainlensI18n;
        const ev = {scores: {artifact: 0, spike: .63, ehfo: .81}, labels: {artifact: 0, spike: 0, ehfo: 1}};
        for (const lang of ["en", "zh-CN"]) {
            setLanguage(lang);
            console.log(...Object.keys(ev.scores).map(stage => modelScoreLabel(ev, stage)));
        }
    """
    lines = subprocess.check_output([node, "-e", script, str(SRC_DIR / "brainlens/assets/i18n.js")], text=True).splitlines()
    assert lines == [
        "Artifact model: artifact (real-HFO p 0.00) Spike model: no spike (spike p 0.63) eHFO model: eHFO (eHFO p 0.81)",
        "伪迹模型：伪迹（真实 HFO 概率 0.00） 棘波模型：无棘波（棘波概率 0.63） eHFO 模型：eHFO（eHFO 概率 0.81）",
    ]


def test_triads_sheet_and_slides_show_classifications_and_only_rated_roles(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    store = WorkbenchStore(_build_two_case_fixture(tmp_path))
    case = "CaseA"
    bundle = store.load_run(case, case, "baseline")
    for key in ("classification_json_path", "detection_json_path"):
        path = Path(bundle["meta"][key])
        stage = read_json(path)
        for ev in stage["events"]:
            ev.update(scores={"artifact": .91, "spike": .63, "ehfo": .81}, labels={"artifact": 1, "spike": 0, "ehfo": 1})
        atomic_write_json(path, stage)
    expected = ["Artifact model: non-artifact (real-HFO p 0.91)", "Spike model: no spike (spike p 0.63)",
                "eHFO model: eHFO (eHFO p 0.81)"]
    ev = bundle["events"][0]
    anchor = dict(recording=case, run_name="baseline", channel=ev["channel_name"], event_id=ev["id"])
    store.record_judgment(case, {**anchor, "kind": "event_clip", "clipped": True, "role": "researcher", "author": "reader"})
    store.record_judgment(case, {**anchor, "kind": "event_verdict", "verdict": "uncertain",
                                 "role": "clinician", "author": "physician"})
    refresh_verdict_tallies(store, case, case, "baseline")
    drawn = set()
    draw = Text.draw

    def capture(artist, renderer):
        if artist.get_visible():
            drawn.add(artist.get_text())
        return draw(artist, renderer)

    monkeypatch.setattr(Text, "draw", capture)
    report = Path(export_case_sheet(case, workspace_dir=store.root, max_cards=2)["report_path"])
    assert verify_sheet_trust(store.workspace_id, case, report.name, report.read_bytes())
    page = html.parse(str(report))
    caption = page.xpath("//div[@class='fig-cap']")[0].text_content()
    assert all(label in caption and label in drawn for label in expected)
    assert "Clinician: uncertain" in caption and "Agent: unreviewed" in caption
    roles = page.xpath("//div[@class='fig-verdicts']")[0].text_content()
    assert roles == "Clinician: uncertain · Agent: unreviewed"
    kpi = page.xpath("//div[@data-rev-rec]")[0].text_content()
    assert "Clinician review: 1 of 2 focal candidates (1 uncertain)." in kpi
    assert "Agent verdicts: 0" in kpi

    # A correctly signed report from the previous renderer remains a valid
    # artifact, but will be regenerated when someone opens the live report tab.
    old_layout = report.read_bytes().replace(SHEET_DISPLAY_META.encode(), b"")
    report.write_bytes(old_layout)
    record_verified_sheet(store.workspace_id, case, report.name, old_layout, "baseline",
                          run_signature=store.case_sheet_run_signature(case, "baseline"))
    assert verify_sheet_trust(store.workspace_id, case, report.name, old_layout)
    boot = WorkbenchHTTPServer(store.root)._bootstrap({"case": case, "lite": "1"})
    assert boot["active"]["sheet_url"] is None

    def slide_text():
        deck = Presentation(export_mdt_slides(case, workspace_dir=store.root, max_cards=2)["slides_path"])
        return "\n".join(text for slide in deck.slides for text in slide._element.xpath(".//a:t/text()"))

    slides = slide_text()
    assert all(label in slides for label in [*expected, "Clinician", "Agent"])
    assert "Researcher:" not in slides and "\nResearcher\n" not in slides and "Engineering rehearsal" not in slides
    store.record_judgment(case, {**anchor, "kind": "event_verdict", "verdict": "artifact",
                                 "role": "engineering_rehearsal", "author": "engineer"})
    assert "Engineering rehearsal: artifact" in slide_text()


def _register_scalp_admission(tmp_path):
    channels = """FP1-F7 F7-T7 T7-P7 P7-O1 FP1-F3 F3-C3 C3-P3 P3-O1
        FP2-F4 F4-C4 C4-P4 P4-O2 FP2-F8 F8-T8 T8-P8 P8-O2 FZ-CZ CZ-PZ
        P7-T7 T7-FT9 FT9-FT10 FT10-T8 T8-P7""".split()
    directory = tmp_path / "Admission"
    scans = ["filename\tacq_time"]
    for i in range(1, 43):
        sidecar = {"modality": "scalp", "reference": "bipolar", "line_freq": 50}
        if i == 3:
            sidecar["markers"] = [{"label": "Seizure", "start": 8, "end": 9, "source": "fixture"}]
        path = _write_fif(directory / f"night_{i:02d}.fif", channels, 256, 20 * 256,
                          ch_types="eeg", sidecar=sidecar, seed=i)
        scans.append(f"{path.name}\t2020-01-02T10:{i:02d}:00")
    (directory / "scans.tsv").write_text("\n".join(scans) + "\n")
    result = inspect_case(str(directory), workspace_dir=tmp_path / "ws")
    assert len(result["recordings"]) == 42
    return channels


def test_cdp_many_recording_scalp_marker_and_agent_reply(reader_cdp, tmp_path, monkeypatch):
    from brainlens import mcp_server

    cdp = reader_cdp
    cdp.send("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 810, "deviceScaleFactor": 1, "mobile": False})
    # Exercise the header with both the rendered SEEG label and a large burst label.
    assert cdp.eval("""(() => {
        const badge = document.getElementById('syncClassBadge');
        const panel = badge.closest('.verdict-panel').getBoundingClientRect();
        const burstFits = [badge.textContent, 'SYNC BURST · 12 shafts (83 ch)', 'Synchronous burst · 83 ch / 12 shafts'].every(label => {
            badge.textContent = label;
            const {left, right} = badge.getBoundingClientRect();
            return left >= panel.left && right <= panel.right && badge.clientWidth >= badge.scrollWidth
                && badge.clientHeight >= badge.scrollHeight;
        });
        return burstFits && [...document.querySelectorAll('.q-badge, #provBadge, #agentSyncBadge')].every(el => {
            const r = el.getBoundingClientRect();
            const bounds = el.closest('.reader-sidebar, .topbar').getBoundingClientRect();
            return r.left >= bounds.left && r.right <= bounds.right && el.clientWidth >= el.scrollWidth;
        });
    })()""")

    channels = _register_scalp_admission(tmp_path)
    monkeypatch.setattr(mcp_server, "_WORKSPACE_ROOT", tmp_path / "ws")
    origin = cdp.eval("location.origin")
    cdp.send("Page.navigate", {"url": origin + "/?case=Admission&recording=night_01&view=overview"})

    def active_recording(name):
        _reader_wait(cdp, f"document.querySelector('.recording-picker summary')?.textContent === '{name}'")

    def click(selector):
        cdp.eval(f"document.querySelector({json.dumps(selector)}).click()")

    active_recording("night_01")
    assert cdp.eval("""[...document.querySelectorAll('.topbar .view-tab, #provBadge, #agentSyncBadge, #themeBtn')]
        .every(el => {
            const r = el.getBoundingClientRect();
            return r.width > 0 && r.left >= 0 && r.right <= innerWidth &&
                el.contains(document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2));
        }) && document.querySelector('.topbar').scrollWidth <= innerWidth""")
    click("#recPills [aria-label='Next recording']")
    active_recording("night_02")
    click("#recPills [aria-label='Previous recording']")
    active_recording("night_01")
    click(".recording-picker summary")
    _reader_wait(cdp, "document.querySelectorAll('.recording-menu button').length === 42")
    cdp.eval("""(() => {
        const input = document.querySelector('.recording-menu input[type=search]');
        input.value = '42'; input.dispatchEvent(new Event('input'));
    })()""")
    assert cdp.eval("Array.from(document.querySelectorAll('.recording-menu button'), b => b.textContent)") == ["night_42"]
    click(".recording-menu button")
    active_recording("night_42")
    click("#admissionPanel > summary")
    _reader_wait(cdp, "document.querySelectorAll('#admissionFiles .admission-file').length === 42")
    assert cdp.eval("document.querySelectorAll('#admissionFiles .admission-track .admission-gap').length") == 41
    assert cdp.eval("document.querySelector('#admissionGaps summary').textContent") == "41 gaps · show"
    assert not cdp.eval("document.querySelector('#admissionGaps details').open")
    cdp.eval("document.querySelector('#admissionPanel').scrollIntoView({block: 'start'})")
    assert cdp.eval("""['#admissionSearch', '#admissionEvents button'].every(selector => {
        const r = document.querySelector(selector).getBoundingClientRect();
        return r.height > 0 && r.top >= 0 && r.bottom <= innerHeight;
    })""")

    def ask(box, question):
        _reader_wait(cdp, f"document.querySelector('#{box} button[type=submit]:enabled')?.offsetWidth")
        cdp.eval(f"""(() => {{
            const form = document.querySelector('#{box} form');
            const field = form.querySelector('input[name=question]');
            field.value = {json.dumps(question)};
            field.dispatchEvent(new Event('input', {{bubbles: true}}));
            form.querySelector('button[type=submit]').click();
        }})()""")
        _reader_wait(cdp, f"document.querySelector('#{box} .ask-thread')?.textContent.includes({json.dumps(question)})")
        inbox = mcp_server.open_workbench(action="status", case="Admission")["pending_report_actions"]
        return next(row for row in inbox if row["comment"] == question)

    click("#admissionEvents .marker-ask")
    marker_action = ask("admissionAskAgent", "Which marker is selected?")
    assert marker_action["source"] == "admission_timeline"
    assert marker_action["recording"] == "night_03"
    assert marker_action["marker"] == {"label": "Seizure", "start_seconds": 8, "end_seconds": 9}
    assert [marker_action["start_seconds"], marker_action["end_seconds"]] == [8, 9]
    click("#admissionEvents .admission-event > button:first-child")
    active_recording("night_03")
    _reader_wait(cdp, "document.getElementById('pageLabel').textContent === 'All (23 ch)'")
    assert cdp.eval("""[document.getElementById('readerDurSelect').value, document.getElementById('dispFilterSelect').value,
        document.getElementById('displayNotchChk').checked, document.getElementById('notchLbl').textContent,
        document.getElementById('toggleTriadBtn').getAttribute('aria-expanded')]""") == ["10", "eeg_1_70", True, "Notch 50 Hz", "false"]
    # Inspect the response from the reader's actual recorded request, including acquisition order and effective filters.
    waveform = cdp.eval("""(async () => {
        const page = await (await fetch('/api/multichannel?' + window.readerRequest())).json();
        return page.traces.map(t => [t.channel_name, t.display_shaft, t.display_filter_band_hz, t.effective_notch_hz[0]]);
    })()""")
    assert waveform == [[ch, None, [1, 70], 50] for ch in channels]
    _reader_wait(cdp, f"{json.dumps(channels)}.every(name => window.readerFrame.labels.includes(name))")
    question = "Where are the seizures?"
    action = ask("readerAskAgent", question)
    assert action["source"] == "waveform_reader" and action["recording"] == "night_03" and action["run_name"] is None
    assert [action["start_seconds"], action["end_seconds"], action["channel"]] == [3, 13, channels[0]]
    assert action["marker"] == marker_action["marker"]
    reply = "Review the marker at 8 seconds."
    result = mcp_server.open_workbench(action="resolve_actions", case="Admission", action_ids=[action["id"]], agent_reply=reply)
    assert not result.get("error"), result
    _reader_wait(cdp, f"""[...document.querySelectorAll('#readerAskAgent .ask-thread article')].some(el =>
        [{json.dumps(question)}, {json.dumps(reply)}, 'Resolved'].every(s => el.textContent.includes(s)))""")
    click("#toggleTriadBtn")
    click("#viewNav [data-view=overview]")
    click("#viewNav [data-view=reader]")
    _reader_wait(cdp, "document.getElementById('pageLabel').textContent === 'All (23 ch)'")
    assert cdp.eval("document.getElementById('toggleTriadBtn').getAttribute('aria-expanded')") == "true"


def test_unanalysed_questions_preserve_scope_mask_identity_and_resolve(tmp_path, monkeypatch):
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from brainlens import mcp_server
    from test_release_gate import _write_fif_with_subject

    # Both the case slug and the free-text marker contain a real header identifier.
    native = "PVT143759"
    path = tmp_path / "eeg_raw.fif"
    _write_fif_with_subject(path, ["FP1-F7", "F7-T7"], 256, 12, his_id=native)
    marker = {"label": f"Seizure {native} 2021-03-04", "start_seconds": 4, "end_seconds": 4}
    atomic_write_json(path.with_suffix(".json"), {
        "modality": "scalp", "line_freq": 50,
        "markers": [{"label": marker["label"], "start": 4, "end": 4, "source": "fixture"}],
    })
    store = WorkbenchStore(tmp_path / "ws")
    case = inspect_case(str(path), case_name=native, workspace_dir=store.root)["case"]
    rec = store.list_recordings(case)[0]
    monkeypatch.setattr(mcp_server, "_WORKSPACE_ROOT", store.root)
    srv = WorkbenchHTTPServer(store.root, port=0)
    base = srv.start().rstrip("/")

    def post(body, **headers):
        request = Request(base + "/api/report-action", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", **headers})
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    auth = {"X-BrainLens-Token": srv.auth_token}
    ask = {"action": "ask_agent", "source": "waveform_reader", "case": case, "recording": rec,
           "channel": "FP1-F7", "start_seconds": 1.25, "end_seconds": 11.25, "marker": marker,
           "role": "researcher", "author": "reader", "comment": f"Review {native} on 2021-03-04?",
           "operation_id": "reader-question"}
    try:
        for headers, code in [({}, 401), ({**auth, "Origin": "https://wrong.invalid"}, 403),
                              ({**auth, "Host": "wrong.invalid"}, 403)]:
            with pytest.raises(HTTPError) as caught:
                post(ask, **headers)
            assert caught.value.code == code
        for change in ({"role": "agent"}, {"verdict": "confirm"}, {"channel": "No such channel"},
                       {"marker": {**marker, "start_seconds": 3}}, {"end_seconds": 13}, {"start_seconds": None}):
            with pytest.raises(HTTPError) as caught:
                post({**ask, **change}, **auth)
            assert caught.value.code == 400
        assert store.load_report_actions()["actions"] == []

        sent = post(ask, **auth)["action"]
        assert post(ask, **auth)["deduplicated"]
        assert sent["marker"] == marker and sent["role"] == "researcher"
        assert sent["run_name"] is None
        timeline = post({"action": "ask_agent", "source": "admission_timeline", "case": case,
                         "role": "clinician", "comment": "Any other marked recordings?"}, **auth)["action"]
        assert timeline["recording"] is None and "start_seconds" not in timeline
        status = mcp_server.open_workbench(action="status")["pending_report_actions"]
        assert len(status) == 2 and native not in json.dumps(status) and "2021-03-04" not in json.dumps(status)
        received = next(row for row in status if row["id"] == sent["id"])
        assert (received["role"], received["recording_id"], received["channel"],
                received["start_seconds"], received["end_seconds"]) == (
                    "researcher", store.load_recording(case, rec).id, "FP1-F7", 1.25, 11.25)
        assert received["marker"]["start_seconds"] == received["marker"]["end_seconds"] == 4
        reply = "See the marker at 4 s."
        resolved = mcp_server.open_workbench(action="resolve_actions", case=received["case_ref"],
                                             action_ids=[sent["id"]], agent_reply=reply)
        assert resolved["resolved_ids"] == [sent["id"]]
        with urlopen(base + "/api/report-actions?case=" + case, timeout=5) as response:
            history = json.load(response)["actions"]
        assert [(row["id"], row["status"], row["agent_reply"]) for row in history] == [
            (sent["id"], "resolved", reply), (timeline["id"], "pending", ""),
        ]
        assert not store.load_judgments(case)
        assert not store.has_run(case, rec, "baseline")
    finally:
        srv.stop()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required to replay reader requests")
def test_cursor_click_wins_over_overlapping_triad_refresh():
    # Run the reader's actual request/commit path with deferred network responses.
    # The CDP test exercises hit testing and drafts; this one forces the rare response order.
    program = """
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const app = fs.readFileSync(process.argv[1], 'utf8');
const loadFunction = app.slice(app.indexOf('  async function loadTriadCanvas('),
                               app.indexOf('  function updateVerdictPanel('));
const st = { caseSlug: 'case', recording: 'rec', runName: 'baseline', channel: 'LB1',
             triadReqSeq: 0, triadSpan: 0.6 };
const requests = [];
function getJSON(path, params) {
  assert.equal(path, '/api/triad');
  return new Promise(resolve => {
    requests.push({params, finish() {
      resolve({case: params.case, recording: params.recording, run_name: params.run_name,
               channel_name: params.channel,
               window_seconds: [params.center_seconds - 0.3, params.center_seconds + 0.3]});
    }});
  });
}
const env = {st, getJSON, setTimeout, clearTimeout};
for (const name of ['setVerdictButtonsBusy', 'windowDraftToken', 'swapNoteDraftForItem', 'updateVerdictPanel',
                    'drawTriad', 'drawMultichannel', 'drawReaderMinimap']) env[name] = () => {};
const load = vm.runInNewContext(loadFunction + '\\nloadTriadCanvas', env);
(async () => {
  const first = load({eventId: '', centerSeconds: 6.755});
  requests[0].finish();
  await first;
  st.channel = 'LB2';
  const click = load({eventId: '', centerSeconds: 7.805});
  const refresh = load();  // A clip save or SSE refresh while the click is in flight.
  requests[2].finish();
  await refresh;
  requests[1].finish();
  await click;
  assert.equal(requests[2].params.center_seconds, 7.805);
  assert.equal(st.triadData.channel_name, 'LB2');
})();
"""
    subprocess.run([shutil.which("node"), "-e", program, SRC_DIR / "brainlens/assets/app.js"], check=True)
