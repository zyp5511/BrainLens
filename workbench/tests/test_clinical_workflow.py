"""Clinical reader exports and per-recording de-identification."""

from __future__ import annotations

import asyncio
import csv
import datetime as dt
import importlib.util
import json
import socket
import sys
import threading
from collections import Counter
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import edfio
import mne
import numpy as np
import pytest

WORKBENCH_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = WORKBENCH_DIR / "src"
if importlib.util.find_spec("brainlens") is None and SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlens.bids_export import export_bids_events
from brainlens.case_sheet import export_case_sheet
from brainlens.edf_export import export_edf_annotations
from brainlens.io_readers import inspect_case
from brainlens.models import ToolkitError, atomic_write_json, read_json, record_verified_sheet, verify_sheet_trust
from brainlens.pipeline import analyze_case, refresh_verdict_tallies
from brainlens.storage import WorkbenchStore
from brainlens.workbench_http import WorkbenchHTTPServer


CASE = "VisitTaylor"
IDENTIFIERS = (b"MRN123456", b"DOE_JANE", b"06-JUL-1974", b"HOSP983", b"12-MAY-2024")
SOURCE_ANNOTATION = edfio.EdfAnnotation(0.375, 0.25, "Source marker")


def _write_edf(path: Path, code: str = "MRN123456") -> None:
    signals = []
    for i, name in enumerate(("LA1", "LB1", "LC1", "LD1", "LE1")):
        rate = 256 if i == 1 else 1024
        data = np.sin(2 * np.pi * 8 * np.arange(4 * rate) / rate) * 40
        signals.append(edfio.EdfSignal(
            data, rate, label=name, physical_dimension="uV",
            physical_range=(-70.0 - i, 110.0 + i), digital_range=(-24000 + i, 25000 - i),
        ))
    edfio.Edf(
        signals,
        patient=edfio.Patient(code=code, name="DOE_JANE", sex="F", birthdate=dt.date(1974, 7, 6)),
        recording=edfio.Recording(startdate=dt.date(2024, 5, 12), hospital_administration_code="HOSP983"),
        starttime=dt.time(12, 34, 56, 125000), annotations=[SOURCE_ANNOTATION],
        data_record_duration=1,
    ).write(path)


def _analyze(path: Path, ws: Path) -> None:
    analyze_case(str(path), workspace_dir=ws, run_name="baseline", overrides={"detector_enabled": False})


@pytest.fixture
def reader_run(tmp_path: Path) -> WorkbenchStore:
    source = tmp_path / f"{CASE}.edf"
    _write_edf(source)
    store = WorkbenchStore(tmp_path / "ws")
    _analyze(source, store.root)
    bundle = store.load_run(CASE, CASE, "baseline")
    assert bundle["events"] == []
    channels = bundle["summary"]["channels"]
    # Two isolated detections (the first on the native 256 Hz channel) and a five-shaft burst.
    starts = [(channels[1], 1.125), (channels[0], 2.125), *((ch, 3.25) for ch in channels)]
    events = [
        {
            "id": f"ev_{i}", "channel_name": ch["channel_name"], "channel_id": ch["channel_id"],
            "shaft": ch["shaft"], "start_sample": round(onset * 1024), "end_sample": round(onset * 1024) + 64,
            "start_seconds": onset, "end_seconds": onset + 0.0625, "labels": {}, "scores": {},
        }
        for i, (ch, onset) in enumerate(starts)
    ]
    for key in ("detection_json_path", "classification_json_path"):
        path = Path(bundle["meta"][key])
        stage = read_json(path)
        stage.update(status="complete", events=events)
        atomic_write_json(path, stage)
    reviewers = (("clinician", "Dr A"), ("agent", "agent"),
                 ("researcher", "analyst"), ("engineering_rehearsal", "engineer"))
    for event, (role, author) in zip(events[:4], reviewers, strict=True):
        store.record_judgment(CASE, {
            "kind": "event_verdict", "recording": CASE, "run_name": "baseline",
            "channel": event["channel_name"], "event_id": event["id"], "role": role,
            "author": author, "verdict": "confirm",
        })
    summary = refresh_verdict_tallies(store, CASE, CASE, "baseline")
    assert len(summary["synchronous_burst_windows"]) == 1
    return store


def _rows(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == ["onset", "duration", "description"]
        return list(reader)


def _assert_masked(payload: bytes) -> None:
    for identifier in IDENTIFIERS:
        assert identifier not in payload


@pytest.mark.parametrize("previously_withheld", [False, True], ids=["original-workspace", "always-masked-workspace"])
def test_reader_roundtrip_and_legacy_provenance(
    reader_run: WorkbenchStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, previously_withheld: bool,
) -> None:
    store = reader_run
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    source = tmp_path / f"{CASE}.edf"
    original = source.read_bytes()
    assert all(value in original for value in IDENTIFIERS)
    # Simulate an already registered, pre-feature recording.json and run. The legacy
    # provenance schema had no field distinguishing an ID from the native header.
    path = store.recording_path(CASE, CASE)
    registered = read_json(path)
    old_provenance = {
        "origin_kind": "local_recording", "participant_id": "MRN123456",
        "summary": f"Local recording ({CASE}.edf); external dataset provenance not declared. · MRN123456",
    }
    registered.pop("native_header_terms", None)
    if previously_withheld:
        old_provenance.update(participant_id=None, participant_id_source="withheld")
        old_provenance["summary"] = old_provenance["summary"].replace("MRN123456", "header subject ID withheld")
    else:
        registered.pop("subject_comparison_digest", None)
    registered["spec"]["provenance"] = old_provenance
    atomic_write_json(path, registered)
    case_path = store.case_json_path(CASE)
    case_doc = read_json(case_path)
    case_doc["recordings"][0]["provenance"] = old_provenance
    atomic_write_json(case_path, case_doc)

    internal = export_edf_annotations(CASE, workspace_dir=store.root)
    internal_edf = Path(internal["edf_path"])
    assert internal["deidentified"] is False
    assert internal_edf.read_bytes()[8:184] == original[8:184]
    share = {"deidentify": True, "pseudonym": "P001"}
    result = export_edf_annotations(CASE, workspace_dir=store.root, **share)
    assert result["run_name"] == "baseline" and result["deidentified"] is True
    assert internal_edf != Path(result["edf_path"]) and internal_edf.read_bytes()[8:184] == original[8:184]
    assert Path(result["edf_path"]).name.endswith("_brainlens.edf")
    assert Path(result["csv_path"]).name.endswith("_brainlens_annotations.csv")
    for key in ("edf_path", "csv_path", "json_path"):
        artifact = Path(result[key])
        assert artifact.parent == store.case_dir(CASE) / "exports"
        assert "P001" in artifact.name and CASE not in artifact.name
        _assert_masked(artifact.read_bytes())
        assert CASE.encode() not in artifact.read_bytes()

    saved = edfio.read_edf(result["edf_path"])
    old = edfio.read_edf(source)
    assert source.read_bytes() == original
    assert saved.local_patient_identification == "X X X X"
    assert saved.local_recording_identification == "Startdate X X X X"
    assert Path(result["edf_path"]).read_bytes()[168:176] == b"01.01.85"
    assert saved.starttime == old.starttime == dt.time(12, 34, 56, 125000)
    assert len(saved.signals) == len(old.signals)
    for before, after in zip(old.signals, saved.signals, strict=True):
        assert before.label == after.label
        assert before.sampling_frequency == after.sampling_frequency
        assert before.physical_range == after.physical_range
        assert before.digital_range == after.digital_range
        assert before.physical_dimension == after.physical_dimension
        np.testing.assert_array_equal(before.digital, after.digital)

    rows = _rows(result["csv_path"])
    annotations = [ann for ann in saved.annotations if ann != SOURCE_ANNOTATION]
    assert SOURCE_ANNOTATION in saved.annotations
    assert len(rows) == result["annotations_exported"] == len(annotations) == 9
    assert "P001" in rows[0]["description"] and "baseline" in rows[0]["description"]
    assert rows[0]["description"].endswith("(not a medical device)")
    assert float(rows[0]["onset"]) == 0.0
    assert all(row["description"].isascii() and len(row["description"]) <= 80 for row in rows)
    assert Counter(
        (float(row["onset"]), float(row["duration"]), row["description"]) for row in rows
    ) == Counter((ann.onset, ann.duration or 0.0, ann.text) for ann in annotations)
    events = [row for row in rows if row["description"].startswith("BL HFO")]
    bundle = store.load_run(CASE, CASE, "baseline")
    for ev in bundle["events"]:
        matches = [
            row for row in events if ev["channel_name"] in row["description"].split()
            and float(row["onset"]) == pytest.approx(ev["start_seconds"], abs=1 / 1024)
        ]
        assert len(matches) == 1
        row = matches[0]
        assert float(row["duration"]) == pytest.approx(ev["end_seconds"] - ev["start_seconds"], abs=1 / 1024)
        assert ("conf" in row["description"].lower()) == (ev["id"] == "ev_0")
    episode, = [r for r in rows if r["description"].startswith("BL burst episode")]
    assert "n_shafts=5" in episode["description"]
    assert float(episode["onset"]) == pytest.approx(bundle["summary"]["synchronous_burst_windows"][0]["start_seconds"])

    sidecar = read_json(Path(result["json_path"]))
    assert sidecar["recording_fingerprint"] == registered["spec"]["fingerprint"]
    assert sidecar["analysis_generation_id"] == bundle["meta"]["analysis_generation_id"]
    assert sidecar["case"] == sidecar["recording"] == "P001"
    assert "De-identified export" in Path(result["json_path"]).read_text()
    assert sidecar["run_name"] == "baseline" and sidecar["deidentified"]
    assert sidecar["brainlens_version"]
    assert "not a medical device" in sidecar["scope"].lower()
    assert "not a standalone surgical resection boundary" in sidecar["scope"].lower()
    sheet = export_case_sheet(CASE, workspace_dir=store.root, max_cards=0)
    bids = export_bids_events(CASE, workspace_dir=store.root)
    assert read_json(Path(bids["json_path"]))["Case"] == CASE
    boot = WorkbenchHTTPServer(store.root)._bootstrap({"case": CASE, "lite": "1"})
    for payload in (
        Path(sheet["report_path"]).read_bytes(),
        json.dumps(boot).encode(), json.dumps(inspect_case(CASE, workspace_dir=store.root)).encode(),
    ):
        assert b"MRN123456" in payload and b"header subject ID withheld" not in payload

    shared_sheet = export_case_sheet(
        CASE, workspace_dir=store.root, clinical_question="Review DOE_JANE (MRN123456), born 06-JUL-1974.", **share,
    )
    shared_bids = export_bids_events(CASE, workspace_dir=store.root, **share)
    shared_html = Path(shared_sheet["report_path"])
    for artifact in (shared_html, Path(shared_bids["tsv_path"]), Path(shared_bids["json_path"])):
        assert "P001" in artifact.name and CASE not in artifact.name
        data = artifact.read_bytes()
        _assert_masked(data)
        assert CASE.encode() not in data
        if artifact.suffix in (".json", ".html"):
            assert b"P001" in data and b"De-identified export" in data
    assert shared_html != Path(sheet["report_path"])
    assert verify_sheet_trust(store.workspace_id, CASE, shared_html.name, shared_html.read_bytes())
    assert b"MRN123456" in Path(sheet["report_path"]).read_bytes()


@pytest.mark.parametrize("record_duration", [1.0, 0.1])
def test_classic_edf_promoted_to_edf_plus(tmp_path: Path, record_duration: float) -> None:
    source = tmp_path / "classic.edf"
    ws = tmp_path / "ws"
    edfio.Edf(
        [edfio.EdfSignal(np.arange(4000.0), 1000, label="LA1", physical_dimension="uV")],
        data_record_duration=record_duration,
    ).write(source)
    assert edfio.read_edf(source).reserved == ""
    # Classic EDF permits free text and arbitrary space padding in these two fields.
    with source.open("r+b") as fh:
        fh.seek(8)
        fh.write(b"Legacy patient   043".ljust(80) + b"Hospital study   076".ljust(80))
    _analyze(source, ws)
    export = export_edf_annotations("classic", workspace_dir=ws)
    assert Path(export["edf_path"]).read_bytes()[8:184] == source.read_bytes()[8:184]
    result = edfio.read_edf(export["edf_path"])
    assert result.reserved == "EDF+C"
    assert result.is_continuous
    assert result.data_record_duration == record_duration
    marker, = result.annotations
    assert marker.onset == 0.0 and marker.text == "BrainLens baseline export (not a medical device)"
    np.testing.assert_array_equal(result.signals[0].digital, edfio.read_edf(source).signals[0].digital)


def test_verdict_and_burst_filters(reader_run: WorkbenchStore) -> None:
    for kwargs, event_count, episode_count in [
        ({"only_confirmed": True}, 1, 1),
        ({"exclude_widespread_bursts": True}, 2, 0),
        ({"include_burst_episodes": False}, 7, 0),
    ]:
        result = export_edf_annotations(CASE, workspace_dir=reader_run.root, **kwargs)
        rows = _rows(result["csv_path"])
        events = [r["description"] for r in rows if r["description"].startswith("BL HFO ")]
        assert len(events) == event_count
        if kwargs.get("only_confirmed"):
            assert "conf" in events[0]
        assert sum(r["description"].startswith("BL burst episode") for r in rows) == episode_count
        filters = read_json(Path(result["json_path"]))["filters"]
        assert all(filters[key] is expected for key, expected in kwargs.items())


def test_long_derived_channel_and_unattributed_verdict() -> None:
    from brainlens.edf_export import _run_annotations

    # A derived label may not exist in the source file. Reserve space for the
    # verdict and keep an unattributed legacy "confirm" out of clinician results.
    channel = "LA1-LA2 (référence " + "A" * 80 + ")"
    event = {
        "start_seconds": 1.25, "end_seconds": 1.5, "start_sample": 17,
        "channel_name": channel, "synchrony_class": "regional", "clinician_verdict": "confirm",
    }
    bundle = {"events": [event, {**event, "verdict_role": "clinician"}], "summary": {}}
    header, finding = _run_annotations(
        bundle, "r" * 80, only_confirmed=True, exclude_widespread_bursts=False, include_burst_episodes=False,
    )
    assert header.text.endswith(" export (not a medical device)") and len(header.text) == 80
    assert finding.onset == 1.25 and finding.duration == 0.25
    assert finding.text.startswith("BL HFO regional LA1-LA2") and finding.text.endswith(" conf")
    assert len(finding.text) == 80 and finding.text.isascii()


@pytest.mark.parametrize("label_source", ["bids", "sidecar"])
def test_explicit_participant_label_preserved(tmp_path: Path, label_source: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from brainlens import mcp_server

    public_id = "sub-P018"
    source = tmp_path / ("sub-P018_task-rest_ieeg.edf" if label_source == "bids" else "local.edf")
    _write_edf(source, code="P018")
    if label_source == "sidecar":
        atomic_write_json(source.with_suffix(".json"), {"participant_id": public_id})
    result = inspect_case(str(source), workspace_dir=tmp_path / "ws")
    prov = result["recordings"][0]["provenance"]
    assert prov["participant_id"] == public_id
    assert public_id in prov["summary"]
    monkeypatch.setattr(mcp_server, "_WORKSPACE_ROOT", tmp_path / "ws")
    agent_record = mcp_server.inspect_case(target=result["case"])["recordings"][0]
    assert agent_record["provenance"]["participant_id"] == public_id
    assert "DOE_JANE" not in json.dumps(agent_record)
    # A pre-phase registration has no origin tag. Its declared ID must still
    # appear in shared files while the other native identity fields remain private.
    _analyze(source, tmp_path / "ws")
    store = WorkbenchStore(tmp_path / "ws")
    case = result["case"]
    entry = store.recording_path(case, result["recordings"][0]["recording"])
    doc = read_json(entry)
    doc["spec"]["provenance"].pop("participant_id_source")
    doc.pop("native_header_terms")
    atomic_write_json(entry, doc)
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    options = {"deidentify": True, "workspace_dir": store.root, "pseudonym": "P001"}
    sheet = Path(export_case_sheet(case, max_cards=0, **options)["report_path"])
    bids = Path(export_bids_events(case, **options)["json_path"])
    edf = Path(export_edf_annotations(case, **options)["json_path"])
    for artifact in (sheet, bids, edf):
        data = artifact.read_text()
        assert public_id in data and "DOE_JANE" not in data


def test_header_identity_still_rejects_conflicting_subject(tmp_path: Path) -> None:
    source = tmp_path / "sub-P018_task-rest_ieeg.edf"
    _write_edf(source)
    with pytest.raises(ToolkitError) as exc:
        inspect_case(str(source), workspace_dir=tmp_path / "ws")
    assert exc.value.code == "conflicting_subjects"
    _assert_masked(str(exc.value).encode() + json.dumps(exc.value.details).encode())


def test_mcp_subject_code_disclosure_is_per_recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brainlens import mcp_server

    ws = tmp_path / "ws"
    monkeypatch.setattr(mcp_server, "_WORKSPACE_ROOT", ws)
    cases = []
    for label in ("LocalA", "DeclaredB"):
        source = tmp_path / label / f"{label}.edf"
        source.parent.mkdir()
        _write_edf(source)
        if label == "DeclaredB":
            atomic_write_json(source.with_suffix(".json"), {"participant_id": "MRN123456"})
        cases.append(inspect_case(str(source), workspace_dir=ws)["case"])
    native, declared = [mcp_server.inspect_case(target=case) for case in cases]
    assert not native.get("error") and not declared.get("error")
    _assert_masked(json.dumps(native).encode())
    assert declared["recordings"][0]["provenance"]["participant_id"] == "MRN123456"
    assert "DOE_JANE" not in json.dumps(declared)
    # An offline legacy recording needs its verified source before disclosure.
    # It must not suppress a different recording's explicitly declared ID.
    legacy = WorkbenchStore(ws).recording_path(cases[0], "LocalA")
    doc = read_json(legacy)
    doc.pop("native_header_terms")
    atomic_write_json(legacy, doc)
    (tmp_path / "LocalA" / "LocalA.edf").unlink()
    assert mcp_server.inspect_case(target=cases[0])["code"] == "source_identity_unavailable"
    other = mcp_server.inspect_case(target=cases[1])
    assert other["recordings"][0]["provenance"]["participant_id"] == "MRN123456"


def test_mcp_redacts_the_registration_read_before_a_replacement(reader_run: WorkbenchStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from brainlens import mcp_server

    monkeypatch.setattr(mcp_server, "_WORKSPACE_ROOT", reader_run.root)
    scrub = mcp_server.scrub_agent_payload

    def replace_then_scrub(payload, store):
        # Simulate a concurrent re-registration after the tool read the original
        # clinical summary and before it handed the response to the agent.
        entry = store.recording_path(CASE, CASE)
        doc = read_json(entry)
        doc["native_header_terms"] = ["REPLACEMENT"]
        doc["spec"]["provenance"]["participant_id"] = "REPLACEMENT"
        atomic_write_json(entry, doc)
        return scrub(payload, store)

    monkeypatch.setattr(mcp_server, "scrub_agent_payload", replace_then_scrub)
    result = mcp_server.query_evidence(target=CASE, mode="recording_summary")
    assert not result.get("error")
    _assert_masked(json.dumps(result).encode())


def test_sharing_rejects_a_native_patient_code_as_pseudonym(reader_run: WorkbenchStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(reader_run.root.parent / "trust"))
    for export in (export_edf_annotations, export_case_sheet, export_bids_events):
        with pytest.raises(ToolkitError) as exc:
            export(CASE, workspace_dir=reader_run.root, deidentify=True, pseudonym="MRN123456")
        assert exc.value.code == "invalid_pseudonym"
        assert "MRN123456" not in str(exc.value)


def test_sharing_scrubs_native_identity_repeated_in_signal_metadata(tmp_path: Path) -> None:
    source = tmp_path / "PrivateTaylor.edf"
    ws = tmp_path / "workspace"
    signals = [
        edfio.EdfSignal(np.arange(2048.0), 512, label=label, physical_dimension="uV",
                        transducer_type="Contact for MRN123456")
        for label in ("MRN123456-LA1", "Channel 1")
    ]
    note = edfio.EdfAnnotation(0.375, None, "Marker for DOE_JANE, born 06-JUL-1974, PrivateTaylor")
    edfio.Edf(signals, patient=edfio.Patient(code="MRN123456", name="DOE_JANE", birthdate=dt.date(1974, 7, 6)),
              annotations=[note]).write(source)
    _analyze(source, ws)
    clinical = export_edf_annotations(source.stem, workspace_dir=ws)
    identified = edfio.read_edf(clinical["edf_path"])
    assert note in identified.annotations and identified.labels[0] == "MRN123456-LA1"
    shared = export_edf_annotations(source.stem, workspace_dir=ws, deidentify=True)
    anonymous = edfio.read_edf(shared["edf_path"])
    assert len(set(anonymous.labels)) == 2 and "Channel 1" in anonymous.labels
    assert any(a.onset == note.onset and "Marker for" in a.text for a in anonymous.annotations)
    for a, b in zip(identified.signals, anonymous.signals, strict=True):
        np.testing.assert_array_equal(a.digital, b.digital)
    for key in ("edf_path", "json_path", "csv_path"):
        artifact = Path(shared[key])
        _assert_masked(artifact.read_bytes())
        assert source.stem not in artifact.name and source.stem.encode() not in artifact.read_bytes()


def test_shared_multirecording_labels_match_across_formats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    source = tmp_path / "LocalTaylor"
    source.mkdir()
    ws = tmp_path / "ws"
    recs = ["TaylorMorning", "TaylorNight"]
    for rec in recs:
        _write_edf(source / f"{rec}.edf")
    _analyze(source, ws)
    options = {"workspace_dir": ws, "deidentify": True, "pseudonym": "P001"}
    sheet = export_case_sheet(source.name, max_cards=0, **options)
    html = Path(sheet["report_path"]).read_text()
    aliases = set()
    for rec in recs:
        edf = export_edf_annotations(source.name, recording=rec, **options)
        bids = export_bids_events(source.name, recording=rec, **options)
        label = read_json(Path(edf["json_path"]))["recording"]
        assert label in html and label in Path(edf["edf_path"]).name and label.startswith("P001")
        assert read_json(Path(bids["json_path"]))["Recording"] == label
        assert rec not in html and source.name not in html
        aliases.add(label)
    assert len(aliases) == 2


def _write_brainvision(base: Path) -> Path:
    data = np.arange(8000, dtype="<i2")
    data.tofile(base.with_suffix(".eeg"))
    base.with_suffix(".vmrk").write_text(
        f"Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\nDataFile={base.name}.eeg\n[Marker Infos]\n",
        encoding="utf-8",
    )
    header = base.with_suffix(".vhdr")
    header.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\n"
        f"DataFile={base.name}.eeg\nMarkerFile={base.name}.vmrk\n"
        "DataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=2\nSamplingInterval=1000\n"
        "[Binary Infos]\nBinaryFormat=INT_16\n[Channel Infos]\nCh1=LA1,,1,µV\nCh2=LA2,,1,µV\n",
        encoding="utf-8",
    )
    return header


@pytest.mark.parametrize("source_format", ["FIF", "BrainVision", "BDF"])
def test_annotation_only_for_non_edf(tmp_path: Path, source_format: str) -> None:
    ws = tmp_path / "ws"
    if source_format == "FIF":
        source = tmp_path / "reader_raw.fif"
        info = mne.create_info(["LA1"], sfreq=512, ch_types="seeg")
        info["subject_info"] = {"his_id": "MRN123456"}
        mne.io.RawArray(np.zeros((1, 2048)), info, verbose=False).save(source, verbose=False)
    elif source_format == "BDF":
        source = tmp_path / "reader.bdf"
        edfio.Bdf([edfio.BdfSignal(np.arange(2048.0), 512, label="LA1", physical_dimension="uV")],
                  patient=edfio.Patient(code="MRN123456", name="DOE_JANE")).write(source)
    else:
        source = _write_brainvision(tmp_path / "reader")
    _analyze(source, ws)
    result = export_edf_annotations(source.stem, workspace_dir=ws, deidentify=True)
    assert result["edf_copy"]["status"] == "unsupported_format"
    assert result["edf_path"] is None and not list(ws.rglob("*.edf"))
    rows = _rows(result["csv_path"])
    assert len(rows) == result["annotations_exported"] == 1
    assert "baseline" in rows[0]["description"] and rows[0]["description"].endswith("(not a medical device)")
    for key in ("json_path", "csv_path"):
        _assert_masked(Path(result[key]).read_bytes())
    boot = WorkbenchHTTPServer(ws)._bootstrap({"case": source.stem, "lite": "1"})
    if source_format in {"FIF", "BDF"}:
        assert b"MRN123456" in json.dumps(boot).encode()


def test_rejects_modified_source_and_superseded_run(reader_run: WorkbenchStore, tmp_path: Path) -> None:
    source = tmp_path / f"{CASE}.edf"
    with source.open("r+b") as fh:
        fh.seek(8)
        fh.write(b"NEW")
    with pytest.raises(ToolkitError) as exc:
        export_edf_annotations(CASE, workspace_dir=reader_run.root)
    assert exc.value.code == "source_recording_modified"
    meta_path = reader_run.run_generation_dir(CASE, CASE, "baseline") / "run_meta.json"
    meta = read_json(meta_path)
    atomic_write_json(meta_path, {**meta, "recording_fingerprint": "superseded"})
    with pytest.raises(ToolkitError) as exc:
        export_edf_annotations(CASE, workspace_dir=reader_run.root)
    assert exc.value.code == "run_recording_mismatch"
    assert not list((reader_run.case_dir(CASE) / "exports").glob("*_brainlens.edf"))


@pytest.mark.parametrize("corruption", ["sample", "annotations"])
def test_rejects_corrupt_edf_write(
    reader_run: WorkbenchStore, monkeypatch: pytest.MonkeyPatch, corruption: str,
) -> None:
    original_write = edfio.Edf.write

    def corrupt_write(edf: edfio.Edf, path: Path) -> None:
        if corruption == "annotations":
            edf.set_annotations([SOURCE_ANNOTATION])
        original_write(edf, path)
        if corruption == "sample":
            with Path(path).open("r+b") as fh:
                fh.seek(184)  # Size of the complete EDF header, in ASCII.
                first_sample = int(fh.read(8))
                fh.seek(first_sample)
                byte = fh.read(1)[0] ^ 1
                fh.seek(first_sample)
                fh.write(bytes([byte]))

    monkeypatch.setattr(edfio.Edf, "write", corrupt_write)
    with pytest.raises(ToolkitError) as exc:
        export_edf_annotations(CASE, workspace_dir=reader_run.root)
    assert exc.value.code == "edf_verification_failed"
    assert not list((reader_run.case_dir(CASE) / "exports").glob("*_brainlens.edf"))


def test_mcp_action_and_cli(reader_run: WorkbenchStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    from brainlens import mcp_server
    from brainlens.cli import main

    monkeypatch.setattr(mcp_server, "_WORKSPACE_ROOT", reader_run.root)

    async def call_export():
        async with create_connected_server_and_client_session(mcp_server.mcp) as client:
            listed = await client.list_tools()
            for tool_name, arguments in (
                ("inspect_case", {"target": CASE}),
                ("query_evidence", {"target": CASE, "mode": "recording_summary"}),
            ):
                answer = await client.call_tool(tool_name, arguments)
                assert not answer.isError
                parsed = json.loads(answer.content[0].text)
                assert not parsed.get("error"), parsed
                _assert_masked(answer.content[0].text.encode())
            response = await client.call_tool("export_or_protocol", {
                "action": "export_edf_annotations", "case": CASE, "recording": CASE, "primary_run": "baseline",
                "only_confirmed": True, "exclude_widespread_bursts": True,
                "deidentify": True, "pseudonym": "P001",
            })
            assert not response.isError
            return listed.tools, json.loads(response.content[0].text)

    tools, result = asyncio.run(call_export())
    assert len(tools) == 7 and {t.name for t in tools} == {
        "inspect_case", "analyze_case", "query_evidence", "render_evidence_card",
        "open_workbench", "record_judgment", "export_or_protocol",
    }
    tool, = [t for t in tools if t.name == "export_or_protocol"]
    assert "export_edf_annotations" in tool.inputSchema["properties"]["action"]["enum"]
    assert tool.inputSchema["properties"]["deidentify"]["default"] is False
    assert result["annotations_exported"] == 2
    capsys.readouterr()
    assert main(["--workspace", str(reader_run.root), "export-edf", CASE, "--recording", CASE,
                 "--run-name", "baseline", "--only-confirmed", "--exclude-bursts", "--deidentify", "--pseudonym", "P001"]) == 0
    cli = json.loads(capsys.readouterr().out)
    assert cli["edf_path"] == result["edf_path"] and cli["annotations_exported"] == 2
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(reader_run.root.parent / "trust"))
    for command, key in (("export", "report_path"), ("export-bids", "json_path")):
        assert main(["--workspace", str(reader_run.root), command, CASE, "--deidentify", "--pseudonym", "P002"]) == 0
        path = Path(json.loads(capsys.readouterr().out)[key])
        assert "P002" in path.name
        _assert_masked(path.read_bytes())


def test_http_export_and_attachment_downloads(
    reader_run: WorkbenchStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    # Use a socket pair to run the real request handler without binding an internet port.
    server = WorkbenchHTTPServer(reader_run.root)
    with monkeypatch.context() as patch:
        patch.setattr(ThreadingHTTPServer, "server_bind", lambda listener: setattr(listener, "server_port", 8765))
        patch.setattr(ThreadingHTTPServer, "server_activate", lambda listener: None)
        patch.setattr(threading.Thread, "start", lambda thread: None)
        server.start()

    def request(method, path, body=None, token=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-BrainLens-Token"] = token
        with _http_pair(server) as conn:
            conn.request(method, path, body=json.dumps(body) if body else None, headers=headers)
            reply = conn.getresponse()
            return reply.status, reply.headers, reply.read()

    sheet = export_case_sheet(CASE, workspace_dir=reader_run.root, max_cards=0)
    report = Path(sheet["report_path"])
    try:
        status, _, internal = request("GET", f"/exports/{CASE}/{report.name}")
        assert status == 200 and b"MRN123456" in internal
        assert server.auth_token.encode() in internal
        status, _, bootstrap = request("GET", f"/api/bootstrap?case={CASE}&lite=1")
        assert status == 200 and b"MRN123456" in bootstrap
        assert json.loads(bootstrap)["active"]["sheet_url"] == f"/exports/{CASE}/{report.name}"
        sharing = {"case": CASE, "deidentify": True, "pseudonym": "P001"}
        for endpoint, key in (("export-sheet", "sheet_url"), ("export-bids", "json_url")):
            status, _, data = request("POST", f"/api/{endpoint}", sharing, server.auth_token)
            assert status == 200, data
            result = json.loads(data)
            url = result[key]
            assert "P001" in url.rsplit("/", 1)[1]
            status, headers, data = request("GET", url)
            assert status == 200 and b"De-identified export" in data
            _assert_masked(data)
            assert server.auth_token.encode() not in data
            if endpoint == "export-sheet":
                assert headers.get_content_disposition() == "attachment"
                assert data == Path(result["report_path"]).read_bytes()
        status, _, _ = request("POST", "/api/export-edf", {"case": CASE})
        assert status in (401, 403)
        status, _, data = request("POST", "/api/export-edf", sharing, server.auth_token)
        assert status == 200
        result = json.loads(data)
        for ext in ("edf", "csv", "json"):
            path = Path(result[f"{ext}_path"])
            assert result[f"{ext}_url"] == f"/exports/{CASE}/{path.name}"
            status, headers, body = request("GET", result[f"{ext}_url"])
            assert status == 200 and body == path.read_bytes()
            if ext in ("edf", "csv"):
                assert headers.get_content_disposition() == "attachment"
        # Signed local sheets from the always-masked version are rebuilt on next use.
        old = report.read_bytes().replace(
            b"<head>", b'<head><meta name="brainlens-deidentification" content="header-subject-v1" />',
        )
        report.write_bytes(old)
        record_verified_sheet(reader_run.workspace_id, CASE, report.name, old, "baseline",
                              run_signature=reader_run.case_sheet_run_signature(CASE, "baseline"))
        status, _, data = request("GET", f"/api/bootstrap?case={CASE}&lite=1")
        assert status == 200 and json.loads(data)["active"]["sheet_url"] is None
    finally:
        server.httpd.server_close()


@contextmanager
def _http_pair(server):
    client, peer = socket.socketpair()
    with client, peer:
        conn = HTTPConnection("127.0.0.1", server.port, timeout=5)
        conn.sock = client
        handler = threading.Thread(target=server.httpd.finish_request, args=(peer, ("127.0.0.1", 12345)))
        handler.start()
        try:
            yield conn
        finally:
            conn.close()
            handler.join(timeout=5)


def test_shared_sheet_masks_names_in_evidence_and_agreement_sections(
    reader_run: WorkbenchStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Sections rendered by the evidence-map and reader-agreement modules obey the sharing rules too."""
    from brainlens.evidence_map import get_evidence_map

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    bundle = reader_run.load_run(CASE, CASE, "baseline")
    for event in bundle["events"][:3]:
        for role, author in (("clinician", "Dr A"), ("researcher", "analyst")):
            reader_run.record_judgment(CASE, {
                "kind": "event_verdict", "recording": CASE, "run_name": "baseline",
                "channel": event["channel_name"], "event_id": event["id"], "role": role,
                "author": author, "verdict": "confirm" if author == "Dr A" else "uncertain",
            })
    refresh_verdict_tallies(reader_run, CASE, CASE, "baseline")
    get_evidence_map(reader_run, CASE, recording=CASE, run_name="baseline")

    internal = Path(export_case_sheet(CASE, workspace_dir=reader_run.root)["report_path"]).read_text(encoding="utf-8")
    assert "Inter-rater agreement" in internal and "Evidence map" in internal
    assert CASE in internal

    shared = export_case_sheet(CASE, workspace_dir=reader_run.root, deidentify=True, pseudonym="P001")
    html_text = Path(shared["report_path"]).read_text(encoding="utf-8")
    assert "Inter-rater agreement" in html_text and "Evidence map" in html_text
    assert CASE not in html_text and CASE.lower() not in html_text.lower()
    _assert_masked(html_text.encode("utf-8"))


def test_i18n_local_assets_and_english_case_sheet(
    reader_run: WorkbenchStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    server = WorkbenchHTTPServer(reader_run.root)
    with monkeypatch.context() as patch:
        patch.setattr(ThreadingHTTPServer, "server_bind", lambda httpd: setattr(httpd, "server_port", 8765))
        patch.setattr(ThreadingHTTPServer, "server_activate", lambda httpd: None)
        patch.setattr(threading.Thread, "start", lambda thread: None)
        server.start()

    def get(path: str) -> bytes:
        with _http_pair(server) as conn:
            conn.request("GET", path, headers={"Accept-Language": "zh-CN"})
            response = conn.getresponse()
            body = response.read()
            assert response.status == 200, body[:200]
            return body

    try:
        page = get("/")
        assert page.index(b'src="/i18n.js"') < page.index(b'src="/app.js"')
        assert get("/i18n.js") == (SRC_DIR / "brainlens" / "assets" / "i18n.js").read_bytes()
        result = export_case_sheet(CASE, workspace_dir=reader_run.root, max_cards=0)
        sheet = get(f"/exports/{CASE}/{Path(result['report_path']).name}")
        assert b'<html lang="en">' in sheet
        assert b"BrainLens is not a medical device. This sheet is not a standalone surgical resection boundary." in sheet
        assert b"/i18n.js" not in sheet
    finally:
        server.httpd.server_close()


def test_placeholder_edf_patient_field_is_not_identity_but_real_names_still_are():
    """CHB-MIT ships 'Surrogate Name' in the EDF patient field: a placeholder, not identity.
    A real EDF+ patient field must still yield its code and name as private terms."""
    from types import SimpleNamespace

    from brainlens.deid import native_header_terms

    placeholder = SimpleNamespace(local_patient_identification="Surrogate Name", local_recording_identification="")
    assert not {t.lower() for t in native_header_terms(None, placeholder)} & {"surrogate", "name", "surrogate name"}

    real = SimpleNamespace(local_patient_identification="MCH-0234567 F 02-MAY-1951 Haagse_Harry",
                           local_recording_identification="")
    terms = {t.lower() for t in native_header_terms(None, real)}
    assert "mch-0234567" in terms
    assert any("haagse" in t for t in terms)
