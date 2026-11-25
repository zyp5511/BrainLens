"""Small synthetic recordings exercise the large-source and discontinuous-clock paths."""

import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brainlens import io_integrity, pipeline, pyhfo_runner, signal_prep, triad_views
from brainlens.analytics import select_control_windows
from brainlens.edf_export import export_edf_annotations
from brainlens.io_readers import inspect_case
from brainlens.models import EngineSettings, ToolkitError
from brainlens.pipeline import analyze_case
from brainlens.pyhfo_runner import PyHFORunner
from brainlens.storage import WorkbenchStore


def _synthetic_edf(path, onsets, *, subtype="EDF+D", pulse_time=None, fs=1024):
    """Write packed EDF/BDF data and independent TAL clocks. Each record lasts one second."""
    bdf = subtype.startswith("BDF")
    width = 3 if bdf else 2
    header = bytearray(b" " * 256)
    header[:8] = b"\xffBIOSEMI" if bdf else b"0       "
    header[8:88] = b"X X X Synthetic".ljust(80)
    header[88:168] = b"Startdate 01-JAN-2020 X X Synthetic".ljust(80)
    header[168:184] = b"01.01.2000.00.00"
    header[184:192] = b"768     "
    header[192:236] = subtype.encode().ljust(44)
    header[236:256] = f"{len(onsets):<8}1       2   ".encode()
    maximum = 2 ** (8 * width - 1)
    for fields, n in [
        (["LA1", f"{subtype[:3]} Annotations"], 16), (["", ""], 80),
        (["uV", ""], 8), (["-1000", "-1"], 8), (["1000", "1"], 8),
        ([str(-maximum)] * 2, 8), ([str(maximum - 1)] * 2, 8),
        (["", ""], 80), ([str(fs), "48"], 8), (["", ""], 32),
    ]:
        for field in fields:
            header.extend(field.encode().ljust(n))
    with path.open("wb") as out:
        out.write(header)
        for onset in onsets:
            t = np.arange(fs) / fs
            signal = 10 * np.sin(2 * np.pi * 8 * t)
            tal = f"+{onset:.12g}\x14\x14\0".encode()
            if pulse_time is not None and onset <= pulse_time < onset + 1:
                x = t - (pulse_time - onset)
                signal += np.where((x >= 0) & (x < .1), 350 * np.sin(2 * np.pi * 160 * x), 0.)
                tal += f"+{pulse_time:g}\x150.1\x14Synthetic pulse\x14\0".encode()
            samples = np.round(signal * (maximum - 1) / 1000).astype("<i4")
            out.write(samples.view("u1").reshape(-1, 4)[:, :width].tobytes())
            out.write(tal.ljust(48 * width, b"\0"))
    path.with_suffix(".json").write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    return path


def _register(path, workspace):
    inspected = inspect_case(str(path), workspace_dir=workspace)
    case = inspected["case"]
    slug = inspected["recordings"][0]["recording"]
    store = WorkbenchStore(workspace)
    return store, case, store.load_recording(case, slug)


@pytest.mark.parametrize("subtype,suffix", [("EDF+D", "edf"), ("BDF+D", "bdf"), ("EDF+C", "edf")])
def test_tal_clock_and_gap_pages(tmp_path, subtype, suffix):
    starts = [0, 1, 5.125, 6.125]
    path = _synthetic_edf(tmp_path / f"synthetic.{suffix}", starts, subtype=subtype, pulse_time=6.375)
    store, case, rec = _register(path, tmp_path / "ws")
    assert rec.n_samples == 4096 and rec.duration == 7.125
    assert rec.gaps() == [(2., 5.125)]
    assert [(a, b) for _, a, b in rec.time_slices(1.75, 5.375)] == [(1792, 2048), (2048, 2304)]
    assert rec.time_slices(2., 5.125) == []
    assert any(m.label == "Synthetic pulse" and m.start == 6.375 for m in rec.markers)
    crossing = triad_views.compute_multichannel_batch(store, case, rec.slug, start_seconds=1.5, duration_seconds=4)
    trace = crossing["traces"][0]
    assert any(v is None for v in trace["values_uv"])
    measured = [t for t, v in zip(trace["times"], trace["values_uv"]) if v is not None]
    assert min(measured) == 1.5 and 5.125 <= max(measured) < 5.5
    assert all(t < 2 or t >= 5.125 for t in measured)
    page = triad_views.compute_multichannel_batch(store, case, rec.slug, start_seconds=3., duration_seconds=1)
    assert page["gaps"]
    assert not any(v is not None for tr in page["traces"] for v in tr["values_uv"])
    triad = triad_views.compute_triad_view(case, workspace_dir=store.root, recording=rec.slug, channel="LA1",
                                           event_id=None, center_seconds=3., window_seconds=.6)
    assert triad["gaps"]


def test_window_budget_and_partial_piece_filters(tmp_path, monkeypatch):
    path = _synthetic_edf(tmp_path / "synthetic.edf", [0, 1, 2, 8, 9, 10])
    store, case, rec = _register(path, tmp_path / "ws")
    t = f"{case}/{rec.slug}"
    monkeypatch.setattr(signal_prep, "PREPARED_DATA_BUDGET_BYTES", 2 * 1024 * 9)
    with pytest.raises(ToolkitError, match="overrides.start_seconds") as exc:
        analyze_case(t, workspace_dir=store.root, overrides={"detector_enabled": False})
    assert exc.value.code == "analysis_window_too_large" and exc.value.details["prepared_bytes"] == 6 * 1024 * 9
    assert not store.has_run(case, rec.slug, "baseline")
    # Seven wall-clock seconds, just two seconds of acquired data. Each SOS call stays inside its piece.
    spans = []
    filt = signal_prep.sosfiltfilt

    def observed(sos, data, *args, **kwargs):
        spans.append(len(data))
        return filt(sos, data, *args, **kwargs)

    monkeypatch.setattr(signal_prep, "sosfiltfilt", observed)
    analyze_case(t, workspace_dir=store.root, run_name="window",
                 overrides={"detector_enabled": False, "start_seconds": 2, "end_seconds": 9, "lowpass": 200})
    assert spans == [1024, 1024]
    prep = json.loads(Path(store.load_run(case, rec.slug, "window")["meta"]["prepared_json_path"]).read_text())
    edge = prep["effective_filters"]["edge_samples_each_side"]
    mask = np.load(prep["valid_path"])[0]
    assert mask.shape == (2048,)
    assert not mask[1024 - edge:1024 + edge].any() and mask[512] and mask[1536]


def test_true_time_detection_controls_and_csv(tmp_path, monkeypatch):
    path = _synthetic_edf(tmp_path / "synthetic.edf", [0, 1, 2, 3, 4, 5, 20.125, 21.125, 22.125, 23.125,
                                                       24.125, 25.125], pulse_time=23.375)
    store, case, rec = _register(path, tmp_path / "ws")
    calls = []

    def synthetic_detector(self, action, params, directory):
        left, right = params["start"], params["stop"]
        row = np.load(params["data_path"], mmap_mode="r")[0, left:right]
        hits = np.flatnonzero(row > 100e-6)
        events = [dict(channel_name="LA1", index=0, start_sample=int(hits[0]), end_sample=int(hits[-1]) + 1)] if hits.size else []
        calls.append((left, right))
        return dict(events=events, backend_sfreq=1024, backend_samples=right-left, session_path=str(directory / "session"))

    monkeypatch.setattr(pipeline, "resolve_engine_settings", lambda _: EngineSettings(python_executable=sys.executable))
    monkeypatch.setattr(PyHFORunner, "__init__", lambda self, settings: setattr(self, "settings", settings))
    monkeypatch.setattr(PyHFORunner, "source_identity", lambda self: {})
    monkeypatch.setattr(PyHFORunner, "_invoke", synthetic_detector)
    monkeypatch.setattr(pyhfo_runner, "MAX_CHUNK_SECONDS", 4)
    analyze_case(f"{case}/{rec.slug}", workspace_dir=store.root)
    assert all(b-a <= 4096 and not a < 6144 < b for a, b in calls)
    bundle = store.load_run(case, rec.slug, "baseline")
    assert len(bundle["events"]) == 1
    event = bundle["events"][0]
    assert 23.375 <= event["start_seconds"] < 23.38
    assert [(a, b) for _, a, b in rec.time_slices(event["start_seconds"], event["end_seconds"])] == [
        (event["start_sample"], event["end_sample"])
    ]
    controls = select_control_windows(store, case, rec.slug, count=12)
    assert any(c["start_seconds"] >= 20.125 for c in controls)
    assert all(c["end_seconds"] <= 6 or c["start_seconds"] >= 20.125 for c in controls)
    exp = export_edf_annotations(case, rec.slug, workspace_dir=store.root)
    assert exp["edf_path"] is None and "clock" in exp["edf_copy"]["reason"]
    with Path(exp["csv_path"]).open() as f:
        rows = list(csv.DictReader(f))
    assert [float(r["onset"]) for r in rows if "BL HFO" in r["description"]] == [event["start_seconds"]]


def test_large_source_hash_shared_by_inspect_pages_and_card(tmp_path, monkeypatch):
    path = _synthetic_edf(tmp_path / "synthetic.edf", range(6), subtype="EDF+C")
    monkeypatch.setattr(io_integrity, "LARGE_FILE_THRESHOLD_BYTES", 1000)
    monkeypatch.setattr(io_integrity, "stat_signature_is_settled", lambda *args: True)
    monkeypatch.setattr(triad_views, "_FULL_REVERIFY_INTERVAL_S", 0)
    full = io_integrity._full_large_hash
    hashed = []

    def counted(fh, size, name, progress):
        hashed.append(name)
        return full(fh, size, name, progress)

    monkeypatch.setattr(io_integrity, "_full_large_hash", counted)
    progress = []
    store = WorkbenchStore(tmp_path / "ws")
    inspection = inspect_case(str(path), workspace_dir=store.root,
                              progress_cb=lambda *args: progress.append(args))
    case, slug = inspection["case"], inspection["recordings"][0]["recording"]
    rec = store.load_recording(case, slug)
    assert any(stage == "hash" and path.name in detail for stage, _, detail in progress)
    for start in (1, 2, 4):
        assert triad_views.compute_multichannel_batch(store, case, slug, start_seconds=start,
                                                      duration_seconds=1)["traces"][0]["times"]
    card = triad_views.render_evidence_card(case, recording=slug, workspace_dir=store.root, channel="LA1",
                                            event_id=None, center_seconds=2)
    assert Path(card["image_path"]).read_bytes().startswith(b"\x89PNG")
    assert hashed == [path.name]
    # A metadata-only change forces a full verification and refreshes the source identity.
    old_mtime = path.stat().st_mtime_ns
    os.utime(path, ns=(old_mtime, old_mtime + 1_000_000_000))
    with ThreadPoolExecutor(4) as pool:
        signatures = list(pool.map(lambda _: io_integrity.verify_source_recording_integrity(rec), range(4)))
    assert len(set(signatures)) == 1 and hashed == [path.name] * 2
    monkeypatch.setattr(io_integrity, "_sample_digest", lambda *args: "sample mismatch")
    io_integrity.verify_source_recording_integrity(rec)
    assert len(hashed) == 3
    monkeypatch.setattr(io_integrity, "stat_signature_is_settled", lambda *args: False)
    io_integrity.verify_source_recording_integrity(rec)
    assert len(hashed) == 4
    with path.open("r+b") as f:
        f.seek(768)
        f.write(b"XX")
    with pytest.raises(ToolkitError) as exc:
        triad_views.compute_multichannel_batch(store, case, slug, start_seconds=0, duration_seconds=1)
    assert exc.value.code == "source_recording_modified"


@pytest.mark.parametrize("stage", ["read", "publish"])
def test_source_change_aborts_analysis(tmp_path, monkeypatch, stage):
    path = _synthetic_edf(tmp_path / "synthetic.edf", range(6), subtype="EDF+C")
    monkeypatch.setattr(io_integrity, "LARGE_FILE_THRESHOLD_BYTES", 1000)
    store, case, rec = _register(path, tmp_path / "ws")
    module, function = (signal_prep, "read_scaled_volts") if stage == "read" else (pipeline, "summarize_run")
    original = getattr(module, function)

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        # Keep the same size and change one signal sample after the verified read.
        with path.open("r+b") as f:
            f.seek(768)
            f.write(b"XX")
        return result

    monkeypatch.setattr(module, function, mutate)
    with pytest.raises(ToolkitError) as exc:
        analyze_case(f"{case}/{rec.slug}", workspace_dir=store.root, run_name="changed",
                     overrides={"detector_enabled": False, "start_seconds": 1, "end_seconds": 3})
    assert exc.value.code == "source_recording_modified"
    assert not store.has_run(case, rec.slug, "changed")
    cache = store.recording_dir(case, rec.slug) / "_stage_cache"
    assert not list(cache.rglob("*.ref.json"))
    assert not list(cache.rglob("data.npy"))


def test_mixed_rate_resampling_stops_at_gap(tmp_path):
    import edfio

    path = tmp_path / "synthetic_mixed.edf"
    # A step across unrecorded time must not ring into either of the constant acquired pieces.
    edfio.Edf([
        edfio.EdfSignal(np.zeros(1024), sampling_frequency=256, label="LA1", physical_dimension="uV"),
        edfio.EdfSignal(np.repeat([-100., 100.], 128), sampling_frequency=64, label="LA2", physical_dimension="uV"),
    ], data_record_duration=1, annotations=[edfio.EdfAnnotation(0, 0, "Synthetic")]).write(path)
    contents = bytearray(path.read_bytes())
    for original, onset in ((b"+2\x14\x14", b"+8\x14\x14"), (b"+3\x14\x14", b"+9\x14\x14")):
        assert contents.count(original) == 1
        contents = contents.replace(original, onset)
    path.write_bytes(contents)
    _, _, rec = _register(path, tmp_path / "ws")
    assert rec.gaps() == [(2, 8)]
    samples = triad_views._READER_CACHE.read(rec, [1], 490, 535)[0] * 1e6
    np.testing.assert_allclose(samples[:22], -100., atol=.01)
    np.testing.assert_allclose(samples[22:], 100., atol=.01)


def test_days_long_fractional_clock_sample_roundtrip(tmp_path):
    # Fractional TAL offset plus a non-power-of-two sampling rate exercises precision loss.
    path = _synthetic_edf(tmp_path / "synthetic.edf", [0, 1, 800000.12345], fs=1000)
    _, _, rec = _register(path, tmp_path / "ws")
    p = rec.pieces[-1]
    for a in range(2000, 3000):
        t0, t1 = p.time_at(a, rec.sfreq), p.time_at(a + 1, rec.sfreq)
        assert rec.time_slices(t0, t1) == [(p, a, a + 1)]
        assert rec.time_slices(t0, t0) == []


def test_chunk_plan_with_large_minimum():
    chunks = list(pyhfo_runner._detection_chunks(0, 1250, limit=600, margin=1, minimum=599))
    assert len(chunks) <= 4
    assert chunks[0][0] == 0 and chunks[-1][1] == 1250
    assert all(599 <= b - a <= 600 for a, b in chunks)
    assert all(a2 <= b1 - 2 for (_, b1), (a2, _) in zip(chunks, chunks[1:]))


def _synthetic_segment_case(tmp_path):
    import shutil

    fixture = Path(__file__).parent / "fixtures" / "synthetic_seeg.edf"
    source = tmp_path / "synthetic.edf"
    shutil.copyfile(fixture, source)
    metadata = json.loads(fixture.with_suffix(".json").read_text())
    metadata["markers"].extend([
        {"start": 8, "end": 10, "label": "Synthetic sleep 1", "source": "synthetic"},
        {"start": 28, "end": 32, "label": "Synthetic sleep 2", "source": "synthetic"},
    ])
    source.with_suffix(".json").write_text(json.dumps(metadata))
    return _register(source, tmp_path / "workspace")


def _mock_segment_detector(monkeypatch, events_for_window):
    def invoke(self, action, params, directory):
        events = events_for_window(params)
        return {"events": events, "backend_sfreq": params["sfreq"], "backend_samples": params["stop"] - params["start"],
                "session_path": str(directory / "synthetic-session")}

    monkeypatch.setattr(pipeline, "resolve_engine_settings", lambda _: EngineSettings(python_executable=sys.executable))
    monkeypatch.setattr(PyHFORunner, "__init__", lambda self, settings: setattr(self, "settings", settings))
    monkeypatch.setattr(PyHFORunner, "source_identity", lambda self: {})
    monkeypatch.setattr(PyHFORunner, "_invoke", invoke)


def test_marker_segments_weighted_rates_and_clinical_outputs(tmp_path, monkeypatch):
    import edfio
    from pptx import Presentation
    from brainlens.case_sheet import export_case_sheet
    from brainlens.mdt_slides import export_mdt_slides
    from brainlens.trends import recording_trends

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    store, case, rec = _synthetic_segment_case(tmp_path)
    calls = []

    def synthetic_events(params):
        calls.append((params["start"], params["stop"]))
        return [{"index": 0, "channel_name": "SYN_A1", "start_sample": 2500, "end_sample": 2550}]

    _mock_segment_detector(monkeypatch, synthetic_events)
    target = f"{case}/{rec.slug}"
    overrides = {"channels": ["SYN_A1"], "marker_segments": {"pattern": "sleep", "before_seconds": 1, "after_seconds": 1}}
    # Ten acquired seconds fit. A full allocation between the two samples would exceed budget.
    monkeypatch.setattr(signal_prep, "PREPARED_DATA_BUDGET_BYTES", 10 * 2000 * 9)
    analyze_case(target, workspace_dir=store.root, run_name="nights", overrides=overrides)
    run = store.load_run(case, rec.slug, "nights")
    assert calls == [(0, 8000), (8000, 20000)]
    segments = run["summary"]["segments"]
    assert [(s["start_seconds"], s["end_seconds"]) for s in segments] == [(7, 11), (27, 33)]
    rows = [s["channels"][0] for s in segments]
    total = run["summary"]["channels"][0]
    assert [r["raw_count"] for r in rows] == [1, 1]
    assert [r["detector_analyzed_seconds"] for r in rows] == [3, 5]
    assert total["raw_count"] == total["focal_count"] == 2
    assert total["valid_signal_seconds"] == 10 and total["detector_analyzed_seconds"] == 8
    assert total["raw_rate_per_min"] == total["focal_rate_per_min"] == 15
    assert [(e["segment_index"], e["segment_label"], e["start_seconds"]) for e in run["events"]] == [
        (0, "Synthetic sleep 1", 8.25), (1, "Synthetic sleep 2", 28.25)]
    trend = recording_trends(store, case, rec.slug, measure="hfo_rate", run="nights", epoch_seconds=5)
    # Partial assessed epochs carry only their own detector exposure. Zero means observed and quiet.
    assert trend["matrix"][0] == [None, 24, 0, None, None, 24, 0, None]
    from brainlens.analytics import query_case
    assert query_case(target, store.root, mode="trends", measure="hfo_rate")["run"] == "nights"

    controls = select_control_windows(store, case, rec.slug, "nights")
    assert controls and all(any(s["start_seconds"] <= c["start_seconds"] < c["end_seconds"] <= s["end_seconds"]
                                for s in segments) for c in controls)
    event = run["events"][-1]
    triad = triad_views.compute_triad_view(case, workspace_dir=store.root, recording=rec.slug, run_name="nights",
                                           event_id=event["id"])
    assert triad["focus_event"]["id"] == event["id"] and triad["times"][0] < 28.25 < triad["times"][-1]

    annotations = export_edf_annotations(case, rec.slug, workspace_dir=store.root, run_name="nights")
    with Path(annotations["csv_path"]).open() as stream:
        assert [float(row["onset"]) for row in csv.DictReader(stream) if "BL HFO" in row["description"]] == [8.25, 28.25]
    assert [ann.onset for ann in edfio.read_edf(annotations["edf_path"]).annotations if "BL HFO" in ann.text] == [8.25, 28.25]
    sheet = export_case_sheet(case, store.root, primary_run="nights", max_cards=1)
    assert "Synthetic sleep 2 (27–33 s)" in Path(sheet["report_path"]).read_text()
    slides = export_mdt_slides(case, store.root, primary_run="nights", max_cards=1)
    texts = [shape.text for slide in Presentation(slides["slides_path"]).slides
             for shape in slide.shapes if shape.has_text_frame]
    assert any("2 selected segments" in t and "Synthetic sleep 2" in t for t in texts)

    monkeypatch.setattr(signal_prep, "PREPARED_DATA_BUDGET_BYTES", 10 * 2000 * 9 - 1)
    with pytest.raises(ToolkitError) as ex:
        analyze_case(target, workspace_dir=store.root, run_name="too_big", overrides=overrides)
    assert ex.value.code == "analysis_window_too_large" and ex.value.details["prepared_bytes"] == 180000
    assert not store.has_run(case, rec.slug, "too_big")


def test_adjacent_segments_split_synchrony_and_coverage(tmp_path, monkeypatch):
    store, case, rec = _synthetic_segment_case(tmp_path)
    fs = int(rec.sfreq)

    def near_boundary(params):
        left = params["start"] == 0
        tick = params["stop"] - 2 if left else 0
        return [{"index": 0, "channel_name": "SYN_A1" if left else "SYN_A3", "start_sample": tick,
                 "end_sample": tick + 2}]

    _mock_segment_detector(monkeypatch, near_boundary)
    segments = [{"start_seconds": 7, "end_seconds": 11, "label": "first"},
                {"start_seconds": 11, "end_seconds": 15, "label": "second"}]
    overrides = {"channels": ["SYN_A1", "SYN_A3"], "segments": segments, "detector": {"edge_seconds": 0}}
    target = f"{case}/{rec.slug}"
    analyze_case(target, workspace_dir=store.root, run_name="adjacent", overrides=overrides)
    run = store.load_run(case, rec.slug, "adjacent")
    assert [(e["end_sample"], e["coactive_channel_count"]) for e in run["events"]] == [(11*fs, 1), (11*fs+2, 1)]
    assert run["summary"]["selection"]["segments"] == segments
    det = json.loads(Path(run["meta"]["detection_json_path"]).read_text())
    assert det["valid_intervals"]["ch0000"] == [[7*fs, 11*fs], [11*fs, 15*fs]]
    for change in ([segments[0], segments[0]], [], [dict(segments[0], end_seconds=100)]):
        with pytest.raises(ToolkitError) as ex:
            analyze_case(target, workspace_dir=store.root, run_name="invalid", overrides={"segments": change})
        assert ex.value.code == "invalid_segments"


def test_admission_clocks_mcp_and_reader_focus(tmp_path, monkeypatch):
    import urllib.request
    from brainlens import mcp_server
    from brainlens.timeline import admission_timeline
    from brainlens.workbench_http import WorkbenchHTTPServer

    folder = tmp_path / "SyntheticAdmission"
    folder.mkdir()
    # The last file uses tomorrow's date. Names deliberately oppose acquisition order.
    for name, date, hour in [("night_z", "01-JAN-2020", "23.59.50"),
                             ("night_a", "02-JAN-2020", "00.00.10"),
                             ("night_b", "03-JAN-2020", "00.00.00")]:
        path = _synthetic_edf(folder / f"{name}.edf", [0.125, 1.125, 4.25, 5.25], pulse_time=4.5)
        with path.open("r+b") as f:
            f.seek(88)
            f.write(f"Startdate {date} X X Synthetic".encode().ljust(80))
            f.write((date[:2] + ".01.20" + hour).encode())
        path.with_suffix(".json").write_text(json.dumps({"synthetic": True, "markers": [
            {"start": 1, "end": 1.5, "label": "Synthetic Sz 2020-01-02", "source": "synthetic_fixture"}]}))
    workspace = tmp_path / "ws"
    store = WorkbenchStore(workspace)
    case = inspect_case(str(folder), workspace_dir=workspace)["case"]
    monkeypatch.setattr(mcp_server, "_WORKSPACE_ROOT", workspace)
    response = mcp_server.query_evidence(case, mode="timeline", text="sZ")
    assert [r["offset_seconds"] for r in response["recordings"]] == [0, 20, 86410]
    assert response["recordings"][2]["day_index"] == 1
    assert response["recordings"][2]["seconds_of_day"] == 10
    assert response["gaps"][0] == {"start_seconds": 6.125, "end_seconds": 20.0}
    assert response["recordings"][0]["gaps"] == [{"start_seconds": 2, "end_seconds": 4.125}]
    assert [e["admission_seconds"] for e in response["events"]] == [1, 21, 86411]
    assert "2020-01-" not in json.dumps(response)
    assert all("absolute_start" not in row for row in response["recordings"])
    rec = response["recordings"][0]["recording"]
    assert mcp_server.query_evidence(case, mode="trends")["code"] == "missing_recording"
    for selection in ({"segments": [{"start_seconds": 0, "end_seconds": 1, "label": "sample"}]},
                      {"marker_segments": {"pattern": "sz", "after_seconds": 1}}):
        with pytest.raises(ToolkitError) as exc:
            analyze_case(case, workspace_dir=workspace, overrides=selection)
        assert exc.value.code == "missing_recording"
    rec_id = response["recordings"][0]["recording_id"]
    tr = mcp_server.query_evidence(case, recording=rec_id, mode="trends", measure="rms", epoch_seconds=2)
    assert "2020-01-" not in json.dumps(tr) and "markers" not in tr and len(tr["matrix"]) == 1

    server = WorkbenchHTTPServer(workspace, port=0)
    base = server.start()
    try:
        def get(route):
            with urllib.request.urlopen(base + route) as resp:
                return json.load(resp)
        target = f"case={case}&recording={rec}"
        http = get(f"/api/timeline?case={case}&text=sz")
        assert http["recordings"][0]["absolute_start"] == "2020-01-01T23:59:50.125000"
        assert len(http["events"]) == 3 and http["events"][0]["label"].endswith("2020-01-02")
        full = get(f"/api/trends?{target}&measure=rms&epoch_seconds=2")
        assert full["matrix"] == tr["matrix"] and full["markers"]
        req = urllib.request.Request(base + "/api/marker-segments", method="POST", data=json.dumps({
            "case": case, "recording": rec, "pattern": "sz", "before_seconds": .5, "after_seconds": .5,
        }).encode(), headers={"Content-Type": "application/json", "X-BrainLens-Token": server.auth_token})
        with urllib.request.urlopen(req) as resp:
            assert json.load(resp)["segments"] == [{"start_seconds": .5, "end_seconds": 2, "label": "Synthetic Sz 2020-01-02"}]
        monkeypatch.setattr(mcp_server, "ensure_workbench_server", lambda **_: {
            "url": base, "auth_token": server.auth_token, "reused": True, "pid": os.getpid()})
        store.update_ui_state({"event_id": "previous", "shaft": "LB"})
        focus = mcp_server.open_workbench(case=case, recording=rec_id, start_seconds=4.375, channel="LA1")
        assert "start_seconds=4.375" in focus["deep_link"]
        state = store.load_ui_state()
        assert (state["view"], state["recording"], state["start_seconds"], state["event_id"], state["shaft"]) == (
            "reader", rec, 4.375, None, None)
    finally:
        server.stop()

    # Legacy registration can obtain its clock directly from the A1 verified source.
    reg_path = store.recording_path(case, rec)
    doc = json.loads(reg_path.read_text())
    del doc["header_clock"]
    reg_path.write_text(json.dumps(doc))
    assert admission_timeline(store, case, include_absolute=True)["recordings"] == http["recordings"]


def test_bids_time_fallback_and_unknown_file_offsets(tmp_path):
    from brainlens.timeline import admission_timeline
    from brainlens.models import atomic_write_json

    folder = tmp_path / "sub-Synthetic" / "ses-one" / "ieeg"
    folder.mkdir(parents=True)
    for name in ["a", "b", "c"]:
        _synthetic_edf(folder / f"{name}.edf", [0, 1], subtype="EDF+C")
    store = WorkbenchStore(tmp_path / "ws")
    case = inspect_case(str(folder), workspace_dir=store.root)["case"]
    unknown = admission_timeline(store, case)
    assert all(r["offset_seconds"] is None and r["clock_reason"] == "identical_start_times" for r in unknown["recordings"])
    assert not unknown["gaps"]
    # Absolute clock from scans is navigation metadata. It can be fixed without
    # changing any recording fingerprint or repeating signal analysis.
    table = folder.parent / "sub-Synthetic_ses-one_scans.tsv"
    table.write_text("filename\tacq_time\nieeg/a.edf\t2022-09-01T23:50:00-04:00\n"
                     "ieeg/b.edf\t2022-09-02T04:05:00Z\nieeg/c.edf\t2022-09-02T04:07:00+00:00\n")
    dated = admission_timeline(store, case)
    assert [r["clock_source"] for r in dated["recordings"]] == ["bids_scans"] * 3
    assert [r["offset_seconds"] for r in dated["recordings"]] == [0, 900, 1020]
    table.write_text(table.read_text().replace("ieeg/c.edf", "elsewhere/c.edf"))
    rec = dated["recordings"][-1]["recording"]
    p = store.recording_path(case, rec)
    doc = json.loads(p.read_text())
    doc["header_clock"] = {"start": "1985-01-01T00:00:00"}
    atomic_write_json(p, doc)
    partial = admission_timeline(store, case)
    assert [r["offset_seconds"] for r in partial["recordings"]] == [0, 900, None]
    assert partial["recordings"][-1]["clock_reason"] == "missing_or_placeholder_start"
    assert not partial["gaps"]


def test_fif_first_sample_and_marker_time(tmp_path):
    import mne
    from datetime import datetime, timezone
    from brainlens.timeline import admission_timeline

    file = tmp_path / "synthetic_raw.fif"
    raw = mne.io.RawArray(np.zeros((1, 512)), mne.create_info(["A1"], 128, ["seeg"]),
                          first_samp=64, verbose=False)
    raw.set_meas_date(datetime(2021, 4, 5, 12, tzinfo=timezone.utc))
    raw.set_annotations(mne.Annotations([1.25], [.25], ["Synthetic sleep"]))
    raw.save(file, overwrite=True, verbose=False)
    file.with_suffix(".json").write_text('{"synthetic": true}')
    store, case, _ = _register(file, tmp_path / "ws")
    clock = admission_timeline(store, case, include_absolute=True)
    assert clock["recordings"][0]["absolute_start"] == "2021-04-05T12:00:00.500000+00:00"
    assert [(e["label"], e["start_seconds"]) for e in clock["events"]] == [("Synthetic sleep", 1.25)]


def test_trend_power_bandwidth_and_compact_peaks(tmp_path, monkeypatch):
    import edfio
    from brainlens import trends

    # Synthetic 4-channel EDF. Only R1 has the bandwidth and planted ripple.
    signals = []
    for name, fs, lowpass in [("R1", 1024, 400), ("R2", 500, 250), ("R3", 1024, 75), ("R4", 128, 64)]:
        t = np.arange(256 * fs) / fs
        values = 10 * np.sin(2 * np.pi * 6 * t)
        if name == "R1":
            values += 500 * np.sin(2 * np.pi * 175 * t) * ((t >= 222) & (t < 224))
        signals.append(edfio.EdfSignal(values, sampling_frequency=fs, label=name, physical_dimension="uV",
                                       prefiltering=f"LP:{lowpass}Hz"))
    source = tmp_path / "synthetic_mixed.edf"
    edfio.Edf(signals, data_record_duration=1,
              annotations=[edfio.EdfAnnotation(2, 0, "Synthetic 2001-01-01")]).write(source)
    source.with_suffix(".json").write_text('{"synthetic":true}', encoding="utf-8")
    store, case, rec = _register(source, tmp_path / "workspace")
    calls = []
    original = trends.read_scaled_volts

    def counted(raw, recording, start, stop):
        calls.append((start, stop))
        return original(raw, recording, start, stop)

    monkeypatch.setattr(trends, "read_scaled_volts", counted)
    progress = []
    full = trends.recording_trends(store, case, rec.slug, epoch_seconds=2, progress_cb=lambda *xs: progress.append(xs))
    assert calls == [(i, i + 2048) for i in range(0, rec.n_samples, 2048)]
    assert progress[0][1] == 0 and progress[-1][1] == 1
    assert full["units"] == "dB re 1 µV²"
    assert full["top_channels"][0]["name"] == "R1" and full["top_channels"][0]["peak_seconds"] == 222
    # Integrated Hann-windowed power of a 500 µV amplitude sinusoid = A²/2.
    assert full["top_channels"][0]["peak_value"] == pytest.approx(10 * np.log10(500**2 / 2), abs=.04)
    assert all(all(v is None for v in row) for row in full["matrix"][1:])
    assert full["file_boundaries"] == [0., 256.] and full["markers"][0]["label"] == "Synthetic 2001-01-01"
    calls.clear()
    compact = trends.recording_trends(store, case, rec.slug, epoch_seconds=2, compact=True)
    assert 1 < len(compact["times_seconds"]) <= 120
    assert compact["top_channels"] == full["top_channels"] and "markers" not in compact
    assert max(compact["matrix"][0]) == full["top_channels"][0]["peak_value"]
    theta = trends.recording_trends(store, case, rec.slug, epoch_seconds=2, measure="theta")
    assert theta["matrix"][0][0] == pytest.approx(10 * np.log10(10**2 / 2), abs=.04)
    assert not calls  # Signal measures and display resolutions share one streaming pass.


def test_trend_gaps_no_line_length_bridge_and_cache_identity(tmp_path):
    from brainlens.trends import recording_trends
    from brainlens.models import read_json

    fs = 1024
    source = _synthetic_edf(tmp_path / "synthetic_steps.edf", [0, 1, 6.25, 7.25], fs=fs)
    # Two flat pieces at opposite voltages, in one 10 s epoch. Changing
    # level in an unrecorded interval contributes zero measured line length.
    with source.open("r+b") as f:
        for index, level in enumerate([3277, 3277, -3277, -3277]):
            f.seek(768 + index * (fs + 48) * 2)
            f.write(np.full(fs, level, dtype="<i2").tobytes())
    store, case, rec = _register(source, tmp_path / "workspace")
    line = recording_trends(store, case, rec.slug, measure="line_length")
    assert line["units"] == "µV" and line["matrix"] == [[0.]]
    assert line["gaps"] == [{"start_seconds": 2., "end_seconds": 6.25}]
    rms = recording_trends(store, case, rec.slug, epoch_seconds=2, measure="rms")
    assert rms["units"] == "µV" and rms["matrix"][0][1:3] == [None, None]
    assert rms["matrix"][0][0] == pytest.approx(100, abs=.03)
    assert rms["matrix"][0][3] == pytest.approx(100, abs=.03)
    with source.open("r+b") as f:
        f.seek(768)
        f.write(np.zeros(fs, dtype="<i2").tobytes())
    with pytest.raises(ToolkitError) as stale:
        recording_trends(store, case, rec.slug, epoch_seconds=2, measure="rms")
    assert stale.value.code == "source_recording_modified"
    _, new_case, changed = _register(source, store.root)
    assert new_case == case and changed.slug == rec.slug and changed.fingerprint != rec.fingerprint
    fresh = recording_trends(store, case, rec.slug, epoch_seconds=2, measure="rms")
    assert fresh["matrix"][0][0] == pytest.approx(100 / np.sqrt(2), abs=.03)
    cache_dir = store.recording_dir(case, rec.slug) / "trends"
    manifests = [read_json(p) for p in cache_dir.glob("*/manifest.json")]
    assert {m["recording_fingerprint"] for m in manifests} == {rec.fingerprint, changed.fingerprint}
    assert all(m["code_version"] and m["parameters"]["epoch_seconds"] in (2, 10) for m in manifests)
    for bad in (float("nan"), float("inf"), 1, 61, "unrecognized"):
        with pytest.raises(ToolkitError) as exc:
            recording_trends(store, case, rec.slug, epoch_seconds=bad)
        assert exc.value.code == "invalid_trend_epoch"
    with pytest.raises(ToolkitError) as exc:
        recording_trends(store, case, rec.slug, measure="unknown")
    assert exc.value.code == "invalid_trend_measure"


def test_brainvision_timeline_marker_with_missing_clock(tmp_path):
    from brainlens.timeline import admission_timeline

    path = tmp_path / "synthetic.vhdr"
    path.write_text("Brain Vision Data Exchange Header File Version 1.0\n"
                    "[Common Infos]\nDataFile=synthetic.eeg\nMarkerFile=synthetic.vmrk\nDataFormat=BINARY\n"
                    "DataOrientation=MULTIPLEXED\nNumberOfChannels=1\nSamplingInterval=1000\n"
                    "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\nCh1=LA1,,1,µV\n")
    path.with_suffix(".eeg").write_bytes(np.zeros(2000, dtype="<f4").tobytes())
    path.with_suffix(".vmrk").write_text("Brain Vision Data Exchange Marker File, Version 1.0\n"
                                        "[Common Infos]\nDataFile=synthetic.eeg\n[Marker Infos]\n"
                                        "Mk1=Comment,Synthetic sleep,1001,100,0\n")
    path.with_suffix(".json").write_text('{"synthetic": true}')
    store, case, _ = _register(path, tmp_path / "ws")
    result = admission_timeline(store, case, text="sleep")
    row, = result["recordings"]
    assert row["offset_seconds"] is None and row["clock_reason"] == "missing_or_placeholder_start"
    marker, = result["events"]
    assert marker["start_seconds"] == 1 and marker["end_seconds"] == pytest.approx(1.1)
    assert marker["admission_seconds"] is None


def test_raw_trend_contact_in_bipolar_run(tmp_path):
    store, case, rec = _synthetic_segment_case(tmp_path)
    target = f"{case}/{rec.slug}"
    analyze_case(target, workspace_dir=store.root, run_name="bipolar",
                 overrides={"reference_mode": "bipolar", "detector_enabled": False})
    center = 8.125
    raw = triad_views.compute_triad_view(target, workspace_dir=store.root, run_name="bipolar",
                                         channel="SYN_A1", center_seconds=center, source_only=True)
    assert raw["channel_name"] == "SYN_A1" and raw["analysis_signature"] is None
    assert raw["times"][0] < center < raw["times"][-1]
    page = triad_views.compute_multichannel_batch(store, case, rec.slug, run_name="bipolar", shaft="ALL",
                                                  focus_channel="SYN_A1", start_seconds=center, source_only=True)
    assert page["effective_reference_mode"] == "original"
    assert any(trace["channel_name"] == "SYN_A1" for trace in page["traces"])
