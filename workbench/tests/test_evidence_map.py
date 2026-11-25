"""On-demand evidence layer and recording-bound algorithm regressions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

WORKBENCH_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = WORKBENCH_DIR / "src"
if importlib.util.find_spec("brainlens") is None and SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlens.ictal import compute_ei, resolve_onset
from brainlens.models import MarkerSpec, ToolkitError
from brainlens.spikes import detect_ieds


def test_ied_known_spikes_clean_controls_and_gaps() -> None:
    sfreq = 512
    seconds = 120
    rng = np.random.default_rng(233)
    t = np.arange(seconds * sfreq) / sfreq
    data = rng.normal(0, 2e-6, (5, len(t)))
    valid = np.ones_like(data, dtype=bool)
    injections = np.arange(3.0, seconds - 2, 4.0)
    for i, centre in enumerate(injections):
        data[0] += (-1) ** i * 65e-6 * np.exp(-0.5 * ((t - centre) / 0.009) ** 2)
    valid[0, (t >= 48) & (t < 60)] = False
    data[0, (t >= 80) & (t < 88)] = np.nan
    valid[-1] = False
    result = detect_ieds(data, valid, sfreq, ["A1", "A2", "A3", "A4", "A5"], start_seconds=10)
    rows = result["channels"]
    found = np.array(rows[0]["event_times"]) - 10
    truth = [v for v in injections if not (48 <= v < 60 or 80 <= v < 88)]
    recall = sum(np.any(abs(found - v) <= 0.05) for v in truth) / len(truth)
    assert recall >= 0.9
    assert not any(48 <= v < 60 or 80 <= v < 88 for v in found)
    assert rows[0]["valid_seconds"] == seconds - 20
    assert rows[0]["rate_per_min"] == pytest.approx(rows[0]["count"] * 60 / (seconds - 20))
    assert all(row["rate_per_min"] <= 1 for row in rows[1:4])
    assert rows[-1]["state"] == "unassessed" and rows[-1]["rate_per_min"] is None


def test_ei_early_and_delayed_fast_recruitment() -> None:
    rng = np.random.default_rng(89)
    sfreq = 256
    t = np.arange(90 * sfreq) / sfreq
    channels = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    data = rng.normal(0, 1e-6, (len(channels), len(t)))
    # Two adjacent early contacts share a fast generator; A4 is recruited later.
    slow = 3e-6 * np.sin(2 * np.pi * 6 * t)
    data += slow
    fast_filter = butter(4, [30, 80], btype="bandpass", fs=sfreq, output="sos")
    fast = sosfiltfilt(fast_filter, rng.normal(0, 1.0, len(t)))
    for i, onset in [(0, 30), (1, 30), (3, 38)]:
        data[i] += 6e-6 * np.clip((t - onset) / 2, 0, 1) * fast
    valid = np.ones_like(data, dtype=bool)
    valid[5, t < 28] = False
    data[6, t >= 33] = np.nan
    result = compute_ei(data, valid, sfreq, channels, onset_seconds=30, lowpass_hz=[97, 97, 55, 97, 97, 97, 97])
    rows = {row["channel"]: row for row in result["channels"]}
    assert 0.8 <= rows["A1"]["ei"] <= 1 and 0.8 <= rows["A2"]["ei"] <= 1
    assert 0.02 < rows["A4"]["ei"] < 0.35
    assert rows["A4"]["detection_seconds"] - rows["A1"]["detection_seconds"] == pytest.approx(8, abs=1)
    assert rows["A3"]["ei"] == rows["A5"]["ei"] == 0.0
    assert rows["A3"]["gamma_upper_hz"] == 55
    assert rows["A6"]["ei"] is None and rows["A7"]["ei"] is None
    assert any(v is None for v in rows["A7"]["er"])
    assert result["reference_detection_seconds"] == min(row["detection_seconds"] for row in rows.values() if row["ei"])


def test_ei_global_onset_and_insufficient_bandwidth() -> None:
    def marker(label: str, seconds: float, channels: list[str] | None = None) -> MarkerSpec:
        return MarkerSpec(label=label, start=seconds, end=seconds, source="test", channels=channels or [])

    markers = [marker("sz", 1, ["A1"]), marker("seizure_end", 2), marker("seizure free", 3)]
    with pytest.raises(ToolkitError) as error:
        resolve_onset(None, markers)
    assert error.value.code == "seizure_onset_unavailable"
    markers += [marker("SZ2", 15), marker("Seizure onset", 10)]
    assert resolve_onset(None, markers) == 10
    assert resolve_onset(11, markers) == 11
    data = np.ones((1, 2000))
    result = compute_ei(data, np.ones_like(data, dtype=bool), 40, ["A1"], onset_seconds=10)
    assert result["channels"][0]["gamma_upper_hz"] == 20
    assert result["channels"][0]["reason"] == "insufficient_gamma_bandwidth"
    assert result["channels"][0]["ei"] is None


@pytest.fixture
def evidence_case(tmp_path, monkeypatch):
    """Two registered synthetic recordings. Only the first has an HFO run."""
    import mne
    from brainlens import pipeline
    from brainlens.io_readers import inspect_case
    from brainlens.models import atomic_write_json
    from brainlens.storage import WorkbenchStore

    monkeypatch.setenv("BRAINLENS_TRUST_DIR", str(tmp_path / "trust"))
    folder = tmp_path / "participant"
    folder.mkdir()
    fs = 400
    t = np.arange(50 * fs) / fs
    rng = np.random.default_rng(221)
    for name in ("rest", "episode"):
        names = ["A1", "A2", "A3" if name == "rest" else "B3", "A4", "ECG"]
        data = rng.normal(0, 1e-6, (len(names), t.size)) + 3e-6 * np.sin(2 * np.pi * 6 * t)
        if name == "rest":
            data[0] += 80e-6 * np.exp(-0.5 * ((t - 5) / 0.008) ** 2)
        else:
            data[1] += 8e-6 * np.clip((t - 20) / 2, 0, 1) * np.sin(2 * np.pi * 48 * t)
        raw = mne.io.RawArray(data, mne.create_info(names, fs, ["seeg"] * 4 + ["ecg"]), verbose="ERROR")
        raw.info["bads"] = ["A4"]
        if name == "episode":
            raw.set_annotations(mne.Annotations([20], [0], ["Seizure onset"]))
        raw.save(folder / f"{name}_raw.fif", verbose="ERROR")

    class SyntheticDetector:
        """Provide only candidate events. The real pipeline calculates all coverage and spectral summaries."""
        def __init__(self, _settings):
            pass

        def detect(self, prepared, config, output_dir):
            event = {"id": "known-ripple", "channel_id": "ch0000", "channel_name": "A1", "shaft": "A",
                     "start_sample": 4 * fs, "end_sample": 4 * fs + 30, "start_seconds": 4,
                     "end_seconds": 4 + 30 / fs, "labels": {}, "scores": {}}
            det = {"status": "complete", "events": [event], "valid_intervals": {
                ch["id"]: ([[0, prepared["n_samples"]]] if ch["name"] != "A4" else [])
                for ch in prepared["channels"]
            }}
            atomic_write_json(output_dir / "detection.json", det)
            return det

        def classify(self, detection, config, output_dir, **kwargs):
            result = {**detection, "events": [{**e, "labels": {"spike": 1}} for e in detection["events"]],
                      "classifier_status": "complete", "classification": {"stages": {"spike": {"status": "complete"}}}}
            atomic_write_json(output_dir / "classification.json", result)
            return result

    monkeypatch.setattr(pipeline, "resolve_engine_settings", lambda _: None)
    monkeypatch.setattr(pipeline, "PyHFORunner", SyntheticDetector)
    store = WorkbenchStore(tmp_path / "workspace")
    case = inspect_case(str(folder), workspace_dir=store.root)["case"]
    rest, episode = sorted(store.list_recordings(case), reverse=True)
    pipeline.analyze_case(f"{case}/{rest}", workspace_dir=store.root, run_name="baseline", overrides={"band": [80, 165]})
    return store, case, rest, episode


def test_evidence_rates_ranks_tags_and_byte_identical_runs(evidence_case):
    from brainlens.evidence_map import get_evidence_map, get_epileptogenicity
    from brainlens.pipeline import analyze_case
    from brainlens.storage_mvcc import run_analysis_signature

    store, case, rest, episode = evidence_case
    bundle = store.load_run(case, rest, "baseline", reconcile=False)
    signature = run_analysis_signature(bundle)
    paths = [Path(bundle["generation_dir"]) / name for name in ("run_meta.json", "summary.json", "events.csv", "channel_rates.csv")]
    before = [p.read_bytes() for p in paths]
    assert get_evidence_map(store, case, rest, cached_only=True) is None
    with pytest.raises(ToolkitError) as err:
        get_epileptogenicity(store, case, rest)
    assert err.value.code == "seizure_onset_unavailable"

    plain = get_evidence_map(store, case, rest)
    rows = {r["channel"]: r for r in plain["rows"]}
    assert {c["key"] for c in plain["columns"]} == {"ripple", "fast_ripple", "spike_hfo", "ied", "ei"}
    assert plain["ei_source"] is None
    assert rows["A1"]["biomarkers"]["ripple"]["value"] == pytest.approx(60 / 50)
    assert rows["A1"]["biomarkers"]["ripple"]["rank"] == 1
    assert rows["A1"]["biomarkers"]["ripple"]["percentile"] == 100
    assert rows["A1"]["biomarkers"]["ied"]["count"] >= 1
    assert rows["A1"]["convergence_count"] == 2
    assert rows["A3"]["state"] == "quiet_measured"
    assert rows["A3"]["convergence_count"] == 0
    assert rows["A3"]["biomarkers"]["ripple"]["percentile"] == 0
    assert "classifier" in rows["A1"]["biomarkers"]["spike_hfo"]["reason"]
    assert "Nyquist" in rows["A1"]["biomarkers"]["fast_ripple"]["reason"]
    assert {rows[k]["state"] for k in ("A4", "ECG")} == {"unassessed"}
    assert rows["A4"]["convergence_count"] is None

    store.save_clinical_overlay(case, {"source": "external fixture", "soz_channels": ["a2"], "resected_channels": ["A1"]})
    assert not store.has_run(case, episode, "baseline")
    ei = get_epileptogenicity(store, case, episode)
    assert ei["onset_seconds"] == 20 and ei["onset_source"] == "global_marker"
    assert ei["window_seconds"] == [0, 50]
    assert get_epileptogenicity(store, case, episode, cached_only=True) == ei
    with_ei = get_evidence_map(store, case, rest, cached_only=True)
    assert with_ei["ei_source"]["recording_id"] == episode
    assert with_ei["ei_source"]["recording_fingerprint"] == store.load_recording(case, episode).fingerprint
    linked = {r["channel"]: r for r in with_ei["rows"]}
    for name, row in linked.items():
        for key in ("ripple", "fast_ripple", "spike_hfo", "ied"):
            assert row["biomarkers"][key] == rows[name]["biomarkers"][key]
    assert linked["A2"]["biomarkers"]["ei"]["value"] == 1
    assert linked["A2"]["biomarkers"]["ei"]["rank"] == 1
    assert linked["A3"]["biomarkers"]["ei"]["value"] is None
    assert "matching" in linked["A3"]["biomarkers"]["ei"]["reason"]
    assert linked["A2"]["external_tags"] == ["SOZ"]
    assert linked["A1"]["external_tags"] == ["Resected"]
    assert linked["A2"]["convergence_count"] == 1
    assert [p.read_bytes() for p in paths] == before
    assert run_analysis_signature(store.load_run(case, rest, "baseline", reconcile=False)) == signature

    analyze_case(f"{case}/{rest}", workspace_dir=store.root, run_name="onnx",
                 overrides={"band": [80, 165], "classifier_enabled": True})
    scored = get_evidence_map(store, case, rest, "onnx")
    assert next(r for r in scored["rows"] if r["channel"] == "A1")["biomarkers"]["spike_hfo"]["value"] == pytest.approx(60 / 50)


def test_evidence_cards_sheet_and_cache_boundaries(evidence_case, monkeypatch):
    from brainlens import evidence_map
    from brainlens.case_sheet import export_case_sheet
    from brainlens.evidence_cards import render_ei_card, render_map_card
    from brainlens.models import verify_sheet_trust
    from brainlens.workbench_http import WorkbenchHTTPServer

    store, case, rest, episode = evidence_case
    initial_sig = store.case_sheet_run_signature(case, "baseline")

    def sheet():
        return Path(export_case_sheet(case, store.root, max_cards=0)["report_path"])

    with monkeypatch.context() as m:
        m.setattr(evidence_map, "_base_rows", lambda *a: pytest.fail("export triggered IED detection"))
        original = sheet()
        assert "Evidence map ·" not in original.read_text()
        server = WorkbenchHTTPServer(store.root)
        assert server._check_verified_sheet_state(case, original, check_judgments=False) == "fresh"
    assert evidence_map.get_evidence_map(store, case, rest, cached_only=True) is None
    first = evidence_map.get_evidence_map(store, case, rest)
    assert server._check_verified_sheet_state(case, original, check_judgments=False) == "stale"
    ei = evidence_map.get_epileptogenicity(store, case, episode, onset_seconds=19)
    assert evidence_map.get_latest_ei(store, case, episode)["cache_key"] == ei["cache_key"]
    assert evidence_map.get_epileptogenicity(store, case, episode, onset_seconds=15, cached_only=True) is None
    leaders = evidence_map.get_interictal_leaders(store, case, episode)
    assert leaders["recording_id"] == rest and leaders["leaders"][0]["channel"] == "A1"
    with monkeypatch.context() as m:
        m.setattr(evidence_map, "_base_rows", lambda *a: pytest.fail("cache miss on repeated read"))
        m.setattr(evidence_map, "compute_ei", lambda *a, **k: pytest.fail("cached EI recomputed"))
        for card in (render_map_card(store, case, rest), render_ei_card(store, case, episode)):
            assert Path(card["image_path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        report = sheet()
    html = report.read_text()
    assert "Evidence map ·" in html and episode in html
    assert "descriptive, not a validated predictor" in html
    assert verify_sheet_trust(store.workspace_id, case, report.name, report.read_bytes())
    assert server._check_verified_sheet_state(case, report) == "fresh"
    assert initial_sig != store.case_sheet_run_signature(case, "baseline")
    assert first["evidence_digest"] != evidence_map.get_evidence_map(store, case, rest)["evidence_digest"]


def test_evidence_public_interfaces_and_async_cache(evidence_case, monkeypatch, capsys):
    import asyncio
    import json
    import time

    from brainlens import mcp_server
    from brainlens.cli import main
    from brainlens.workbench_http import WorkbenchHTTPServer

    store, case, rest, episode = evidence_case
    monkeypatch.setattr(mcp_server, "_WORKSPACE_ROOT", store.root)
    schema = {tool.name: tool for tool in asyncio.run(mcp_server.mcp.list_tools())}
    assert set(schema) == {
        "inspect_case", "analyze_case", "query_evidence", "render_evidence_card",
        "open_workbench", "record_judgment", "export_or_protocol",
    }
    query_schema = schema["query_evidence"].inputSchema
    assert "onset_seconds" in query_schema["properties"] and "onset_seconds" not in query_schema["required"]

    server = WorkbenchHTTPServer(store.root)
    for path, recording in (("/api/evidence-map", rest), ("/api/epileptogenicity", episode)):
        params = {"case": case, "recording": recording}
        status, job = server._evidence_response(path, params)
        assert status == 202
        deadline = time.monotonic() + 20
        while job["status"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.02)
            job = server.jobs.get(job["job_id"])
        assert job["status"] == "complete", job["error"] or job["detail"]
        assert server._evidence_response(path, params) == (200, job["result"])

    cli = ["--workspace", str(store.root)]
    assert main([*cli, "evidence", case, "--recording", rest, "--run-name", "baseline"]) == 0
    assert json.loads(capsys.readouterr().out)["rows"]
    assert main([*cli, "ei", case, "--recording", episode, "--onset-seconds", "19"]) == 0
    assert json.loads(capsys.readouterr().out)["onset_seconds"] == 19

    # Selecting the original cached marker analysis via HTTP must rebind the map and EI card to it.
    status, marker = server._evidence_response("/api/epileptogenicity", {"case": case, "recording": episode})
    assert status == 200 and marker["onset_seconds"] == 20
    evidence = mcp_server.query_evidence(f"{case}/{rest}", mode="evidence_map", run_a="baseline")
    assert evidence["ei_source"]["cache_key"] == marker["cache_key"]
    explicit = mcp_server.query_evidence(case, mode="epileptogenicity", recording=episode, onset_seconds=19)
    assert explicit["onset_seconds"] == 19
    for selector, recording in (("evidence_map", rest), ("epileptogenicity", episode)):
        card = mcp_server.render_evidence_card(case, event_id=selector, recording=recording)
        assert card["view_mode"] == selector and card["recording"] == recording
        assert Path(card["image_path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_evicted_prepared_data_use_verified_snapshot(evidence_case):
    from brainlens.evidence_map import get_evidence_map
    from brainlens.models import read_json

    store, case, rest, _ = evidence_case
    run = store.load_run(case, rest, "baseline", reconcile=False)
    prep = read_json(run["meta"]["prepared_json_path"])
    data = Path(prep["data_path"])
    data.unlink()
    result = get_evidence_map(store, case, rest)
    row = next(r for r in result["rows"] if r["channel"] == "A1")
    assert row["biomarkers"]["ied"]["count"] > 0
    assert not data.exists()


def test_ties_with_missing_and_quiet_channels():
    from brainlens.evidence_map import _cell, _rank

    cells = [_cell(v) for v in [0, 6.0, 8.0, None, 8.0, 0]]
    _rank(cells)
    assert [c["rank"] for c in cells] == [4, 3, 1, None, 1, 4]
    assert [c["percentile"] for c in cells] == [0, 50, 100, None, 100, 0]


def test_ripple_and_fast_ripple_partition_focal_candidates(evidence_case, tmp_path):
    from brainlens.evidence_map import _measure_rows

    store, case, rest, _ = evidence_case
    # Same contact evaluated at full bandwidth and then with a run low-pass.
    rec = store.load_recording(case, rest).model_copy(update={"sfreq": 1024.0, "lowpass": 500.0})
    ch = rec.channels[0].model_copy(update={"source_sfreq": 1024.0, "source_lowpass": 500.0})
    data = np.random.default_rng(41).normal(size=(1, 5 * 1024))
    data_path, valid_path = tmp_path / "data.npy", tmp_path / "valid.npy"
    np.save(data_path, data)
    np.save(valid_path, np.ones_like(data, dtype=bool))
    prep = {"sfreq": 1024, "start_sample": 0, "n_samples": data.shape[1], "channels": [ch.model_dump()]}
    rate = {"channel_id": ch.id, "valid_signal_seconds": 5.0, "detector_analyzed_seconds": 4.0,
            "focal_count": 5, "fast_ripple_count": 2, "spike_hfo_count": None}
    bundle = {"config": {"analysis_config": {"detector": {"band": [80, 450]}}}, "summary": {"channels": [rate]}}
    both = _measure_rows(rec, bundle, prep, data_path, valid_path)[0]["biomarkers"]
    assert both["ripple"]["value"] == 45 and both["fast_ripple"]["value"] == 30
    prep["effective_filters"] = {"lowpass": 200.0}
    limited = _measure_rows(rec, bundle, prep, data_path, valid_path)[0]["biomarkers"]
    assert limited["fast_ripple"]["value"] is None
    assert "lowpass (200 Hz)" in limited["fast_ripple"]["reason"]
