"""Clinical reader display filters, tested against known frequency components."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

import edfio
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if importlib.util.find_spec("brainlens") is None and SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlens.io_readers import inspect_case
from brainlens.models import ToolkitError
from brainlens.storage import WorkbenchStore
from brainlens.triad_views import compute_multichannel_batch
from brainlens.workbench_http import WorkbenchHTTPServer


def _tone_signal(name: str, rate: int, tones: tuple[int, ...], prefilter: str = "") -> edfio.EdfSignal:
    seconds = np.arange(6 * rate) / rate
    data_uv = sum(20 * np.sin(2 * np.pi * hz * seconds) for hz in tones)
    return edfio.EdfSignal(data_uv, rate, label=name, physical_dimension="uV", prefiltering=prefilter)


def _register(tmp_path: Path, signals: list[edfio.EdfSignal], record_seconds: float = 1.0) -> WorkbenchStore:
    path = tmp_path / "tones.edf"
    edfio.Edf(signals, data_record_duration=record_seconds).write(path)
    ws = tmp_path / "ws"
    inspect_case(str(path), workspace_dir=ws)
    return WorkbenchStore(ws)


@pytest.mark.parametrize(("lff", "hff"), [
    ("", None), ("abc", None), (float("nan"), 70), (1, "inf"),
    (0, 70), (1, -5), (True, None), (70, 70), (71, 70),
])
def test_invalid_display_cutoffs(tmp_path: Path, lff, hff) -> None:
    with pytest.raises(ToolkitError) as exc:
        compute_multichannel_batch(WorkbenchStore(tmp_path), "tones", "tones", display_lff=lff, display_hff=hff)
    assert exc.value.code == "invalid_display_filter"


@pytest.mark.parametrize(("edges", "keep"), [
    ((None, 12), [True, False, False]),
    ((60, None), [False, False, True]),
    ((15, 55), [False, True, False]),
])
def test_single_and_double_edge_filter_spectrum(tmp_path: Path, edges, keep) -> None:
    tones = (3, 30, 110)
    store = _register(tmp_path, [_tone_signal("LA1", 400, tones)])
    lff, hff = edges
    params = dict(start_seconds=2, duration_seconds=2)
    unfiltered = compute_multichannel_batch(store, "tones", "tones", **params)
    data = compute_multichannel_batch(store, "tones", "tones", display_lff=lff, display_hff=hff,
                                      display_filter="fast_ripple_250_450", **params)
    assert data["requested_display_filter"] == data["effective_display_filter"] == "custom"
    assert data["display_filter_band_hz"] == data["traces"][0]["display_filter_band_hz"] == list(edges)
    assert data["recording_fingerprint"] == unfiltered["recording_fingerprint"]
    raw = np.asarray(unfiltered["traces"][0]["values_uv"])
    filtered = np.asarray(data["traces"][0]["values_uv"])
    freq_bins = np.array(tones) * 2  # two-second window
    ratios = np.abs(np.fft.rfft(filtered)[freq_bins] / np.fft.rfft(raw)[freq_bins])
    assert np.all(ratios[keep] > 0.9)
    assert np.all(ratios[np.logical_not(keep)] < 0.05)


def test_nyquist_clamp_and_partial_broadband(tmp_path: Path) -> None:
    store = _register(tmp_path, [_tone_signal("LA1", 400, (3, 110)), _tone_signal("LB1", 128, (3, 30))])
    data = compute_multichannel_batch(store, "tones", "tones", display_lff=60, display_hff=500, display_notch=True)
    first, second = data["traces"]
    assert data["effective_display_filter"] == "partial"
    assert data["display_filter_available"] is False
    assert first["display_filter_band_hz"] == [60, 184]  # 92% of 200 Hz
    assert first["effective_notch_hz"] == [60, 120]
    assert second["effective_display_filter"] == "broadband"
    assert second["display_filter_band_hz"] is None
    assert second["effective_notch_hz"] == []
    notice = data["montage_notice"]
    assert "LB1" in notice and "broadband" in notice
    assert "HFF clamped to 184 Hz on 1/2" in notice
    assert "Notch 60 Hz active on 1/2" in notice
    data = compute_multichannel_batch(store, "tones", "tones", display_lff=300)
    assert data["effective_display_filter"] == "broadband"
    assert "showing broadband" in data["montage_notice"]


def test_acquisition_lowpass_and_preset_compatibility(tmp_path: Path) -> None:
    store = _register(tmp_path, [_tone_signal("LA1", 400, (3, 30), "LP:70Hz")])
    custom = compute_multichannel_batch(store, "tones", "tones", display_lff="1", display_hff="999")
    assert custom["display_filter_band_hz"] == [1, 70]
    assert "HFF clamped to 70 Hz" in custom["montage_notice"]
    preset = compute_multichannel_batch(store, "tones", "tones", display_filter="eeg_1_70")
    assert preset["effective_display_filter"] == "eeg_1_70"
    assert preset["traces"][0]["values_uv"] == custom["traces"][0]["values_uv"]


def test_unresolvable_cutoff_reports_client_error(tmp_path: Path) -> None:
    store = _register(tmp_path, [_tone_signal("LA1", 400, (30,))])
    with pytest.raises(ToolkitError) as exc:
        compute_multichannel_batch(store, "tones", "tones", display_hff=1e-300)
    assert exc.value.code == "invalid_display_filter"


def test_short_window_reports_broadband(tmp_path: Path) -> None:
    signal = edfio.EdfSignal(np.arange(32, dtype=float), 256, label="LA1", physical_dimension="uV")
    store = _register(tmp_path, [signal], record_seconds=0.125)
    raw = compute_multichannel_batch(store, "tones", "tones")
    for params in ({"display_filter": "eeg_1_70"}, {"display_lff": 1, "display_hff": 70}):
        data = compute_multichannel_batch(store, "tones", "tones", display_notch=True, **params)
        trace = data["traces"][0]
        assert trace["values_uv"] == raw["traces"][0]["values_uv"]
        assert trace["effective_display_filter"] == data["effective_display_filter"] == "broadband"
        assert trace["display_filter_available"] is False
        assert trace["display_filter_band_hz"] is None
        assert trace["effective_notch_hz"] == []
        assert "short window with 32 samples" in data["montage_notice"]


def test_multichannel_http_cutoffs(tmp_path: Path) -> None:
    store = _register(tmp_path, [_tone_signal("LA1", 400, (3, 30))])
    server = WorkbenchHTTPServer(workspace_dir=store.root)
    # Capture the production request handler at the point the listener would bind.
    with patch.object(ThreadingHTTPServer, "__init__", side_effect=RuntimeError("capture handler")) as listener:
        with pytest.raises(RuntimeError, match="capture handler"):
            server.start()
    handler = listener.call_args.args[-1]

    def get(cutoffs: str) -> tuple[int, dict]:
        path = f"/api/multichannel?case=tones&recording=tones&{cutoffs}"
        request = Mock()
        request.makefile.return_value = io.BytesIO(f"GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n".encode())
        response = io.BytesIO()
        request.sendall.side_effect = response.write
        handler(request, ("127.0.0.1", 1), None)
        headers, body = response.getvalue().split(b"\r\n\r\n", 1)
        return int(headers.split()[1]), json.loads(body)

    status, data = get("display_lff=1&display_hff=70")
    assert status == 200 and data["display_filter_band_hz"] == [1, 70]
    for query in ("display_hff=", "display_lff=oops", "display_lff=70&display_hff=10"):
        status, data = get(query)
        assert status == 400 and data["error"]["code"] == "invalid_display_filter"
