"""Streaming quantitative trends for navigating a single recording."""

from __future__ import annotations

import math
import os
from bisect import bisect_right
from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from filelock import FileLock
from scipy.fft import next_fast_len, rfft
from scipy.signal.windows import hann

from .channel_order import ChannelDisplayOrder
from .io_integrity import ProgressCallback, verified_window_source, verify_raw_filenames, verify_source_recording_integrity
from .io_signals import open_mne_raw, read_scaled_volts
from .models import RecordingSpec, ToolkitError, atomic_write_json, canonical_digest, read_json
from .storage import WorkbenchStore

_BANDS = {
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30),
    "gamma": (30, 80), "ripple": (80, 250), "fast_ripple": (250, 500),
}
_SIGNAL_MEASURES = (*_BANDS, "line_length", "rms")
_UNITS = {**dict.fromkeys(_BANDS, "dB re 1 µV²"), "line_length": "µV", "rms": "µV", "hfo_rate": "detections/min"}
_VERSION = 1
_MAX_COLUMNS = 120
_MAX_ROWS = 32
_MAX_GAPS = 120


@lru_cache(maxsize=16)
def _taper_and_bins(n: int, sfreq: float):
    nfft = next_fast_len(n, real=True)
    taper = hann(n, sym=False).astype(np.float32)
    # Convert squared one-sided FFT coefficients to integrated power. DC and
    # the Nyquist bin are outside the half-open bands.
    scale = 2 / (nfft * np.sum(taper * taper, dtype=np.float64))
    freqs = np.fft.rfftfreq(nfft, 1 / sfreq)
    bins = [tuple(freqs.searchsorted(edges)) for edges in _BANDS.values()]
    return nfft, taper, bins, scale


def _band_powers(samples: np.ndarray, sfreq: float) -> Iterator[tuple[int, np.ndarray]]:
    """Yield the integrated periodogram (µV²) of each band with sampled bins."""
    n = samples.shape[1]
    if n < 2:
        return
    nfft, taper, bins, scale = _taper_and_bins(n, sfreq)
    samples -= samples.mean(axis=1, keepdims=True)
    samples *= taper
    spectrum = rfft(samples, n=nfft)
    magnitude = np.abs(spectrum)
    np.square(magnitude, out=magnitude)
    for k, (left, right) in enumerate(bins):
        if left < right:
            yield k, magnitude[:, left:right].sum(axis=1, dtype=np.float64) * scale


def _epoch_count(rec: RecordingSpec, epoch: float) -> int:
    return max(1, math.ceil(rec.duration / epoch))


def _signal_trends(rec: RecordingSpec, epoch: float, progress: ProgressCallback | None) -> np.ndarray:
    n_ch, n_epochs = len(rec.channels), _epoch_count(rec, epoch)
    totals = np.zeros((len(_SIGNAL_MEASURES), n_ch, n_epochs), dtype=np.float64)
    # Count samples only in windows with frequency bins for that band. A piece
    # ending within an epoch never creates a spectral or line-length transition.
    power_weights = np.zeros((len(_BANDS), n_epochs), dtype=np.int64)
    acquired = np.zeros(n_epochs, dtype=np.int64)
    max_block = max(2, min(int(rec.sfreq * 10), 4_194_304 // max(1, n_ch)))
    with verified_window_source(rec) as (path, directory, _, sig):
        with open_mne_raw(path, preload=False) as raw:
            verify_raw_filenames(rec, raw.filenames, directory, sig, opened_stat=getattr(raw, "_brainlens_fd_stat", None))
            for piece, first, stop in rec.time_slices(0, rec.duration):
                cursor = first
                col = math.floor(piece.time_at(first, rec.sfreq) / epoch)
                while cursor < stop and col < n_epochs:
                    right = min(stop, piece.start_sample + math.ceil(piece.sample_offset((col + 1) * epoch, rec.sfreq)))
                    previous = None
                    while cursor < right:
                        end = min(right, cursor + max_block)
                        n = end - cursor
                        block = read_scaled_volts(raw, rec, cursor, end)
                        block *= 1e6
                        totals[-1, :, col] += np.einsum("ij,ij->i", block, block)
                        totals[-2, :, col] += np.abs(np.diff(block, axis=1)).sum(axis=1)
                        if previous is not None:
                            totals[-2, :, col] += np.abs(block[:, 0] - previous)
                        previous = block[:, -1].copy()
                        for k, power in _band_powers(block.astype(np.float32), rec.sfreq):
                            totals[k, :, col] += power * n
                            power_weights[k, col] += n
                        acquired[col] += n
                        cursor = end
                    if progress:
                        progress("trends", cursor / rec.n_samples, f"Computed trends through {piece.time_at(cursor, rec.sfreq):.1f} s")
                    col += 1
    totals[:, :, acquired == 0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        totals[-1] = np.sqrt(totals[-1] / acquired)
        totals[:len(_BANDS)] /= power_weights[:, None, :]
        totals[:len(_BANDS)] = 10 * np.log10(np.maximum(totals[:len(_BANDS)], 1e-30))
    edf = Path(rec.path).suffix.lower() in {".edf", ".bdf"}
    for i, channel in enumerate(rec.channels):
        if channel.unit != "V":
            totals[:, i] = np.nan
            continue
        nyquist = min(channel.source_sfreq or rec.sfreq, rec.sfreq) / 2
        # For EDF/BDF, MNE stores the minimum of dissimilar per-signal
        # prefilters in info.lowpass. Prefer the channel's own header.
        lowpasses = (channel.source_lowpass or rec.lowpass,) if edf else (rec.lowpass, channel.source_lowpass)
        limit = min([nyquist] + [lp for lp in lowpasses if lp is not None])
        for k, (_, upper) in enumerate(_BANDS.values()):
            if upper >= nyquist or upper > limit:
                totals[k, i] = np.nan
    return totals


def _hfo_rates(store: WorkbenchStore, bundle: dict, rec: RecordingSpec, channels: list[dict], epoch: float):
    n = _epoch_count(rec, epoch)
    counts = np.zeros((len(channels), n), dtype=np.float64)
    duration = np.zeros_like(counts)
    rows = {ch["id"]: i for i, ch in enumerate(channels)}
    path = store.resolve_workspace_artifact_path(bundle["meta"].get("detection_json_path"))
    recording_dir = store.recording_dir(bundle["meta"]["case"], rec.slug)
    if path is None or not path.is_file() or not path.resolve().is_relative_to(recording_dir.resolve()):
        raise ToolkitError("detection_unavailable", "Detector coverage is unavailable for this saved run.")
    det = read_json(path)
    pieces = [p for p, _, _ in rec.time_slices(0, rec.duration)]
    stops = [p.stop_sample for p in pieces]
    for cid, intervals in det.get("valid_intervals", {}).items():
        if cid not in rows:
            continue
        covered = duration[rows[cid]]
        for left, right in intervals:
            i = bisect_right(stops, left)
            while i < len(pieces) and pieces[i].start_sample < right:
                p = pieces[i]
                start = p.time_at(max(left, p.start_sample), rec.sfreq)
                end = p.time_at(min(right, p.stop_sample), rec.sfreq)
                first, last = int(start // epoch), min(n, math.ceil(end / epoch))
                edges = np.arange(first, last) * epoch
                covered[first:last] += np.minimum(edges + epoch, end) - np.maximum(edges, start)
                i += 1
    for event in bundle["events"]:
        col = int(event["start_seconds"] // epoch)
        if event["channel_id"] in rows and 0 <= col < n:
            counts[rows[event["channel_id"]], col] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        return (60 * counts / duration)[None, :, :]


def _cached_matrix(directory: Path, manifest: dict, compute: Callable[[], np.ndarray], index: int,
                   verify: Callable[[], Any]) -> np.ndarray:
    directory.mkdir(parents=True, exist_ok=True)
    meta, data = directory / "manifest.json", directory / "values.npy"
    with FileLock(str(directory / ".lock")):
        if meta.is_file() and data.is_file():
            try:
                if read_json(meta) == manifest:
                    result = np.load(data, mmap_mode="r", allow_pickle=False)[index]
                    verify()
                    return result
            except (ValueError, OSError, IndexError):
                pass
        values = compute()
        tmp = NamedTemporaryFile(dir=directory, delete=False)
        tmp_path = Path(tmp.name)
        try:
            with tmp:
                np.save(tmp, values, allow_pickle=False)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, data)
        finally:
            tmp_path.unlink(missing_ok=True)
        atomic_write_json(meta, manifest)
        return values[index]


def _view(values, channels, epoch, compact, indices):
    peaks = []
    for i in indices:
        row = values[i]
        if np.isfinite(row).any():
            peak = int(np.argmax(np.where(np.isfinite(row), row, -np.inf)))
            peaks.append((float(row[peak]), peak, i))
    peaks.sort(key=lambda item: -item[0])
    top = [{**channels[i], "peak_seconds": col * epoch, "peak_value": value} for value, col, i in peaks[:8]]
    if compact and len(channels) > _MAX_ROWS:
        strongest = set(list(dict.fromkeys([i for _, _, i in peaks] + indices))[:_MAX_ROWS])
        indices = [i for i in indices if i in strongest]
    if indices != list(range(len(channels))):
        values, channels = values[indices], [channels[i] for i in indices]
    width = math.ceil(values.shape[1] / _MAX_COLUMNS) if compact else 1
    starts = np.arange(0, values.shape[1], width)
    if width > 1:
        values = np.fmax.reduceat(values, starts, axis=1)
    matrix = values.astype(object)
    matrix[~np.isfinite(values)] = None
    return channels, top, (starts * epoch).tolist(), matrix.tolist(), width


def recording_trends(
    store: WorkbenchStore, case: str, recording: str, *, measure: str = "ripple", epoch_seconds: float = 10,
    run: str | None = None, compact: bool = False, progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Return one navigation measure on the recording's true-time clock."""
    if measure not in _UNITS:
        raise ToolkitError("invalid_trend_measure", "Unknown trend measure.", measures=list(_UNITS))
    try:
        epoch = float(epoch_seconds)
    except (TypeError, ValueError) as exc:
        raise ToolkitError("invalid_trend_epoch", "epoch_seconds must be between 2 and 60.") from exc
    if not 2 <= epoch <= 60:
        raise ToolkitError("invalid_trend_epoch", "epoch_seconds must be between 2 and 60.")
    rec = store.load_recording(case, recording)
    order = ChannelDisplayOrder(rec)
    channels = [{"id": c.id, "name": c.name} for c in rec.channels]
    indices = sorted(range(len(channels)), key=lambda i: order.position(rec.channels[i].id, rec.channels[i].name))
    params = {"epoch_seconds": epoch, "measures": list(_SIGNAL_MEASURES)}
    manifest = {"recording_fingerprint": rec.fingerprint, "parameters": params, "code_version": _VERSION}
    if progress_cb:
        progress_cb("trends", 0., "Loading quantitative trends")
    if measure == "hfo_rate":
        if run is None:
            runs = sorted((m for m in store.list_runs(case, recording, reconcile=False)
                           if m.get("recording_fingerprint") == rec.fingerprint), key=lambda m: m.get("created_at") or "")
            run = next((m["run_name"] for m in runs if m["run_name"] == "baseline"), runs[-1]["run_name"] if runs else None)
        if run is None:
            raise ToolkitError("run_not_found", "Select a saved run for HFO detection rates.")
        bundle = store.load_run(case, recording, run, reconcile=False)
        meta = bundle["meta"]
        if meta.get("recording_fingerprint") != rec.fingerprint:
            raise ToolkitError("stale_run", "Selected run belongs to an earlier recording fingerprint.")
        params.update(measures=[measure], run=run, generation=meta.get("analysis_generation_id") or meta.get("generation_id"),
                      analysis_signature=meta.get("analysis_signature"))
        ordered = sorted(bundle["summary"]["channels"], key=order.row_position)
        channels = [{"id": ch["channel_id"], "name": ch["channel_name"]} for ch in ordered]
        # Older HFO-rate caches used summary rank as their matrix row order.
        params["channel_ids"] = [ch["id"] for ch in channels]
        indices = list(range(len(channels)))
        index = 0

        def compute():
            verify_source_recording_integrity(rec)
            return _hfo_rates(store, bundle, rec, channels, epoch)
    else:
        index = _SIGNAL_MEASURES.index(measure)

        def compute():
            return _signal_trends(rec, epoch, progress_cb)

    directory = store.recording_dir(case, recording) / "trends" / canonical_digest(manifest)
    values = _cached_matrix(directory, manifest, compute, index, lambda: verify_source_recording_integrity(rec))
    if store.load_recording(case, recording).fingerprint != rec.fingerprint:
        raise ToolkitError("source_recording_modified", "Recording registration changed while loading trends.")
    channel_count = len(channels)
    selected, top, times, matrix, width = _view(values, channels, epoch, compact, indices)
    gaps = rec.gaps()
    result = {
        "mode": "trends", "case": case, "recording": recording, "recording_id": rec.id,
        "measure": measure, "units": _UNITS[measure],
        "purpose": "Quantitative trend for navigation.",
        "epoch_seconds": epoch, "duration_seconds": rec.duration, "run": run if measure == "hfo_rate" else None,
        "channels": selected, "channel_count": channel_count, "top_channels": top,
        "times_seconds": times, "matrix": matrix, "epochs_per_column": width, "column_reduction": "max",
        "gaps": [{"start_seconds": a, "end_seconds": b} for a, b in (gaps[:_MAX_GAPS] if compact else gaps)],
        "gap_count": len(gaps),
    }
    if not compact:
        result.update(file_boundaries=[0, rec.duration], markers=[m.model_dump(mode="json") for m in rec.markers])
    if progress_cb:
        progress_cb("trends", 1., "Quantitative trends ready")
    return result
