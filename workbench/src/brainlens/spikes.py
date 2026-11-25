"""IED candidates — require visual verification.

For each contiguous valid voltage segment, apply a zero-phase third-order
Butterworth band-pass (10–60 Hz by default) and take the Hilbert envelope.
Estimate the median and IQR in five-second windows with 50% overlap. Linearly
interpolate median + 5 IQR between window centres. Mark the envelope peak in
each 8–200 ms excursion above threshold, keeping the first peak in a 100 ms
refractory interval. Reflected padding is confined to the segment. Both
polarities are detected. Rates use the combined duration of assessed segments.
These are transparent screening candidates and require visual verification.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from .models import ToolkitError, true_runs_mask

IED_VERSION = "1"
IED_LABEL = "IED candidates — require visual verification"
IED_DEFAULTS = {
    "low_hz": 10.0, "high_hz": 60.0, "window_seconds": 5.0, "threshold_iqr": 5.0,
    "min_duration_seconds": 0.008, "max_duration_seconds": 0.2, "refractory_seconds": 0.1,
}


def _signal_inputs(
    data: np.ndarray, valid: np.ndarray, channels: Sequence[str], sfreq: float,
    start_seconds: float, lowpass_hz: float | Sequence[float | None] | None,
) -> list[float]:
    """Check the channel/time geometry shared by the two on-demand signal analyses."""
    if data.ndim != 2 or data.shape != valid.shape or len(channels) != data.shape[0]:
        raise ToolkitError("invalid_signal_shape", "Expected matching channel × sample data and validity arrays.")
    if not np.isfinite(sfreq) or sfreq <= 0 or not np.isfinite(start_seconds) or start_seconds < 0:
        raise ToolkitError("invalid_timebase", "Sampling frequency must be positive and recording start finite and nonnegative.")
    if lowpass_hz is None or np.isscalar(lowpass_hz):
        limits = [lowpass_hz] * len(channels)
    else:
        limits = list(lowpass_hz)
    if len(limits) != len(channels) or any(x is not None and (not np.isfinite(x) or x <= 0) for x in limits):
        raise ToolkitError("invalid_lowpass", "Supply a positive low-pass ceiling or None for each channel.")
    return [min(sfreq / 2, x) if x is not None else sfreq / 2 for x in limits]


def _threshold(envelope: np.ndarray, width: int, iqr_factor: float) -> np.ndarray:
    width = min(width, len(envelope))
    last = len(envelope) - width
    starts = np.arange(0, last + 1, max(1, width // 2))
    if starts[-1] != last:
        starts = np.append(starts, last)
    thresholds = []
    for start in starts:
        q1, median, q3 = np.percentile(envelope[start:start + width], [25, 50, 75])
        thresholds.append(median + iqr_factor * (q3 - q1))
    return np.interp(np.arange(len(envelope)), starts + (width - 1) / 2, thresholds)


def detect_ieds(
    data: np.ndarray, valid: np.ndarray, sfreq: float, channels: Sequence[str], *,
    start_seconds: float = 0.0, lowpass_hz: float | Sequence[float | None] | None = None,
    low_hz: float = IED_DEFAULTS["low_hz"], high_hz: float = IED_DEFAULTS["high_hz"],
    window_seconds: float = IED_DEFAULTS["window_seconds"], threshold_iqr: float = IED_DEFAULTS["threshold_iqr"],
    min_duration_seconds: float = IED_DEFAULTS["min_duration_seconds"],
    max_duration_seconds: float = IED_DEFAULTS["max_duration_seconds"],
    refractory_seconds: float = IED_DEFAULTS["refractory_seconds"],
) -> dict[str, Any]:
    """Detect events independently per channel, reporting recording-relative seconds."""
    ceilings = _signal_inputs(data, valid, channels, sfreq, start_seconds, lowpass_hz)
    params = dict(low_hz=low_hz, high_hz=high_hz, window_seconds=window_seconds, threshold_iqr=threshold_iqr,
                  min_duration_seconds=min_duration_seconds, max_duration_seconds=max_duration_seconds,
                  refractory_seconds=refractory_seconds)
    if (not all(np.isfinite(x) and x > 0 for x in params.values())
            or high_hz <= low_hz or min_duration_seconds > max_duration_seconds):
        raise ToolkitError("invalid_ied_parameters", "Check the IED passband, threshold and duration parameters.")
    width = max(2, round(window_seconds * sfreq))
    min_samples = max(1, int(np.ceil(min_duration_seconds * sfreq)))
    max_samples = int(np.floor(max_duration_seconds * sfreq))
    rows = []
    for idx, name in enumerate(channels):
        high = min(high_hz, ceilings[idx], 0.99 * sfreq / 2)
        times: list[float] = []
        assessed_samples = 0
        if high > low_hz:
            sos = butter(3, (low_hz, high), btype="bandpass", fs=sfreq, output="sos")
            # Use at least five low-band cycles for filtering and envelope statistics.
            min_segment = max(24, int(np.ceil(5 * sfreq / low_hz)))
            for a, b in true_runs_mask(valid[idx] & np.isfinite(data[idx])):
                if b - a < min_segment:
                    continue
                seg = np.asarray(data[idx, a:b], dtype=float)
                pad = min(len(seg) - 1, min_segment)
                padded = np.pad(seg - np.median(seg), pad, mode="reflect")
                envelope = np.abs(hilbert(sosfiltfilt(sos, padded)))[pad:-pad]
                assessed_samples += len(seg)
                above = envelope > _threshold(envelope, width, threshold_iqr)
                for left, right in true_runs_mask(above):
                    if not min_samples <= right - left <= max_samples:
                        continue
                    peak = a + left + int(np.argmax(envelope[left:right]))
                    time = start_seconds + peak / sfreq
                    if not times or time - times[-1] >= refractory_seconds:
                        times.append(time)
        seconds = assessed_samples / sfreq
        state = "measured" if times else "quiet_measured"
        reason = None
        if not seconds:
            state = "unassessed"
            reason = "insufficient_bandwidth" if high <= low_hz else "insufficient_valid_data"
        rows.append({
            "channel": name, "state": state, "reason": reason, "event_times": times,
            "valid_seconds": seconds, "count": len(times) if seconds else None,
            "rate_per_min": len(times) * 60 / seconds if seconds else None,
            "band_hz": [low_hz, high] if high > low_hz else None,
        })
    return {"label": IED_LABEL, "version": IED_VERSION, "parameters": params, "channels": rows}
