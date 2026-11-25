"""Windowed Epileptogenicity Index (EI), following Bartolomei et al., Brain 2008.

Demeaned two-second Hann spectra every 0.25 s give ER = (beta + gamma) /
(theta + alpha) at each window centre. Bands in Hz are theta [3.5, 7.4),
alpha [7.4, 12.4), beta [12.4, 24) and gamma [24, 97] limited to the
individual channel's Nyquist/acquisition low-pass. Only whole valid windows
enter the calculation.

The upward Page–Hinkley statistic accumulates (ER - running mean) in units
of the pre-onset ER standard deviation, minus a bias (2 by default) per
step. Starting at onset, a rise of 30 units above the cumulative minimum
declares recruitment at the last minimum (first window of the rise).
Missing windows reset the sum. The running mean starts with the pre-onset
samples. EI is the sum of ER in the next five seconds divided by recruitment
delay relative to the earliest detection plus one second. Normalization uses
the largest raw score in this recording. This parameterized implementation
is for evidence review. A zero means no detected change with adequate
coverage. Missing baseline, coverage or follow-up produces an unassessed EI.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from .models import MarkerSpec, ToolkitError, true_runs_mask
from .spikes import _signal_inputs

EI_VERSION = "1"
EI_DEFAULTS = {
    "window_seconds": 2.0, "step_seconds": 0.25, "bias": 2.0,
    "threshold": 30.0, "tau_seconds": 1.0, "horizon_seconds": 5.0,
}


def resolve_onset(onset_seconds: float | None, markers: Sequence[MarkerSpec]) -> float:
    """Use explicit seconds or the earliest global seizure/start marker."""
    if onset_seconds is not None:
        if not np.isfinite(onset_seconds) or onset_seconds < 0:
            raise ToolkitError("invalid_seizure_onset", "Seizure onset must be finite and nonnegative.")
        return float(onset_seconds)
    starts = []
    for marker in markers:
        if marker.channels:
            continue
        # Recognize labels such as "SZ1", "Seizure onset", "ictal_onset" and
        # "Onset". End markers, artifact markers and seizure-free annotations
        # must never establish an onset.
        label = re.sub(r"[_\W]+", " ", marker.label.casefold()).strip()
        if re.search(r"\b(end\w*|stop\w*|off\w*|terminat\w*|free|no|non|bad|pre|post)\b", label):
            continue
        if re.search(r"\b(seizure|sz)\d*\b", label) or re.fullmatch(r"(?:(?:ictal|clinical|electrographic) )?onset(?: \d+)?", label):
            starts.append(marker.start)
    if not starts:
        raise ToolkitError("seizure_onset_unavailable", "Provide onset_seconds or a global seizure onset marker.")
    return float(min(starts))


def _er_trace(
    signal: np.ndarray, valid: np.ndarray, starts: np.ndarray, taper: np.ndarray,
    frequencies: np.ndarray, gamma_upper: float,
) -> np.ndarray:
    er = np.full(len(starts), np.nan)
    if gamma_upper <= 24:
        return er
    low_band = (frequencies >= 3.5) & (frequencies < 12.4)
    high_band = (frequencies >= 12.4) & (frequencies <= gamma_upper)
    if not low_band.any() or not high_band.any():
        return er
    for i, start in enumerate(starts):
        stop = start + len(taper)
        x = np.asarray(signal[start:stop], dtype=float)
        if not valid[start:stop].all() or not np.isfinite(x).all():
            continue
        power = np.abs(np.fft.rfft((x - x.mean()) * taper)) ** 2
        low_energy = power[low_band].sum()
        if low_energy > np.finfo(float).tiny:
            er[i] = power[high_band].sum() / low_energy
    return er


def _recruitment(
    er: np.ndarray, baseline: np.ndarray, post: np.ndarray, bias: float, threshold: float,
) -> int | None:
    initial = er[baseline]
    mean = float(initial.mean())
    # A floor keeps an almost constant baseline numerically well-defined.
    scale = max(float(initial.std()), abs(mean) * 0.05, np.finfo(float).eps)
    seen = len(initial)
    score = 0.0
    first = None
    for i in np.flatnonzero(post):
        if not np.isfinite(er[i]):
            score, first = 0.0, None
            continue
        seen += 1
        mean += (er[i] - mean) / seen
        score = max(0.0, score + (er[i] - mean) / scale - bias)
        if score == 0.0:
            first = None
        elif first is None:
            first = int(i)
        if score >= threshold:
            return first
    return None


def compute_ei(
    data: np.ndarray, valid: np.ndarray, sfreq: float, channels: Sequence[str], *,
    onset_seconds: float | None, start_seconds: float = 0.0,
    lowpass_hz: float | Sequence[float | None] | None = None,
    window_seconds: float = EI_DEFAULTS["window_seconds"], step_seconds: float = EI_DEFAULTS["step_seconds"],
    bias: float = EI_DEFAULTS["bias"], threshold: float = EI_DEFAULTS["threshold"],
    tau_seconds: float = EI_DEFAULTS["tau_seconds"], horizon_seconds: float = EI_DEFAULTS["horizon_seconds"],
) -> dict[str, Any]:
    """Calculate EI within the supplied recording slice using absolute recording times."""
    onset = resolve_onset(onset_seconds, [])
    ceilings = _signal_inputs(data, valid, channels, sfreq, start_seconds, lowpass_hz)
    if not start_seconds <= onset < start_seconds + data.shape[1] / sfreq:
        raise ToolkitError("invalid_seizure_onset", "Seizure onset lies outside the supplied recording window.")
    params = dict(window_seconds=window_seconds, step_seconds=step_seconds, bias=bias, threshold=threshold,
                  tau_seconds=tau_seconds, horizon_seconds=horizon_seconds)
    if (not all(np.isfinite(x) for x in params.values()) or bias < 0 or threshold <= 0 or tau_seconds <= 0
            or horizon_seconds <= 0 or not 0 < step_seconds <= window_seconds or window_seconds * sfreq < 4):
        raise ToolkitError("invalid_ei_parameters", "Check EI spectral windows, Page–Hinkley and integration parameters.")
    width = round(window_seconds * sfreq)
    hop = max(1, round(step_seconds * sfreq))
    starts = np.arange(0, data.shape[1] - width + 1, hop)
    times = start_seconds + (starts + width / 2) / sfreq
    baseline = start_seconds + (starts + width) / sfreq <= onset
    post = times >= onset
    taper = np.hanning(width)
    frequencies = np.fft.rfftfreq(width, d=1 / sfreq)
    rows = []
    for idx, name in enumerate(channels):
        gamma_upper = min(97.0, ceilings[idx])
        er = _er_trace(data[idx], valid[idx], starts, taper, frequencies, gamma_upper)
        finite = np.isfinite(er)
        pre_valid = baseline & finite
        post_valid = post & finite
        row = {
            "channel": name, "gamma_upper_hz": gamma_upper, "ei": None,
            "detection_seconds": None, "er_sum": None,
            "er": [float(x) if np.isfinite(x) else None for x in er],
            "state": "unassessed", "reason": None,
        }
        rows.append(row)
        # Five seconds of baseline support and >=80% coverage on each side.
        # There must also be a contiguous post-onset span covering the horizon.
        if gamma_upper <= 24:
            row["reason"] = "insufficient_gamma_bandwidth"
        elif pre_valid.sum() < max(2, int(np.ceil(5 * sfreq / hop)), 0.8 * baseline.sum()):
            row["reason"] = "insufficient_valid_baseline"
        elif (post_valid.sum() < 0.8 * post.sum()
              or not any((b - a - 1) * hop / sfreq >= horizon_seconds for a, b in true_runs_mask(post_valid))):
            row["reason"] = "insufficient_valid_ictal_data"
        else:
            detection = _recruitment(er, pre_valid, post, bias, threshold)
            if detection is None:
                row.update(state="quiet_measured", ei=0.0)
                continue
            nd = float(times[detection])
            row["detection_seconds"] = nd
            end = nd + horizon_seconds
            interval = (times >= nd) & (times <= end + 1e-9)
            if times[-1] + 1e-9 < end or not finite[interval].all():
                row["reason"] = "insufficient_valid_integration_window"
            else:
                row.update(state="measured", er_sum=float(er[interval].sum()))

    detections = [row["detection_seconds"] for row in rows if row["detection_seconds"] is not None]
    n0 = min(detections, default=None)
    if n0 is not None:
        for row in rows:
            if row["er_sum"] is not None:
                row["ei"] = row["er_sum"] / (row["detection_seconds"] - n0 + tau_seconds)
        maximum = max((row["ei"] for row in rows if row["ei"] is not None), default=0.0)
        if maximum > 0:
            for row in rows:
                if row["ei"] is not None:
                    row["ei"] /= maximum
    return {
        "version": EI_VERSION, "parameters": params, "onset_seconds": onset,
        "reference_detection_seconds": n0, "times_seconds": times.tolist(), "channels": rows,
        "bands_hz": {"theta": [3.5, 7.4], "alpha": [7.4, 12.4], "beta": [12.4, 24.0], "gamma": [24.0, 97.0]},
    }
