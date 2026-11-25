"""Standalone windowed QC, Welch line-noise check, verified referencing, and SOS filtering."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch

from .analysis_windows import AnalysisWindow, selected_windows
from .io_integrity import verified_window_source, verify_raw_filenames, verify_source_recording_integrity
from .io_signals import open_mne_raw, read_scaled_volts
from .models import (
    AnalysisConfig,
    BipolarPair,
    ChannelSpec,
    RecordingSpec,
    ToolkitError,
    atomic_write_json,
    canonical_digest,
    complement_intervals,
    intersect_intervals,
    merge_intervals,
    parse_shaft_group,
    true_runs_mask,
)


# float64 signal plus the one-byte validity mask written for each selected sample.
PREPARED_DATA_BUDGET_BYTES = int(os.environ.get("BRAINLENS_PREPARED_DATA_BUDGET_BYTES", 4 * 1024**3))


def _check_prepared_size(n_samples: int, n_channels: int) -> None:
    size = n_samples * n_channels * (np.dtype(np.float64).itemsize + np.dtype(bool).itemsize)
    if size > PREPARED_DATA_BUDGET_BYTES:
        raise ToolkitError(
            "analysis_window_too_large",
            f"Preparing {n_samples} samples × {n_channels} channels needs {size:,} bytes "
            f"(budget {PREPARED_DATA_BUDGET_BYTES:,} bytes). Select a shorter window with "
            "overrides.start_seconds and overrides.end_seconds.",
            prepared_bytes=size, budget_bytes=PREPARED_DATA_BUDGET_BYTES,
        )


def check_analysis_window(recording: RecordingSpec, config: AnalysisConfig) -> None:
    windows = selected_windows(recording, config)
    channels, _ = _build_reference_plan(recording, config)
    _check_prepared_size(windows[-1].array_stop, len(channels))


def _sample_spans(recording: RecordingSpec, start: float, end: float) -> list[tuple[int, int]]:
    return [(a, b) for _, a, b in recording.time_slices(start, end)]


def _blocks(windows: list[AnalysisWindow], width: int) -> Iterator[tuple[AnalysisWindow, int, int]]:
    for window in windows:
        for a in range(window.start, window.stop, width):
            yield window, a, min(window.stop, a + width)


def resolve_channel(recording: RecordingSpec, token: str) -> ChannelSpec:
    for c in recording.channels:
        if token in {c.id, c.name, c.original_label} or c.contacts == [token]:
            return c
    low = token.lower()
    for c in recording.channels:
        if low in {c.id.lower(), c.name.lower(), c.original_label.lower()}:
            return c
    raise ToolkitError("unknown_channel", f"Channel {token!r} not found in recording {recording.slug}.")


def _bad_marker_spans(recording: RecordingSpec, ch: ChannelSpec) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in recording.markers:
        if not m.label.upper().startswith("BAD"):
            continue
        if not m.channels or any(resolve_channel(recording, tok).id == ch.id for tok in m.channels):
            a, b = max(0.0, m.start), min(recording.duration, m.end)
            if b > a:
                spans.extend(_sample_spans(recording, a, b))
    return spans


def _measure_line_noise(
    recording: RecordingSpec,
    valid_chunks_by_ch: dict[str, dict[int, list[np.ndarray]]],
    freqs_hz: list[float],
) -> dict[str, Any]:
    sfreq = recording.sfreq
    evaluated_freqs = [f for f in freqs_hz if f + 5.0 < sfreq / 2.0]
    per_ch: list[dict[str, Any]] = []
    for ch in recording.channels:
        if ch.unit != "V" or ch.is_auxiliary:
            continue
        groups = valid_chunks_by_ch.get(ch.id, {})
        blocks = [np.concatenate(chunks) for chunks in groups.values()]
        blocks = [block for block in blocks if len(block) >= 64]
        if not blocks:
            per_ch.append({"channel_id": ch.id, "channel_name": ch.name, "ratios": {str(int(f)): None for f in evaluated_freqs}})
            continue
        nperseg = min(max(len(b) for b in blocks), max(64, int(round(2.0 * sfreq))))
        psds = []
        for block in blocks:
            f_grid, spectrum = welch(block, fs=sfreq, nperseg=min(len(block), nperseg), nfft=nperseg)
            psds.append(spectrum)
        psd = psds[0] if len(psds) == 1 else np.average(psds, axis=0, weights=[len(b) for b in blocks])
        ratios: dict[str, float | None] = {}
        for f0 in evaluated_freqs:
            center_mask = (f_grid >= f0 - 1.0) & (f_grid <= f0 + 1.0)
            flank_mask = ((f_grid >= f0 - 5.0) & (f_grid <= f0 - 2.0)) | ((f_grid >= f0 + 2.0) & (f_grid <= f0 + 5.0))
            if not np.any(center_mask) or not np.any(flank_mask):
                ratios[str(int(f0))] = None
                continue
            flank_med = float(np.median(psd[flank_mask]))
            center_mean = float(np.mean(psd[center_mask]))
            ratios[str(int(f0))] = round(center_mean / flank_med, 3) if flank_med > 0 and math.isfinite(flank_med) else None
        per_ch.append({"channel_id": ch.id, "channel_name": ch.name, "ratios": ratios})
    return {"evaluated_frequencies_hz": evaluated_freqs, "channels": per_ch}


def compute_qc(recording: RecordingSpec, config: AnalysisConfig, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = selected_windows(recording, config)
    start, stop = windows[0].start, windows[-1].stop
    win_samples = max(1, int(round(config.qc.window_seconds * recording.sfreq)))

    invalid: dict[str, list[tuple[int, int]]] = {c.id: _bad_marker_spans(recording, c) for c in recording.channels}
    excl_reasons: dict[str, list[str]] = {c.id: [] for c in recording.channels}
    for ex in config.exclusions:
        if ex.scope == "acquired":
            sp = _sample_spans(recording, ex.start, ex.end)
            for tok in ex.channels:
                ch = resolve_channel(recording, tok)
                invalid[ch.id].extend(sp)
                excl_reasons[ch.id].append(ex.reason)

    amp_warn_channels: set[str] = set()
    flat_channels: set[str] = set()
    valid_samples_for_ln: dict[str, dict[int, list[np.ndarray]]] = {c.id: {} for c in recording.channels}
    max_ln_samples = int(round(60.0 * recording.sfreq))
    collected_ln: dict[str, int] = {c.id: 0 for c in recording.channels}

    with verified_window_source(recording) as (snap_main, snap_dir, _bp_snap, sig_before):
        raw = open_mne_raw(snap_main, preload=False)
        try:
            verify_raw_filenames(recording, raw.filenames, snap_dir, sig_before,
                                 opened_stat=getattr(raw, "_brainlens_fd_stat", None))
            for window, a, b in _blocks(windows, win_samples):
                block = read_scaled_volts(raw, recording, a, b)
                for idx, ch in enumerate(recording.channels):
                    sig = block[idx]
                    finite = np.isfinite(sig)
                    good = sig[finite]
                    if not finite.all():
                        invalid[ch.id].extend((x + a, y + a) for x, y in true_runs_mask(~finite))
                    ptp = float(np.ptp(good)) if good.size else None
                    is_flat = good.size == sig.size and ptp is not None and ptp <= config.qc.flat_tolerance
                    if is_flat:
                        flat_channels.add(ch.name)
                        if config.qc.exclude_flat:
                            invalid[ch.id].append((a, b))
                    if ch.unit == "V" and good.size and float(np.max(np.abs(good))) > config.qc.amplitude_warning_volts:
                        amp_warn_channels.add(ch.name)
                    if config.qc.line_noise_enabled and good.size and collected_ln[ch.id] < max_ln_samples:
                        valid_samples_for_ln[ch.id].setdefault(window.piece.start_sample, []).append(good)
                        collected_ln[ch.id] += int(good.size)
        finally:
            raw.close()

    merged_invalid = {cid: [list(p) for p in merge_intervals(spans, start, stop)] for cid, spans in invalid.items()}
    if config.segments:
        selected = [(w.start, w.stop) for w in windows]
        merged_invalid = {cid: [list(p) for p in intersect_intervals(spans, selected)]
                          for cid, spans in merged_invalid.items()}
    covered = windows[-1].array_stop
    per_channel: list[dict[str, Any]] = []
    for ch in recording.channels:
        inv_samps = sum(b - a for a, b in merged_invalid[ch.id])
        val_samps = covered - inv_samps
        per_channel.append({
            "channel_id": ch.id,
            "channel_name": ch.name,
            "covered_samples": covered,
            "invalid_samples": inv_samps,
            "valid_samples": val_samps,
            "valid_seconds": round(val_samps / recording.sfreq, 4),
            "exclusion_reasons": excl_reasons[ch.id],
        })

    ln_summary = (
        _measure_line_noise(recording, valid_samples_for_ln, config.qc.line_noise_frequencies_hz)
        if config.qc.line_noise_enabled
        else {"evaluated_frequencies_hz": [], "channels": []}
    )
    qc_doc = {
        "status": "complete",
        "recording_fingerprint": recording.fingerprint,
        "coverage": {
            "start_sample": start,
            "end_sample": stop,
            "sfreq": recording.sfreq,
            "channels": [c.id for c in recording.channels],
        },
        "invalid_intervals": merged_invalid,
        "per_channel": per_channel,
        "amplitude_warning_channels": sorted(amp_warn_channels),
        "flat_window_channels": sorted(flat_channels),
        "line_noise": ln_summary,
        "warnings": list(recording.warnings),
    }
    if recording.pieces:
        intervals = ([(s.start_seconds, s.end_seconds) for s in config.segments] if config.segments
                     else [(config.start_seconds, config.end_seconds or recording.duration)])
        qc_doc["coverage"]["gaps_seconds"] = [
            [max(a, left), min(b, right)] for left, right in intervals
            for a, b in recording.gaps() if a < right and b > left
        ]
    if config.segments:
        qc_doc["coverage"]["segments"] = [s.model_dump() for s in config.segments]
    atomic_write_json(output_dir / "qc.json", qc_doc)
    return qc_doc


def _auto_bipolar_pairs(recording: RecordingSpec, *, bp_snap_path: Path | None = None) -> list[BipolarPair]:
    # 1. Check published bipolar pairs file if present (e.g. ZurichHFO) — prefer verified snapshot bytes!
    if bp_snap_path is not None and bp_snap_path.is_file():
        doc = json.loads(bp_snap_path.read_bytes().decode("utf-8"))
        raw_pairs = doc.get("pairs") if isinstance(doc, dict) else doc
        if isinstance(raw_pairs, list) and raw_pairs:
            out: list[BipolarPair] = []
            for item in raw_pairs:
                if isinstance(item, dict) and "anode" in item and "cathode" in item:
                    out.append(BipolarPair(anode=item["anode"], cathode=item["cathode"], name=item.get("name")))
            if out:
                return out
    elif recording.published_bipolar_pairs_path:
        bp_path = Path(recording.published_bipolar_pairs_path)
        if not bp_path.is_file():
            raise ToolkitError(
                "source_recording_missing",
                f"Registered published bipolar pairs sidecar is missing on disk: {bp_path.name}",
                recording=recording.slug,
            )
        verify_source_recording_integrity(recording)
        raw_bytes = bp_path.read_bytes()
        comp_meta = (recording.companion_files or {}).get(str(bp_path.resolve()))
        exp_sha = comp_meta.get("sha256") if isinstance(comp_meta, dict) else None
        if exp_sha and hashlib.sha256(raw_bytes).hexdigest() != exp_sha:
            raise ToolkitError(
                "source_recording_modified",
                f"Published bipolar pairs sidecar {bp_path.name!r} content (SHA-256) changed on disk since registration; "
                "refusing to derive altered montage under an earlier recording fingerprint.",
                recording=recording.slug,
            )
        doc = json.loads(raw_bytes.decode("utf-8"))
        raw_pairs = doc.get("pairs") if isinstance(doc, dict) else doc
        if isinstance(raw_pairs, list) and raw_pairs:
            out = []
            for item in raw_pairs:
                if isinstance(item, dict) and "anode" in item and "cathode" in item:
                    out.append(BipolarPair(anode=item["anode"], cathode=item["cathode"], name=item.get("name")))
            if out:
                return out

    # 2. Otherwise build adjacent pairs along each depth shaft (LA1-LA2, LB1-LB2, ...)
    by_shaft: dict[str, list[ChannelSpec]] = {}
    for ch in recording.channels:
        if not ch.is_auxiliary and ch.unit == "V" and ch.shaft not in {"AUX", "OTHER"} and ch.contact_index < 9999:
            by_shaft.setdefault(ch.shaft, []).append(ch)
    pairs: list[BipolarPair] = []
    for sh in sorted(by_shaft):
        ordered = sorted(by_shaft[sh], key=lambda c: c.contact_index)
        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]
            pairs.append(BipolarPair(anode=a.id, cathode=b.id, name=f"{a.name}-{b.name}"))
    if not pairs:
        raise ToolkitError("bipolar_pairs_missing", "Could not infer shaft-adjacent bipolar pairs; provide explicit pairs.")
    return pairs


def _build_reference_plan(
    recording: RecordingSpec,
    config: AnalysisConfig,
    *,
    bp_snap_path: Path | None = None,
) -> tuple[list[ChannelSpec], dict[str, list[tuple[ChannelSpec, float]]]]:
    selected = (
        [resolve_channel(recording, tok) for tok in config.channels]
        if config.channels
        else [c for c in recording.channels if c.kind in {"seeg", "ecog", "eeg"}]
    )
    if not selected:
        raise ToolkitError("empty_selection", "No EEG/SEEG/iEEG channels selected for analysis.")

    mode = config.reference.mode
    if mode == "original":
        return selected, {c.id: [(c, 1.0)] for c in selected}

    if recording.reference not in {"common", "average"}:
        raise ToolkitError(
            "reference_incompatible",
            f"Pipeline re-referencing ('{mode}') requires documented 'common' or 'average' input reference; "
            f"this recording's acquired reference is '{recording.reference}'.",
        )

    defs: list[tuple[str, ChannelSpec, list[tuple[ChannelSpec, float]]]] = []
    if mode == "bipolar":
        pairs = config.reference.pairs or _auto_bipolar_pairs(recording, bp_snap_path=bp_snap_path)
        for p in pairs:
            an = resolve_channel(recording, p.anode)
            ca = resolve_channel(recording, p.cathode)
            defs.append((p.name or f"{an.name}-{ca.name}", an, [(an, 1.0), (ca, -1.0)]))
    else:
        refs = (
            [resolve_channel(recording, tok) for tok in config.reference.average_channels]
            if config.reference.average_channels
            else [c for c in selected if not c.is_auxiliary and c.unit == "V"]
        )
        if len(refs) < 2:
            raise ToolkitError("invalid_average_ref", "Average reference requires at least 2 valid EEG channels.")
        for ch in selected:
            if ch.is_auxiliary:
                continue
            weights = {r.id: -1.0 / len(refs) for r in refs}
            weights[ch.id] = weights.get(ch.id, 0.0) + 1.0
            defs.append((ch.name, ch, [(resolve_channel(recording, cid), w) for cid, w in weights.items() if abs(w) > 1e-12]))

    outputs: list[ChannelSpec] = []
    deps: dict[str, list[tuple[ChannelSpec, float]]] = {}
    for name, primary, parents in defs:
        did = "drv_" + canonical_digest({"mode": mode, "parents": [(c.id, w) for c, w in parents]})[:16]
        if mode == "bipolar":
            contacts = list(dict.fromkeys(ct for c, _ in parents for ct in (c.contacts or [c.name])))
        else:
            contacts = list(primary.contacts or [primary.name])
        s_rates = [c.source_sfreq for c, _ in parents if c.source_sfreq is not None]
        s_lps = [c.source_lowpass for c, _ in parents if c.source_lowpass is not None]
        sgroup = parse_shaft_group(primary.name, primary.kind)
        out_ch = ChannelSpec(
            id=did,
            name=name,
            original_label=name,
            kind=primary.kind,
            unit=primary.unit,
            source_unit="V" if primary.unit == "V" else primary.source_unit,
            source_sfreq=min(s_rates) if s_rates else None,
            source_lowpass=min(s_lps) if s_lps else None,
            contacts=contacts,
            parents=[c.id for c, _ in parents],
            weights=[w for _, w in parents],
            shaft=sgroup["shaft"],
            contact_index=sgroup["contact_index"],
            is_auxiliary=False,
        )
        outputs.append(out_ch)
        deps[did] = parents
    return outputs, deps


def _build_sos_filters(filters_cfg, sfreq: float) -> tuple[np.ndarray | None, int, dict[str, Any]]:
    sos_parts = []
    freqs = [f for f in (filters_cfg.highpass, filters_cfg.lowpass, *filters_cfg.notch) if f is not None]
    if any(f >= sfreq / 2.0 for f in freqs):
        raise ToolkitError("filter_above_nyquist", f"Filter cutoff exceeds Nyquist ({sfreq / 2.0} Hz).")
    if filters_cfg.highpass and filters_cfg.lowpass:
        sos_parts.append(butter(4, [filters_cfg.highpass, filters_cfg.lowpass], btype="bandpass", fs=sfreq, output="sos"))
    elif filters_cfg.highpass:
        sos_parts.append(butter(4, filters_cfg.highpass, btype="highpass", fs=sfreq, output="sos"))
    elif filters_cfg.lowpass:
        sos_parts.append(butter(4, filters_cfg.lowpass, btype="lowpass", fs=sfreq, output="sos"))
    for nf in filters_cfg.notch:
        b, a = iirnotch(nf, Q=30.0, fs=sfreq)
        sos_parts.append(tf2sos(b, a))
    if not sos_parts:
        return None, 0, {"applied": False, **filters_cfg.model_dump()}
    sos = np.vstack(sos_parts)
    times = ([3.0 / filters_cfg.highpass] if filters_cfg.highpass else []) + (
        [3.0 / filters_cfg.lowpass] if filters_cfg.lowpass else []
    )
    times += [90.0 / nf for nf in filters_cfg.notch]
    edge = max(3 * (2 * len(sos) + 1), int(math.ceil(max(times) * sfreq)))
    return sos, edge, {"applied": True, "edge_samples_each_side": edge, **filters_cfg.model_dump()}


def prepare_signals(
    recording: RecordingSpec,
    config: AnalysisConfig,
    qc: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = selected_windows(recording, config)
    start, stop = windows[0].start, windows[-1].stop
    n_sel = windows[-1].array_stop

    with verified_window_source(recording) as (snap_main, snap_dir, bp_snap, sig_before):
        out_channels, deps = _build_reference_plan(recording, config, bp_snap_path=bp_snap)
        _check_prepared_size(n_sel, len(out_channels))

        # State selection (Lesson L5: requires explicit global time markers)
        state_spans = [(start, stop)]
        if config.state is not None:
            matches = [m for m in recording.markers if m.label == config.state and not m.channels and m.end > m.start]
            if not matches:
                raise ToolkitError(
                    "state_unavailable",
                    f"Requested state {config.state!r} has no explicit time markers in {recording.slug}; state is never inferred from filename.",
                )
            state_spans = merge_intervals(
                [span for m in matches for span in _sample_spans(recording, m.start, m.end)],
                start,
                stop,
            )

        parents_needed = {c.id: c for plist in deps.values() for c, _ in plist}
        parent_invalid: dict[str, list[tuple[int, int]]] = {}
        state_comp = complement_intervals(state_spans, start, stop)
        for pid in parents_needed:
            raw_inv = qc.get("invalid_intervals", {}).get(pid, [])
            parent_invalid[pid] = merge_intervals([*raw_inv, *state_comp], start, stop)

        derived_invalid: dict[str, list[tuple[int, int]]] = {c.id: [] for c in out_channels}
        for ex in config.exclusions:
            if ex.scope == "derived":
                sp = _sample_spans(recording, ex.start, ex.end)
                for tok in ex.channels:
                    matched = [c for c in out_channels if tok in {c.id, c.name}]
                    if not matched:
                        raise ToolkitError("unknown_derived_channel", f"Derived channel {tok!r} not found.")
                    derived_invalid[matched[0].id].extend(sp)

        sos, edge, filter_meta = _build_sos_filters(config.filters, recording.sfreq)
        data_path = output_dir / "data.npy"
        valid_path = output_dir / "valid.npy"
        data = np.lib.format.open_memmap(data_path, mode="w+", dtype=np.float64, shape=(len(out_channels), n_sel))
        valid = np.lib.format.open_memmap(valid_path, mode="w+", dtype=bool, shape=(len(out_channels), n_sel))

        rec_idx_by_id = {c.id: i for i, c in enumerate(recording.channels)}
        parent_id_list = sorted(parents_needed)
        parent_row = {pid: k for k, pid in enumerate(parent_id_list)}
        picks = [rec_idx_by_id[pid] for pid in parent_id_list]

        raw = open_mne_raw(snap_main, preload=False)
        try:
            verify_raw_filenames(recording, raw.filenames, snap_dir, sig_before,
                                 opened_stat=getattr(raw, "_brainlens_fd_stat", None))
            chunk_samples = max(1, int(round(recording.sfreq * 30.0)))
            for window, a, b in _blocks(windows, chunk_samples):
                src = read_scaled_volts(raw, recording, a, b, picks=picks)
                for row_i, out_ch in enumerate(out_channels):
                    seg = np.zeros(b - a, dtype=np.float64)
                    msk = np.ones(b - a, dtype=bool)
                    for p_ch, w in deps[out_ch.id]:
                        psig = src[parent_row[p_ch.id]]
                        seg += psig * w
                        msk &= np.isfinite(psig)
                        for ix, iy in parent_invalid[p_ch.id]:
                            if iy > a and ix < b:
                                msk[max(0, ix - a) : min(b - a, iy - a)] = False
                    for ix, iy in derived_invalid[out_ch.id]:
                        if iy > a and ix < b:
                            msk[max(0, ix - a) : min(b - a, iy - a)] = False
                    seg[~msk] = np.nan
                    data[row_i, a - window.offset : b - window.offset] = seg
                    valid[row_i, a - window.offset : b - window.offset] = msk
        finally:
            raw.close()

    if sos is not None:
        for row_i in range(len(out_channels)):
            for window in windows:
                left, right = window.array_start, window.array_stop
                for x, y in true_runs_mask(valid[row_i, left:right]):
                    ra, rb = left + x, left + y
                    if rb - ra <= 2 * edge:
                        valid[row_i, ra:rb] = False
                        data[row_i, ra:rb] = np.nan
                    else:
                        data[row_i, ra:rb] = sosfiltfilt(sos, np.asarray(data[row_i, ra:rb]))
                        valid[row_i, ra : ra + edge] = False
                        valid[row_i, rb - edge : rb] = False
                        data[row_i, ra : ra + edge] = np.nan
                        data[row_i, rb - edge : rb] = np.nan

    data.flush()
    valid.flush()
    prep_doc = {
        "data_path": str(data_path.resolve()),
        "valid_path": str(valid_path.resolve()),
        "sfreq": recording.sfreq,
        "start_sample": start,
        "n_samples": n_sel,
        "source_first_sample": recording.source_first_sample,
        "channels": [c.model_dump(mode="json") for c in out_channels],
        "effective_filters": filter_meta,
        "reference": config.reference.model_dump(mode="json"),
        "input_lowpass": recording.lowpass,
        "input_highpass": recording.highpass,
        "recording_fingerprint": recording.fingerprint,
    }
    if recording.pieces or config.segments:
        prep_doc["pieces"] = [w.piece.model_dump() for w in windows]
    if config.segments:
        prep_doc["segment_indexes"] = [w.segment_index for w in windows]
    atomic_write_json(output_dir / "prepared.json", prep_doc)
    return prep_doc
