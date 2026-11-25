"""Neurophysiological analytics: synchrony stratification, tied ranks, control windows, and comparisons."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import welch
from scipy.stats import spearmanr

from .analysis_windows import prepared_windows
from .anatomy import AnatomyMatcher, anatomy_coverage, load_anatomy_labels, match_anatomy
from .channel_order import ChannelDisplayOrder
from .models import (
    AnalysisConfig,
    ChannelSpec,
    RecordingPiece,
    RecordingSpec,
    ToolkitError,
    atomic_write_json,
    intersect_intervals,
    merge_intervals,
    true_runs_mask,
)
from .storage import WorkbenchStore, sanitize_slug, write_csv_file

_UNSET: Any = object()


def _leaders(rows: list[dict[str, Any]], rank: str) -> list[dict[str, Any]]:
    active = (row for row in rows if row.get(rank) is not None)
    # Keep raw rate as the secondary focal criterion; tied rows retain their montage order.
    return sorted(active, key=lambda row: (row[rank], -(row.get("raw_rate_per_min") or 0)))


def annotate_summary_anatomy(
    summary: dict[str, Any], overlay: dict[str, Any] | None, *, recording: RecordingSpec | None = None,
) -> dict[str, Any]:
    """Join external contact labels to display rows without changing the saved analysis."""
    rows = summary.get("channels", [])
    matcher = AnatomyMatcher(overlay)
    by_name = {
        row["channel_name"]: matcher.match(row["channel_name"], row.get("contacts") or [])
        for row in rows
    }

    def annotate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        decorated = []
        for row in items:
            name = row.get("channel_name") or row.get("channel") or ""
            anatomy = by_name[name] if name in by_name else matcher.match(name, row.get("contacts") or [])
            decorated.append({**row, "anatomy": anatomy})
        return decorated

    view = {**summary, "anatomy_coverage": anatomy_coverage(overlay, rows)}
    for key in ("channels", "leading_focal_channels", "leading_raw_channels", "band_powers"):
        if key in summary:
            view[key] = annotate(summary[key])
    if summary.get("spatial"):
        spatial = summary["spatial"]
        view["spatial"] = {**spatial, "contacts": annotate(spatial.get("contacts", []))}
    saved_fingerprint = summary.get("recording_fingerprint")
    if recording is not None and (not saved_fingerprint or saved_fingerprint == recording.fingerprint):
        order = ChannelDisplayOrder(recording)
        channels = sorted(view.get("channels", []), key=order.row_position)
        view["channels"] = channels
        for row in channels:
            row["display_shaft"] = order.depth_shaft(row.get("channel_id", ""), row["channel_name"], row.get("parents") or ())
        # Older workspaces saved these tables in rate/name order. Use the
        # acquisition metadata for display without rewriting the saved run.
        positions = {ch["channel_name"]: i for i, ch in enumerate(channels)}
        if "band_powers" in view:
            view["band_powers"].sort(key=lambda row: positions.get(row["channel_name"], len(positions)))
        if view.get("spatial"):
            shafts = {row["channel_name"]: row["display_shaft"] for row in channels}
            for row in view["spatial"]["contacts"]:
                row["display_shaft"] = shafts.get(row["channel_name"])
            view["spatial"]["contacts"].sort(key=lambda row: positions.get(row["channel_name"], len(positions)))
        for key in ("quiet_measured_channel_names", "unassessed_channel_names"):
            if key in view:
                view[key] = sorted(view[key], key=lambda name: positions.get(name, len(positions)))
        for key, rank in (("leading_focal_channels", "focal_rank"), ("leading_raw_channels", "raw_rank")):
            if (key in view and channels and all(rank in row for row in channels)
                    and all(row["channel_name"] in positions for row in view[key])):
                view[key] = _leaders(channels, rank)[:15]
        if "segments" in view:
            view["segments"] = [{**seg, "channels": sorted(seg["channels"], key=lambda r: positions.get(r["channel_name"], len(positions)))}
                                for seg in view["segments"]]
    return view


def _interval_seconds(spans: list[tuple[int, int]] | list[list[int]], sfreq: float) -> float:
    return sum(int(b) - int(a) for a, b in merge_intervals(spans)) / sfreq


def compute_tied_ranks(items: list[tuple[str, float | None]]) -> dict[str, int | None]:
    """Standard competition ranking ('1224') strictly for channels with rate > 0.

    Zero-rate channels receive rank=None so 0-event channels are NEVER labeled '#2, #3...'.
    """
    positive = sorted(
        [(name, round(float(val), 6)) for name, val in items if val is not None and val > 0],
        key=lambda x: (-x[1], x[0]),
    )
    ranks: dict[str, int | None] = {name: None for name, _ in items}
    current_rank = 1
    for idx, (name, val) in enumerate(positive):
        if idx > 0 and val < positive[idx - 1][1]:
            current_rank = idx + 1
        ranks[name] = current_rank
    return ranks


def annotate_event_synchrony(
    events: list[dict[str, Any]],
    channels: list[ChannelSpec],
    sfreq: float,
    window_ms: float = 30.0,
    min_burst_shafts: int = 5,
    piece: RecordingPiece | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify each event as 'focal', 'regional', or 'widespread_burst' by cross-shaft simultaneity."""
    if not events:
        return [], []

    distinct_eeg_shafts = {
        c.shaft for c in channels if not c.is_auxiliary and c.shaft not in {"AUX", "OTHER"}
    }
    n_total_shafts = len(distinct_eeg_shafts)
    # Honor min_burst_shafts strictly whenever enough shafts exist; never silently lower e.g. min=5 to 3.
    if n_total_shafts >= min_burst_shafts:
        burst_shaft_threshold = min_burst_shafts
    else:
        burst_shaft_threshold = 9999
    burst_channel_threshold = max(8, int(math.ceil(0.35 * max(1, sum(1 for c in channels if not c.is_auxiliary)))))

    pad = max(1, int(round((window_ms / 1000.0) * sfreq)))
    ordered = sorted(events, key=lambda e: (e["start_sample"], e["end_sample"], e["channel_id"]))
    n = len(ordered)

    # Two-pointer sweep over exact ±window_ms ([start_sample - pad, end_sample + pad])
    left = 0
    right = 0
    annotated: list[dict[str, Any]] = []
    burst_spans: list[tuple[int, int]] = []

    for i in range(n):
        ev = dict(ordered[i])
        lo_i = ev["start_sample"] - pad
        hi_i = ev["end_sample"] + pad
        while left < n and ordered[left]["end_sample"] < lo_i:
            left += 1
        if right < i:
            right = i
        while right + 1 < n and ordered[right + 1]["start_sample"] <= hi_i:
            right += 1

        co_chs: set[str] = set()
        co_shafts: set[str] = set()
        for j in range(left, right + 1):
            cand = ordered[j]
            if cand["end_sample"] >= lo_i and cand["start_sample"] <= hi_i:
                co_chs.add(cand["channel_name"])
                sh = cand.get("shaft") or "OTHER"
                if sh not in {"AUX", "OTHER"}:
                    co_shafts.add(sh)

        n_co_ch = len(co_chs)
        n_co_sh = len(co_shafts)
        if n_co_sh >= burst_shaft_threshold or (n_total_shafts < 3 and len(channels) >= 12 and n_co_ch >= burst_channel_threshold):
            sync_cls = "widespread_burst"
            burst_spans.append((ev["start_sample"] - pad, ev["end_sample"] + pad))
        elif n_co_sh >= 3 or n_co_ch >= 5:
            sync_cls = "regional"
        else:
            sync_cls = "focal"

        ev["coactive_channel_count"] = n_co_ch
        ev["coactive_shaft_count"] = n_co_sh
        ev["coactive_shafts"] = sorted(co_shafts)[:8]
        ev["synchrony_class"] = sync_cls
        annotated.append(ev)

    # Summarize contiguous widespread burst episodes
    merged_bursts = merge_intervals(burst_spans)
    burst_episodes: list[dict[str, Any]] = []
    for ba, bb in merged_bursts:
        members = [e for e in annotated if e["start_sample"] < bb and e["end_sample"] > ba]
        if not members:
            continue
        chs = sorted({e["channel_name"] for e in members})
        shs = sorted({e["shaft"] for e in members if e.get("shaft") not in {"AUX", "OTHER", None}})
        burst_episodes.append({
            "start_seconds": piece.time_at(max(piece.start_sample, ba), sfreq) if piece else round(max(0, ba) / sfreq, 3),
            "end_seconds": piece.time_at(min(piece.stop_sample, bb), sfreq) if piece else round(bb / sfreq, 3),
            "event_count": len(members),
            "channel_count": len(chs),
            "shaft_count": len(shs),
            "shafts": shs,
        })

    return annotated, burst_episodes


def _annotate_event_spectral_metrics(
    events: list[dict[str, Any]],
    out_channels: list[ChannelSpec],
    data_arr: Any,
    start_sample: int,
    sfreq: float,
    input_lowpass: float | None = None,
) -> None:
    """Compute per-event peak HFO frequency, Ripple vs Fast-Ripple dominance, and peak-to-peak µV."""
    if not events:
        return
    idx_by_cid = {c.id: i for i, c in enumerate(out_channels)}
    ch_by_cid = {c.id: c for c in out_channels}
    nyq = 0.5 * sfreq
    n_total_samps = int(data_arr.shape[1])
    pad = max(4, int(round(0.015 * sfreq)))

    for ev in events:
        cidx = idx_by_cid.get(ev["channel_id"])
        if cidx is None:
            ev["peak_freq_hz"] = None
            ev["dominant_band"] = "ripple"
            ev["fast_ripple_power_ratio"] = None
            ev["peak_to_peak_uv"] = None
            continue
        ch_spec = ch_by_cid[ev["channel_id"]]
        ch_nyq = 0.5 * min(sfreq, float(ch_spec.source_sfreq)) if (ch_spec.source_sfreq and float(ch_spec.source_sfreq) > 0) else nyq
        lp_vals = [float(v) for v in (ch_spec.source_lowpass, input_lowpass) if v is not None and float(v) > 0]
        bw_ceil = min([ch_nyq * 0.92, *lp_vals])
        fr_hi = min(450.0, bw_ceil)
        has_fr = fr_hi > 270.0
        a = max(0, int(ev["start_sample"]) - start_sample - pad)
        b = min(n_total_samps, int(ev["end_sample"]) - start_sample + pad)
        if b - a < 8:
            ev["peak_freq_hz"] = None
            ev["dominant_band"] = "ripple"
            ev["fast_ripple_power_ratio"] = None
            ev["peak_to_peak_uv"] = None
            continue
        seg_uv = np.asarray(data_arr[cidx, a:b], dtype=np.float64) * 1e6
        seg_uv = np.nan_to_num(seg_uv - np.nanmedian(seg_uv), nan=0.0)
        p2p = float(np.max(seg_uv) - np.min(seg_uv)) if seg_uv.size else 0.0
        nfft = max(256, 1 << (seg_uv.size - 1).bit_length())
        win = np.hanning(seg_uv.size)
        spec = np.abs(np.fft.rfft(seg_uv * win, n=nfft)) ** 2
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sfreq)
        rip_hi = min(250.0, bw_ceil)
        rip_mask = (freqs >= 80.0) & (freqs < rip_hi)
        fr_mask = (freqs >= 250.0) & (freqs <= fr_hi) if has_fr else np.zeros_like(rip_mask, dtype=bool)
        hfo_mask = rip_mask | fr_mask
        peak_hz = None
        if np.any(hfo_mask):
            sub_f = freqs[hfo_mask]
            sub_p = spec[hfo_mask]
            if sub_p.size and float(np.max(sub_p)) > 0:
                peak_hz = round(float(sub_f[int(np.argmax(sub_p))]), 1)
        rip_p = float(np.sum(spec[rip_mask])) if np.any(rip_mask) else 0.0
        fr_p = float(np.sum(spec[fr_mask])) if np.any(fr_mask) else 0.0
        denom = rip_p + fr_p
        fr_ratio = round(fr_p / denom, 4) if (has_fr and denom > 0) else 0.0
        dom = "fast_ripple" if (has_fr and ((peak_hz is not None and peak_hz >= 250.0) or fr_ratio >= 0.45)) else "ripple"
        ev["peak_freq_hz"] = peak_hz
        ev["dominant_band"] = dom
        ev["fast_ripple_power_ratio"] = fr_ratio if has_fr else None
        ev["peak_to_peak_uv"] = round(p2p, 2)


def summarize_run(
    recording: RecordingSpec,
    prepared: dict[str, Any],
    qc: dict[str, Any],
    detection: dict[str, Any],
    classification: dict[str, Any],
    verdicts_by_event: dict[str, dict[str, Any]],
    config: AnalysisConfig,
    run_dir: Path,
    generation_id: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sfreq = float(prepared["sfreq"])
    windows = prepared_windows(prepared)
    out_channels = [ChannelSpec.model_validate(c) for c in prepared["channels"]]
    valid_arr = np.load(prepared["valid_path"], mmap_mode="r", allow_pickle=False)
    data_arr = np.load(prepared["data_path"], mmap_mode="r", allow_pickle=False)

    eff_lp_vals = [
        float(v)
        for v in (recording.lowpass, (prepared.get("effective_filters") or {}).get("lowpass"))
        if v is not None and float(v) > 0
    ]

    raw_events = classification.get("events", detection.get("events", []))
    events, burst_windows = [], []
    for window in windows:
        piece, left, right = window.piece, window.start, window.stop
        piece_events = ([e for e in raw_events if left <= e["start_sample"] < right]
                        if prepared.get("pieces") else raw_events)
        annotated, bursts = annotate_event_synchrony(
            piece_events, out_channels, sfreq=sfreq, window_ms=config.synchrony_window_ms,
            min_burst_shafts=config.widespread_burst_min_shafts, piece=piece if prepared.get("pieces") else None,
        )
        _annotate_event_spectral_metrics(
            annotated, out_channels, data_arr[:, window.array_start:window.array_stop], left, sfreq,
            input_lowpass=min(eff_lp_vals) if eff_lp_vals else None,
        )
        if window.segment_index is not None:
            provenance = {"segment_index": window.segment_index,
                          "segment_label": config.segments[window.segment_index].label}
            for item in [*annotated, *bursts]:
                item.update(provenance)
        events.extend(annotated)
        burst_windows.extend(bursts)
    for ev in events:
        vd = verdicts_by_event.get(ev["id"], {})
        ev["clinician_verdict"] = vd.get("verdict")
        ev["clipped"] = bool(vd.get("clipped", False))
        ev["clinician_note"] = vd.get("note", "")
        ev["verdict_role"] = vd.get("role")
        ev["researcher_verdict"] = vd.get("researcher_verdict")
        ev["researcher_clipped"] = bool(vd.get("researcher_clipped", False))
        ev["rehearsal_verdict"] = vd.get("rehearsal_verdict")
        ev["rehearsal_clipped"] = bool(vd.get("rehearsal_clipped", False))
        ev["agent_verdict"] = vd.get("agent_verdict")
        ev["agent_note"] = vd.get("agent_note", "")
        ev["agent_clipped"] = bool(vd.get("agent_clipped", False))

    det_valid = detection.get("valid_intervals", {})
    ev_by_cid: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        ev_by_cid.setdefault(ev["channel_id"], []).append(ev)

    clf_stages = (classification.get("classification") or {}).get("stages") or {}
    art_stage_ok = (clf_stages.get("artifact") or {}).get("status") == "complete"
    spike_stage_ok = (clf_stages.get("spike") or {}).get("status") == "complete"
    ehfo_stage_ok = (clf_stages.get("ehfo") or {}).get("status") == "complete"

    ch_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    psd_rows: list[dict[str, Any]] = []
    ep_width = max(1, int(round(config.epoch_seconds * sfreq)))
    epochs = [(w, ea, min(w.stop, ea + ep_width)) for w in windows for ea in range(w.start, w.stop, ep_width)]
    segment_rows = [
        {**seg.model_dump(), "segment_index": i, "channels": []}
        for i, seg in enumerate(config.segments or [])
    ]
    segment_spans: list[list[tuple[int, int]]] = [[] for _ in segment_rows]
    for w in windows:
        if w.segment_index is not None:
            segment_spans[w.segment_index].append((w.start, w.stop))
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

    order = ChannelDisplayOrder(recording)
    channel_indices = sorted(range(len(out_channels)), key=lambda i: order.position(
        out_channels[i].id, out_channels[i].name, out_channels[i].parents))
    for idx in channel_indices:
        ch = out_channels[idx]
        good_spans = [(w.start + a, w.start + b) for w in windows
                      for a, b in true_runs_mask(valid_arr[idx, w.array_start:w.array_stop])]
        analyzed_spans = det_valid.get(ch.id, [])
        valid_sec = _interval_seconds(good_spans, sfreq)
        analyzed_sec = _interval_seconds(analyzed_spans, sfreq)

        ch_evs = ev_by_cid.get(ch.id, [])
        focal_evs = [e for e in ch_evs if e["synchrony_class"] != "widespread_burst"]
        burst_evs = [e for e in ch_evs if e["synchrony_class"] == "widespread_burst"]
        non_art_evs = [e for e in ch_evs if (e.get("labels") or {}).get("artifact") == 1]
        spike_evs = [e for e in ch_evs if (e.get("labels") or {}).get("spike") == 1]
        ehfo_evs = [e for e in ch_evs if (e.get("labels") or {}).get("ehfo") == 1]
        fr_evs = [e for e in focal_evs if e.get("dominant_band") == "fast_ripple"]

        confirmed_n = sum(1 for e in ch_evs if e.get("clinician_verdict") == "confirm")
        rejected_n = sum(1 for e in ch_evs if e.get("clinician_verdict") == "artifact")
        uncertain_n = sum(1 for e in ch_evs if e.get("clinician_verdict") == "uncertain")

        has_cov = analyzed_sec > 0
        raw_rate = (len(ch_evs) * 60.0 / analyzed_sec) if has_cov else None
        focal_rate = (len(focal_evs) * 60.0 / analyzed_sec) if has_cov else None
        non_art_rate = (len(non_art_evs) * 60.0 / analyzed_sec) if (has_cov and art_stage_ok) else None
        ehfo_rate = (len(ehfo_evs) * 60.0 / analyzed_sec) if (has_cov and ehfo_stage_ok) else None

        ch_rows.append({
            "channel_id": ch.id,
            "channel_name": ch.name,
            "shaft": ch.shaft,
            "contact_index": ch.contact_index,
            "is_auxiliary": ch.is_auxiliary,
            "contacts": ch.contacts,
            "parents": ch.parents,
            "unit": ch.unit,
            "status": "measured" if has_cov else "unassessed",
            "valid_signal_seconds": valid_sec if config.segments else round(valid_sec, 4),
            "detector_analyzed_seconds": analyzed_sec if config.segments else round(analyzed_sec, 4),
            "raw_count": len(ch_evs) if has_cov else None,
            "raw_rate_per_min": round(raw_rate, 4) if raw_rate is not None else None,
            "focal_count": len(focal_evs) if has_cov else None,
            "focal_rate_per_min": round(focal_rate, 4) if focal_rate is not None else None,
            "fast_ripple_count": len(fr_evs) if has_cov else None,
            "widespread_burst_count": len(burst_evs) if has_cov else None,
            "model_non_artifact_count": len(non_art_evs) if (has_cov and art_stage_ok) else None,
            "model_non_artifact_rate_per_min": round(non_art_rate, 4) if non_art_rate is not None else None,
            "spike_hfo_count": len(spike_evs) if (has_cov and spike_stage_ok) else None,
            "ehfo_count": len(ehfo_evs) if (has_cov and ehfo_stage_ok) else None,
            "ehfo_rate_per_min": round(ehfo_rate, 4) if ehfo_rate is not None else None,
            "clinician_confirmed": confirmed_n,
            "clinician_artifact": rejected_n,
            "clinician_uncertain": uncertain_n,
        })

        for row, spans in zip(segment_rows, segment_spans):
            index = row["segment_index"]
            seg_events = [e for e in ch_evs if e["segment_index"] == index]
            focal_count = sum(e["synchrony_class"] != "widespread_burst" for e in seg_events)
            n_valid = sum(b - a for a, b in intersect_intervals(good_spans, spans))
            n_analyzed = sum(b - a for a, b in intersect_intervals(analyzed_spans, spans))
            row["channels"].append({
                "channel_id": ch.id, "channel_name": ch.name,
                "valid_signal_samples": n_valid, "detector_analyzed_samples": n_analyzed,
                "valid_signal_seconds": n_valid / sfreq, "detector_analyzed_seconds": n_analyzed / sfreq,
                "raw_count": len(seg_events) if n_analyzed else None,
                "focal_count": focal_count if n_analyzed else None,
                "raw_rate_per_min": round(len(seg_events) * 60 * sfreq / n_analyzed, 4) if n_analyzed else None,
                "focal_rate_per_min": round(focal_count * 60 * sfreq / n_analyzed, 4) if n_analyzed else None,
            })
        for w, ea, eb in epochs:
            piece = w.piece
            denom = _interval_seconds(intersect_intervals(analyzed_spans, [(ea, eb)]), sfreq)
            ep_all = [e for e in ch_evs if ea <= e["start_sample"] < eb]
            ep_focal = [e for e in ep_all if e["synchrony_class"] != "widespread_burst"]
            epoch_rows.append({
                "channel_id": ch.id,
                "channel_name": ch.name,
                **({"segment_index": w.segment_index} if config.segments else {}),
                "start_seconds": round(piece.time_at(ea, sfreq), 4),
                "end_seconds": round(piece.time_at(eb, sfreq), 4),
                "analyzed_seconds": round(denom, 4),
                "raw_count": len(ep_all) if denom > 0 else None,
                "focal_count": len(ep_focal) if denom > 0 else None,
                "raw_rate_per_min": round(len(ep_all) * 60.0 / denom, 4) if denom > 0 else None,
                "focal_rate_per_min": round(len(ep_focal) * 60.0 / denom, 4) if denom > 0 else None,
            })

        # Compact Welch band powers (delta/theta/alpha/beta/ripple) for scalp & spatial views
        nperseg = max(16, int(round(2.0 * sfreq)))
        chunks = [data_arr[idx, a + w.array_start:b + w.array_start] for w in windows
                  for a, b in true_runs_mask(valid_arr[idx, w.array_start:w.array_stop]) if b - a >= 16]
        if chunks and ch.unit == "V":
            blocks = chunks[:4] if prepared.get("pieces") else [np.concatenate(chunks[:4])]
            powers, weights = [], []
            for block in blocks:
                f_grid, pxx = welch(block, fs=sfreq, nperseg=min(nperseg, len(block)))
                alpha_mask = (f_grid >= 8.0) & (f_grid <= 13.0)
                if np.any(alpha_mask):
                    powers.append(float(_trapz(pxx[alpha_mask], f_grid[alpha_mask]) * 1e12))
                    weights.append(len(block))
            alpha_uv2 = float(np.average(powers, weights=weights)) if powers else None
            psd_rows.append({"channel_id": ch.id, "channel_name": ch.name, "alpha_power_uv2": round(alpha_uv2, 3) if alpha_uv2 is not None else None})

    # Assign tied competition ranks (ONLY for count > 0!)
    focal_ranks = compute_tied_ranks([(r["channel_name"], r["focal_rate_per_min"]) for r in ch_rows])
    raw_ranks = compute_tied_ranks([(r["channel_name"], r["raw_rate_per_min"]) for r in ch_rows])
    for r in ch_rows:
        r["focal_rank"] = focal_ranks[r["channel_name"]]
        r["raw_rank"] = raw_ranks[r["channel_name"]]

    n_total = len(events)
    n_focal = sum(1 for e in events if e["synchrony_class"] == "focal")
    n_regional = sum(1 for e in events if e["synchrony_class"] == "regional")
    n_burst = sum(1 for e in events if e["synchrony_class"] == "widespread_burst")

    # Build 10s recording-wide timeline histogram separating focal vs burst events + assessed-channel QC bad-segment fraction
    assessed_row_idxs = [i for i, ch in enumerate(out_channels) if det_valid.get(ch.id)]
    timeline_bins: list[dict[str, Any]] = []
    for w, ea, eb in epochs:
        piece = w.piece
        t0, t1 = round(piece.time_at(ea, sfreq), 3), round(piece.time_at(eb, sfreq), 3)
        in_bin = [e for e in events if ea <= e["start_sample"] < eb]
        rel_a, rel_b = ea - w.offset, eb - w.offset
        inv_frac = (
            round(float(1.0 - np.mean(valid_arr[assessed_row_idxs, rel_a:rel_b])), 4)
            if rel_b > rel_a and assessed_row_idxs
            else 0.0
        )
        timeline_bins.append({
            "start_seconds": t0,
            **({"segment_index": w.segment_index} if config.segments else {}),
            "end_seconds": t1,
            "focal_events": sum(1 for e in in_bin if e["synchrony_class"] != "widespread_burst"),
            "widespread_burst_events": sum(1 for e in in_bin if e["synchrony_class"] == "widespread_burst"),
            "total_events": len(in_bin),
            "invalid_channel_fraction": inv_frac,
        })

    # Spatial projection mapping
    spatial_contacts: list[dict[str, Any]] = []
    if recording.spatial is not None:
        pos_map = recording.spatial.positions
        for r in ch_rows:
            keys = [*r.get("contacts", []), r["channel_name"], r["channel_id"]]
            xyz = next((pos_map[k] for k in keys if k in pos_map), None)
            if xyz is not None:
                spatial_contacts.append({
                    "channel_id": r["channel_id"],
                    "channel_name": r["channel_name"],
                    "shaft": r["shaft"],
                    "contact_index": r["contact_index"],
                    "xyz": [round(float(v), 4) for v in xyz],
                    "focal_count": r["focal_count"],
                    "focal_rate_per_min": r["focal_rate_per_min"],
                    "raw_count": r["raw_count"],
                    "raw_rate_per_min": r["raw_rate_per_min"],
                    "analyzed_seconds": r["detector_analyzed_seconds"],
                })

    leading_focal = _leaders(ch_rows, "focal_rank")
    leading_raw = _leaders(ch_rows, "raw_rank")
    quiet_measured = [r["channel_name"] for r in ch_rows if r["status"] == "measured" and (r["raw_count"] or 0) == 0]
    unassessed = [r["channel_name"] for r in ch_rows if r["status"] != "measured"]

    distinct_eeg_shafts = {
        c.shaft for c in out_channels if not c.is_auxiliary and c.shaft not in {"AUX", "OTHER"}
    }
    n_total_shafts = len(distinct_eeg_shafts)
    eff_burst_shafts = (
        config.widespread_burst_min_shafts
        if n_total_shafts >= config.widespread_burst_min_shafts
        else None
    )

    warnings = list(dict.fromkeys([
        *recording.warnings,
        *qc.get("warnings", []),
        *detection.get("warnings", []),
        *classification.get("warnings", []),
    ]))
    if n_total > 0 and n_burst / n_total >= 0.5:
        sh_lbl = f">={eff_burst_shafts} electrode shafts" if eff_burst_shafts is not None else "widespread channels"
        warnings.insert(
            0,
            f"High multi-shaft synchrony: {n_burst}/{n_total} ({100.0 * n_burst / n_total:.1f}%) raw candidates "
            f"occurred simultaneously across {sh_lbl} "
            f"(likely common-reference or global transients). Inspect 'Focal Candidates' ({n_focal + n_regional} events) separately.",
        )

    measured_n = sum(1 for r in ch_rows if r["status"] == "measured")
    focal_quiet_n = max(0, measured_n - len(leading_focal))
    eff_mode = config.reference.mode
    if eff_mode == "bipolar":
        eff_ref_label = f"bipolar ({len(out_channels)} pairs)"
    elif eff_mode == "average":
        eff_ref_label = f"common average ({len(out_channels)} ch)"
    else:
        eff_ref_label = f"original ({recording.reference})"

    qc_ln = qc.get("line_noise") or {}
    qc_overview = {
        "line_freq_hz": qc_ln.get("line_freq_hz") or recording.line_freq,
        "median_line_noise_snr_db": qc_ln.get("median_snr_db"),
        "max_line_noise_snr_db": qc_ln.get("max_snr_db"),
        "high_line_noise_channels": qc_ln.get("elevated_channels", []),
        "flat_window_channels": qc.get("flat_window_channels", []),
        "amplitude_warning_channels": qc.get("amplitude_warning_channels", []),
    }

    summary = {
        "generation_id": generation_id,
        "recording_id": recording.id,
        "recording": recording.slug,
        "recording_fingerprint": recording.fingerprint,
        "modality": recording.modality,
        "acquired_reference": recording.reference,
        "acquired_reference_description": recording.reference_description,
        "effective_reference": config.reference.model_dump(mode="json"),
        "effective_reference_label": eff_ref_label,
        "synchrony_thresholds": {
            "window_ms": config.synchrony_window_ms,
            "configured_min_burst_shafts": config.widespread_burst_min_shafts,
            "effective_min_burst_shafts": eff_burst_shafts,
            "total_eeg_shafts": n_total_shafts,
        },
        "qc_overview": qc_overview,
        "selection": {
            "start_seconds": round(windows[0].piece.time_at(windows[0].start, sfreq), 4),
            "end_seconds": round(windows[-1].piece.time_at(windows[-1].stop, sfreq), 4),
            **({"pieces": prepared["pieces"]} if prepared.get("pieces") else {}),
            **({"segments": [s.model_dump() for s in config.segments]} if config.segments else {}),
            "sfreq_hz": sfreq,
            "state": config.state,
        },
        "detector_status": detection.get("status", "skipped"),
        "classifier_status": classification.get("classifier_status", "skipped"),
        "classification": {
            "requested": (classification.get("classification") or {}).get("requested", []),
            "stages": (classification.get("classification") or {}).get("stages", {}),
        },
        "counts": {
            "total_raw_events": n_total,
            "focal_events": n_focal,
            "regional_events": n_regional,
            "non_burst_events": n_focal + n_regional,
            "fast_ripple_focal_events": sum(1 for e in events if e["synchrony_class"] != "widespread_burst" and e.get("dominant_band") == "fast_ripple"),
            "widespread_burst_events": n_burst,
            "widespread_burst_fraction": round(n_burst / n_total, 4) if n_total else 0.0,
            "widespread_burst_episodes": len(burst_windows),
            "non_artifact_events": (
                sum((r["model_non_artifact_count"] or 0) for r in ch_rows)
                if art_stage_ok and any(r["model_non_artifact_count"] is not None for r in ch_rows)
                else None
            ),
            "spike_hfo_events": (
                sum((r["spike_hfo_count"] or 0) for r in ch_rows)
                if spike_stage_ok and any(r["spike_hfo_count"] is not None for r in ch_rows)
                else None
            ),
            "ehfo_events": (
                sum((r["ehfo_count"] or 0) for r in ch_rows)
                if ehfo_stage_ok and any(r["ehfo_count"] is not None for r in ch_rows)
                else None
            ),
            "measured_channels": measured_n,
            "active_focal_channels": len(leading_focal),
            "active_raw_channels": len(leading_raw),
            "focal_quiet_measured_channels": focal_quiet_n,
            "quiet_measured_channels": len(quiet_measured),
            "unassessed_channels": len(unassessed),
            "reviewed_focal_events": sum(
                1 for e in events
                if e["synchrony_class"] != "widespread_burst" and e.get("clinician_verdict") in {"confirm", "artifact", "uncertain"}
            ),
            "clinician_confirmed": sum(r["clinician_confirmed"] for r in ch_rows),
            "clinician_artifact": sum(r["clinician_artifact"] for r in ch_rows),
            "clinician_uncertain": sum(r["clinician_uncertain"] for r in ch_rows),
            "researcher_confirmed": sum(1 for e in events if e.get("researcher_verdict") == "confirm"),
            "researcher_reviewed": sum(1 for e in events if e.get("researcher_verdict") in {"confirm", "artifact", "uncertain"}),
            "rehearsal_confirmed": sum(1 for e in events if e.get("rehearsal_verdict") == "confirm"),
            "rehearsal_reviewed": sum(1 for e in events if e.get("rehearsal_verdict") in {"confirm", "artifact", "uncertain"}),
            "agent_confirmed": sum(1 for e in events if e.get("agent_verdict") == "confirm"),
            "agent_artifact": sum(1 for e in events if e.get("agent_verdict") == "artifact"),
            "agent_uncertain": sum(1 for e in events if e.get("agent_verdict") == "uncertain"),
            "agent_reviewed": sum(1 for e in events if e.get("agent_verdict") in {"confirm", "artifact", "uncertain"}),
            "clipped_for_discussion": sum(1 for e in events if e.get("clipped")),
        },
        "synchronous_burst_windows": burst_windows,
        "leading_focal_channels": leading_focal[:15],
        "leading_raw_channels": leading_raw[:15],
        "quiet_measured_channel_names": quiet_measured,
        "unassessed_channel_names": unassessed,
        "channels": ch_rows,
        **({"segments": segment_rows} if segment_rows else {}),
        "timeline_bins": timeline_bins,
        "band_powers": psd_rows,
        "spatial": {
            "available": bool(spatial_contacts),
            "coordinate_system": recording.spatial.coordinate_system if recording.spatial else None,
            "units": recording.spatial.units if recording.spatial else None,
            "template": recording.spatial.template if recording.spatial else False,
            "source": recording.spatial.source if recording.spatial else None,
            "mapped_count": len(spatial_contacts),
            "contacts": spatial_contacts,
        },
        "rate_formula": "events/min = 60 * count / detector_analyzed_seconds (unassessed = null, never 0)",
        "warnings": warnings,
    }

    # Write clean CSVs for researchers
    write_csv_file(
        run_dir / "channel_rates.csv",
        ch_rows,
        [
            "focal_rank",
            "raw_rank",
            "channel_name",
            "channel_id",
            "shaft",
            "status",
            "valid_signal_seconds",
            "detector_analyzed_seconds",
            "focal_count",
            "focal_rate_per_min",
            "fast_ripple_count",
            "raw_count",
            "raw_rate_per_min",
            "widespread_burst_count",
            "model_non_artifact_count",
            "model_non_artifact_rate_per_min",
            "spike_hfo_count",
            "ehfo_count",
            "ehfo_rate_per_min",
            "clinician_confirmed",
            "clinician_artifact",
            "clinician_uncertain",
        ],
    )
    write_csv_file(
        run_dir / "epoch_rates.csv",
        epoch_rows,
        [
            "channel_name",
            "channel_id",
            *(["segment_index"] if config.segments else []),
            "start_seconds",
            "end_seconds",
            "analyzed_seconds",
            "focal_count",
            "focal_rate_per_min",
            "raw_count",
            "raw_rate_per_min",
        ],
    )
    write_csv_file(
        run_dir / "events.csv",
        [
            {
                "id": e["id"],
                **({"segment_index": e["segment_index"], "segment_label": e["segment_label"]} if config.segments else {}),
                "channel_name": e["channel_name"],
                "shaft": e.get("shaft", ""),
                "start_seconds": e["start_seconds"],
                "end_seconds": e["end_seconds"],
                "duration_ms": round((e["end_seconds"] - e["start_seconds"]) * 1000.0, 2),
                "synchrony_class": e["synchrony_class"],
                "dominant_band": e.get("dominant_band"),
                "peak_freq_hz": e.get("peak_freq_hz"),
                "peak_to_peak_uv": e.get("peak_to_peak_uv"),
                "coactive_channel_count": e["coactive_channel_count"],
                "coactive_shaft_count": e["coactive_shaft_count"],
                "artifact_score": (e.get("scores") or {}).get("artifact"),
                "spike_score": (e.get("scores") or {}).get("spike"),
                "ehfo_score": (e.get("scores") or {}).get("ehfo"),
                "artifact_label": (e.get("labels") or {}).get("artifact"),
                "spike_label": (e.get("labels") or {}).get("spike"),
                "ehfo_label": (e.get("labels") or {}).get("ehfo"),
                "clinician_verdict": e.get("clinician_verdict"),
                "clipped": e.get("clipped", False),
            }
            for e in events
        ],
        [
            "id",
            *(["segment_index", "segment_label"] if config.segments else []),
            "channel_name",
            "shaft",
            "start_seconds",
            "end_seconds",
            "duration_ms",
            "synchrony_class",
            "dominant_band",
            "peak_freq_hz",
            "peak_to_peak_uv",
            "coactive_channel_count",
            "coactive_shaft_count",
            "artifact_score",
            "spike_score",
            "ehfo_score",
            "artifact_label",
            "spike_label",
            "ehfo_label",
            "clinician_verdict",
            "clipped",
        ],
    )
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(run_dir / "events.json", events)
    return summary, events


def select_control_windows(
    store: WorkbenchStore,
    case_slug: str,
    rec_slug: str,
    run_name: str = "baseline",
    channel: str | None = None,
    count: int = 3,
    window_seconds: float = 0.6,
) -> list[dict[str, Any]]:
    """Deterministically select clean, valid, candidate-free background control intervals on a channel."""
    bundle = store.load_run(case_slug, rec_slug, run_name)
    summary = bundle["summary"]
    if summary.get("detector_status") != "complete":
        return []
    events = bundle["events"]
    sfreq = float(summary["selection"]["sfreq_hz"])
    pieces = [RecordingPiece.model_validate(p) for p in summary["selection"].get("pieces", [])]

    ch_rows = summary["channels"]
    if channel:
        target_row = next((r for r in ch_rows if channel in {r["channel_name"], r["channel_id"]}), None)
    else:
        lead = summary["leading_focal_channels"] or summary["leading_raw_channels"] or ch_rows
        target_row = lead[0] if lead else None
    if target_row is None:
        raise ToolkitError("unknown_channel", f"Channel {channel!r} not found.")

    cid = target_row["channel_id"]
    cname = target_row["channel_name"]
    anatomy = match_anatomy(load_anatomy_labels(store, case_slug), cname, target_row.get("contacts") or [])
    det_path = store.resolve_workspace_artifact_path(bundle["meta"].get("detection_json_path"))
    if det_path is None:
        return []
    det = json_load_safe(det_path)
    valid_spans = det.get("valid_intervals", {}).get(cid, [])
    if not valid_spans:
        return []

    # Exclude +-1.5s around every event on this channel AND around global bursts
    guard = int(round(1.5 * sfreq))
    busy: list[tuple[int, int]] = []
    for ev in events:
        if ev["channel_id"] == cid or ev.get("synchrony_class") == "widespread_burst":
            piece = next((p for p in pieces if p.start_sample <= ev["start_sample"] < p.stop_sample), None)
            left = piece.start_sample if piece else 0
            end = ev["end_sample"] + guard
            busy.append((max(left, ev["start_sample"] - guard), min(piece.stop_sample, end) if piece else end))

    rec_dur = float(summary.get("selection", {}).get("duration_seconds") or 30.0)
    eff_win = float(np.clip(window_seconds, 0.2, min(30.0, rec_dur)))
    win_samps = max(16, int(round(eff_win * sfreq)))
    quiet_candidates: list[tuple[int, int]] = []
    for va, vb in valid_spans:
        # Subtract busy from [va, vb]
        cursor = int(va)
        for ba, bb in merge_intervals(busy, int(va), int(vb)):
            if ba - cursor >= win_samps:
                quiet_candidates.append((cursor, ba))
            cursor = max(cursor, bb)
        if int(vb) - cursor >= win_samps:
            quiet_candidates.append((cursor, int(vb)))

    if not quiet_candidates:
        return []

    # Sample evenly spaced centers across quiet spans
    total_quiet = sum(b - a - win_samps for a, b in quiet_candidates)
    picks: list[dict[str, Any]] = []
    for k in range(count):
        frac = (k + 0.5) / max(1, count)
        target_offset = frac * max(0, total_quiet)
        acc = 0.0
        chosen_s0 = quiet_candidates[0][0]
        for a, b in quiet_candidates:
            usable = max(0, b - a - win_samps)
            if acc + usable >= target_offset:
                chosen_s0 = int(round(a + (target_offset - acc)))
                break
            acc += usable
        s1 = chosen_s0 + win_samps
        piece = next((p for p in pieces if p.start_sample <= chosen_s0 < p.stop_sample), None)
        t0 = piece.time_at(chosen_s0, sfreq) if piece else chosen_s0 / sfreq
        t1 = piece.time_at(s1, sfreq) if piece else s1 / sfreq
        picks.append({
            "id": f"ctrl_{k + 1}_{cname}_{round(t0, 2)}s",
            "kind": "background_control",
            "channel_id": cid,
            "channel_name": cname,
            "anatomy": anatomy,
            "start_seconds": t0 if piece else round(t0, 4),
            "end_seconds": t1 if piece else round(t1, 4),
            "center_seconds": (t0 + t1) * 0.5 if piece else round(0.5 * (chosen_s0 + s1) / sfreq, 4),
            "duration_ms": round(window_seconds * 1000.0, 1),
        })
    return picks


def json_load_safe(path: Path) -> dict[str, Any]:
    from .models import read_json

    return read_json(path) if path.is_file() else {}


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    r, _ = spearmanr(xs, ys)
    return round(float(r), 4) if r is not None and math.isfinite(float(r)) else None


def compare_runs(
    store: WorkbenchStore,
    case_slug: str,
    rec_slug: str,
    run_a: str,
    run_b: str,
    metric: str = "focal",
    top_k: int = 16,
    *,
    bundle_a: dict[str, Any] | None = None,
    bundle_b: dict[str, Any] | None = None,
    anatomy_labels: dict[str, Any] | None = _UNSET,
) -> dict[str, Any]:
    """Symmetric sensitivity comparison showing BOTH baseline leaders and branch risers."""
    ba = bundle_a if bundle_a is not None else store.load_run(case_slug, rec_slug, run_a)
    bb = bundle_b if bundle_b is not None else store.load_run(case_slug, rec_slug, run_b)
    if anatomy_labels is _UNSET:
        anatomy_labels = load_anatomy_labels(store, case_slug)
    recording = store.load_recording(case_slug, rec_slug)
    sa = annotate_summary_anatomy(ba["summary"], anatomy_labels, recording=recording)
    sb = annotate_summary_anatomy(bb["summary"], anatomy_labels, recording=recording)

    map_a = {r["channel_name"]: r for r in sa["channels"]}
    map_b = {r["channel_name"]: r for r in sb["channels"]}

    fp_a = (ba.get("meta") or {}).get("recording_fingerprint") or sa.get("recording_fingerprint")
    fp_b = (bb.get("meta") or {}).get("recording_fingerprint") or sb.get("recording_fingerprint")
    same_rec_fp = not (fp_a and fp_b and str(fp_a) != str(fp_b))

    same_ref = (
        same_rec_fp
        and sa["effective_reference"] == sb["effective_reference"]
        and sa["acquired_reference"] == sb["acquired_reference"]
    )
    eff_mode = (sa.get("effective_reference") or {}).get("mode", "original")
    if not same_rec_fp:
        ref_mismatch_reason = (
            f"Runs were computed on different recording snapshots (recording_fingerprint {str(fp_a)[:12]} vs {str(fp_b)[:12]}); "
            "per-channel rate subtraction and event-fate attribution are undefined. Inspect each run's own ranking below."
        )
    else:
        ref_mismatch_reason = "Reference derivations differ (Lesson L7); per-channel rate subtraction is undefined. Inspect each derivation's own ranking below."
    if same_ref and eff_mode != "original":
        shared_names = sorted(set(map_a) & set(map_b))
        for n in shared_names:
            ra, rb = map_a[n], map_b[n]
            if ra.get("channel_id") != rb.get("channel_id") or sorted(ra.get("parents") or []) != sorted(rb.get("parents") or []):
                same_ref = False
                ref_mismatch_reason = (
                    f"Effective {eff_mode} reference pool differs across runs (e.g. channel '{n}' derived from different parent channels); "
                    "per-channel rate subtraction is undefined. Inspect each run's own ranking below."
                )
                break
        if same_ref and eff_mode == "average":
            pool_a = sorted({pid for r in sa["channels"] for pid in (r.get("parents") or [])})
            pool_b = sorted({pid for r in sb["channels"] for pid in (r.get("parents") or [])})
            if pool_a != pool_b:
                same_ref = False
                ref_mismatch_reason = (
                    "Common-average reference (CAR) channel pool differs between runs; "
                    "per-channel rate subtraction is undefined. Inspect each run's own ranking below."
                )

    rate_key = "focal_rate_per_min" if metric == "focal" else "raw_rate_per_min"
    rank_key = "focal_rank" if metric == "focal" else "raw_rank"
    count_key = "focal_count" if metric == "focal" else "raw_count"

    if not same_ref:
        return {
            "mode": "compare_runs",
            "case": case_slug,
            "recording": rec_slug,
            "run_a": sanitize_slug(run_a),
            "run_b": sanitize_slug(run_b),
            "directly_comparable_rates": False,
            "reason": ref_mismatch_reason,
            "run_a_reference": sa["effective_reference"],
            "run_b_reference": sb["effective_reference"],
            "run_a_leading": sa["leading_focal_channels"][:10],
            "run_b_leading": sb["leading_focal_channels"][:10],
        }

    shared_measured = [
        n for n in sorted(set(map_a) & set(map_b))
        if map_a[n]["status"] == "measured" and map_b[n]["status"] == "measured"
    ]

    rho_focal = _spearman(
        [float(map_a[n]["focal_rate_per_min"] or 0.0) for n in shared_measured],
        [float(map_b[n]["focal_rate_per_min"] or 0.0) for n in shared_measured],
    )
    rho_raw = _spearman(
        [float(map_a[n]["raw_rate_per_min"] or 0.0) for n in shared_measured],
        [float(map_b[n]["raw_rate_per_min"] or 0.0) for n in shared_measured],
    )

    lead_a = [
        r["channel_name"]
        for r in sorted(
            (r for r in sa["channels"] if r.get(rank_key) is not None),
            key=lambda x: x[rank_key],
        )
    ]
    lead_b = [
        r["channel_name"]
        for r in sorted(
            (r for r in sb["channels"] if r.get(rank_key) is not None),
            key=lambda x: x[rank_key],
        )
    ]
    # SYMMETRIC UNION sorted by best rank in EITHER run (safe for subset-channel branches!)
    order = ChannelDisplayOrder(recording)
    active_union = sorted(
        dict.fromkeys(lead_a[:12] + lead_b[:12]),
        key=lambda n: (
            min(
                (map_a[n].get(rank_key) if n in map_a else None) or 9999,
                (map_b[n].get(rank_key) if n in map_b else None) or 9999,
            ),
            -((map_b[n].get(rate_key) if n in map_b else None) or (map_a[n].get(rate_key) if n in map_a else None) or 0.0),
            order.row_position(map_a.get(n) or map_b[n]),
        ),
    )

    rows: list[dict[str, Any]] = []
    for name in active_union[:top_k]:
        ra = map_a.get(name)
        rb = map_b.get(name)
        sample_r = ra or rb
        if sample_r is None:
            continue
        va = ra.get(rate_key) if ra else None
        vb = rb.get(rate_key) if rb else None
        rows.append({
            "channel": name,
            "anatomy": sample_r["anatomy"],
            "shaft": sample_r["shaft"],
            "display_shaft": sample_r.get("display_shaft"),
            "rank_a": ra.get(rank_key) if ra else None,
            "rank_b": rb.get(rank_key) if rb else None,
            "count_a": ra.get(count_key) if ra else None,
            "count_b": rb.get(count_key) if rb else None,
            "analyzed_seconds_a": ra["detector_analyzed_seconds"] if ra else 0.0,
            "analyzed_seconds_b": rb["detector_analyzed_seconds"] if rb else 0.0,
            "rate_a": va,
            "rate_b": vb,
            "rate_delta": round(float(vb) - float(va), 4) if (va is not None and vb is not None) else None,
        })

    # Coverage-attributed event fate (resolve relocated workspace paths cleanly!)
    det_b_p = store.resolve_workspace_artifact_path((bb.get("meta") or {}).get("detection_json_path"))
    det_b = json_load_safe(det_b_p) if det_b_p is not None else {}
    cov_b = det_b.get("valid_intervals", {})
    ids_a = {e["id"]: e for e in ba["events"]}
    ids_b = {e["id"]: e for e in bb["events"]}
    shared_ids = set(ids_a) & set(ids_b)
    removed_outside_cov = 0
    removed_inside_cov = 0
    for eid in set(ids_a) - shared_ids:
        ev = ids_a[eid]
        spans = cov_b.get(ev["channel_id"], [])
        inside = any(a <= ev["start_sample"] and ev["end_sample"] <= b for a, b in spans)
        if inside:
            removed_inside_cov += 1
        else:
            removed_outside_cov += 1

    return {
        "mode": "compare_runs",
        "case": case_slug,
        "recording": rec_slug,
        "run_a": sanitize_slug(run_a),
        "run_b": sanitize_slug(run_b),
        "metric": metric,
        "directly_comparable_rates": True,
        "config_diff": {
            "run_a_reason": ba["meta"].get("reason"),
            "run_b_reason": bb["meta"].get("reason"),
            "run_b_overrides": bb["config"].get("overrides", {}),
        },
        "summary_metrics": {
            "spearman_focal_rho": rho_focal,
            "spearman_raw_rho": rho_raw,
            "top5_a": lead_a[:5],
            "top5_b": lead_b[:5],
            "top5_shared": [c for c in lead_a[:5] if c in set(lead_b[:5])],
            "raw_events_a": sa["counts"]["total_raw_events"],
            "raw_events_b": sb["counts"]["total_raw_events"],
            "focal_events_a": sa["counts"]["non_burst_events"],
            "focal_events_b": sb["counts"]["non_burst_events"],
        },
        "event_fate": {
            "unchanged_exact_id": len(shared_ids),
            "excluded_by_changed_time_or_channel_mask": removed_outside_cov,
            "disappeared_within_analyzed_coverage": removed_inside_cov,
            "newly_detected_in_run_b": len(set(ids_b) - shared_ids),
        },
        "channels": rows,
    }


def compare_recordings(
    store: WorkbenchStore,
    case_slug: str,
    run_name: str = "baseline",
    top_k: int = 16,
    *,
    bundles: dict[str, dict[str, Any]] | None = None,
    anatomy_labels: dict[str, Any] | None = _UNSET,
) -> dict[str, Any]:
    """Cross-snapshot consistency comparing BOTH focal (<5 shafts) and raw detector rates."""
    if bundles is None:
        bundles = {
            r: store.load_run(case_slug, r, run_name)
            for r in store.list_recordings(case_slug) if store.has_run(case_slug, r, run_name)
        }
    recs = list(bundles)
    if not recs:
        raise ToolkitError("no_runs", f"Case {case_slug!r} has no recordings with run {run_name!r}.")

    if anatomy_labels is _UNSET:
        anatomy_labels = load_anatomy_labels(store, case_slug)
    matcher = AnatomyMatcher(anatomy_labels)
    maps = {r: {row["channel_name"]: row for row in bundles[r]["summary"]["channels"]} for r in recs}
    # Equal evidence ranks follow the first recording's montage, then any
    # additional channels from subsequent recordings.
    positions: dict[str, int] = {}
    display_shafts: dict[str, str | None] = {}
    for r in recs:
        rows = list(maps[r].values())
        order = None
        if store.recording_path(case_slug, r).is_file():
            order = ChannelDisplayOrder(store.load_recording(case_slug, r))
            rows.sort(key=order.row_position)
        for row in rows:
            name = row["channel_name"]
            if name not in positions:
                positions[name] = len(positions)
                display_shafts[name] = (order.depth_shaft(row.get("channel_id", ""), name, row.get("parents") or ())
                                        if order else row.get("display_shaft"))

    # Union of active channels (count > 0 in AT LEAST ONE recording — never polluting top list with all-zero channels!)
    active_focal_names = {
        name
        for r in recs
        for name, row in maps[r].items()
        if (row.get("focal_count") or 0) > 0
    }
    active_raw_names = {
        name
        for r in recs
        for name, row in maps[r].items()
        if (row.get("raw_count") or 0) > 0
    }
    candidate_names = active_focal_names or active_raw_names

    def _cross_sort_key(name: str) -> tuple[int, float, int]:
        in_how_many = sum(1 for r in recs if (maps[r].get(name, {}).get("focal_count") or 0) > 0)
        best_focal_rank = min((maps[r].get(name, {}).get("focal_rank") or 9999) for r in recs)
        max_focal_rate = max((maps[r].get(name, {}).get("focal_rate_per_min") or 0.0) for r in recs)
        return (-in_how_many, best_focal_rank - max_focal_rate * 0.01, positions[name])

    top_names = sorted(candidate_names, key=_cross_sort_key)[:top_k]
    ordered_names = sorted(top_names, key=positions.__getitem__)

    aligned: list[dict[str, Any]] = []
    for name in ordered_names:
        sample_row = next(maps[r][name] for r in recs if name in maps[r])
        per_rec: dict[str, Any] = {}
        for r in recs:
            row = maps[r].get(name)
            if not row or row["status"] != "measured":
                per_rec[r] = None
            else:
                per_rec[r] = {
                    "anatomy": matcher.match(name, row.get("contacts") or []),
                    "focal_rank": row["focal_rank"],
                    "focal_count": row["focal_count"],
                    "focal_rate_per_min": row["focal_rate_per_min"],
                    "raw_rank": row["raw_rank"],
                    "raw_count": row["raw_count"],
                    "raw_rate_per_min": row["raw_rate_per_min"],
                    "analyzed_seconds": row["detector_analyzed_seconds"],
                }
        aligned.append({
            "channel": name,
            "shaft": sample_row["shaft"],
            "display_shaft": display_shafts[name],
            "by_recording": per_rec,
        })

    pairwise: list[dict[str, Any]] = []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            ra, rb = recs[i], recs[j]
            ma, mb = maps[ra], maps[rb]
            shared = [n for n in sorted(set(ma) & set(mb)) if ma[n]["status"] == "measured" and mb[n]["status"] == "measured"]
            f_lead_a = [r["channel_name"] for r in bundles[ra]["summary"]["leading_focal_channels"][:5]]
            f_lead_b = [r["channel_name"] for r in bundles[rb]["summary"]["leading_focal_channels"][:5]]
            r_lead_a = [r["channel_name"] for r in bundles[ra]["summary"]["leading_raw_channels"][:5]]
            r_lead_b = [r["channel_name"] for r in bundles[rb]["summary"]["leading_raw_channels"][:5]]
            pairwise.append({
                "recording_a": ra,
                "recording_b": rb,
                "shared_measured_channels": len(shared),
                "focal_top5_a": f_lead_a,
                "focal_top5_b": f_lead_b,
                "focal_top5_shared": [c for c in f_lead_a if c in set(f_lead_b)],
                "raw_top5_a": r_lead_a,
                "raw_top5_b": r_lead_b,
                "raw_top5_shared": [c for c in r_lead_a if c in set(r_lead_b)],
                "raw_spearman_rho": _spearman(
                    [float(ma[n]["raw_rate_per_min"] or 0.0) for n in shared],
                    [float(mb[n]["raw_rate_per_min"] or 0.0) for n in shared],
                ),
            })

    return {
        "mode": "compare_recordings",
        "case": case_slug,
        "run_name": sanitize_slug(run_name),
        "recordings": recs,
        "recording_overview": {
            r: bundles[r]["summary"]["counts"] for r in recs
        },
        "pairwise": pairwise,
        "aligned_active_channels": aligned,
        "note": "Only channels with >0 events in at least one recording are ranked; 0-event measured channels are excluded from Top lists. Rates are never pooled across separate recordings.",
    }


def compute_overlay_concordance(
    store: WorkbenchStore,
    case_slug: str,
    rec_slug: str | None = None,
    run_name: str = "baseline",
    top_k: int = 5,
    *,
    overlay: dict[str, Any] | None = _UNSET,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare active focal HFO channels against externally supplied SOZ/Resected clinical annotations."""
    cslug = sanitize_slug(case_slug)
    if overlay is _UNSET:
        overlay = store.load_clinical_overlay(cslug)
    if not overlay:
        return {
            "mode": "overlay_concordance",
            "case": cslug,
            "available": False,
            "message": "No external clinical overlay (SOZ / Resected) attached to this case.",
        }
    recs = store.list_recordings(cslug)
    if not recs:
        raise ToolkitError("no_recordings", f"Case {cslug!r} has no recordings.")
    r = sanitize_slug(rec_slug) if rec_slug else recs[0]
    if bundle is None:
        bundle = store.load_run(cslug, r, run_name)
    summary = bundle["summary"]

    soz_set = {c.upper() for c in overlay.get("soz_channels", [])}
    res_set = {c.upper() for c in overlay.get("resected_channels", [])}
    irr_set = {c.upper() for c in overlay.get("irritative_channels", [])}

    def _matches(row: dict[str, Any], target_set: set[str]) -> bool:
        if not target_set:
            return False
        tokens = {row["channel_name"].upper()} | {str(ct).upper() for ct in row.get("contacts", [])}
        return bool(tokens & target_set)

    order = ChannelDisplayOrder(store.load_recording(cslug, r))
    measured = sorted((row for row in summary["channels"] if row["status"] == "measured"), key=order.row_position)
    all_focal_active = sorted(
        [row for row in summary["channels"] if row.get("focal_rank") is not None],
        key=lambda row: (row["focal_rank"], order.row_position(row)),
    )
    top_rows = all_focal_active[:top_k]

    top_in_soz = [row["channel_name"] for row in top_rows if _matches(row, soz_set)]
    top_in_res = [row["channel_name"] for row in top_rows if _matches(row, res_set)]
    active_in_soz = [row["channel_name"] for row in all_focal_active if _matches(row, soz_set)]
    measured_soz = [row["channel_name"] for row in measured if _matches(row, soz_set)]
    measured_res = [row["channel_name"] for row in measured if _matches(row, res_set)]

    union_soz_active = set(measured_soz) | {row["channel_name"] for row in all_focal_active}
    jaccard_soz = (
        round(len(active_in_soz) / len(union_soz_active), 4)
        if union_soz_active and measured_soz
        else None
    )

    all_rec_tokens: set[str] = set()
    per_channel_tags: dict[str, list[str]] = {}
    for row in summary["channels"]:
        all_rec_tokens.add(row["channel_name"].upper())
        all_rec_tokens.update(str(ct).upper() for ct in row.get("contacts", []))
        tags: list[str] = []
        if _matches(row, soz_set):
            tags.append("SOZ")
        if _matches(row, res_set):
            tags.append("Resected")
        if _matches(row, irr_set):
            tags.append("Irritative")
        if tags:
            per_channel_tags[row["channel_name"]] = tags

    unmatched_overlay = {
        "soz": [c for c in overlay.get("soz_channels", []) if str(c).upper() not in all_rec_tokens],
        "resected": [c for c in overlay.get("resected_channels", []) if str(c).upper() not in all_rec_tokens],
        "irritative": [c for c in overlay.get("irritative_channels", []) if str(c).upper() not in all_rec_tokens],
    }

    return {
        "mode": "overlay_concordance",
        "case": cslug,
        "recording": r,
        "run_name": sanitize_slug(run_name),
        "available": True,
        "overlay": overlay,
        "metrics": {
            "measured_soz_channels": len(measured_soz),
            "measured_resected_channels": len(measured_res),
            "focal_active_channels": len(all_focal_active),
            "top_k": top_k,
            "top_k_in_soz": top_in_soz,
            "top_k_in_resected": top_in_res,
            "soz_precision_at_top_k": round(len(top_in_soz) / len(top_rows), 4) if (top_rows and measured_soz) else None,
            "soz_recall_all_focal": round(len(active_in_soz) / len(measured_soz), 4) if measured_soz else None,
            "soz_focal_jaccard": jaccard_soz,
        },
        "per_channel_tags": per_channel_tags,
        "unmatched_overlay_channels": unmatched_overlay,
    }


def query_case(
    target: str,
    workspace_dir: str | Path | None = None,
    mode: str = "auto",
    run_a: str = "baseline",
    run_b: str | None = None,
    recording: str | None = None,
    channel: str | None = None,
    readers: list[str] | None = None,
    review_set: str | None = None,
    onset_seconds: float | None = None,
    measure: str = "ripple",
    epoch_seconds: float = 10,
    text: str | None = None,
    progress_cb=None,
) -> dict[str, Any]:
    from .io_readers import inspect_case, resolve_case_target

    store = WorkbenchStore(workspace_dir)
    if mode == "cohort_summary":
        cohorts = store.list_cohorts()
        if target and target.strip() not in {"*", "all", ""}:
            slug = sanitize_slug(target)
            match = next((c for c in cohorts if c.get("cohort_slug") == slug), None)
            if match is not None:
                return {"mode": "cohort_summary", **match}
        return {"mode": "cohort_summary", "cohorts": cohorts}

    spec = resolve_case_target(target, store)
    if spec["kind"] == "filesystem":
        insp = inspect_case(str(spec["path"]), workspace_dir=store.root, persist=True, progress_cb=progress_cb)
        cslug = insp["case"]
        rec_hint = recording
    elif spec["kind"] == "registered_recording":
        cslug = spec["case"]
        rec_hint = recording or spec["recording"]
    else:
        cslug = spec["case"]
        rec_hint = recording

    recs = store.list_recordings(cslug)
    if not recs:
        raise ToolkitError("no_recordings", f"Case {cslug!r} has no recordings.")

    if mode == "timeline":
        from .timeline import admission_timeline

        return admission_timeline(store, cslug, text=text)

    if mode == "trends":
        from .io_cases import navigation_recording_slug
        from .trends import recording_trends

        if not rec_hint and len(recs) != 1:
            raise ToolkitError("missing_recording", "Choose one recording for quantitative trends.")
        slug = navigation_recording_slug(store, cslug, rec_hint or recs[0])
        # Browsing a file can lead directly to a segment run, with no baseline.
        trend_run = None if measure == "hfo_rate" and run_a == "baseline" and not store.has_run(cslug, slug, run_a) else run_a
        return recording_trends(store, cslug, slug, measure=measure, epoch_seconds=epoch_seconds,
                                run=trend_run, compact=True, progress_cb=progress_cb)

    if mode == "reader_agreement":
        from .agreement import reader_agreement

        if not rec_hint and len(recs) != 1:
            raise ToolkitError("missing_recording", "Choose one recording for reader agreement.")
        return {"mode": mode, **reader_agreement(store, cslug, rec_hint or recs[0], run_a, readers, review_set)}

    if mode == "evidence_map":
        from .evidence_map import get_evidence_map

        return get_evidence_map(store, cslug, recording=rec_hint, run_name=run_a)

    if mode == "epileptogenicity":
        from .evidence_map import get_epileptogenicity

        return get_epileptogenicity(store, cslug, recording=rec_hint, onset_seconds=onset_seconds)

    if mode == "overlay_concordance":
        r = sanitize_slug(rec_hint) if rec_hint else recs[0]
        return compute_overlay_concordance(store, cslug, rec_slug=r, run_name=run_a)

    if mode == "control_windows":
        r = sanitize_slug(rec_hint) if rec_hint else recs[0]
        return {
            "mode": "control_windows",
            "case": cslug,
            "recording": r,
            "run_name": run_a,
            "windows": select_control_windows(store, cslug, r, run_name=run_a, channel=channel),
        }

    if mode == "compare_runs" or (mode == "auto" and run_b):
        if not run_b:
            raise ToolkitError("missing_run_b", "Specify run_b to compare two runs.")
        r = sanitize_slug(rec_hint) if rec_hint else next((x for x in recs if store.has_run(cslug, x, run_b)), recs[0])
        return compare_runs(store, cslug, r, run_a, run_b)

    if mode == "compare_recordings" or (mode == "auto" and len(recs) > 1 and not rec_hint):
        anatomy_labels = load_anatomy_labels(store, cslug)
        bundles = {
            r: store.load_run(cslug, r, run_a)
            for r in recs if store.has_run(cslug, r, run_a)
        }
        per_rec = []
        for r, bundle in bundles.items():
            s = annotate_summary_anatomy(bundle["summary"], anatomy_labels, recording=store.load_recording(cslug, r))
            per_rec.append({
                "recording": r,
                "counts": s["counts"],
                "leading_focal": s["leading_focal_channels"][:6],
                "synchronous_burst_windows": s["synchronous_burst_windows"][:6],
                "anatomy_coverage": s["anatomy_coverage"],
                "warnings": s["warnings"],
            })
        return {
            "mode": "case_overview",
            "case": cslug,
            "run_name": run_a,
            "per_recording": per_rec,
            "cross_recording": compare_recordings(store, cslug, run_name=run_a, bundles=bundles, anatomy_labels=anatomy_labels),
        }

    r = sanitize_slug(rec_hint) if rec_hint else recs[0]
    bundle = store.load_run(cslug, r, run_a)
    recording = store.load_recording(cslug, r)
    s = annotate_summary_anatomy(bundle["summary"], load_anatomy_labels(store, cslug), recording=recording)
    return {
        "mode": "recording_summary",
        "case": cslug,
        "recording": r,
        "run_name": run_a,
        "provenance": recording.provenance.model_dump(mode="json"),
        **({"segments": s["segments"]} if "segments" in s else {}),
        "counts": s["counts"],
        "channels": s["channels"],
        "anatomy_coverage": s["anatomy_coverage"],
        "leading_focal_channels": s["leading_focal_channels"][:12],
        "leading_raw_channels": s["leading_raw_channels"][:12],
        "synchronous_burst_windows": s["synchronous_burst_windows"][:10],
        "quiet_measured_channels_count": len(s["quiet_measured_channel_names"]),
        "unassessed_channels": s["unassessed_channel_names"],
        "spatial_available": s["spatial"]["available"],
        "warnings": s["warnings"],
        "run_dir": bundle["run_dir"],
    }
