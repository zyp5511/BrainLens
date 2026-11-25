"""On-demand, recording-bound evidence. Published analysis generations are read-only."""

from __future__ import annotations

import math
import re
import tempfile
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock

from .ictal import EI_DEFAULTS, EI_VERSION, compute_ei, resolve_onset
from .io_integrity import verified_window_source, verify_raw_filenames
from .io_signals import open_mne_raw, read_scaled_volts
from .analysis_windows import prepared_windows
from .channel_order import ChannelDisplayOrder
from .models import (
    AnalysisConfig, ChannelSpec, RecordingSpec, ToolkitError, atomic_write_json, canonical_digest,
    read_json,
)
from .signal_prep import _bad_marker_spans, compute_qc, prepare_signals
from .spikes import IED_DEFAULTS, IED_VERSION, detect_ieds
from .storage import WorkbenchStore
from .storage_mvcc import run_analysis_signature, sanitize_slug
from .triad_views import _verify_run_recording_binding

MAP_VERSION = "1"
SCOPE = "Research and review only — not a medical device / not a standalone surgical resection boundary."
CONVERGENCE = "Convergence count — descriptive, not a validated predictor"
TOP_TIER_RULE = (
    "Count each available biomarker with a positive value at or above the 80th percentile. "
    "Rank each biomarker within its source recording. Ties share the best tied rank. "
    "Measured zeros have percentile 0. External annotations do not count."
)
COLUMNS = (
    ("ripple", "Focal ripple", "events/min"),
    ("fast_ripple", "Focal fast ripple", "events/min"),
    ("spike_hfo", "ONNX spike-HFO", "events/min"),
    ("ied", "IED candidates — require visual verification", "candidates/min"),
    ("ei", "Ictal Epileptogenicity Index", "0–1"),
)


def _recording(store: WorkbenchStore, case: str, recording: str | None) -> RecordingSpec:
    if not recording:
        recordings = store.list_recordings(case)
        if len(recordings) != 1:
            raise ToolkitError("recording_required", "Select a registered recording.", recordings=recordings)
        recording = recordings[0]
    return store.load_recording(case, recording)


def _cell(value: float | None, reason: str | None = None, *, state: str = "unavailable") -> dict[str, Any]:
    if value is not None:
        state = "measured" if value > 0 else "quiet_measured"
    return {"state": state, "value": value, "reason": reason, "percentile": None, "rank": None}


def _rank(cells: list[dict[str, Any]]) -> None:
    values = sorted((c["value"] for c in cells if c["value"] is not None), reverse=True)
    ranks = {}
    for rank, value in enumerate(values, start=1):
        ranks.setdefault(value, rank)
    for cell in cells:
        v = cell["value"]
        if v is not None:
            rank = ranks[v]
            cell["rank"] = rank
            if v == 0:
                cell["percentile"] = 0.0
            elif len(values) == 1:
                cell["percentile"] = 100.0
            else:
                cell["percentile"] = 100.0 * (len(values) - rank) / (len(values) - 1)


def _cache_dir(store: WorkbenchStore, case: str, rec: RecordingSpec) -> Path:
    # Namespace the active selection as well as the immutable results by recording identity.
    identity = canonical_digest([rec.fingerprint, rec.source_sha256, EI_VERSION])
    return store.recording_dir(case, rec.slug) / "evidence" / "ei" / identity


def get_latest_ei(store: WorkbenchStore, case_id: str, recording: str | None = None) -> dict | None:
    rec = _recording(store, case_id, recording)
    directory = _cache_dir(store, case_id, rec)
    active = directory / "active.json"
    if active.is_file():
        key = read_json(active).get("cache_key", "")
        if re.fullmatch(r"[0-9a-f]{64}", key):
            path = directory / f"{key}.json"
            if path.is_file():
                return read_json(path)
    return None


def _nyquist_lowpass(ch: ChannelSpec, rec: RecordingSpec, run_lowpass: float | None = None) -> tuple[float, float | None]:
    nyquist = min(rec.sfreq, ch.source_sfreq or rec.sfreq) / 2
    limits = [v for v in (rec.lowpass, ch.source_lowpass, run_lowpass) if v is not None and v > 0]
    return nyquist, min(limits) if limits else None


def _ceiling(ch: ChannelSpec, rec: RecordingSpec, run_lowpass: float | None = None) -> float:
    nyquist, lowpass = _nyquist_lowpass(ch, rec, run_lowpass)
    return min(nyquist, lowpass) if lowpass is not None else nyquist


def get_epileptogenicity(
    store: WorkbenchStore, case_id: str, recording: str | None = None,
    onset_seconds: float | None = None, run_name: str | None = None, *,
    cached_only: bool = False, before_seconds: float = 30.0, after_seconds: float = 60.0,
    parameters: dict[str, float] | None = None,
) -> dict | None:
    """Compute EI on the acquired montage. A registered source suffices without any HFO run."""
    rec = _recording(store, case_id, recording)
    onset = resolve_onset(onset_seconds, rec.markers)
    if not 0 <= onset < rec.duration:
        raise ToolkitError("invalid_onset", "Seizure onset must lie within the recording.", duration=rec.duration)
    if not all(math.isfinite(x) and x > 0 for x in (before_seconds, after_seconds)):
        raise ToolkitError("invalid_ei_window", "EI windows must have finite positive lengths.")
    if parameters and (parameters.keys() - EI_DEFAULTS.keys()):
        raise ToolkitError("invalid_ei_parameters", "Unknown EI parameter.", parameters=sorted(parameters))
    params = {**EI_DEFAULTS, **(parameters or {})}
    if any(not isinstance(v, Real) or not math.isfinite(v) for v in params.values()):
        raise ToolkitError("invalid_ei_parameters", "EI parameters must be finite numbers.")
    params = {name: float(value) for name, value in params.items()}
    # EI needs one uninterrupted peri-onset baseline and one post-onset segment.
    candidates = rec.time_slices(onset - before_seconds, onset + after_seconds)
    containing = next(((p, a, b) for p, a, b in candidates
                       if p.onset_seconds <= onset < p.time_at(p.stop_sample, rec.sfreq)), None)
    if containing is None:
        raise ToolkitError("invalid_onset", "The selected seizure onset falls in a recording gap.")
    piece, start, stop = containing
    onset_source = "explicit" if onset_seconds is not None else "global_marker"
    identity = {
        "recording_fingerprint": rec.fingerprint, "source_sha256": rec.source_sha256,
        "module_version": EI_VERSION, "parameters": params, "samples": [start, stop],
        "onset_seconds": onset, "onset_source": onset_source,
    }
    key = canonical_digest(identity)
    directory = _cache_dir(store, case_id, rec)
    path = directory / f"{key}.json"
    if cached_only:
        return read_json(path) if path.is_file() else None
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.with_suffix(".lock"))):
        if path.is_file():
            result = read_json(path)
        else:
            with verified_window_source(rec) as (source, snapshot, _, signature):
                with open_mne_raw(source, preload=False) as raw:
                    verify_raw_filenames(rec, raw.filenames, snapshot, signature,
                                         opened_stat=getattr(raw, "_brainlens_fd_stat", None))
                    data = read_scaled_volts(raw, rec, start, stop)
            valid = np.isfinite(data)
            for i, ch in enumerate(rec.channels):
                if ch.is_auxiliary or ch.unit != "V":
                    valid[i] = False
                else:
                    for a, b in _bad_marker_spans(rec, ch):
                        if a < stop and b > start:
                            valid[i, max(a, start) - start:min(b, stop) - start] = False
            result = compute_ei(
                data, valid, rec.sfreq, [ch.name for ch in rec.channels], onset_seconds=onset,
                start_seconds=piece.time_at(start, rec.sfreq), lowpass_hz=[_ceiling(ch, rec) for ch in rec.channels], **params,
            )
            ranked = [_cell(row["ei"]) for row in result["channels"]]
            _rank(ranked)
            for row, stats, ch in zip(result["channels"], ranked, rec.channels, strict=True):
                row.update(rank=stats["rank"], percentile=stats["percentile"], contacts=ch.contacts)
            result.update(
                **identity, cache_key=key, mode="epileptogenicity", case=case_id, recording_id=rec.slug,
                window_seconds=[piece.time_at(start, rec.sfreq), piece.time_at(stop, rec.sfreq)], reference=rec.reference,
                provenance=rec.provenance.model_dump(mode="json"), scope=SCOPE,
            )
            atomic_write_json(path, result)
        atomic_write_json(directory / "active.json", {"cache_key": key})
    return result


def _band_reason(key: str, ch: ChannelSpec, rec: RecordingSpec, run_lp: float | None,
                 detector_band: list[float]) -> str | None:
    nyquist, lowpass = _nyquist_lowpass(ch, rec, run_lp)
    lo, hi = (80, 250) if key == "ripple" else (250, 450)
    # Use the ceiling and fast-ripple margin from analytics._annotate_event_spectral_metrics.
    minimum_hi = lo if key == "ripple" else 270
    bandwidth = min(0.92 * nyquist, lowpass) if lowpass is not None else 0.92 * nyquist
    if bandwidth <= minimum_hi:
        if lowpass is not None and lowpass < 0.92 * nyquist:
            return f"{lo}–{hi} Hz exceeds effective lowpass ({lowpass:g} Hz)"
        return f"{lo}–{hi} Hz exceeds channel native Nyquist ({nyquist:g} Hz)"
    if detector_band and (detector_band[1] <= lo or detector_band[0] >= hi):
        return f"Outside the selected HFO detector band ({detector_band[0]:g}–{detector_band[1]:g} Hz)"
    return None


def _artifact(store: WorkbenchStore, path: str | None, recording_dir: Path, fallback: Path | None = None) -> Path | None:
    candidate = store.resolve_workspace_artifact_path(path, fallback_dir=fallback)
    if candidate is not None and not candidate.resolve().is_relative_to(recording_dir):
        raise ToolkitError("prepared_data_unavailable", "Prepared artifact belongs to a different recording directory.")
    return candidate


def _base_rows(store: WorkbenchStore, case: str, rec: RecordingSpec, bundle: dict) -> list[dict]:
    rec_dir = store.recording_dir(case, rec.slug)
    path = _artifact(store, bundle["meta"].get("prepared_json_path"), rec_dir)
    if path is not None:
        prep = read_json(path)
        if prep.get("recording_fingerprint", rec.fingerprint) != rec.fingerprint:
            raise ToolkitError("run_recording_mismatch", "Prepared signal belongs to a different recording.")
        data_path = _artifact(store, prep.get("data_path"), rec_dir, path.parent)
        valid_path = _artifact(store, prep.get("valid_path"), rec_dir, path.parent)
        if data_path is not None and valid_path is not None:
            return _measure_rows(rec, bundle, prep, data_path, valid_path)

    # Older workspaces may have evicted the arrays. Rebuild the same selection and montage
    # in a private temporary directory from a verified source, without touching run artifacts.
    config_doc = bundle["config"]
    config = AnalysisConfig.model_validate(config_doc.get("analysis_config", config_doc))
    with tempfile.TemporaryDirectory(prefix="brainlens-evidence-") as temporary:
        scratch = Path(temporary)
        qc = bundle["qc"]
        if "invalid_intervals" not in qc:
            qc = compute_qc(rec, config, scratch / "qc")
        prep = prepare_signals(rec, config, qc, scratch / "prepare")
        return _measure_rows(rec, bundle, prep, Path(prep["data_path"]), Path(prep["valid_path"]))


def _measure_rows(rec: RecordingSpec, bundle: dict, prep: dict, data_path: Path, valid_path: Path) -> list[dict]:
    data = np.load(data_path, mmap_mode="r", allow_pickle=False)
    valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
    channels = [ChannelSpec.model_validate(ch) for ch in prep["channels"]]
    if data.shape != valid.shape or data.shape != (len(channels), prep["n_samples"]):
        raise ToolkitError("prepared_data_unavailable", "Prepared data dimensions differ from the manifest.")
    windows = prepared_windows(prep)
    run_lp = (prep.get("effective_filters") or {}).get("lowpass")
    run_hp = (prep.get("effective_filters") or {}).get("highpass")
    summary = bundle["summary"]
    rates = {r["channel_id"]: r for r in summary["channels"]}
    detector_band = (bundle["config"].get("analysis_config", {}).get("detector") or {}).get("band", [])
    spike_stage = ((summary.get("classification") or {}).get("stages") or {}).get("spike") or {}
    analyzed_ids = {ch.id for ch in channels}
    # Raw contact subsets and omitted auxiliary inputs remain visible as unassessed.
    original = (prep.get("reference") or {}).get("mode", "original") == "original"
    display = channels + [ch for ch in rec.channels if ch.id not in analyzed_ids and (original or ch.is_auxiliary)]
    rows = []
    for i, ch in enumerate(display):
        rate = rates.get(ch.id, {})
        seconds = rate.get("detector_analyzed_seconds") or 0
        reason = None
        if ch.is_auxiliary or ch.unit != "V":
            reason = "Auxiliary channel or unverified voltage units"
        elif ch.id not in analyzed_ids:
            reason = "Outside the run's selected channels"
        elif not rate.get("valid_signal_seconds"):
            reason = "No valid prepared signal coverage"
        row = {
            "channel": ch.name, "channel_id": ch.id, "contacts": ch.contacts, "state": "unassessed",
            "unassessed_reason": reason, "detector_analyzed_seconds": seconds,
            "valid_signal_seconds": rate.get("valid_signal_seconds", 0), "biomarkers": {},
        }
        cells = row["biomarkers"]
        if reason:
            cells.update({k: _cell(None, reason, state="unassessed") for k, _, _ in COLUMNS[:-1]})
        else:
            counts = {"fast_ripple": rate.get("fast_ripple_count"), "spike_hfo": rate.get("spike_hfo_count")}
            focal = rate.get("focal_count")
            counts["ripple"] = (focal - counts["fast_ripple"] if focal is not None and
                                counts["fast_ripple"] is not None else None)
            for k, count in counts.items():
                if k == "spike_hfo":
                    why = None if spike_stage.get("status") == "complete" else "ONNX spike classifier did not complete"
                else:
                    why = _band_reason(k, ch, rec, run_lp, detector_band)
                if not why and not seconds:
                    why = "No HFO detector coverage"
                if not why and count is None:
                    why = "Biomarker count unavailable for this run"
                cells[k] = _cell(None, why) if why else _cell(60.0 * count / seconds)
            if max(run_hp or 0.0, rec.highpass or 0.0) > IED_DEFAULTS["low_hz"]:
                cells["ied"] = _cell(None, "High-pass filter excludes the IED detection band")
            else:
                # Use contiguous mmap row views, so a long run never requires a full signal copy.
                results = []
                for window in windows:
                    samples = slice(window.array_start, window.array_stop)
                    result = detect_ieds(
                        data[i:i + 1, samples], valid[i:i + 1, samples], prep["sfreq"], [ch.name],
                        start_seconds=window.piece.time_at(window.start, rec.sfreq),
                        lowpass_hz=_ceiling(ch, rec, run_lp),
                    )["channels"][0]
                    results.append(result)
                seconds = sum(r["valid_seconds"] for r in results)
                times = [t for r in results for t in r["event_times"]]
                cells["ied"] = _cell(len(times) * 60 / seconds if seconds else None,
                                     None if seconds else results[0].get("reason"), state="unassessed")
                cells["ied"].update(count=len(times) if seconds else None, event_times=times,
                                    valid_seconds=seconds, band_hz=results[0]["band_hz"])
        rows.append(row)
    for key, _, _ in COLUMNS[:-1]:
        _rank([r["biomarkers"][key] for r in rows])
    return rows


def _ei_source(store: WorkbenchStore, case: str, recording: str) -> tuple[dict | None, list[str]]:
    """Prefer this recording, then a single named, deterministic source. Never combine seizures."""
    sources = []
    for slug in store.list_recordings(case):
        rec = store.load_recording(case, slug)
        if (_cache_dir(store, case, rec) / "active.json").is_file():
            sources.append(slug)
    if not sources:
        return None, sources
    selected = recording if recording in sources else sources[0]
    return get_latest_ei(store, case, selected), sources


def get_evidence_map(
    store: WorkbenchStore, case_id: str, recording: str | None = None,
    run_name: str | None = None, *, cached_only: bool = False,
) -> dict | None:
    """Read or materialize run-local rates; align current external tags and one cached ictal result."""
    rec = _recording(store, case_id, recording)
    run = sanitize_slug(run_name or "baseline")
    directory = store.run_dir(case_id, rec.slug, run) / "evidence" / "map"
    if cached_only and not directory.is_dir():
        return None
    bundle = store.load_run(case_id, rec.slug, run, reconcile=False)
    _verify_run_recording_binding(rec, bundle, run)
    meta = bundle["meta"]
    analysis_id = meta.get("analysis_generation_id") or meta.get("generation_id")
    analysis_sig = run_analysis_signature(bundle)
    prepare_fp = (meta.get("stage_fingerprints") or {}).get("prepare")
    identity = {
        "module_version": MAP_VERSION, "recording_fingerprint": rec.fingerprint,
        "prepare_fingerprint": prepare_fp, "source_sha256": rec.source_sha256,
        "analysis_signature": analysis_sig, "analysis_generation_id": analysis_id,
        "ied_version": IED_VERSION, "ied_parameters": IED_DEFAULTS,
    }
    key = canonical_digest(identity)
    path = directory / f"{key}.json"
    if path.is_file():
        rows = read_json(path)["rows"]
    elif cached_only:
        return None
    else:
        directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path.with_suffix(".lock"))):
            if path.is_file():
                rows = read_json(path)["rows"]
            else:
                rows = _base_rows(store, case_id, rec, bundle)
                atomic_write_json(path, {**identity, "rows": rows})
    ei, sources = _ei_source(store, case_id, rec.slug)
    ei_rows = {r["channel"]: r for r in ei["channels"]} if ei else {}
    source = None
    if ei:
        source = {k: ei[k] for k in ("recording_id", "recording_fingerprint", "cache_key",
                                      "onset_seconds", "onset_source", "provenance", "reference")}
    overlay = store.load_clinical_overlay(case_id)
    tags = {label: {ch.upper() for ch in (overlay or {}).get(field, [])}
            for label, field in (("SOZ", "soz_channels"), ("Resected", "resected_channels"))}
    reference_mode = (bundle["summary"].get("effective_reference") or {}).get("mode", "original")
    bipolar = reference_mode == "bipolar" or (reference_mode == "original" and rec.reference == "bipolar")
    for row in rows:
        measures = row["biomarkers"]
        ei_row = ei_rows.get(row["channel"])
        if row["unassessed_reason"]:
            cell = _cell(None, row["unassessed_reason"], state="unassessed")
        elif not ei:
            cell = _cell(None, "No cached ictal analysis for this case")
        elif not ei_row:
            cell = _cell(None, "No matching acquired channel in the ictal recording")
        elif bipolar != (ei["reference"] == "bipolar") or (
            row["contacts"] and ei_row.get("contacts") and row["contacts"] != ei_row["contacts"]
        ):
            cell = _cell(None, "Contact definitions or bipolar montages differ between recordings")
        else:
            cell = _cell(ei_row["ei"], ei_row.get("reason"), state="unassessed")
            cell.update(rank=ei_row["rank"], percentile=ei_row["percentile"])
        measures["ei"] = cell
        tokens = {row["channel"].upper(), *(c.upper() for c in row["contacts"])}
        row["external_tags"] = [name for name, contacts in tags.items() if contacts & tokens]
        available = [c for c in measures.values() if c["value"] is not None]
        row["available_count"] = len(available)
        row["convergence_count"] = (sum(c["value"] > 0 and c["percentile"] >= 80 for c in available)
                                    if available else None)
        if available:
            row["state"] = "measured" if any(c["value"] > 0 for c in available) else "quiet_measured"
    overlay_digest = canonical_digest(overlay)
    provenance = rec.provenance.model_dump(mode="json")
    order = ChannelDisplayOrder(rec)
    parents = {ch["channel_id"]: ch.get("parents") or () for ch in bundle["summary"]["channels"]}
    rows.sort(key=lambda row: order.position(row["channel_id"], row["channel"], parents.get(row["channel_id"], ())))
    return {
        **identity, "mode": "evidence_map", "case": case_id, "recording_id": rec.slug, "run_name": run,
        "generation_id": meta.get("generation_id"),
        "provenance": provenance, "selection": bundle["summary"].get("selection"),
        "reference": bundle["summary"].get("effective_reference_label"), "ei_source": source,
        "available_ei_recordings": sources, "clinical_overlay_digest": overlay_digest,
        "external_overlay_available": bool(overlay), "external_overlay_source": (overlay or {}).get("source"),
        "scope": SCOPE, "convergence_label": CONVERGENCE,
        "evidence_digest": canonical_digest([key, provenance, source, sources, overlay_digest]),
        "top_tier_rule": TOP_TIER_RULE, "rows": rows,
        "columns": [
            {"key": k, "label": label, "unit": unit,
             "recording_id": (source["recording_id"] if source else None) if k == "ei" else rec.slug}
            for k, label, unit in COLUMNS
        ],
    }


def get_interictal_leaders(store: WorkbenchStore, case_id: str, ictal_recording_id: str) -> dict | None:
    """Return a named comparison run with focal HFOs, preserving uncertainty about unmarked state."""
    fallback = None
    for slug in store.list_recordings(case_id):
        rec = store.load_recording(case_id, slug)
        has_seizure = slug == ictal_recording_id or (_cache_dir(store, case_id, rec) / "active.json").is_file()
        if not has_seizure:
            try:
                resolve_onset(None, rec.markers)
            except ToolkitError as exc:
                if exc.code != "seizure_onset_unavailable":
                    raise
            else:
                has_seizure = True
        runs = sorted(store.list_runs(case_id, slug, reconcile=False),
                      key=lambda m: (m["run_name"] != "baseline", m["run_name"]))
        for meta in runs:
            if meta.get("recording_fingerprint") != rec.fingerprint:
                continue
            bundle = store.load_run(case_id, slug, meta["run_name"], reconcile=False)
            _verify_run_recording_binding(rec, bundle, meta["run_name"])
            summary = bundle["summary"]
            state = (summary.get("selection") or {}).get("state") or ""
            state_tokens = set(re.findall(r"[a-z]+", state.casefold()))
            marked_interictal = "interictal" in state_tokens
            if has_seizure and not marked_interictal:
                continue
            if state_tokens & {"ictal", "seizure", "sz", "postictal", "preictal"}:
                continue
            eligible = [r for r in summary["channels"] if not r.get("is_auxiliary")
                        and r.get("detector_analyzed_seconds", 0) and r.get("focal_rate_per_min") is not None]
            if eligible:
                order = ChannelDisplayOrder(rec)
                ranked = sorted(eligible, key=lambda r: (-r["focal_rate_per_min"], order.row_position(r)))
                comparison = {
                    "recording_id": slug, "run_name": meta["run_name"],
                    "state": state if marked_interictal else " · ".join(filter(None, (state, "interictal state unverified"))),
                    "provenance": rec.provenance.model_dump(mode="json"),
                    "leaders": [{"channel": r["channel_name"], "value": r["focal_rate_per_min"]}
                                for r in ranked if r["focal_rate_per_min"] > 0][:10],
                }
                if marked_interictal:
                    return comparison
                if fallback is None:
                    fallback = comparison
    return fallback
