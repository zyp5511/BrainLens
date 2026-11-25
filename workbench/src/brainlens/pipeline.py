"""Analysis pipeline & branch execution with automatic QC/Prepare/Detect stage reuse."""

from __future__ import annotations

import errno
import json
import os
import secrets
import shutil
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout as FileLockTimeout

from .analytics import summarize_run
from .anatomy import AnatomyMatcher, load_anatomy_labels
from .io_readers import inspect_case, resolve_case_target, verify_source_recording_integrity
from .models import (
    AnalysisConfig,
    AnalysisSegment,
    BipolarPair,
    DetectorConfig,
    ExclusionSpec,
    FilterConfig,
    QCConfig,
    RecordingSpec,
    ReferenceConfig,
    ToolkitError,
    atomic_write_json,
    canonical_digest,
    get_env,
    read_json,
    utc_now,
)
from .protocols import load_protocol_config
from .pyhfo_runner import PyHFORunner, resolve_engine_settings
from .signal_prep import check_analysis_window, compute_qc, prepare_signals, resolve_channel
from .storage import WorkbenchStore, release_gc_lock, run_analysis_signature, sanitize_slug, try_lock_generation_for_gc


def _auto_branch_slug(overrides: dict[str, Any] | None, protocol: str | None) -> str:
    if protocol and not overrides:
        return sanitize_slug(f"proto_{protocol}")
    if not overrides:
        return "baseline"
    parts: list[str] = []
    if protocol:
        parts.append(f"proto_{protocol}")
    needs_hash = False
    if overrides.get("start_seconds") is not None or overrides.get("end_seconds") is not None:
        s0 = overrides.get("start_seconds", 0)
        s1 = overrides.get("end_seconds")
        s0_txt = f"{float(s0):g}"
        s1_txt = f"{float(s1):g}" if s1 is not None else "end"
        parts.append(f"t{s0_txt}_{s1_txt}s")
    if overrides.get("exclude_intervals"):
        sp = overrides["exclude_intervals"][0]
        parts.append(f"excl_{int(sp.get('start', 0))}s_{int(sp.get('end', 0))}s")
        if len(overrides["exclude_intervals"]) > 1 or sp.get("channels"):
            needs_hash = True
    if overrides.get("exclude_channels"):
        parts.append("excl_" + "_".join(str(c) for c in overrides["exclude_channels"][:2]))
        if len(overrides["exclude_channels"]) > 2:
            needs_hash = True
    if overrides.get("reference_mode") and overrides["reference_mode"] != "original":
        parts.append(f"ref_{overrides['reference_mode']}")
    if isinstance(overrides.get("detector"), str):
        parts.append(overrides["detector"].lower())
    elif isinstance(overrides.get("detector"), dict):
        parts.append(str(overrides["detector"].get("name", "det")).lower())
        needs_hash = True
    if overrides.get("rms_thres") is not None:
        parts.append(f"rms_{overrides['rms_thres']}")
    if overrides.get("band"):
        b = overrides["band"]
        parts.append(f"band_{int(b[0])}_{int(b[1])}")
    if overrides.get("classifier_enabled") is True:
        parts.append("onnx_clf")
    if overrides.get("state"):
        parts.append(f"state_{overrides['state']}")

    readable_keys = {
        "start_seconds",
        "end_seconds",
        "exclude_intervals",
        "exclude_channels",
        "reference_mode",
        "detector",
        "rms_thres",
        "band",
        "classifier_enabled",
        "state",
        "reason",
    }
    if any(k not in readable_keys for k in overrides):
        needs_hash = True
    if not parts:
        parts.append("branch")
        needs_hash = True
    if needs_hash:
        parts.append(canonical_digest({"protocol": protocol, "overrides": overrides})[:6])
    return sanitize_slug("_".join(parts))


_DETECTOR_ALLOWED_PARAMS: dict[str, frozenset[str]] = {
    "STE": frozenset({"epoch_len", "min_gap", "min_osc", "min_window", "peak_thres", "rms_thres", "rms_window"}),
    "MNI": frozenset({
        "base_min", "base_seg", "base_shift", "base_thrd",
        "epo_CHF", "epoch_time", "min_gap", "min_win", "per_CHF", "thrd_perc",
    }),
    "HIL": frozenset({"epoch_time", "min_window", "sd_threshold"}),
}
_DETECTOR_PARAM_ALIASES: dict[str, dict[str, str]] = {
    "STE": {"min_win": "min_window", "epoch_time": "epoch_len"},
    "MNI": {"min_window": "min_win", "epoch_len": "epoch_time", "percentile": "thrd_perc"},
    "HIL": {"min_win": "min_window", "epoch_len": "epoch_time"},
}
_FLAT_DETECTOR_PARAM_KEYS: frozenset[str] = frozenset().union(
    *_DETECTOR_ALLOWED_PARAMS.values(),
    *(d.keys() for d in _DETECTOR_PARAM_ALIASES.values()),
)
_INT_DETECTOR_PARAMS: frozenset[str] = frozenset({"min_osc", "epoch_len", "epoch_time", "epo_CHF", "base_min"})
_ALLOWED_OVERRIDE_KEYS: frozenset[str] = frozenset({
    "start_seconds",
    "end_seconds",
    "segments",
    "marker_segments",
    "state",
    "channels",
    "exclude_intervals",
    "exclude_channels",
    "filters",
    "highpass",
    "lowpass",
    "notch",
    "qc",
    "reference_mode",
    "bipolar_pairs",
    "average_channels",
    "detector",
    "detector_enabled",
    "band",
    "detector_params",
    "classifier_enabled",
    "classifier_stages",
    "classifier_thresholds",
    "reason",
}) | _FLAT_DETECTOR_PARAM_KEYS


def _normalize_detector_params(detector_name: str, raw_params: dict[str, Any]) -> dict[str, Any]:
    det_key = str(detector_name).upper()
    allowed = _DETECTOR_ALLOWED_PARAMS.get(det_key)
    if allowed is None:
        raise ToolkitError("invalid_detector", f"Unsupported detector {detector_name!r}; expected STE, MNI, or HIL.")
    aliases = _DETECTOR_PARAM_ALIASES.get(det_key, {})
    out: dict[str, Any] = {}
    for k, v in raw_params.items():
        if v is None:
            continue
        canon = aliases.get(str(k), str(k))
        if canon not in allowed:
            raise ToolkitError(
                "invalid_detector_parameter",
                f"Detector {det_key} does not support parameter {k!r}. Supported parameters: {', '.join(sorted(allowed))}.",
                detector=det_key,
                parameter=str(k),
                supported_parameters=sorted(allowed),
            )
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ToolkitError(
                "invalid_detector_parameter",
                f"Detector parameter {canon!r} must be numeric, got {type(v).__name__}.",
                detector=det_key,
                parameter=canon,
            )
        val_f = float(v)
        out[canon] = int(val_f) if (canon in _INT_DETECTOR_PARAMS and val_f.is_integer()) else val_f
    return out


def apply_config_overrides(
    recording: RecordingSpec,
    base: AnalysisConfig,
    overrides: dict[str, Any] | None,
) -> AnalysisConfig:
    cfg = base.model_copy(deep=True)
    if not overrides:
        return cfg
    ov = dict(overrides)
    unknown_keys = sorted(set(ov.keys()) - _ALLOWED_OVERRIDE_KEYS)
    if unknown_keys:
        raise ToolkitError(
            "unknown_override_key",
            f"Unsupported override key(s): {', '.join(unknown_keys)}.",
            unknown_keys=unknown_keys,
        )

    selections = {"segments", "marker_segments"}.intersection(ov)
    if selections:
        if len(selections) > 1 or "start_seconds" in ov or "end_seconds" in ov:
            raise ToolkitError("invalid_segments", "Choose a single window, explicit segments, or a marker selection.")
        values = ov.get("segments")
        if "marker_segments" in ov:
            from .timeline import segments_from_markers

            options = ov["marker_segments"]
            if not isinstance(options, dict) or "pattern" not in options or set(options) - {"pattern", "before_seconds", "after_seconds"}:
                raise ToolkitError("invalid_segments", "Marker selection requires a pattern and optional before_seconds and after_seconds.")
            values = segments_from_markers(recording, **options)
        if not isinstance(values, list) or not values:
            raise ToolkitError("invalid_segments", "Provide at least one analysis segment.")
        try:
            segments = [AnalysisSegment.model_validate(s) for s in values]
        except ValueError as exc:
            raise ToolkitError("invalid_segments", "Each segment needs a label and a finite, increasing time interval.") from exc
        cfg.start_seconds, cfg.end_seconds = 0, None
        cfg.segments = sorted(segments, key=lambda s: s.start_seconds)
    elif "start_seconds" in ov or "end_seconds" in ov:
        cfg.segments = None

    if "start_seconds" in ov and ov["start_seconds"] is not None:
        cfg.start_seconds = float(ov["start_seconds"])
    if "end_seconds" in ov:
        cfg.end_seconds = float(ov["end_seconds"]) if ov["end_seconds"] is not None else None
    if "state" in ov:
        cfg.state = ov["state"]
    if "channels" in ov and ov["channels"] is not None:
        cfg.channels = [resolve_channel(recording, str(c)).id for c in ov["channels"]]

    eeg_ids = [c.id for c in recording.channels if not c.is_auxiliary] or [c.id for c in recording.channels]
    if isinstance(ov.get("exclude_intervals"), list):
        for item in ov["exclude_intervals"]:
            chs = (
                [resolve_channel(recording, str(c)).id for c in item["channels"]]
                if item.get("channels")
                else list(eeg_ids)
            )
            cfg.exclusions.append(
                ExclusionSpec(
                    start=float(item["start"]),
                    end=float(item["end"]),
                    channels=chs,
                    reason=str(item.get("reason") or f"Excluded [{item['start']}, {item['end']})s"),
                    scope=item.get("scope", "acquired"),
                )
            )
    if isinstance(ov.get("exclude_channels"), list):
        for tok in ov["exclude_channels"]:
            ch = resolve_channel(recording, str(tok))
            cfg.exclusions.append(
                ExclusionSpec(
                    start=0.0,
                    end=recording.duration,
                    channels=[ch.id],
                    reason=str(ov.get("reason") or f"Excluded channel {ch.name}"),
                    scope="acquired",
                )
            )

    if "exclude_intervals" in ov and ov["exclude_intervals"] is not None and not isinstance(ov["exclude_intervals"], list):
        raise ToolkitError("invalid_override_value", "exclude_intervals must be a list of interval specs.")
    if "exclude_channels" in ov and ov["exclude_channels"] is not None and not isinstance(ov["exclude_channels"], list):
        raise ToolkitError("invalid_override_value", "exclude_channels must be a list of channel names/IDs.")
    if "filters" in ov and ov["filters"] is not None and not isinstance(ov["filters"], dict):
        raise ToolkitError("invalid_override_value", "filters must be an object/dict.")
    if "qc" in ov and ov["qc"] is not None and not isinstance(ov["qc"], dict):
        raise ToolkitError("invalid_override_value", "qc must be an object/dict.")
    if "detector_params" in ov and ov["detector_params"] is not None and not isinstance(ov["detector_params"], dict):
        raise ToolkitError("invalid_detector_parameter", "detector_params must be an object/dict mapping parameter names to numeric values.")
    if "classifier_stages" in ov and ov["classifier_stages"] is not None and not isinstance(ov["classifier_stages"], list):
        raise ToolkitError("invalid_override_value", "classifier_stages must be a list.")
    if "classifier_thresholds" in ov and ov["classifier_thresholds"] is not None and not isinstance(ov["classifier_thresholds"], dict):
        raise ToolkitError("invalid_override_value", "classifier_thresholds must be an object/dict.")

    if isinstance(ov.get("filters"), dict):
        cfg.filters = FilterConfig.model_validate({**cfg.filters.model_dump(), **ov["filters"]})
    for fk in ("highpass", "lowpass", "notch"):
        if fk in ov:
            setattr(cfg.filters, fk, ov[fk])

    if isinstance(ov.get("qc"), dict):
        cfg.qc = QCConfig.model_validate({**cfg.qc.model_dump(), **ov["qc"]})

    if "reference_mode" in ov and ov["reference_mode"] is not None:
        rmode = str(ov["reference_mode"])
        pairs = [
            BipolarPair(
                anode=resolve_channel(recording, p["anode"]).id,
                cathode=resolve_channel(recording, p["cathode"]).id,
                name=p.get("name"),
            )
            for p in (ov.get("bipolar_pairs") or [])
        ]
        avg_chs = [resolve_channel(recording, c).id for c in (ov.get("average_channels") or [])]
        cfg.reference = ReferenceConfig(mode=rmode, pairs=pairs, average_channels=avg_chs)  # type: ignore[arg-type]

    if isinstance(ov.get("detector"), str):
        cfg.detector = DetectorConfig(
            enabled=True,
            name=ov["detector"].upper(),  # type: ignore[arg-type]
            band=cfg.detector.band,
            parameters={},
        )
    elif isinstance(ov.get("detector"), dict):
        cfg.detector = DetectorConfig.model_validate({**cfg.detector.model_dump(), **ov["detector"]})
    if "detector_enabled" in ov and ov["detector_enabled"] is not None:
        cfg.detector.enabled = bool(ov["detector_enabled"])
    if ov.get("band"):
        cfg.detector.band = (float(ov["band"][0]), float(ov["band"][1]))
    merged_det_params = dict(cfg.detector.parameters)
    if isinstance(ov.get("detector_params"), dict):
        merged_det_params.update(ov["detector_params"])
    for fk in sorted(_FLAT_DETECTOR_PARAM_KEYS):
        if fk in ov and ov[fk] is not None:
            merged_det_params[fk] = ov[fk]
    if merged_det_params:
        cfg.detector.parameters = _normalize_detector_params(cfg.detector.name, merged_det_params)

    if "classifier_enabled" in ov and ov["classifier_enabled"] is not None:
        cfg.classifier.enabled = bool(ov["classifier_enabled"])
    if isinstance(ov.get("classifier_stages"), list):
        sts = {str(s).lower() for s in ov["classifier_stages"]}
        cfg.classifier.enabled = bool(sts)
        cfg.classifier.artifact = "artifact" in sts or bool(sts & {"spike", "ehfo"})
        cfg.classifier.spike = "spike" in sts
        cfg.classifier.ehfo = "ehfo" in sts
    if isinstance(ov.get("classifier_thresholds"), dict):
        cfg.classifier.thresholds = {**cfg.classifier.thresholds, **{str(k): float(v) for k, v in ov["classifier_thresholds"].items()}}

    return AnalysisConfig.model_validate(cfg.model_dump())


def _stage_fps(recording: RecordingSpec, cfg: AnalysisConfig) -> dict[str, str]:
    norm = json.loads(cfg.model_dump_json())
    acq_ex = [e for e in norm["exclusions"] if e.get("scope", "acquired") == "acquired"]
    qc_fp = canonical_digest({
        "rec": recording.fingerprint,
        "s0": float(norm["start_seconds"]),
        "s1": float(norm["end_seconds"]) if norm["end_seconds"] is not None else None,
        "qc": norm["qc"],
        "acq_ex": acq_ex,
        **({"segments": norm["segments"]} if "segments" in norm else {}),
    })
    prep_fp = canonical_digest({
        "qc_fp": qc_fp,
        "channels": norm["channels"],
        "state": norm["state"],
        "exclusions": norm["exclusions"],
        "ref": norm["reference"],
        "filters": norm["filters"],
    })
    det_fp = canonical_digest({"prep_fp": prep_fp, "detector": norm["detector"]})
    return {"qc": qc_fp, "prepare": prep_fp, "detect": det_fp}


def _find_stage_donors(
    store: WorkbenchStore,
    case_slug: str,
    rec_slug: str,
    current_run: str,
    parent_run: str | None,
    fps: dict[str, str],
) -> dict[str, tuple[str, Path]]:
    donors: list[str] = []
    if parent_run:
        donors.append(sanitize_slug(parent_run))
    if "baseline" not in donors:
        donors.append("baseline")
    for m in store.list_runs(case_slug, rec_slug):
        rn = m.get("run_name")
        if rn and rn not in donors:
            donors.append(rn)

    found: dict[str, tuple[str, Path]] = {}
    for dn in donors:
        if dn == current_run or not store.has_run(case_slug, rec_slug, dn):
            continue
        try:
            meta = store.load_run(case_slug, rec_slug, dn)["meta"]
        except (ToolkitError, FileNotFoundError, OSError):
            continue
        dfps = meta.get("stage_fingerprints", {})
        qc_p = store.resolve_workspace_artifact_path(meta.get("qc_json_path"))
        prep_p = store.resolve_workspace_artifact_path(meta.get("prepared_json_path"))
        det_p = store.resolve_workspace_artifact_path(meta.get("detection_json_path"))

        if "qc" not in found and dfps.get("qc") == fps["qc"] and qc_p is not None and qc_p.is_file():
            found["qc"] = (dn, qc_p)
        if "prepare" not in found and dfps.get("prepare") == fps["prepare"] and prep_p is not None and prep_p.is_file():
            try:
                pdoc = read_json(prep_p)
                dp = store.resolve_workspace_artifact_path(pdoc.get("data_path"), fallback_dir=prep_p.parent)
                vp = store.resolve_workspace_artifact_path(pdoc.get("valid_path"), fallback_dir=prep_p.parent)
                if dp is not None and dp.is_file() and vp is not None and vp.is_file():
                    found["prepare"] = (dn, prep_p)
            except Exception:
                pass
        if "detect" not in found and dfps.get("detect") == fps["detect"] and det_p is not None and det_p.is_file():
            found["detect"] = (dn, det_p)

    # Fallback to content-addressed immutable stage cache (pointer .ref.json -> private immutable dir,
    # with legacy direct <fp>/ fallback for pre-ref workspaces)
    sc_root = store.recording_dir(case_slug, rec_slug) / "_stage_cache"

    def _resolve_stage_cache(stage: str, marker: str) -> Path | None:
        ref_p = sc_root / stage / f"{fps[stage][:24]}.ref.json"
        if ref_p.is_file():
            try:
                ref_doc = read_json(ref_p)
                mp = store.resolve_workspace_artifact_path(ref_doc.get("marker"))
                if mp is not None and mp.is_file():
                    return mp
            except Exception:
                pass
        legacy_p = sc_root / stage / fps[stage][:24] / marker
        return legacy_p if legacy_p.is_file() else None

    if "qc" not in found:
        sc_qc = _resolve_stage_cache("qc", "qc.json")
        if sc_qc is not None:
            found["qc"] = ("stage_cache", sc_qc)
    if "prepare" not in found:
        sc_prep = _resolve_stage_cache("prepare", "prepared.json")
        if sc_prep is not None:
            try:
                pdoc = read_json(sc_prep)
                dp = store.resolve_workspace_artifact_path(pdoc.get("data_path"), fallback_dir=sc_prep.parent)
                vp = store.resolve_workspace_artifact_path(pdoc.get("valid_path"), fallback_dir=sc_prep.parent)
                if dp is not None and dp.is_file() and vp is not None and vp.is_file():
                    found["prepare"] = ("stage_cache", sc_prep)
            except Exception:
                pass
    if "detect" not in found:
        sc_det = _resolve_stage_cache("detect", "detection.json")
        if sc_det is not None:
            found["detect"] = ("stage_cache", sc_det)
    return found


def _find_matching_classification_donor(
    store: WorkbenchStore,
    case_slug: str,
    rec_slug: str,
    current_run: str,
    parent_run: str | None,
    detect_fp: str,
    det_path: Path,
) -> dict[str, Any] | None:
    """Return prior classification ONLY when an existing complete run used the EXACT same detection output."""
    candidates: list[str] = []
    if parent_run:
        candidates.append(sanitize_slug(parent_run))
    for m in store.list_runs(case_slug, rec_slug):
        rn = m.get("run_name")
        if rn and rn not in candidates:
            candidates.append(rn)
    det_resolved = str(det_path.resolve())
    for dn in candidates:
        if dn == current_run or not store.has_run(case_slug, rec_slug, dn):
            continue
        try:
            meta = store.load_run(case_slug, rec_slug, dn)["meta"]
        except (ToolkitError, FileNotFoundError, OSError):
            continue
        if (meta.get("stage_fingerprints") or {}).get("detect") != detect_fp:
            continue
        d_p = store.resolve_workspace_artifact_path(meta.get("detection_json_path"))
        if d_p is None or not d_p.is_file() or str(d_p.resolve()) != det_resolved:
            continue
        c_p = store.resolve_workspace_artifact_path(meta.get("classification_json_path"))
        if c_p is not None and c_p.is_file():
            try:
                doc = read_json(c_p)
                if (doc.get("classification") or {}).get("session_path_by_segment"):
                    return doc
            except Exception:
                continue
    return None


_MVCC_TOP_LINKS = ("CURRENT.json", "channel_rates.csv", "epoch_rates.csv", "events.csv")


def _atomic_symlink(target_rel: str, dest: Path) -> None:
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    os.symlink(target_rel, tmp)
    os.replace(tmp, dest)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _migrate_legacy_flat_run_to_mvcc(rdir: Path) -> None:
    """Materialize a pre-MVCC flat run into generations/<legacy_id> and link `current` + CSVs BEFORE switching."""
    cur_link = rdir / "current"
    if cur_link.is_symlink() or (rdir / "CURRENT.json").is_file() or not (rdir / "run_meta.json").is_file():
        return
    try:
        old_meta = read_json(rdir / "run_meta.json")
        old_cfg = read_json(rdir / "config.json")
        old_sum = read_json(rdir / "summary.json")
    except Exception:
        return
    old_gen = str(old_meta.get("generation_id") or f"legacy_{secrets.token_hex(6)}")
    if not old_gen or not all(ch.isalnum() or ch in "_-" for ch in old_gen):
        old_gen = f"legacy_{secrets.token_hex(6)}"
    stg_leg = rdir / "_staging" / f"mig_{old_gen}_{secrets.token_hex(4)}"
    stg_leg.mkdir(parents=True, exist_ok=True)
    try:
        old_meta = {**old_meta, "generation_id": old_gen}
        old_cfg = {**old_cfg, "generation_id": old_gen}
        old_sum = {**old_sum, "generation_id": old_gen}
        atomic_write_json(stg_leg / "run_meta.json", old_meta)
        atomic_write_json(stg_leg / "config.json", old_cfg)
        atomic_write_json(stg_leg / "summary.json", old_sum)
        if (rdir / "qc_summary.json").is_file():
            shutil.copy2(rdir / "qc_summary.json", stg_leg / "qc_summary.json")
        else:
            atomic_write_json(stg_leg / "qc_summary.json", {})
        if (rdir / "events.json").is_file():
            shutil.copy2(rdir / "events.json", stg_leg / "events.json")
        else:
            atomic_write_json(stg_leg / "events.json", [])
        for csv_name in ("channel_rates.csv", "epoch_rates.csv", "events.csv"):
            src_c = rdir / csv_name
            if src_c.is_file() and not src_c.is_symlink():
                shutil.copy2(src_c, stg_leg / csv_name)
        atomic_write_json(
            stg_leg / "CURRENT.json",
            {
                "generation_id": old_gen,
                "status": "complete",
                "recording_fingerprint": old_meta.get("recording_fingerprint"),
                "config_fingerprint": old_meta.get("config_fingerprint"),
                "committed_at": old_meta.get("created_at") or utc_now(),
            },
        )
        gen_root = rdir / "generations"
        gen_root.mkdir(parents=True, exist_ok=True)
        leg_final = gen_root / old_gen
        if leg_final.exists():
            shutil.rmtree(leg_final, ignore_errors=True)
        os.replace(stg_leg, leg_final)
        # Point `current` and top-level symlinks at the OLD historical generation BEFORE flipping to the new one!
        _atomic_symlink(f"generations/{old_gen}", cur_link)
        for fname in _MVCC_TOP_LINKS:
            if (leg_final / fname).is_file():
                _atomic_symlink(f"current/{fname}", rdir / fname)
        _fsync_dir(rdir)
    finally:
        if stg_leg.exists():
            shutil.rmtree(stg_leg, ignore_errors=True)


def _publish_mvcc_generation(
    rdir: Path,
    gen_id: str,
    final_gdir: Path,
    *,
    recording_fingerprint: str | None,
    config_fingerprint: str | None,
) -> None:
    """Atomically switch CURRENT.json + top-level CSVs + generation dir in ONE POSIX symlink swap."""
    cur_doc = {
        "generation_id": gen_id,
        "status": "complete",
        "recording_fingerprint": recording_fingerprint,
        "config_fingerprint": config_fingerprint,
        "committed_at": utc_now(),
    }
    atomic_write_json(final_gdir / "CURRENT.json", cur_doc)

    cur_link = rdir / "current"
    try:
        # 1) Smooth zero-crash-window upgrade if rdir previously used legacy flat layout (pre-MVCC)
        _migrate_legacy_flat_run_to_mvcc(rdir)

        # 2) Smooth zero-crash-window upgrade if rdir previously used regular CURRENT.json without `current` symlink
        if not cur_link.is_symlink() and (rdir / "CURRENT.json").is_file() and not (rdir / "CURRENT.json").is_symlink():
            try:
                old_cur = read_json(rdir / "CURRENT.json")
                old_gen = str((old_cur or {}).get("generation_id") or "").strip()
                old_gdir = rdir / "generations" / old_gen if old_gen else None
                if old_gdir and old_gdir.is_dir():
                    if not (old_gdir / "CURRENT.json").is_file():
                        atomic_write_json(old_gdir / "CURRENT.json", old_cur)
                    _atomic_symlink(f"generations/{old_gen}", cur_link)
            except Exception:
                pass

        # 3) Ensure ALL permanent relative top-level symlinks point through `current/` BEFORE the atomic flip
        # so ONE single os.replace(..., rdir/"current") flips CURRENT.json + ALL 3 CSVs simultaneously!
        for fname in _MVCC_TOP_LINKS:
            fpath = rdir / fname
            want = f"current/{fname}"
            if fpath.is_symlink():
                try:
                    if os.readlink(fpath) == want:
                        continue
                except OSError:
                    pass
            _atomic_symlink(want, fpath)

        # 4) SINGLE ATOMIC COMMIT SYSCALL: flips current, CURRENT.json, events.csv, channel_rates.csv, epoch_rates.csv
        _atomic_symlink(f"generations/{gen_id}", cur_link)
        _fsync_dir(rdir)
    except OSError as exc:
        # Fallback ONLY for unprivileged Windows environments that lack symlink privileges;
        # NEVER swallow real POSIX / I/O errors (EIO, ENOSPC, etc.) or tear an existing symlink layout!
        win_symlink_unsupported = (
            os.name == "nt"
            and not cur_link.is_symlink()
            and exc.errno in {errno.EPERM, errno.EACCES, errno.ENOTSUP, errno.EOPNOTSUPP}
        )
        if not win_symlink_unsupported:
            raise
        atomic_write_json(rdir / "CURRENT.json", cur_doc)
        for csv_name in ("channel_rates.csv", "epoch_rates.csv", "events.csv"):
            src_csv = final_gdir / csv_name
            if src_csv.is_file():
                tmp_csv = rdir / f".{csv_name}.{gen_id}.tmp"
                shutil.copy2(src_csv, tmp_csv)
                os.replace(tmp_csv, rdir / csv_name)


def _prune_old_generations(rdir: Path, keep_gen: str, *, min_keep: int = 8) -> None:
    """Lock-aware grace-period GC: NEVER deletes active generation, reader-pinned generations, or live writer staging."""
    grace_s = float(get_env("BRAINLENS_MVCC_GC_GRACE_SECONDS", "BTK_MVCC_GC_GRACE_SECONDS", "300") or "300")
    now = time.time()
    gen_root = rdir / "generations"
    if gen_root.is_dir():
        candidates: list[tuple[float, Path]] = []
        for d in gen_root.iterdir():
            if not d.is_dir() or d.name == keep_gen:
                continue
            try:
                mt = d.stat().st_mtime
            except OSError:
                continue
            candidates.append((mt, d))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        for mt, old_g in candidates[min_keep:]:
            if (now - mt) >= grace_s:
                gc_fd = try_lock_generation_for_gc(old_g)
                if gc_fd is None:
                    # Active reader holds LOCK_SH on this generation; NEVER delete!
                    continue
                try:
                    shutil.rmtree(old_g, ignore_errors=True)
                finally:
                    release_gc_lock(old_g, gc_fd)
    stg_root = rdir / "_staging"
    if stg_root.is_dir():
        for d in stg_root.iterdir():
            if not d.is_dir():
                continue
            try:
                if (now - d.stat().st_mtime) < grace_s:
                    continue
            except OSError:
                continue
            lock_p = stg_root / f".{d.name}.lock"
            wlock = FileLock(str(lock_p))
            try:
                wlock.acquire(timeout=0)
            except FileLockTimeout:
                # Active writer still owns this staging directory; NEVER delete even past grace!
                continue
            try:
                shutil.rmtree(d, ignore_errors=True)
            finally:
                wlock.release()
                try:
                    if not d.exists():
                        lock_p.unlink(missing_ok=True)
                except OSError:
                    pass


def _execute_one_recording(
    store: WorkbenchStore,
    case_slug: str,
    rec_slug: str,
    run_slug: str,
    overrides: dict[str, Any] | None,
    parent_run: str | None,
    protocol: str | None,
    reason: str | None,
    overwrite: bool,
    progress_cb: Callable[[str, float, str], None] | None = None,
) -> dict[str, Any]:
    def _emit(stage: str, pct: float, detail: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(stage, pct, detail)
            except Exception:
                pass

    recording = store.load_recording(case_slug, rec_slug)
    # Refuse to analyze or stamp a registered recording if its source file(s) mutated on disk!
    def hash_progress(stage: str, fraction: float, detail: str) -> None:
        _emit(stage, 0.05 + 0.05 * fraction, detail)

    source_signature = verify_source_recording_integrity(recording, progress_cb=hash_progress if progress_cb else None)
    eff_parent = (
        sanitize_slug(parent_run)
        if parent_run
        else ("baseline" if run_slug != "baseline" and not protocol and store.has_run(case_slug, rec_slug, "baseline") else None)
    )
    if protocol:
        base_cfg = load_protocol_config(protocol, store, recording_modality=recording.modality)
    elif eff_parent and store.has_run(case_slug, rec_slug, eff_parent):
        base_cfg = AnalysisConfig.model_validate(store.load_run(case_slug, rec_slug, eff_parent)["config"]["analysis_config"])
    else:
        base_cfg = AnalysisConfig.for_modality(recording.modality)

    cfg = apply_config_overrides(recording, base_cfg, overrides)
    cfg_dump = json.loads(cfg.model_dump_json())
    cfg_fp = canonical_digest({"recording_fingerprint": recording.fingerprint, "analysis_config": cfg_dump})

    if store.has_run(case_slug, rec_slug, run_slug) and not overwrite:
        b = store.load_run(case_slug, rec_slug, run_slug)
        saved_rec_fp = b["meta"].get("recording_fingerprint") or b["summary"].get("recording_fingerprint")
        saved_cfg_dump = b["config"].get("analysis_config")
        same_rec = saved_rec_fp == recording.fingerprint
        same_cfg = saved_cfg_dump == cfg_dump
        if same_rec and same_cfg:
            _emit("complete", 1.0, f"Reused saved run {case_slug}/{rec_slug}:{run_slug}")
            return _format_receipt(store, case_slug, rec_slug, run_slug, b["summary"], b["meta"], reused_run=True)
        raise ToolkitError(
            "run_conflict",
            f"Run {run_slug!r} already exists for {case_slug}/{rec_slug} with a different recording or configuration; "
            f"choose a new run_name or pass overwrite=True.",
            existing_recording_fingerprint=saved_rec_fp,
            current_recording_fingerprint=recording.fingerprint,
        )

    # Preflight L7 reference compatibility check BEFORE creating directories on disk
    if cfg.reference.mode != "original" and recording.reference not in {"common", "average"}:
        raise ToolkitError(
            "reference_incompatible",
            f"Pipeline '{cfg.reference.mode}' re-referencing requires documented 'common' or 'average' reference; "
            f"{rec_slug} has acquired reference '{recording.reference}'.",
        )

    check_analysis_window(recording, cfg)
    fps = _stage_fps(recording, cfg)
    donors = {} if overwrite else _find_stage_donors(store, case_slug, rec_slug, run_slug, eff_parent, fps)

    cdir = store.case_dir(case_slug)
    cdir.mkdir(parents=True, exist_ok=True)
    rdir = store.run_dir(case_slug, rec_slug, run_slug)
    runs_parent = rdir.parent
    runs_parent.mkdir(parents=True, exist_ok=True)
    gen_root = rdir / "generations"
    gen_id = secrets.token_hex(8)
    # Stage OUTSIDE `generations/` AND pin with a live writer FileLock so concurrent GC can NEVER delete
    # an active writer's staging dir even if PyHFO detection exceeds the grace period!
    stg_root = rdir / "_staging"
    stg_root.mkdir(parents=True, exist_ok=True)
    stage_lock_path = stg_root / f".{gen_id}.lock"
    stage_lock = FileLock(str(stage_lock_path))
    stage_lock.acquire()
    stage_dir = stg_root / gen_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    committed = False
    t0 = time.monotonic()
    reused: dict[str, str] = {}
    private_stages: list[Path] = []

    def _alloc_stage_dir(stage: str) -> Path:
        # ALWAYS allocate a globally unique private directory per writer so NO TWO concurrent writers
        # can ever open/truncate each other's data.npy / valid.npy / qc.json / detection.json!
        directory = store.stage_cache_dir(
            case_slug,
            rec_slug,
            stage,
            fps[stage],
            nonce=f"{gen_id[:6]}{secrets.token_hex(3)}",
        )
        private_stages.append(directory)
        return directory

    def _publish_stage_ref(stage: str, marker_path: Path) -> None:
        ref_p = store.recording_dir(case_slug, rec_slug) / "_stage_cache" / stage / f"{fps[stage][:24]}.ref.json"
        atomic_write_json(
            ref_p,
            {
                "fingerprint": fps[stage],
                "stage_dir": str(marker_path.parent),
                "marker": str(marker_path),
                "published_at": utc_now(),
            },
        )

    def _check_source_unchanged() -> None:
        if verify_source_recording_integrity(recording) != source_signature:
            raise ToolkitError("source_recording_modified", "Source changed during analysis; discarding this run.",
                               recording=rec_slug)

    try:
        # 1. QC
        _emit("qc", 0.12, f"[{rec_slug}] Running full-coverage signal QC...")
        if "qc" in donors:
            reused["qc"], qc_path = donors["qc"]
            qc = read_json(qc_path)
        else:
            qc_dir = _alloc_stage_dir("qc")
            qc = compute_qc(recording, cfg, qc_dir)
            qc_path = qc_dir / "qc.json"
        atomic_write_json(stage_dir / "qc_summary.json", qc)

        # 2. Prepare
        _emit("prepare", 0.30, f"[{rec_slug}] Preparing reference & zero-phase filters...")
        if "prepare" in donors:
            reused["prepare"], prep_path = donors["prepare"]
            prepared = dict(read_json(prep_path))
            for arr_key in ("valid_path", "data_path"):
                if prepared.get(arr_key):
                    arr_res = store.resolve_workspace_artifact_path(prepared[arr_key], fallback_dir=prep_path.parent)
                    if arr_res is not None:
                        prepared[arr_key] = str(arr_res)
        else:
            prep_dir = _alloc_stage_dir("prepare")
            prepared = prepare_signals(recording, cfg, qc, prep_dir)
            prep_path = prep_dir / "prepared.json"

        # 3. Detect
        _emit("detect", 0.55, f"[{rec_slug}] Running PyHFO {cfg.detector.name} detector...")
        runner = PyHFORunner(resolve_engine_settings(store.root)) if (cfg.detector.enabled or cfg.classifier.enabled) else None
        if "detect" in donors:
            reused["detect"], det_path = donors["detect"]
            detection = dict(read_json(det_path))
            # Remap cached segment session.pybrain paths when workspace was relocated!
            segs_raw = detection.get("segments")
            if isinstance(segs_raw, list):
                norm_segs: list[dict[str, Any]] = []
                for seg in segs_raw:
                    if isinstance(seg, dict) and seg.get("session_path"):
                        seg_copy = dict(seg)
                        sid = str(seg.get("id") or "")
                        sp_res = (
                            store.resolve_workspace_artifact_path(seg["session_path"], fallback_dir=det_path.parent / sid)
                            if sid
                            else None
                        ) or store.resolve_workspace_artifact_path(seg["session_path"], fallback_dir=det_path.parent)
                        if sp_res is not None:
                            seg_copy["session_path"] = str(sp_res)
                        norm_segs.append(seg_copy)
                    else:
                        norm_segs.append(seg)
                detection["segments"] = norm_segs
        elif cfg.detector.enabled:
            assert runner is not None
            det_dir = _alloc_stage_dir("detect")
            detection = runner.detect(prepared, cfg, det_dir)
            det_path = det_dir / "detection.json"
        else:
            det_dir = _alloc_stage_dir("detect")
            detection = {
                "status": "skipped",
                "reason": "detector_disabled",
                "events": [],
                "valid_intervals": {c["id"]: [] for c in prepared["channels"]},
                "segments": [],
                "backend": {},
                "skipped_channels": {},
                "event_backend": {},
                "warnings": [],
            }
            det_path = det_dir / "detection.json"
            atomic_write_json(det_path, detection)

        # 4. Classify (stored in immutable content-addressed stage cache; never cross-wires across branches!)
        clf_fp = canonical_digest({"det_fp": fps["detect"], "clf": cfg_dump["classifier"], "gen": gen_id})
        clf_dir = store.stage_cache_dir(case_slug, rec_slug, "classify", clf_fp)
        private_stages.append(clf_dir)
        if cfg.classifier.enabled:
            _emit("classify", 0.78, f"[{rec_slug}] Scoring candidates with PyHFO ONNX models...")
            assert runner is not None
            prior_clf = (
                None
                if overwrite
                else _find_matching_classification_donor(store, case_slug, rec_slug, run_slug, eff_parent, fps["detect"], det_path)
            )
            if isinstance(prior_clf, dict):
                pc_meta = prior_clf.get("classification")
                if isinstance(pc_meta, dict) and isinstance(pc_meta.get("session_path_by_segment"), dict):
                    norm_prior_sess: dict[str, Any] = {}
                    for sid, sp_old in pc_meta["session_path_by_segment"].items():
                        sp_res = (
                            store.resolve_workspace_artifact_path(sp_old, fallback_dir=det_path.parent / str(sid))
                            or store.resolve_workspace_artifact_path(sp_old, fallback_dir=det_path.parent)
                        )
                        norm_prior_sess[sid] = str(sp_res) if sp_res is not None else sp_old
                    pc_meta["session_path_by_segment"] = norm_prior_sess
            classification = runner.classify(detection, cfg.classifier, clf_dir, prior_classification=prior_clf)
            clf_path = clf_dir / "classification.json"
        else:
            classification = {**detection, "classifier_status": "skipped", "classification": {"requested": [], "stages": {}}}
            clf_path = clf_dir / "classification.json"
            atomic_write_json(clf_path, classification)

        # 5. Analytics & Clean Tables (all written inside isolated stage_dir — NEVER mutates existing run in-place!)
        _emit("summarize", 0.92, f"[{rec_slug}] Computing synchrony, spectral metrics & CSVs...")
        cur_asig = run_analysis_signature({
            "recording_fingerprint": recording.fingerprint,
            "config_fingerprint": cfg_fp,
        })
        pre_wal_floor = 0
        if store.has_run(case_slug, rec_slug, run_slug):
            try:
                ex_pre = store.load_run(case_slug, rec_slug, run_slug, reconcile=False)
                ex_pre_meta = ex_pre.get("meta") or {}
                ex_pre_asig = str(ex_pre_meta.get("analysis_signature") or run_analysis_signature(ex_pre))
                prev_floor = int(ex_pre_meta.get("analysis_superseded_wal_index") or 0)
                pre_wal_floor = (
                    max(prev_floor, len(store.load_judgments(case_slug)))
                    if ex_pre_asig != cur_asig
                    else prev_floor
                )
            except Exception:
                pre_wal_floor = 0
        verdicts = store.latest_event_verdicts(
            case_slug,
            rec_slug,
            run_slug,
            recording_fingerprint=recording.fingerprint,
            analysis_signature=cur_asig,
            analysis_wal_floor=pre_wal_floor,
        )
        summary, _events = summarize_run(
            recording, prepared, qc, detection, classification, verdicts, cfg, stage_dir, generation_id=gen_id
        )

        cfg_doc = {
            "case": case_slug,
            "recording": rec_slug,
            "run_name": run_slug,
            "parent_run": eff_parent,
            "protocol_applied": protocol,
            "reason": reason or ("Baseline analysis" if run_slug == "baseline" else f"Branch {run_slug}"),
            "overrides": overrides or {},
            "analysis_config": cfg_dump,
            "recording_fingerprint": recording.fingerprint,
            "config_fingerprint": cfg_fp,
            "generation_id": gen_id,
            "created_at": utc_now(),
        }
        atomic_write_json(stage_dir / "config.json", cfg_doc)

        rec_spatial_digest = canonical_digest(
            recording.spatial.model_dump(mode="json") if recording.spatial is not None else None
        )
        meta_doc = {
            "case": case_slug,
            "recording": rec_slug,
            "run_name": run_slug,
            "parent_run": eff_parent,
            "reason": cfg_doc["reason"],
            "status": "complete",
            "generation_id": gen_id,
            "analysis_generation_id": gen_id,
            "analysis_signature": cur_asig,
            "analysis_superseded_wal_index": pre_wal_floor,
            "recording_fingerprint": recording.fingerprint,
            "recording_spatial_digest": rec_spatial_digest,
            "config_fingerprint": cfg_fp,
            "detector_status": summary["detector_status"],
            "classifier_status": summary["classifier_status"],
            "counts": summary["counts"],
            "elapsed_seconds": round(time.monotonic() - t0, 3),
            "reused_stages": reused,
            "stage_fingerprints": fps,
            "qc_json_path": str(qc_path),
            "prepared_json_path": str(prep_path),
            "detection_json_path": str(det_path),
            "classification_json_path": str(clf_path),
            "backend": detection.get("backend", {}),
            "created_at": utc_now(),
        }
        atomic_write_json(stage_dir / "run_meta.json", meta_doc)

        # Single-pointer MVCC commit under canonical lock order (.judgments.lock -> .<run>.lock):
        # 1) Re-verify overwrite=False conflict INSIDE the commit lock (closes concurrent duplicate race!)
        # 2) Compute authoritative analysis_superseded_wal_index INSIDE .judgments.lock so config/recording
        #    overwrites cleanly isolate prior clips/notes while identical-config concurrent verdicts survive!
        # 3) Re-read latest_event_verdicts INSIDE .judgments.lock so concurrent accepted verdicts are NEVER dropped!
        # 4) Move staged directory into immutable generations/<gen_id> and flip `current` symlink in ONE syscall
        #    (atomically switches CURRENT.json + summary/events JSON + ALL top-level CSVs simultaneously!)
        with FileLock(str(cdir / ".judgments.lock")):
            with FileLock(str(runs_parent / f".{run_slug}.lock")):
                commit_wal_floor = 0
                if store.has_run(case_slug, rec_slug, run_slug):
                    existing = store.load_run(case_slug, rec_slug, run_slug, reconcile=False)
                    ex_meta = existing["meta"]
                    ex_rec_fp = ex_meta.get("recording_fingerprint") or existing["summary"].get("recording_fingerprint")
                    ex_cfg_dump = existing["config"].get("analysis_config")
                    if not overwrite:
                        if ex_rec_fp == recording.fingerprint and ex_cfg_dump == cfg_dump:
                            _check_source_unchanged()
                            return _format_receipt(store, case_slug, rec_slug, run_slug, existing["summary"], ex_meta, reused_run=True)
                        raise ToolkitError(
                            "run_conflict",
                            f"Run {run_slug!r} already exists for {case_slug}/{rec_slug} with a different recording or configuration; "
                            f"choose a new run_name or pass overwrite=True.",
                            existing_recording_fingerprint=ex_rec_fp,
                            current_recording_fingerprint=recording.fingerprint,
                        )
                    ex_asig = str(ex_meta.get("analysis_signature") or run_analysis_signature(existing))
                    prev_floor = int(ex_meta.get("analysis_superseded_wal_index") or 0)
                    commit_wal_floor = (
                        max(prev_floor, len(store.load_judgments(case_slug)))
                        if ex_asig != cur_asig
                        else prev_floor
                    )
                meta_doc["analysis_superseded_wal_index"] = commit_wal_floor
                fresh_verdicts = store.latest_event_verdicts(
                    case_slug,
                    rec_slug,
                    run_slug,
                    recording_fingerprint=recording.fingerprint,
                    analysis_signature=cur_asig,
                    analysis_wal_floor=commit_wal_floor,
                )
                if fresh_verdicts != verdicts:
                    verdicts = fresh_verdicts
                    summary, _events = summarize_run(
                        recording, prepared, qc, detection, classification, verdicts, cfg, stage_dir, generation_id=gen_id
                    )
                    meta_doc["counts"] = summary["counts"]
                atomic_write_json(stage_dir / "run_meta.json", meta_doc)
                _check_source_unchanged()
                gen_root.mkdir(parents=True, exist_ok=True)
                final_gdir = gen_root / gen_id
                os.replace(stage_dir, final_gdir)
                try:
                    _publish_mvcc_generation(
                        rdir,
                        gen_id,
                        final_gdir,
                        recording_fingerprint=recording.fingerprint,
                        config_fingerprint=cfg_fp,
                    )
                except Exception:
                    shutil.rmtree(final_gdir, ignore_errors=True)
                    raise
                committed = True
                # The run manifest already binds every donor. Publish the optional cache
                # shortcuts only after the source bracket and generation commit succeed.
                for stage, marker in (("qc", qc_path), ("prepare", prep_path), ("detect", det_path)):
                    if stage not in reused:
                        with suppress(OSError):
                            _publish_stage_ref(stage, marker)
                _prune_old_generations(rdir, keep_gen=gen_id)
    finally:
        if not committed:
            shutil.rmtree(stage_dir, ignore_errors=True)
            for directory in private_stages:
                shutil.rmtree(directory, ignore_errors=True)
        try:
            stage_lock.release()
        except Exception:
            pass
        try:
            stage_lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    return _format_receipt(store, case_slug, rec_slug, run_slug, summary, meta_doc, reused_run=False)


def _format_receipt(
    store: WorkbenchStore,
    case_slug: str,
    rec_slug: str,
    run_slug: str,
    summary: dict[str, Any],
    meta: dict[str, Any],
    reused_run: bool,
) -> dict[str, Any]:
    matcher = AnatomyMatcher(load_anatomy_labels(store, case_slug))
    return {
        "case": case_slug,
        "recording": rec_slug,
        "run_name": run_slug,
        "parent_run": meta.get("parent_run"),
        "already_existed": reused_run,
        "elapsed_seconds": meta.get("elapsed_seconds"),
        "reused_stages": meta.get("reused_stages", {}),
        "acquired_reference": summary["acquired_reference"],
        "effective_reference": summary["effective_reference"],
        **({"segments": summary["selection"]["segments"]} if summary.get("selection", {}).get("segments") else {}),
        "counts": summary["counts"],
        "leading_focal_channels": [
            {
                "focal_rank": r["focal_rank"],
                "channel": r["channel_name"],
                "anatomy": matcher.match(r["channel_name"], r.get("contacts", [])),
                "focal_count": r["focal_count"],
                "focal_rate_per_min": r["focal_rate_per_min"],
                "raw_count": r["raw_count"],
                "analyzed_seconds": r["detector_analyzed_seconds"],
            }
            for r in summary["leading_focal_channels"][:8]
        ],
        "synchronous_burst_windows": summary["synchronous_burst_windows"][:5],
        "warnings": summary["warnings"],
    }


def refresh_verdict_tallies(store: WorkbenchStore, case_slug: str, rec_slug: str, run_name: str) -> dict[str, Any]:
    """Refresh human verdict & clip counts atomically via a new MVCC generation without rerunning compute."""
    cdir = store.case_dir(case_slug)
    cdir.mkdir(parents=True, exist_ok=True)
    run_slug = sanitize_slug(run_name)
    rdir = store.run_dir(case_slug, rec_slug, run_slug)
    runs_parent = rdir.parent
    runs_parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cdir / ".judgments.lock")):
        with FileLock(str(runs_parent / f".{run_slug}.lock")):
            cur_bundle = store.load_run(case_slug, rec_slug, run_slug, reconcile=False)
            meta = cur_bundle["meta"]
            cfg_doc = cur_bundle["config"]
            cfg = AnalysisConfig.model_validate(cfg_doc["analysis_config"])
            recording = store.load_recording(case_slug, rec_slug)
            cur_gdir = Path(cur_bundle["generation_dir"])
            qc_p = store.resolve_workspace_artifact_path(meta.get("qc_json_path"))
            if qc_p is not None and qc_p.is_file():
                qc = read_json(qc_p)
            elif (cur_gdir / "qc_summary.json").is_file():
                qc_p = cur_gdir / "qc_summary.json"
                qc = read_json(qc_p)
            else:
                raise FileNotFoundError(f"Missing qc artifact for {case_slug}/{rec_slug}/{run_slug}: {meta.get('qc_json_path')}")
            prep_p = store.resolve_workspace_artifact_path(meta.get("prepared_json_path"))
            det_p = store.resolve_workspace_artifact_path(meta.get("detection_json_path"))
            clf_p = store.resolve_workspace_artifact_path(meta.get("classification_json_path"))
            if prep_p is None or det_p is None or clf_p is None:
                raise FileNotFoundError(
                    f"Missing stage cache artifact for {case_slug}/{rec_slug}/{run_slug} "
                    f"(prep={prep_p}, det={det_p}, clf={clf_p})"
                )
            prepared = dict(read_json(prep_p))
            for arr_key in ("valid_path", "data_path"):
                if prepared.get(arr_key):
                    arr_res = store.resolve_workspace_artifact_path(prepared[arr_key], fallback_dir=prep_p.parent)
                    if arr_res is not None:
                        prepared[arr_key] = str(arr_res)
            detection = read_json(det_p)
            classification = read_json(clf_p)
            run_fp_eff = meta.get("recording_fingerprint") or recording.fingerprint
            run_asig_eff = str(meta.get("analysis_signature") or run_analysis_signature(cur_bundle))
            run_floor_eff = int(meta.get("analysis_superseded_wal_index") or 0)
            verdicts = store.latest_event_verdicts(
                case_slug,
                rec_slug,
                run_slug,
                recording_fingerprint=run_fp_eff,
                analysis_signature=run_asig_eff,
                analysis_wal_floor=run_floor_eff,
            )

            # ALWAYS stage into an isolated MVCC generation and commit via atomic symlink flip
            # (pre-MVCC flat runs are migrated cleanly into generations/<old_gen> first -> zero torn writes!)
            new_gen = secrets.token_hex(8)
            stg_root = rdir / "_staging"
            stg_root.mkdir(parents=True, exist_ok=True)
            stg_lock_p = stg_root / f".{new_gen}.lock"
            stg_lock = FileLock(str(stg_lock_p))
            stg_lock.acquire()
            stg_gdir = stg_root / new_gen
            stg_gdir.mkdir(parents=True, exist_ok=True)
            committed = False
            try:
                atomic_write_json(stg_gdir / "qc_summary.json", qc)
                summary, _ = summarize_run(
                    recording,
                    prepared,
                    qc,
                    detection,
                    classification,
                    verdicts,
                    cfg,
                    stg_gdir,
                    generation_id=new_gen,
                )
                new_cfg = {**cfg_doc, "generation_id": new_gen}
                base_analysis_gen = (
                    meta.get("analysis_generation_id")
                    or meta.get("generation_id")
                    or canonical_digest({
                        "rec": meta.get("recording_fingerprint"),
                        "cfg": meta.get("config_fingerprint"),
                        "created": meta.get("created_at"),
                    })[:16]
                )
                rec_spatial_digest = canonical_digest(
                    recording.spatial.model_dump(mode="json") if recording.spatial is not None else None
                )
                new_meta = {
                    **meta,
                    "generation_id": new_gen,
                    "analysis_generation_id": base_analysis_gen,
                    "analysis_signature": run_asig_eff,
                    "analysis_superseded_wal_index": run_floor_eff,
                    "recording_spatial_digest": rec_spatial_digest,
                    "qc_json_path": str(qc_p),
                    "prepared_json_path": str(prep_p),
                    "detection_json_path": str(det_p),
                    "classification_json_path": str(clf_p),
                    "counts": summary["counts"],
                }
                atomic_write_json(stg_gdir / "config.json", new_cfg)
                atomic_write_json(stg_gdir / "run_meta.json", new_meta)
                gen_root = rdir / "generations"
                gen_root.mkdir(parents=True, exist_ok=True)
                final_gdir = gen_root / new_gen
                os.replace(stg_gdir, final_gdir)
                try:
                    _publish_mvcc_generation(
                        rdir,
                        new_gen,
                        final_gdir,
                        recording_fingerprint=new_meta.get("recording_fingerprint"),
                        config_fingerprint=new_meta.get("config_fingerprint"),
                    )
                except Exception:
                    shutil.rmtree(final_gdir, ignore_errors=True)
                    raise
                committed = True
                _prune_old_generations(rdir, keep_gen=new_gen)
                return summary
            finally:
                if not committed and stg_gdir.exists():
                    shutil.rmtree(stg_gdir, ignore_errors=True)
                try:
                    stg_lock.release()
                except Exception:
                    pass
                try:
                    stg_lock_p.unlink(missing_ok=True)
                except OSError:
                    pass


def analyze_case(
    target: str,
    workspace_dir: str | Path | None = None,
    run_name: str | None = None,
    overrides: dict[str, Any] | None = None,
    parent_run: str | None = None,
    protocol: str | None = None,
    reason: str | None = None,
    spatial_sidecar: str | Path | None = None,
    overwrite: bool = False,
    progress_cb: Callable[[str, float, str], None] | None = None,
) -> dict[str, Any]:
    """Inspect (if needed) and execute QC + PyHFO + Synchrony/ONNX stratification."""
    store = WorkbenchStore(workspace_dir)
    if progress_cb:
        progress_cb("inspect", 0.05, f"Inspecting target {target}...")
    def inspection_progress(stage: str, fraction: float, detail: str) -> None:
        if progress_cb:
            progress_cb(stage, 0.05 + 0.05 * fraction, detail)

    on_inspect = inspection_progress if progress_cb else None
    spec = resolve_case_target(target, store)
    rslug = sanitize_slug(run_name or _auto_branch_slug(overrides, protocol))

    if spec["kind"] == "filesystem":
        fs_target_str = str(spec.get("logical_path") or spec["path"])
        # Preflight non-persisting inspection BEFORE mutating existing case/recording.json so a
        # run_conflict on an in-place modified file NEVER leaves recording.json out of sync with existing runs!
        if not overwrite:
            preview = inspect_case(fs_target_str, workspace_dir=store.root, spatial_sidecar=spatial_sidecar, persist=False, progress_cb=on_inspect)
            prev_case = preview["case"]
            if store.has_case(prev_case):
                prev_specs = preview.get("_recording_specs") or {}
                for rec_slug, rec_model in prev_specs.items():
                    if store.has_run(prev_case, rec_slug, rslug):
                        b = store.load_run(prev_case, rec_slug, rslug)
                        saved_rec_fp = b["meta"].get("recording_fingerprint") or b["summary"].get("recording_fingerprint")
                        if saved_rec_fp != rec_model.fingerprint:
                            raise ToolkitError(
                                "run_conflict",
                                f"Run {rslug!r} already exists for {prev_case}/{rec_slug} with a different recording fingerprint; "
                                f"choose a new run_name or pass overwrite=True (existing recording preserved untouched).",
                                existing_recording_fingerprint=saved_rec_fp,
                                current_recording_fingerprint=rec_model.fingerprint,
                            )
        insp = inspect_case(fs_target_str, workspace_dir=store.root, spatial_sidecar=spatial_sidecar, persist=True, progress_cb=on_inspect)
        cslug = insp["case"]
        recs = [r["recording"] for r in insp["recordings"]]
    elif spec["kind"] == "registered_recording":
        cslug = spec["case"]
        recs = [spec["recording"]]
        if spatial_sidecar:
            inspect_case(f"{cslug}/{spec['recording']}", workspace_dir=store.root, spatial_sidecar=spatial_sidecar, persist=True, progress_cb=on_inspect)
    else:
        cslug = spec["case"]
        if spatial_sidecar:
            inspect_case(cslug, workspace_dir=store.root, spatial_sidecar=spatial_sidecar, persist=True, progress_cb=on_inspect)
        recs = store.list_recordings(cslug)
    if len(recs) != 1 and {"segments", "marker_segments"}.intersection(overrides or {}):
        raise ToolkitError("missing_recording", "Select one recording for a segment run.", recordings=recs)
    results = []
    total_r = max(1, len(recs))
    for r_i, rec in enumerate(recs):
        def _sub_cb(stage: str, frac: float, detail: str, idx: int = r_i) -> None:
            if progress_cb:
                overall = (idx + max(0.0, min(1.0, frac))) / total_r
                progress_cb(stage, round(overall, 4), f"({idx + 1}/{total_r}) {detail}")

        results.append(
            _execute_one_recording(
                store, cslug, rec, rslug, overrides, parent_run, protocol, reason, overwrite, progress_cb=_sub_cb
            )
        )
    if progress_cb:
        progress_cb("complete", 1.0, f"Completed {cslug} ({rslug})")

    ui = store.load_ui_state()
    if not ui.get("case"):
        top_ch = results[0]["leading_focal_channels"][0]["channel"] if results and results[0]["leading_focal_channels"] else None
        store.update_ui_state(
            {"case": cslug, "recording": recs[0], "run_name": rslug, "channel": top_ch},
            updated_by="pipeline",
        )

    return {
        "workspace": str(store.root),
        "case": cslug,
        "run_name": rslug,
        "recordings_analyzed": len(results),
        "runs": results,
    }
