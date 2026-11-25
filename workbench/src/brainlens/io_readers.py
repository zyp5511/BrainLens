"""High-level EEG/SEEG/iEEG recording & case inspection orchestration + public I/O facade."""

from __future__ import annotations

import os
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import edfio
from filelock import FileLock

from .deid import HEADER_SUBJECT_WITHHELD, native_header_terms, registration_provenance, subject_digest
from .io_cases import (
    _allocate_case_slug,
    _allocate_recording_slugs,
    _check_existing_case_subject_consistency,
    _filter_case_overlays,
    _inspect_registered_case,
    _recording_brief,
    resolve_case_target,
    summarize_registered_case,
)

from .io_integrity import (
    ProgressCallback,
    _ancestor_chain_sig,
    _canonical_entry_path,
    _compute_v2_fingerprint,
    _copy_and_hash_file,
    _hash_in_place,
    _large_file,
    _pin_file,
    _stat_fields,
    _stat_tuple,
    create_verified_snapshot_dir,
    verified_source_snapshot,
    verify_source_recording_integrity,
)
from .io_sidecars import (
    _bids_subject_ids_from_path,
    _bipolar_pairs_file_matches_recording,
    _discover_folder_recordings,
    _discover_recording_sidecars,
    _extract_declared_source_files,
    _extract_doc_subject_tokens,
    _extract_native_subject_id,
    _extract_prov_participant_id,
    _extract_provenance,
    _find_folder_provenance_and_overlays,
    _find_recording_folder_context,
    _norm_subject_token,
    _recording_sidecar_dir_sig,
    _source_file_matches_recording,
    discover_available_inputs,
)
from .io_signals import (
    _ALLOWED_KINDS,
    _VOLT_UNITS,
    _extract_fiff_next_filename,
    _extract_vhdr_component_names,
    _infer_modality,
    _parse_bdf_header,
    _parse_brainvision_header,
    _prefilter_lowpass,
    is_recording_file,
    open_mne_raw,
    read_scaled_volts,
    recording_stem,
)
from .io_vendor import (
    check_unsupported_vendor_folder,
    check_unsupported_vendor_target,
    discover_vendor_companions,
    is_vendor_recording_suffix,
    parse_vendor_header,
)
from .models import (
    PROJECT_ROOT,
    ChannelSpec,
    MarkerSpec,
    RecordingSpec,
    SpatialSpec,
    ToolkitError,
    canonical_digest,
    file_sha256,
    parse_shaft_group,
    read_json,
    utc_now,
)
from .storage import WorkbenchStore, sanitize_slug

__all__ = [
    "PROJECT_ROOT",
    "_copy_and_hash_file",
    "_discover_folder_recordings",
    "_discover_recording_sidecars",
    "create_verified_snapshot_dir",
    "discover_available_inputs",
    "inspect_case",
    "inspect_single_recording",
    "is_recording_file",
    "open_mne_raw",
    "read_scaled_volts",
    "recording_stem",
    "resolve_case_target",
    "summarize_registered_case",
    "verified_source_snapshot",
    "verify_source_recording_integrity",
]


def _assert_recording_snapshot_stable(
    rec_entry: Path,
    rec_slug: str,
    anc_before: tuple[Any, Any],
    pre_main_st: tuple[str, tuple[int, int, int, int, int]],
    pre_sidecar_sig: tuple[Any, ...],
    orig_sp_arg: Path | None,
    orig_prov_arg: Path | None,
    eff_stop_root: Path,
    *,
    phase: str,
) -> None:
    cur_target = rec_entry.resolve()
    if (
        (_ancestor_chain_sig(rec_entry), _ancestor_chain_sig(cur_target)) != anc_before
        or (str(cur_target), _stat_tuple(cur_target)) != pre_main_st
        or _recording_sidecar_dir_sig(rec_entry, orig_sp_arg, orig_prov_arg, stop_root=eff_stop_root) != pre_sidecar_sig
    ):
        raise ToolkitError(
            "source_recording_modified",
            f"Source recording or sidecar metadata for {rec_entry.name!r} changed on disk during {phase}; retry on a stable directory.",
            recording=rec_slug,
        )


def _stage_inspection_snapshot(
    snap_dir: Path,
    rec_entry: Path,
    rec_resolved: Path,
    rec_slug: str,
    pre_main_st: tuple[str, tuple[int, int, int, int, int]],
    folder_prov: dict[str, Any] | None,
    folder_prov_path: Path | None,
    ctx_overlay_paths: list[Path],
    explicit_spatial_sidecar: bool,
    stack: ExitStack,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Stage metadata and small files privately. Bind large signal files by digest and inode."""
    orig_parent_res = rec_entry.parent
    sfx = rec_entry.suffix.lower()
    snap_main = snap_dir / rec_entry.name
    live_signals = {}

    def stage_signal(src: Path, dest: Path):
        if not _large_file(src.stat().st_size):
            return _copy_and_hash_file(src, dest)
        identity = _hash_in_place(src, recording=rec_slug, progress_cb=progress_cb)
        if dest == snap_main and sfx in {".edf", ".bdf"}:
            # The native reader pins a descriptor and exposes its verified stat identity.
            dest.symlink_to(src)
        else:
            _pin_file(stack, src, dest, identity[2])
        live_signals[dest] = (src, identity)
        return identity

    rec_sha256, rec_size, rec_st = stage_signal(rec_resolved, snap_main)
    if (str(rec_entry.resolve()), rec_st) != pre_main_st:
        raise ToolkitError(
            "source_recording_modified",
            f"Source recording {rec_entry.name!r} changed during inspection; retry on a stable file.",
            recording=rec_slug,
        )

    companion_hashes: dict[str, str] = {}
    companion_files: dict[str, dict[str, Any]] = {}

    def _stage_companion(orig_comp: Path, *, record_in_signal_hashes: bool, stage_name: str | None = None) -> Path:
        orig_entry = _canonical_entry_path(orig_comp)
        cp_res = orig_entry.resolve()
        if not cp_res.is_file():
            raise ToolkitError("missing_companion", f"Companion recording file missing: {orig_comp.name}")
        eff_stage = stage_name or orig_entry.name
        snap_comp = snap_dir / eff_stage
        if not snap_comp.exists():
            csha, csize, cst = stage_signal(cp_res, snap_comp)
        else:
            if snap_comp in live_signals:
                src, (csha, csize, cst) = live_signals[snap_comp]
                if src != cp_res:
                    raise ToolkitError("source_recording_modified", "Conflicting companion names during inspection.")
            else:
                csha, cst = file_sha256(snap_comp), _stat_tuple(cp_res)
                csize = cst[2]
        if record_in_signal_hashes:
            companion_hashes[str(cp_res)] = csha
        companion_files[str(cp_res)] = {
            "sha256": csha,
            "size": int(csize),
            "mtime_ns": int(cst[3]),
            "stage_name": eff_stage,
            "logical_path": str(orig_entry),
        }
        return snap_comp

    if sfx == ".vhdr":
        for cname in _extract_vhdr_component_names(snap_main):
            raw_cand = orig_parent_res / cname
            cand = raw_cand.resolve()
            if not cand.is_file() or not cand.is_relative_to(orig_parent_res):
                raise ToolkitError("missing_companion", f"BrainVision component missing: {cname}")
            _stage_companion(raw_cand, record_in_signal_hashes=True, stage_name=Path(cname).name)
    elif rec_entry.name.lower().endswith((".fif", ".fif.gz")):
        seen_parts: set[Path] = {rec_resolved}
        cur_snap = snap_main
        while True:
            nxt_name = _extract_fiff_next_filename(cur_snap)
            if not nxt_name:
                break
            raw_cand = orig_parent_res / Path(nxt_name).name
            cand = raw_cand.resolve()
            if cand in seen_parts:
                break
            if not cand.is_file() or not cand.is_relative_to(orig_parent_res):
                raise ToolkitError("missing_companion", f"Split FIFF continuation part missing: {nxt_name}")
            seen_parts.add(cand)
            cur_snap = _stage_companion(raw_cand, record_in_signal_hashes=True, stage_name=Path(nxt_name).name)
    elif is_vendor_recording_suffix(rec_entry):
        for comp_orig, comp_stage in discover_vendor_companions(rec_entry, snap_main):
            _stage_companion(comp_orig, record_in_signal_hashes=True, stage_name=comp_stage)

    snapped_sidecars: dict[str, tuple[Path, str, int, tuple[int, int, int, int, int]]] = {}
    if orig_parent_res.is_dir():
        for entry in sorted(orig_parent_res.iterdir()):
            low = entry.name.lower()
            if entry.is_file() and (low.endswith(".json") or low.endswith("_channels.tsv")):
                dest_sc = snap_dir / entry.name
                if not dest_sc.exists():
                    e_res = entry.resolve()
                    sc_sha, sc_sz, sc_st = _copy_and_hash_file(e_res, dest_sc)
                    snapped_sidecars[entry.name] = (e_res, sc_sha, sc_sz, sc_st)

    eff_folder_prov: dict[str, Any] = dict(folder_prov or {})
    prov_is_ancestor = False
    if folder_prov_path is not None and folder_prov_path.is_file():
        fp_res = folder_prov_path.resolve()
        prov_is_ancestor = fp_res.parent != orig_parent_res
        prov_snap = snap_dir / "__brainlens_folder_prov.json"
        _copy_and_hash_file(fp_res, prov_snap)
        try:
            parsed_fp = read_json(prov_snap)
            if not isinstance(parsed_fp, dict):
                raise ValueError("PROVENANCE.json root must be a JSON object")
            eff_folder_prov = parsed_fp
        except Exception as exc:
            raise ToolkitError("invalid_provenance", f"Malformed PROVENANCE.json at {fp_res.name!r}: {exc}", recording=rec_slug) from exc

    sp_snap_candidates: list[tuple[str, bool, dict[str, Any], Path]] = []
    for cand_sp_p in ctx_overlay_paths:
        if not cand_sp_p.is_file():
            continue
        sp_res = cand_sp_p.resolve()
        sp_snap = snap_dir / f"__brainlens_sp_{len(sp_snap_candidates)}_{sp_res.name}"
        _copy_and_hash_file(sp_res, sp_snap)
        try:
            cand_doc = read_json(sp_snap)
        except Exception:
            continue
        if isinstance(cand_doc, dict):
            sp_snap_candidates.append((cand_sp_p.name, sp_res.parent != orig_parent_res, cand_doc, sp_res))

    sp_selected = (
        sp_snap_candidates[0]
        if (explicit_spatial_sidecar and sp_snap_candidates)
        else (next((c for c in sp_snap_candidates if "correction" in c[0]), sp_snap_candidates[0]) if sp_snap_candidates else None)
    )
    return {
        "live_signals": live_signals,
        "snap_main": snap_main,
        "rec_sha256": rec_sha256,
        "rec_size": rec_size,
        "rec_st": rec_st,
        "companion_hashes": companion_hashes,
        "companion_files": companion_files,
        "snapped_sidecars": snapped_sidecars,
        "eff_folder_prov": eff_folder_prov,
        "prov_is_ancestor": prov_is_ancestor,
        "sp_snap_candidates": sp_snap_candidates,
        "sp_selected": sp_selected,
    }


def _resolve_recording_subject_and_sidecar(
    raw: Any,
    header: Any,
    rec_path: Path,
    rec_entry: Path,
    rec_resolved: Path,
    rec_slug: str,
    root_folder: Path | None,
    logical_input_path: Path | None,
    paired: dict[str, Any],
    snapped_sidecars: dict[str, Any],
    eff_folder_prov: dict[str, Any],
    prov_is_ancestor: bool,
    sp_snap_candidates: list[tuple[str, bool, dict[str, Any], Path]],
    sp_selected: tuple[str, bool, dict[str, Any], Path] | None,
) -> tuple[dict[str, Any], Path | None, dict[str, Any], list[str], list[str], str | None, list[tuple[str, bool, dict[str, Any], Path]], tuple[str, bool, dict[str, Any], Path] | None]:
    """Cross-check subject identities across native header, BIDS path, sidecar, & provenance, and disambiguate fuzzy sidecars."""
    orig_parent_res = rec_entry.parent
    sidecar = paired["sidecar"]
    orig_sidecar_path: Path | None = None
    if paired["sidecar_path"] is not None:
        sc_name = Path(paired["sidecar_path"]).name
        sc_entry = snapped_sidecars.get(sc_name)
        orig_sidecar_path = sc_entry[0] if sc_entry is not None else (orig_parent_res / sc_name)

    native_sub = _extract_native_subject_id(raw, header)
    raw_logical_path = Path(os.path.abspath(str(logical_input_path or rec_path)))
    bids_subs = list(
        dict.fromkeys([*_bids_subject_ids_from_path(raw_logical_path, None), *_bids_subject_ids_from_path(rec_entry, root_folder)])
    )
    prov_sub = _extract_prov_participant_id(eff_folder_prov)
    prov_tok = _norm_subject_token(prov_sub)

    if paired.get("sidecar_fuzzy"):
        rec_hint_tok = next((_norm_subject_token(s) for s in (*bids_subs, native_sub) if _norm_subject_token(s)), None) or prov_tok
        fuzzy_exact: list[tuple[Path, dict[str, Any]]] = []
        fuzzy_untagged: list[tuple[Path, dict[str, Any]]] = []
        for f_path, f_doc in paired.get("sidecar_fuzzy_candidates") or []:
            f_toks, _ = _extract_doc_subject_tokens(f_doc, f_path.name)
            if len(f_toks) > 1:
                continue
            f_tok = next(iter(f_toks), None)
            if rec_hint_tok and f_tok == rec_hint_tok:
                fuzzy_exact.append((f_path, f_doc))
            elif f_tok is None:
                fuzzy_untagged.append((f_path, f_doc))
        chosen_fuzzy = fuzzy_exact[0] if fuzzy_exact else (fuzzy_untagged[0] if fuzzy_untagged else None)
        if chosen_fuzzy is None:
            sidecar, orig_sidecar_path = {}, None
        else:
            f_path, sidecar = chosen_fuzzy
            sc_entry = snapped_sidecars.get(f_path.name)
            orig_sidecar_path = sc_entry[0] if sc_entry is not None else (orig_parent_res / f_path.name)

    _, sidecar_all_subs = _extract_doc_subject_tokens(
        sidecar, orig_sidecar_path.name if orig_sidecar_path is not None else None
    )
    _, bids_json_subs = _extract_doc_subject_tokens(paired["bids_json"])
    sidecar_subs = [*sidecar_all_subs, *bids_json_subs]
    direct_norm = {
        token for label in (*sidecar_subs, *bids_subs, native_sub)
        if (token := _norm_subject_token(label))
    }
    # The normalized header token participates only in comparisons, never in diagnostics.
    public_subs = [*sidecar_subs, *bids_subs]
    if len(direct_norm) > 1:
        raise ToolkitError(
            "conflicting_subjects",
            f"Recording {rec_entry.name!r} declares conflicting subject identities across header/BIDS path/sidecar.",
            recording=rec_slug,
            subjects=sorted(set(public_subs)),
        )
    direct_tok = next(iter(direct_norm), None)
    if direct_tok and prov_tok and direct_tok != prov_tok:
        if prov_is_ancestor:
            eff_folder_prov = {}
            sp_snap_candidates = [c for c in sp_snap_candidates if not c[1]]
            if sp_selected is not None and sp_selected[1]:
                sp_selected = None
        else:
            raise ToolkitError(
                "conflicting_subjects",
                f"Recording {rec_resolved.name!r} subject conflicts with folder PROVENANCE.json ({prov_sub!r}).",
                recording=rec_slug,
                subjects=[*public_subs, str(prov_sub)],
            )
    return sidecar, orig_sidecar_path, eff_folder_prov, sidecar_subs, bids_subs, direct_tok, sp_snap_candidates, sp_selected


def _build_channels_and_markers(
    raw: Any,
    header: Any,
    rec_path: Path,
    modality: str,
    sidecar: dict[str, Any],
    paired: dict[str, Any],
    mne_ch_types: list[str],
    is_fif: bool,
) -> tuple[list[ChannelSpec], list[MarkerSpec], list[str]]:
    sfx = rec_path.suffix.lower()
    sfreq = float(raw.info["sfreq"])
    duration = int(raw.n_times) / sfreq
    if pieces := getattr(raw, "_brainlens_pieces", None):
        duration = pieces[-1].time_at(pieces[-1].stop_sample, sfreq)
    headers = None
    if header is not None and getattr(header, "signals", None) is not None:
        raw_hdrs = list(header.signals)
        if sfx in {".edf", ".bdf"} and len(raw_hdrs) != len(raw.ch_names):
            raw_hdrs = [
                s for s in raw_hdrs
                if str(getattr(s, "label", "")).strip().upper() not in {"EDF ANNOTATIONS", "BDF ANNOTATIONS"}
            ]
        headers = raw_hdrs
    if headers is not None and len(headers) != len(raw.ch_names):
        raise ToolkitError("header_mismatch", f"Header channel count differs from reader in {rec_path.name}")

    ch_meta_map = sidecar.get("channels") if isinstance(sidecar.get("channels"), dict) else {}
    warnings: list[str] = []
    channels: list[ChannelSpec] = []
    default_kind = "seeg" if modality == "seeg" else ("ecog" if modality == "ieeg" else "eeg")

    for i, name in enumerate(raw.ch_names):
        cid = f"ch{i:04d}"
        cinfo = ch_meta_map.get(cid, ch_meta_map.get(name, {}))
        h = headers[i] if headers is not None else None
        src_unit = h.physical_dimension.strip() if h is not None else ("V" if int(raw.info["chs"][i]["unit"]) == 107 else None)
        unit = "V" if (src_unit and src_unit in _VOLT_UNITS) else ("dimensionless" if src_unit in {"1", "dimensionless"} else "unknown")
        if unit != "V":
            warnings.append(f"{name} ({cid}): physical voltage units unverified ({src_unit!r}); HFO detection skipped on this channel.")
        mne_kind = mne_ch_types[i] if i < len(mne_ch_types) else ""
        inferred_mne_kind = mne_kind if (mne_kind in _ALLOWED_KINDS and (is_fif or mne_kind != "eeg")) else None
        kind = str(cinfo.get("kind") or paired["bids_channel_kinds"].get(name) or inferred_mne_kind or default_kind).lower()
        if kind not in _ALLOWED_KINDS:
            kind = default_kind
        src_fs = float(h.sampling_frequency) if h is not None else sfreq
        src_lp = _prefilter_lowpass(h.prefiltering) if h is not None else raw.info.get("lowpass")
        sgroup = parse_shaft_group(name, kind)
        channels.append(
            ChannelSpec(
                id=cid,
                name=name,
                original_label=h.label if h is not None else name,
                kind=kind,
                unit=unit,  # type: ignore[arg-type]
                source_unit=src_unit or None,
                source_sfreq=src_fs,
                source_lowpass=float(src_lp) if src_lp else None,
                contacts=list(cinfo.get("contacts", [])),
                shaft=sgroup["shaft"],
                contact_index=sgroup["contact_index"],
                is_auxiliary=sgroup["is_auxiliary"],
            )
        )

    markers: list[MarkerSpec] = []
    for ann in getattr(raw, "_brainlens_annotations", raw.annotations):
        raw_onset = float(ann["onset"]) - raw.first_time
        raw_dur = max(0.0, float(ann["duration"]))
        onset, end = max(0.0, raw_onset), min(duration, raw_onset + raw_dur)
        if (raw_dur == 0.0 and 0.0 <= raw_onset <= duration) or end > onset:
            markers.append(
                MarkerSpec(
                    start=onset,
                    end=max(onset, end),
                    label=str(ann["description"]),
                    source="recording_annotation",
                    channels=[channels[raw.ch_names.index(n)].id for n in ann.get("ch_names", ()) if n in raw.ch_names],
                )
            )
    for m_item in sidecar.get("markers", []):
        markers.append(MarkerSpec.model_validate(m_item))
    for bad_name in sorted(set(raw.info.get("bads", [])) | set(paired["bids_bad_channels"])):
        if bad_name in raw.ch_names:
            markers.append(MarkerSpec(start=0.0, end=duration, label="BAD_channel", source="channel_status", channels=[bad_name]))
    return channels, markers, warnings


def _resolve_bipolar_and_spatial(
    snap_dir: Path,
    rec_entry: Path,
    rec_slug: str,
    sidecar: dict[str, Any],
    paired: dict[str, Any],
    snapped_sidecars: dict[str, Any],
    companion_files: dict[str, dict[str, Any]],
    direct_subs: list[str],
    eff_subject_tok: str | None,
    explicit_spatial_sidecar: bool,
    sp_snap_candidates: list[tuple[str, bool, dict[str, Any], Path]],
    sp_selected: tuple[str, bool, dict[str, Any], Path] | None,
) -> tuple[Path | None, str | None, SpatialSpec | None, Path | None]:
    orig_parent_res = rec_entry.parent
    orig_bipolar_file: Path | None = None
    bp_sha256: str | None = None

    explicit_bp_rel = (
        str(sidecar["provenance"]["published_bipolar_pairs_file"]).strip()
        if isinstance(sidecar.get("provenance"), dict) and sidecar["provenance"].get("published_bipolar_pairs_file")
        else None
    )
    if explicit_bp_rel:
        bp_logical = _canonical_entry_path(orig_parent_res / explicit_bp_rel)
        cand_bp = bp_logical.resolve()
        if not cand_bp.is_relative_to(orig_parent_res):
            raise ToolkitError("unsafe_sidecar_path", f"Sidecar published_bipolar_pairs_file escapes recording directory: {explicit_bp_rel!r}")
        if cand_bp.is_file():
            bp_snap_cand = snap_dir / Path(explicit_bp_rel).name
            sc_entry = snapped_sidecars.get(bp_snap_cand.name) if (bp_snap_cand.is_file() and Path(explicit_bp_rel).parent == Path(".")) else None
            if sc_entry is not None:
                orig_bipolar_file, bp_sha256, bp_size, bp_st = sc_entry
            else:
                orig_bipolar_file = cand_bp
                bp_snap_cand = snap_dir / f"__brainlens_bp_{cand_bp.name}"
                bp_sha256, bp_size, bp_st = _copy_and_hash_file(cand_bp, bp_snap_cand)
            try:
                bp_doc_chk = read_json(bp_snap_cand)
            except Exception:
                bp_doc_chk = {}
            if isinstance(bp_doc_chk, dict):
                bp_toks, bp_lbls = _extract_doc_subject_tokens(bp_doc_chk, cand_bp.name)
                if len(bp_toks) > 1 or (bp_toks and (not eff_subject_tok or next(iter(bp_toks)) != eff_subject_tok)):
                    raise ToolkitError(
                        "conflicting_subjects",
                        f"Referenced published_bipolar_pairs_file {cand_bp.name!r} declares subject {sorted(set(bp_lbls))} "
                        f"which conflicts with recording {rec_entry.name!r} ({direct_subs[0] if direct_subs else 'anonymous'}).",
                        recording=rec_slug,
                        subjects=sorted(set(bp_lbls + direct_subs)),
                    )
                bp_srcs = _extract_declared_source_files(bp_doc_chk)
                if bp_srcs and not _source_file_matches_recording(bp_srcs, rec_entry, base_dir=orig_parent_res):
                    raise ToolkitError(
                        "conflicting_subjects",
                        f"Referenced published_bipolar_pairs_file {cand_bp.name!r} explicitly targets {bp_srcs}, not {rec_entry.name!r}.",
                        recording=rec_slug,
                    )
            companion_files[str(orig_bipolar_file)] = {
                "sha256": bp_sha256,
                "size": int(bp_size),
                "mtime_ns": int(bp_st[3]),
                "logical_path": str(bp_logical),
            }
    else:
        bp_explicit_matches: list[Path] = []
        bp_generic_matches: list[Path] = []
        all_bp_snaps = list(dict.fromkeys([*(Path(p) for p in (paired.get("bipolar_candidates") or [])), *sorted(snap_dir.glob("*bipolar*pairs*.json"))]))
        for bp_cand in all_bp_snaps:
            if not bp_cand.is_file():
                continue
            ok_bp, is_exp = _bipolar_pairs_file_matches_recording(
                bp_cand, rec_entry, orig_parent_res, target_subject_tok=eff_subject_tok, reject_tagged_if_untargeted=True
            )
            if ok_bp:
                (bp_explicit_matches if is_exp else bp_generic_matches).append(bp_cand)
        chosen_bp_snap = bp_explicit_matches[0] if len(bp_explicit_matches) == 1 else (
            bp_generic_matches[0] if (not bp_explicit_matches and len(bp_generic_matches) == 1) else None
        )
        if chosen_bp_snap is not None:
            sc_entry = snapped_sidecars.get(chosen_bp_snap.name)
            if sc_entry is not None:
                orig_bipolar_file, bp_sha256, bp_size, bp_st = sc_entry
                companion_files[str(orig_bipolar_file)] = {
                    "sha256": bp_sha256,
                    "size": int(bp_size),
                    "mtime_ns": int(bp_st[3]),
                    "logical_path": str(_canonical_entry_path(orig_parent_res / chosen_bp_snap.name)),
                }

    sp_orig_name, _, sp_snap_doc, sp_orig_path = sp_selected if sp_selected is not None else (None, False, None, None)
    if explicit_spatial_sidecar and sp_snap_doc is not None:
        exp_sp_toks, exp_sp_lbls = _extract_doc_subject_tokens(sp_snap_doc, sp_orig_name)
        if len(exp_sp_toks) > 1:
            raise ToolkitError(
                "conflicting_subjects",
                f"Spatial sidecar {sp_orig_name!r} declares contradictory subject identities: {sorted(set(exp_sp_lbls))}.",
                recording=rec_slug,
                subjects=sorted(set(exp_sp_lbls)),
            )
    elif not explicit_spatial_sidecar:
        if sp_snap_candidates:
            subj_exact, subj_untagged = [], []
            for cand in sp_snap_candidates:
                sp_toks, _ = _extract_doc_subject_tokens(cand[2], cand[0])
                if len(sp_toks) > 1:
                    continue
                sp_sub_tok = next(iter(sp_toks), None)
                if eff_subject_tok and sp_sub_tok == eff_subject_tok:
                    subj_exact.append(cand)
                elif sp_sub_tok is None:
                    subj_untagged.append(cand)
            pool_ov = subj_exact or subj_untagged
            if pool_ov:
                chosen_ov = next((c for c in pool_ov if "correction" in c[0]), pool_ov[0])
                sp_orig_name, _, sp_snap_doc, sp_orig_path = chosen_ov
            else:
                sp_snap_doc, sp_orig_path = None, None
        elif sp_snap_doc is not None:
            sp_toks, _ = _extract_doc_subject_tokens(sp_snap_doc, sp_orig_name)
            sp_sub_tok = next(iter(sp_toks), None)
            if len(sp_toks) > 1 or (sp_sub_tok and (not eff_subject_tok or sp_sub_tok != eff_subject_tok)):
                sp_snap_doc, sp_orig_path = None, None

    spatial: SpatialSpec | None = None
    chosen_spatial_overlay_path: Path | None = None
    if explicit_spatial_sidecar and sp_snap_doc is not None:
        sp_raw = sp_snap_doc.get("spatial") if "spatial" in sp_snap_doc else sp_snap_doc
        spatial = SpatialSpec.model_validate({k: v for k, v in sp_raw.items() if k in SpatialSpec.model_fields})
        chosen_spatial_overlay_path = sp_orig_path
    elif isinstance(sidecar.get("spatial"), dict):
        sp_raw = sidecar["spatial"]
        spatial = SpatialSpec.model_validate({k: v for k, v in sp_raw.items() if k in SpatialSpec.model_fields})
    elif sp_snap_doc is not None:
        sp_raw = sp_snap_doc.get("spatial") if "spatial" in sp_snap_doc else sp_snap_doc
        spatial = SpatialSpec.model_validate({k: v for k, v in sp_raw.items() if k in SpatialSpec.model_fields})
        chosen_spatial_overlay_path = sp_orig_path

    return orig_bipolar_file, bp_sha256, spatial, chosen_spatial_overlay_path


def inspect_single_recording(
    rec_path: Path,
    rec_slug: str,
    modality_override: str | None = None,
    folder_prov: dict[str, Any] | None = None,
    spatial_overlay_path: Path | None = None,
    folder_prov_path: Path | None = None,
    root_folder: Path | None = None,
    *,
    explicit_spatial_sidecar: bool = False,
    logical_input_path: Path | None = None,
    progress_cb: ProgressCallback | None = None,
) -> RecordingSpec:
    sfx = rec_path.suffix.lower()
    rec_entry = _canonical_entry_path(rec_path)
    orig_parent_res = rec_entry.parent
    rec_resolved = rec_entry.resolve()
    eff_stop_root = root_folder.resolve() if root_folder is not None else orig_parent_res

    orig_sp_arg, orig_prov_arg = spatial_overlay_path, folder_prov_path
    anc_before = (_ancestor_chain_sig(rec_entry), _ancestor_chain_sig(rec_resolved))
    pre_main_st = (str(rec_resolved), _stat_tuple(rec_resolved))
    pre_sidecar_sig = _recording_sidecar_dir_sig(rec_entry, orig_sp_arg, orig_prov_arg, stop_root=eff_stop_root)

    ctx_overlay_paths: list[Path] = []
    if folder_prov is None and folder_prov_path is None:
        folder_prov_path, ctx_overlays = _find_recording_folder_context(orig_parent_res, stop_root=eff_stop_root, rec_path=rec_entry)
        if spatial_overlay_path is None and ctx_overlays:
            ctx_overlay_paths = [Path(o["path"]) for o in ctx_overlays]
            preferred = next((o for o in ctx_overlays if "correction" in o["name"]), ctx_overlays[0])
            spatial_overlay_path = Path(preferred["path"])
    if spatial_overlay_path is not None and not ctx_overlay_paths:
        ctx_overlay_paths = [spatial_overlay_path]

    with ExitStack() as stack:
        snap_tmp = stack.enter_context(tempfile.TemporaryDirectory(prefix="brainlens_insp_"))
        snap_dir = Path(snap_tmp).resolve()
        try:
            os.chmod(snap_dir, 0o700)
        except OSError:
            pass

        staged = _stage_inspection_snapshot(
            snap_dir,
            rec_entry,
            rec_resolved,
            rec_slug,
            pre_main_st,
            folder_prov,
            folder_prov_path,
            ctx_overlay_paths,
            explicit_spatial_sidecar,
            stack,
            progress_cb,
        )
        _assert_recording_snapshot_stable(
            rec_entry, rec_slug, anc_before, pre_main_st, pre_sidecar_sig, orig_sp_arg, orig_prov_arg, eff_stop_root, phase="snapshot"
        )

        snap_main: Path = staged["snap_main"]
        try:
            paired = _discover_recording_sidecars(snap_main, logical_rec_path=rec_entry, orig_folder=orig_parent_res)
        except TypeError:
            paired = _discover_recording_sidecars(snap_main)
        bids_json = paired["bids_json"]

        header_path = snap_main
        if snap_main in staged["live_signals"] and sfx in {".edf", ".bdf"}:
            # edfio and the waveform reader each open the EDF header. Bind both handles
            # to the hash, even if a parent directory is briefly replaced then restored.
            descriptor = stack.enter_context(snap_main.open("rb"))
            if _stat_fields(os.fstat(descriptor.fileno())) != staged["rec_st"]:
                raise ToolkitError("source_recording_modified", "Source was replaced before opening its EDF/BDF header.",
                                   recording=rec_slug)
            header_path = Path("/dev/fd") / str(descriptor.fileno())
        header = (
            _parse_brainvision_header(snap_main, verified_components={p.resolve() for p in staged["live_signals"]})
            if sfx == ".vhdr" else (edfio.read_edf(header_path, lazy_load_data=True) if sfx == ".edf"
            else (_parse_bdf_header(header_path) if sfx == ".bdf"
            else (parse_vendor_header(snap_main) if is_vendor_recording_suffix(rec_entry) else None)))
        )
        raw = open_mne_raw(snap_main, preload=False)
        try:
            header_terms = native_header_terms(raw, header)
            live_signals = staged["live_signals"]
            live_paths = {p.resolve() for p in live_signals}
            opened_stat = getattr(raw, "_brainlens_fd_stat", None)
            if snap_main.is_symlink() and opened_stat is not None and opened_stat != staged["rec_st"]:
                raise ToolkitError("source_recording_modified", "Signal reader opened a different source during inspection.",
                                   recording=rec_slug)
            for fn in (getattr(raw, "filenames", None) or ()):
                if fn and not (Path(str(fn)).resolve().is_relative_to(snap_dir) or Path(str(fn)).resolve() in live_paths):
                    raise ToolkitError(
                        "source_recording_modified",
                        f"Multi-part recording {rec_resolved.name!r} referenced external unverified part {Path(str(fn)).name!r}; retry on a stable file.",
                        recording=rec_slug,
                    )
            n_samples, sfreq, first_samp = int(raw.n_times), float(raw.info["sfreq"]), int(raw.first_samp)
            mne_ch_types = [str(t).lower() for t in raw.get_channel_types()]
            is_fif = rec_path.name.lower().endswith((".fif", ".fif.gz"))

            (
                sidecar,
                orig_sidecar_path,
                eff_folder_prov,
                sidecar_subs,
                bids_subs,
                direct_tok,
                sp_snap_candidates,
                sp_selected,
            ) = _resolve_recording_subject_and_sidecar(
                raw,
                header,
                rec_path,
                rec_entry,
                rec_resolved,
                rec_slug,
                root_folder,
                logical_input_path,
                paired,
                staged["snapped_sidecars"],
                staged["eff_folder_prov"],
                staged["prov_is_ancestor"],
                staged["sp_snap_candidates"],
                staged["sp_selected"],
            )

            modality = modality_override or _infer_modality(
                rec_path,
                sidecar,
                bids_json,
                eff_folder_prov,
                bids_channel_kinds=paired["bids_channel_kinds"],
                mne_ch_types=mne_ch_types,
                is_fif=is_fif,
            )
            channels, markers, warnings = _build_channels_and_markers(
                raw, header, rec_path, modality, sidecar, paired, mne_ch_types, is_fif
            )
            eff_subject_tok = direct_tok or _norm_subject_token(_extract_prov_participant_id(eff_folder_prov))
            orig_bipolar_file, bp_sha256, spatial, chosen_spatial_overlay_path = _resolve_bipolar_and_spatial(
                snap_dir,
                rec_entry,
                rec_slug,
                sidecar,
                paired,
                staged["snapped_sidecars"],
                staged["companion_files"],
                [*sidecar_subs, *bids_subs],
                eff_subject_tok,
                explicit_spatial_sidecar,
                sp_snap_candidates,
                sp_selected,
            )

            ref_raw = str(sidecar.get("reference") or "").lower()
            if ref_raw not in {"common", "bipolar", "average", "unknown"}:
                bids_ref = str(bids_json.get("iEEGReference") or "").lower()
                ref_raw = "bipolar" if "bipolar" in bids_ref else "unknown"
            line_freq = sidecar.get("line_freq") or bids_json.get("PowerLineFrequency") or raw.info.get("line_freq")

            _assert_recording_snapshot_stable(
                rec_entry, rec_slug, anc_before, pre_main_st, pre_sidecar_sig, orig_sp_arg, orig_prov_arg, eff_stop_root, phase="registration"
            )
            for alias, (src, identity) in live_signals.items():
                if (alias.is_symlink() and _stat_tuple(alias) != identity[2]) or _hash_in_place(src, recording=rec_slug) != identity:
                    raise ToolkitError("source_recording_modified", "Signal file changed during source registration.",
                                       recording=rec_slug)
            for src, meta in staged["companion_files"].items():
                if meta.get("logical_path") and Path(meta["logical_path"]).resolve() != Path(src):
                    raise ToolkitError("source_recording_modified", "Companion filename changed target during inspection.",
                                       recording=rec_slug)

            fp = _compute_v2_fingerprint(
                staged["rec_sha256"], staged["companion_hashes"], modality, ref_raw, line_freq, channels, markers, bp_sha256
            )
            prov = _extract_provenance(rec_entry, sidecar, orig_sidecar_path, bids_json, eff_folder_prov)
            declared_label = prov.participant_id or next(iter(sidecar_subs), None) or _extract_prov_participant_id(eff_folder_prov)
            prov = registration_provenance(
                prov, sidecar_label=declared_label, bids_label=next(iter(bids_subs), None),
                native_label=_extract_native_subject_id(raw, header),
            )
            recording = RecordingSpec(
                id=f"rec_{fp[:16]}",
                slug=sanitize_slug(rec_slug),
                path=str(rec_entry),
                fingerprint=fp,
                source_sha256=staged["rec_sha256"],
                source_file_size=int(staged["rec_size"]),
                source_file_mtime_ns=int(staged["rec_st"][3]),
                companion_files=staged["companion_files"],
                modality=modality,  # type: ignore[arg-type]
                sfreq=sfreq,
                n_samples=n_samples,
                source_first_sample=first_samp,
                pieces=getattr(raw, "_brainlens_pieces", []),
                channels=channels,
                reference=ref_raw,  # type: ignore[arg-type]
                reference_description=sidecar.get("reference_description"),
                highpass=float(raw.info["highpass"]) if raw.info.get("highpass") is not None else None,
                lowpass=float(raw.info["lowpass"]) if raw.info.get("lowpass") is not None else None,
                line_freq=float(line_freq) if line_freq is not None else None,
                markers=markers,
                spatial=spatial,
                provenance=prov,
                published_bipolar_pairs_path=str(orig_bipolar_file) if orig_bipolar_file else None,
                spatial_overlay_path=str(chosen_spatial_overlay_path) if chosen_spatial_overlay_path else None,
                warnings=warnings,
            )
            recording._subject_digest = canonical_digest(eff_subject_tok) if eff_subject_tok else None
            recording._header_terms = tuple(header_terms)
            from .timeline import header_clock
            recording._header_clock = header_clock(raw, rec_entry)
            return recording
        finally:
            raw.close()


def inspect_case(
    target: str,
    workspace_dir: str | Path | None = None,
    modality: str | None = None,
    case_name: str | None = None,
    spatial_sidecar: str | Path | None = None,
    persist: bool = True,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Inspect a local file, patient/BIDS directory, or registered case."""
    store = WorkbenchStore(workspace_dir)
    spec = resolve_case_target(target, store)
    sp_path = Path(spatial_sidecar).expanduser().resolve() if spatial_sidecar else None
    if sp_path is not None and not sp_path.is_file():
        alt = (PROJECT_ROOT / str(spatial_sidecar)).resolve()
        sp_path = alt if alt.is_file() else sp_path

    if spec["kind"] in {"registered_case", "registered_recording"}:
        return _inspect_registered_case(store, spec, sp_path, persist=persist)

    fs_path: Path = spec["path"]
    logical_target_path: Path | None = spec.get("logical_path")
    resolved_src = str(_canonical_entry_path(fs_path))
    check_unsupported_vendor_target(fs_path)
    if fs_path.is_file():
        rec_files, folder, auto_base_slug = [fs_path], fs_path.parent, sanitize_slug(recording_stem(fs_path))
    else:
        rec_files = _discover_folder_recordings(fs_path)
        if not rec_files:
            check_unsupported_vendor_folder(fs_path)
            raise ToolkitError("no_recordings", f"No supported EEG files found in {fs_path}")
        folder, auto_base_slug = fs_path, sanitize_slug(fs_path.name)

    folder_prov, root_overlays, readme = _find_folder_provenance_and_overlays(folder)
    raw_inspected: list[tuple[Path, RecordingSpec, Path | None]] = []
    origins: set[str] = set()
    all_overlays_by_path: dict[str, dict[str, Any]] = {o["path"]: o for o in root_overlays}
    subject_labels: dict[str, str] = {}

    for rf in rec_files:
        if fs_path.is_file() and logical_target_path is not None:
            rf_logical = logical_target_path
        elif fs_path.is_dir() and logical_target_path is not None:
            try:
                rf_logical = logical_target_path / rf.relative_to(fs_path.resolve())
            except ValueError:
                rf_logical = rf
        else:
            rf_logical = rf
        rmodel = inspect_single_recording(
            rf,
            rec_slug=sanitize_slug(recording_stem(rf)),
            modality_override=modality,
            folder_prov=None,
            spatial_overlay_path=sp_path,
            folder_prov_path=None,
            root_folder=folder,
            explicit_spatial_sidecar=sp_path is not None,
            logical_input_path=rf_logical,
            progress_cb=progress_cb,
        )
        rec_sp = Path(rmodel.spatial_overlay_path) if rmodel.spatial_overlay_path else (sp_path if rmodel.spatial is not None else None)
        if sp_path is None:
            _, rec_overlays = _find_recording_folder_context(rf.parent, stop_root=folder, rec_path=rf)
            for ov in rec_overlays:
                all_overlays_by_path[ov["path"]] = ov
        origins.add(rmodel.provenance.origin_kind)
        if rmodel._subject_digest:
            subject_labels[rmodel._subject_digest] = rmodel.provenance.participant_id or HEADER_SUBJECT_WITHHELD
        raw_inspected.append((rf, rmodel, rec_sp if rmodel.spatial is not None else None))

    if len(subject_labels) > 1:
        distinct_subjects = sorted(set(subject_labels.values()))
        raise ToolkitError(
            "conflicting_subjects",
            f"Directory {folder.name!r} contains recordings belonging to multiple distinct subjects "
            f"({', '.join(distinct_subjects)}); refusing to merge distinct patients into one case. "
            "Import each subject's folder separately (or use cohort batch mode).",
            subjects=distinct_subjects,
        )

    folder_subject = subject_digest(_extract_prov_participant_id(folder_prov))
    prov_matches_case = not (folder_subject and subject_labels and folder_subject not in subject_labels)
    case_overlays = _filter_case_overlays(list(all_overlays_by_path.values()), subject_labels)

    def _make_case_doc(cslug: str, existing_case: dict[str, Any], rec_summaries: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "case": cslug,
            "source_target": resolved_src,
            "recording_count": len(rec_summaries),
            "origin_kinds": sorted(origins),
            "same_subject_association": folder_prov.get("same_subject_association") if prov_matches_case else None,
            "readme_note": readme,
            "available_spatial_sidecars": case_overlays,
            "open_questions": existing_case.get("open_questions", []),
            "recordings": rec_summaries,
            "created_at": existing_case.get("created_at", utc_now()),
        }

    runs_to_refresh_spatial: list[tuple[str, str, str]] = []
    if persist:
        store.root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(store.root / ".cases_registry.lock")):
            cslug = _allocate_case_slug(store, case_name, auto_base_slug, fs_path, resolved_src)
            _check_existing_case_subject_consistency(store, cslug, subject_labels)
            store.case_dir(cslug).mkdir(parents=True, exist_ok=True)
            with FileLock(str(store.case_dir(cslug) / ".case.lock")):
                existing_case = store.load_case(cslug) if store.has_case(cslug) else {}
                inspected_recs = _allocate_recording_slugs(store, cslug, folder, raw_inspected)
                rec_summaries: list[dict[str, Any]] = []
                for rslug, rmodel, rec_sp in inspected_recs:
                    prev_sp_digest: str | None = None
                    prev_fp: str | None = None
                    prev_wal_floor = 0
                    try:
                        if store.recording_path(cslug, rslug).is_file():
                            prev_doc = store.load_recording_doc(cslug, rslug)
                            prev_wal_floor = int(prev_doc.get("replaced_at_wal_index") or 0)
                            prev_rec = RecordingSpec.model_validate(prev_doc["spec"])
                            prev_fp = prev_rec.fingerprint
                            prev_sp_digest = canonical_digest(
                                prev_rec.spatial.model_dump(mode="json") if prev_rec.spatial is not None else None
                            )
                    except Exception:
                        prev_sp_digest = None
                    new_wal_floor = prev_wal_floor
                    if prev_fp is not None and prev_fp != rmodel.fingerprint:
                        with FileLock(str(store.case_dir(cslug) / ".judgments.lock")):
                            new_wal_floor = max(prev_wal_floor, len(store.load_judgments(cslug)))
                    new_sp_digest = canonical_digest(
                        rmodel.spatial.model_dump(mode="json") if rmodel.spatial is not None else None
                    )
                    rec_extra: dict[str, Any] = {"spatial_overlay_path": str(rec_sp) if rec_sp else None}
                    if new_wal_floor > 0:
                        rec_extra["replaced_at_wal_index"] = new_wal_floor
                    store.save_recording(cslug, rslug, rmodel, extra=rec_extra)
                    existing_runs = store.list_runs(cslug, rslug)
                    for rm in existing_runs:
                        rn = rm.get("run_name")
                        if not rn or rm.get("recording_fingerprint") != rmodel.fingerprint:
                            continue
                        if (prev_sp_digest is not None and prev_sp_digest != new_sp_digest) or (
                            rm.get("recording_spatial_digest") != new_sp_digest
                        ):
                            runs_to_refresh_spatial.append((cslug, rslug, str(rn)))
                    rec_summaries.append(_recording_brief(rmodel, existing_runs))
                case_doc = _make_case_doc(cslug, existing_case, rec_summaries)
                store.save_case(cslug, case_doc, _locked=True)
        if runs_to_refresh_spatial:
            from .pipeline import refresh_verdict_tallies

            for rc_slug, rr_slug, rn_slug in runs_to_refresh_spatial:
                try:
                    refresh_verdict_tallies(store, rc_slug, rr_slug, rn_slug)
                except Exception:
                    continue
    else:
        cslug = _allocate_case_slug(store, case_name, auto_base_slug, fs_path, resolved_src)
        _check_existing_case_subject_consistency(store, cslug, subject_labels)
        existing_case = store.load_case(cslug) if store.has_case(cslug) else {}
        inspected_recs = _allocate_recording_slugs(store, cslug, folder, raw_inspected)
        case_doc = _make_case_doc(cslug, existing_case, [_recording_brief(rmodel, []) for _, rmodel, _ in inspected_recs])
        case_doc["_recording_specs"] = {rslug: rmodel for rslug, rmodel, _ in inspected_recs}
    return {"workspace": str(store.root), **case_doc}
