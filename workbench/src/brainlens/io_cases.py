"""Case target resolution, collision-free slug allocation, registered case preview/update, and summaries."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from filelock import FileLock

from .deid import HEADER_SUBJECT_WITHHELD
from .io_integrity import _canonical_entry_path
from .io_sidecars import _extract_doc_subject_tokens
from .io_signals import _REC_SUFFIXES, recording_stem
from .models import PROJECT_ROOT, RecordingSpec, SpatialSpec, ToolkitError, canonical_digest, read_json
from .storage import WorkbenchStore, sanitize_slug


def navigation_recording_slug(store: WorkbenchStore, case: str, token: str) -> str:
    """An opaque recording ID remains usable when its filename contains a private date."""
    slug = sanitize_slug(token)
    if re.fullmatch(r"rec_[0-9a-f]{16}", slug):
        names = store.list_recordings(case)
        if slug not in names:
            matches = [name for name in names if store.load_recording(case, name).id == slug]
            if len(matches) == 1:
                return matches[0]
    return slug


def resolve_case_target(target: str, store: WorkbenchStore) -> dict[str, Any]:
    raw = (target or "").strip()
    if not raw:
        raise ToolkitError("empty_target", "Provide a file/folder path or registered case name.")
    for cand in (Path(raw).expanduser(), (PROJECT_ROOT / raw).expanduser()):
        if cand.exists():
            return {"kind": "filesystem", "path": _canonical_entry_path(cand), "logical_path": Path(os.path.abspath(str(cand)))}
    for sep in ("/", ":"):
        if sep in raw:
            left, right = raw.split(sep, 1)
            cslug = sanitize_slug(left)
            rslug = navigation_recording_slug(store, cslug, right)
            if store.has_case(cslug) and rslug in store.list_recordings(cslug):
                return {"kind": "registered_recording", "case": cslug, "recording": rslug}
    cslug = sanitize_slug(raw)
    if store.has_case(cslug):
        return {"kind": "registered_case", "case": cslug}
    inp = (PROJECT_ROOT / "inputs" / raw).expanduser()
    if inp.exists():
        return {"kind": "filesystem", "path": _canonical_entry_path(inp), "logical_path": Path(os.path.abspath(str(inp)))}
    raise ToolkitError("target_not_found", f"No path or registered case matches {target!r}.")


def _recording_brief(rmodel: RecordingSpec, runs: list[dict[str, Any]]) -> dict[str, Any]:
    shaft_counts: dict[str, int] = {}
    for c in rmodel.channels:
        shaft_counts[c.shaft] = shaft_counts.get(c.shaft, 0) + 1
    bad_chs = sorted({ch for m in rmodel.markers if m.label.upper().startswith("BAD") for ch in (m.channels or ["ALL"])})
    return {
        "recording": rmodel.slug,
        "source_file": Path(rmodel.path).name,
        "modality": rmodel.modality,
        "sfreq_hz": rmodel.sfreq,
        "duration_seconds": round(rmodel.duration, 3),
        **({"piece_count": len(rmodel.pieces), "acquired_seconds": rmodel.n_samples / rmodel.sfreq}
           if rmodel.pieces else {}),
        "channel_count": len(rmodel.channels),
        "eeg_channel_count": sum(1 for c in rmodel.channels if not c.is_auxiliary),
        "verified_voltage_channels": sum(1 for c in rmodel.channels if c.unit == "V"),
        "reference": rmodel.reference,
        "reference_description": rmodel.reference_description,
        "line_freq_hz": rmodel.line_freq,
        "lowpass_hz": rmodel.lowpass,
        "declared_bad_channels": bad_chs,
        "published_bipolar_pairs_available": rmodel.published_bipolar_pairs_path is not None,
        "shaft_groups": shaft_counts,
        "spatial": (
            {
                "available": True,
                "coordinate_system": rmodel.spatial.coordinate_system,
                "units": rmodel.spatial.units,
                "template": rmodel.spatial.template,
                "mapped_positions": len(rmodel.spatial.positions),
                "source": rmodel.spatial.source,
            }
            if rmodel.spatial is not None
            else {"available": False}
        ),
        "provenance": rmodel.provenance.model_dump(mode="json"),
        "warnings": rmodel.warnings,
        "saved_runs": [r.get("run_name") for r in runs if r.get("run_name")],
    }


def summarize_registered_case(store: WorkbenchStore, case_slug: str) -> dict[str, Any]:
    cdoc = store.load_case(case_slug)
    recs = [
        _recording_brief(store.load_recording(case_slug, r), store.list_runs(case_slug, r))
        for r in store.list_recordings(case_slug)
    ]
    return {
        "workspace": str(store.root),
        **cdoc,
        "recordings": recs,
        "judgment_count": len(store.load_judgments(case_slug)),
    }


def _inspect_registered_case(
    store: WorkbenchStore,
    spec: dict[str, Any],
    sp_path: Path | None,
    *,
    persist: bool,
) -> dict[str, Any]:
    cslug = spec["case"]
    sp_model: SpatialSpec | None = None
    target_recs: set[str] = set()
    if sp_path is not None:
        if not sp_path.is_file():
            raise ToolkitError("spatial_sidecar_not_found", f"Spatial sidecar file not found: {sp_path}")
        sp_doc = read_json(sp_path)
        sp_raw = sp_doc.get("spatial") if isinstance(sp_doc, dict) and "spatial" in sp_doc else sp_doc
        if not isinstance(sp_raw, dict):
            raise ToolkitError("invalid_spatial_sidecar", f"Spatial sidecar {sp_path.name!r} must be a JSON object.")
        sp_model = SpatialSpec.model_validate({k: v for k, v in sp_raw.items() if k in SpatialSpec.model_fields})
        target_recs = {spec["recording"]} if spec["kind"] == "registered_recording" else set(store.list_recordings(cslug))

    if not persist:
        cdoc = store.load_case(cslug)
        rec_briefs: list[dict[str, Any]] = []
        preview_specs: dict[str, RecordingSpec] = {}
        for rslug in store.list_recordings(cslug):
            rmodel = store.load_recording(cslug, rslug)
            if sp_model is not None and rslug in target_recs:
                rmodel = rmodel.model_copy(update={"spatial": sp_model, "spatial_overlay_path": str(sp_path)})
            preview_specs[rslug] = rmodel
            rec_briefs.append(_recording_brief(rmodel, store.list_runs(cslug, rslug)))
        return {
            "workspace": str(store.root),
            **cdoc,
            "recordings": rec_briefs,
            "judgment_count": len(store.load_judgments(cslug)),
            "_recording_specs": preview_specs,
        }

    if sp_model is not None:
        runs_to_refresh: list[tuple[str, str]] = []
        with FileLock(str(store.case_dir(cslug) / ".case.lock")):
            for rslug in sorted(target_recs):
                prev_doc = store.load_recording_doc(cslug, rslug)
                rmodel = store.load_recording(cslug, rslug)
                rmodel.spatial = sp_model
                rmodel.spatial_overlay_path = str(sp_path)
                extra_meta: dict[str, Any] = {"spatial_overlay_path": str(sp_path)}
                if prev_doc.get("replaced_at_wal_index"):
                    extra_meta["replaced_at_wal_index"] = int(prev_doc["replaced_at_wal_index"])
                store.save_recording(cslug, rslug, rmodel, extra=extra_meta)
                for rm in store.list_runs(cslug, rslug, reconcile=False):
                    rn = rm.get("run_name")
                    if rn:
                        runs_to_refresh.append((rslug, str(rn)))
        from .pipeline import refresh_verdict_tallies

        for rslug, rn in runs_to_refresh:
            try:
                refresh_verdict_tallies(store, cslug, rslug, rn)
            except Exception:
                continue
    return summarize_registered_case(store, cslug)


def _allocate_recording_slugs(
    store: WorkbenchStore,
    cslug: str,
    folder: Path,
    raw_inspected: list[tuple[Path, RecordingSpec, Path | None]],
) -> list[tuple[str, RecordingSpec, Path | None]]:
    existing_path_to_slug: dict[str, str] = {}
    pinned_slugs: set[str] = set()
    if store.has_case(cslug):
        for ex_rslug in store.list_recordings(cslug):
            pinned_slugs.add(ex_rslug)
            try:
                ex_rec = store.load_recording(cslug, ex_rslug)
                existing_path_to_slug[str(_canonical_entry_path(Path(ex_rec.path)))] = ex_rslug
                existing_path_to_slug.setdefault(str(Path(ex_rec.path).resolve()), ex_rslug)
            except Exception:
                continue

    used_rslugs: set[str] = set(pinned_slugs)
    assigned: dict[int, str] = {}
    new_items: list[tuple[int, Path, str]] = []
    for idx, (rf, _rmodel, _rec_sp) in enumerate(raw_inspected):
        rf_entry, rf_res = str(_canonical_entry_path(rf)), str(rf.resolve())
        if rf_entry in existing_path_to_slug:
            assigned[idx] = existing_path_to_slug[rf_entry]
        elif rf_res in existing_path_to_slug:
            assigned[idx] = existing_path_to_slug[rf_res]
        else:
            new_items.append((idx, rf, sanitize_slug(recording_stem(rf))))

    new_counts: dict[str, int] = {}
    for _idx, _rf, base_s in new_items:
        new_counts[base_s] = new_counts.get(base_s, 0) + 1
    reserved_natural = {s for s, cnt in new_counts.items() if cnt == 1 and s not in used_rslugs}

    for idx, rf, base_rslug in new_items:
        if new_counts[base_rslug] == 1 and base_rslug not in used_rslugs:
            rslug = base_rslug
        else:
            rel_no_ext = str(rf.relative_to(folder))
            for sfx in _REC_SUFFIXES:
                if rel_no_ext.lower().endswith(sfx):
                    rel_no_ext = rel_no_ext[: -len(sfx)]
                    break
            rel_clean = sanitize_slug(rel_no_ext.replace("/", "_"))
            h6_r = canonical_digest(str(rf.resolve()))[:6]
            cand = (
                sanitize_slug(f"{rel_clean[:68].rstrip('._-')}_{h6_r}")
                if (len(rel_clean) > 72 or rel_clean in used_rslugs or rel_clean in reserved_natural)
                else rel_clean
            )
            idx_n = 2
            while cand in used_rslugs or cand in reserved_natural:
                cand = sanitize_slug(f"{rel_clean[:62].rstrip('._-')}_{h6_r}_{idx_n}")
                idx_n += 1
            rslug = cand
        used_rslugs.add(rslug)
        assigned[idx] = rslug

    return [
        (assigned[idx], rmodel if rmodel.slug == assigned[idx] else rmodel.model_copy(update={"slug": assigned[idx]}), rec_sp)
        for idx, (_rf, rmodel, rec_sp) in enumerate(raw_inspected)
    ]


def _allocate_case_slug(
    store: WorkbenchStore,
    case_name: str | None,
    auto_base_slug: str,
    fs_path: Path,
    resolved_src: str,
) -> str:
    if case_name:
        slug = sanitize_slug(case_name)
        if store.has_case(slug):
            ex_src = store.load_case(slug).get("source_target")
            if ex_src and str(_canonical_entry_path(Path(ex_src))) != resolved_src:
                raise ToolkitError(
                    "case_name_collision",
                    f"Case {slug!r} already maps to {ex_src}; refusing to overwrite with {resolved_src}.",
                    existing_source=ex_src,
                    requested_source=resolved_src,
                )
        return slug

    matched_slug = next(
        (
            c["case"]
            for c in store.list_cases()
            if c.get("source_target") and str(_canonical_entry_path(Path(c["source_target"]))) == resolved_src
        ),
        None,
    )
    if matched_slug:
        return matched_slug

    slug = auto_base_slug
    if store.has_case(slug):
        ex_src = store.load_case(slug).get("source_target")
        if ex_src and str(_canonical_entry_path(Path(ex_src))) != resolved_src:
            base_pref = auto_base_slug[:54].rstrip("._-") or "case"
            parent_tok = sanitize_slug(fs_path.parent.name, fallback="src")[:14].rstrip("._-") or "src"
            h6 = canonical_digest(resolved_src)[:6]
            cand = sanitize_slug(f"{base_pref}_{parent_tok}_{h6}")
            suffix_n = 2
            while store.has_case(cand):
                ex2 = store.load_case(cand).get("source_target")
                if not ex2 or str(_canonical_entry_path(Path(ex2))) == resolved_src:
                    break
                cand = sanitize_slug(f"{base_pref[:48]}_{parent_tok[:12]}_{h6}_{suffix_n}")
                suffix_n += 1
            slug = cand
    return slug


def _check_existing_case_subject_consistency(
    store: WorkbenchStore,
    cslug: str,
    subject_labels: dict[str, str],
) -> None:
    if not store.has_case(cslug):
        return
    combined = dict(subject_labels)
    for ex_rslug in store.list_recordings(cslug):
        try:
            ex_rec = store.load_recording(cslug, ex_rslug)
        except Exception:
            continue
        if ex_rec._subject_digest:
            combined[ex_rec._subject_digest] = ex_rec.provenance.participant_id or HEADER_SUBJECT_WITHHELD
    if len(combined) > 1:
        distinct_all = sorted(set(combined.values()))
        raise ToolkitError(
            "conflicting_subjects",
            f"Case {cslug!r} already contains recordings for another subject ({', '.join(distinct_all)}); "
            "refusing to merge a different patient into the existing case.",
            subjects=distinct_all,
        )


def _filter_case_overlays(overlays: list[dict[str, Any]], subject_labels: dict[str, str]) -> list[dict[str, Any]]:
    if len(subject_labels) != 1:
        return overlays
    case_subject = next(iter(subject_labels))
    filtered: list[dict[str, Any]] = []
    for ov in overlays:
        try:
            ov_doc = read_json(Path(ov["path"]))
            ov_toks, _ = _extract_doc_subject_tokens(ov_doc, str(ov.get("name") or ""))
            if ov_toks and case_subject not in {canonical_digest(token) for token in ov_toks}:
                continue
        except Exception:
            pass
        filtered.append(ov)
    return filtered
