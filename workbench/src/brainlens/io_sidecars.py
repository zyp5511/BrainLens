"""Sidecar pairing, BIDS/subject-identity extraction, provenance, spatial overlays, and folder discovery."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any

from .deid import _norm_subject_token
from .io_integrity import _canonical_entry_path, _stat_tuple
from .io_signals import (
    _ALLOWED_KINDS,
    _REC_SUFFIXES,
    _filter_split_fif_continuations,
    is_recording_file,
    recording_stem,
)
from .models import PROJECT_ROOT, ProvenanceSpec, ToolkitError, read_json

_GENERIC_FUZZY_PREFIX_TOKENS = frozenset({"raw", "ieeg", "seeg", "eeg", "ecog", "meg", "data", "recording", "signal"})
_RECORDING_SIDECAR_KEYS = frozenset({
    "channels",
    "reference",
    "markers",
    "iEEGReference",
    "SamplingFrequency",
    "TaskName",
    "synthetic",
    "bipolar_pairs",
})
_SUB_TOKEN_RE = re.compile(r"(?:^|[_\-\s/\\])(sub-[A-Za-z0-9]+)(?:[_\-.\s/\\]|$)", re.IGNORECASE)
_BIDS_MOD_SUFFIXES = (
    "_ieeg_raw", "_eeg_raw", "_seeg_raw", "_ecog_raw", "_meg_raw",
    "_ieeg", "_eeg", "_seeg", "_ecog", "_meg",
)


def _extract_prov_participant_id(doc: Any) -> str | None:
    if not isinstance(doc, dict):
        return None
    prov_sub = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
    for cand in (
        ((doc.get("same_subject_association") or {}).get("participant_identity") or {}).get("participant_id")
        if isinstance(doc.get("same_subject_association"), dict) else None,
        ((doc.get("dataset") or {}).get("participant_identity") or {}).get("participant_id")
        if isinstance(doc.get("dataset"), dict) else None,
        ((prov_sub.get("dataset") or {}).get("participant_identity") or {}).get("participant_id")
        if isinstance(prov_sub.get("dataset"), dict) else None,
        (prov_sub.get("participant_identity") or {}).get("participant_id")
        if isinstance(prov_sub.get("participant_identity"), dict) else None,
        prov_sub.get("participant_id"),
        prov_sub.get("subject_id"),
        prov_sub.get("patient_id"),
        prov_sub.get("subject"),
        (doc.get("participant_identity") or {}).get("participant_id")
        if isinstance(doc.get("participant_identity"), dict) else None,
        doc.get("participant_id"),
        doc.get("subject_id"),
        doc.get("patient_id"),
        doc.get("subject"),
    ):
        if cand is not None and _norm_subject_token(cand) is not None:
            return str(cand).strip()
    return None


def _extract_source_files_from_mapping(container: Any) -> list[str]:
    if not isinstance(container, dict):
        return []
    out: list[str] = []
    sf = container.get("source_file")
    if isinstance(sf, str) and sf.strip():
        out.append(sf.strip())
    sfs = container.get("source_files")
    if isinstance(sfs, dict):
        out.extend(str(k).strip() for k in sfs.keys() if str(k).strip())
    elif isinstance(sfs, list):
        out.extend(str(x).strip() for x in sfs if str(x).strip())
    return out


def _extract_declared_source_files(doc: dict[str, Any]) -> list[str]:
    """Extract all source_file / source_files entries declared at top level or under provenance."""
    return _extract_source_files_from_mapping(doc) + _extract_source_files_from_mapping(doc.get("provenance"))


def _extract_doc_subject_tokens(doc: Any, filename: str | None = None) -> tuple[set[str], list[str]]:
    """Extract all distinct normalized subject tokens across top-level, provenance, nested spatial, & filename sub-*."""
    labels: list[str] = []
    if isinstance(doc, dict):
        for sub in (doc, doc.get("spatial"), doc.get("provenance")):
            pid = _extract_prov_participant_id(sub)
            if pid:
                labels.append(pid)
        prov_sub = doc.get("provenance")
        if isinstance(prov_sub, dict) and isinstance(prov_sub.get("source_file"), str):
            labels.extend(m.group(1) for m in _SUB_TOKEN_RE.finditer(prov_sub["source_file"]))
    if filename:
        labels.extend(m.group(1) for m in _SUB_TOKEN_RE.finditer(filename))
    toks = {_norm_subject_token(lbl) for lbl in labels}
    toks.discard(None)
    return toks, labels


def _bids_subject_ids_from_path(rec_path: Path, root_folder: Path | None = None) -> list[str]:
    """Collect all distinct BIDS sub-<label> identities across filename and enclosing BIDS directories."""
    p_lex = Path(os.path.abspath(str(rec_path)))
    p_res = _canonical_entry_path(rec_path)
    found: list[str] = []
    seen_norm: set[str] = set()

    def _add(val: str) -> None:
        tok = _norm_subject_token(val)
        if tok and tok not in seen_norm:
            seen_norm.add(tok)
            found.append(val)

    for cand_p in (p_lex, p_res):
        m_fn = re.search(r"(?:^|[_\-\s])(sub-[A-Za-z0-9]+)(?:[_\-.\s]|$)", cand_p.name, re.IGNORECASE)
        if m_fn:
            _add(m_fn.group(1))

    cand_dirs: list[str] = []
    if root_folder is not None:
        for cand_p, cand_root in ((p_lex, Path(os.path.abspath(str(root_folder)))), (p_res, root_folder.resolve())):
            try:
                for seg in reversed(cand_p.parent.relative_to(cand_root).parts):
                    if seg and seg not in cand_dirs:
                        cand_dirs.append(seg)
            except ValueError:
                pass
    for cand_p in (p_lex, p_res):
        cur = cand_p.parent
        for _ in range(4):
            if cur.name and cur.name not in cand_dirs:
                cand_dirs.append(cur.name)
            if cur.parent == cur:
                break
            cur = cur.parent
    for dname in cand_dirs:
        if re.fullmatch(r"sub-[A-Za-z0-9]+", dname, re.IGNORECASE):
            _add(dname)
    return found


def _bids_subject_id_from_path(rec_path: Path, root_folder: Path | None = None) -> str | None:
    subs = _bids_subject_ids_from_path(rec_path, root_folder)
    return subs[0] if subs else None


def _extract_native_subject_id(raw: Any, header: Any = None) -> str | None:
    si = (raw.info.get("subject_info") or {}) if hasattr(raw, "info") else {}
    if isinstance(si, dict):
        for cand in (si.get("his_id"), f"{si.get('first_name') or ''} {si.get('last_name') or ''}".strip()):
            if cand and _norm_subject_token(cand):
                return str(cand).strip()
    if header is not None and hasattr(header, "patient"):
        pat = getattr(header, "patient", None)
        if pat is not None:
            for attr in ("code", "name"):
                cand = getattr(pat, attr, None)
                if cand and _norm_subject_token(cand):
                    return str(cand).strip()
    return None


def _source_file_matches_recording(
    src_vals: list[str],
    rec_path: Path,
    *,
    base_dir: Path | None = None,
) -> bool:
    """Match declared source_file(s) strictly: respect directory qualifiers, symlink entry identity, and extensions."""
    rec_entry = _canonical_entry_path(rec_path)
    rec_name_low = rec_entry.name.lower()
    rec_stems_low = {rec_entry.stem.lower(), recording_stem(rec_entry).lower()}
    rec_parts_low = tuple(p.lower() for p in rec_entry.parts)
    eff_base = base_dir if base_dir is not None else rec_entry.parent

    for sv in src_vals:
        sv_norm = str(sv).strip().replace("\\", "/")
        if not sv_norm:
            continue
        sv_p = Path(sv_norm)
        if "/" in sv_norm:
            norm_parts = tuple(p for p in sv_p.parts if p not in {"", "."})
            if sv_p.is_absolute():
                if _canonical_entry_path(sv_p) == rec_entry:
                    return True
                continue
            if len(norm_parts) > 1:
                try:
                    joined = eff_base / sv_p
                    if _canonical_entry_path(joined) == rec_entry:
                        return True
                    if joined.exists() or joined.is_symlink():
                        continue
                except OSError:
                    pass
                if ".." not in norm_parts and len(rec_parts_low) >= len(norm_parts):
                    if rec_parts_low[-len(norm_parts) :] == tuple(p.lower() for p in norm_parts):
                        return True
                continue
            sv_p = Path(norm_parts[0]) if norm_parts else sv_p

        sv_name_low = sv_p.name.lower()
        if sv_name_low == rec_name_low:
            return True
        if not sv_p.suffix and sv_name_low in rec_stems_low:
            return True
    return False


def _bipolar_pairs_file_matches_recording(
    bp_path: Path,
    rec_path: Path,
    orig_folder: Path,
    *,
    target_subject_tok: str | None = None,
    reject_tagged_if_untargeted: bool = False,
) -> tuple[bool, bool]:
    """Return (matches, is_explicit_match) for an auto-discovered *bipolar*pairs*.json."""
    try:
        doc = read_json(bp_path)
    except Exception:
        return False, False
    if not isinstance(doc, dict):
        return False, False

    explicit_match = False
    rec_names = {rec_path.name.lower(), rec_path.stem.lower(), recording_stem(rec_path).lower()}

    src_vals = _extract_declared_source_files(doc)
    if src_vals:
        if not _source_file_matches_recording(src_vals, rec_path, base_dir=orig_folder):
            return False, False
        explicit_match = True

    bp_toks, _ = _extract_doc_subject_tokens(doc, bp_path.name)
    if len(bp_toks) > 1:
        return False, False
    bp_sub_tok = next(iter(bp_toks), None)
    if bp_sub_tok:
        if target_subject_tok:
            if bp_sub_tok != target_subject_tok:
                return False, False
            explicit_match = True
        elif reject_tagged_if_untargeted:
            return False, False

    bp_low = bp_path.stem.lower()
    if orig_folder.is_dir():
        for entry in orig_folder.iterdir():
            if not is_recording_file(entry) or entry.name.lower() == rec_path.name.lower():
                continue
            other_names = {entry.stem.lower(), recording_stem(entry).lower()}
            if other_names & rec_names:
                continue
            for ostem in other_names:
                if len(ostem) >= 2 and (bp_low == ostem or bp_low.startswith(ostem + "_") or bp_low.startswith(ostem + "-")):
                    return False, False

    for rstem in (rec_path.stem.lower(), recording_stem(rec_path).lower()):
        if len(rstem) >= 2 and (bp_low == rstem or bp_low.startswith(rstem + "_") or bp_low.startswith(rstem + "-")):
            explicit_match = True

    return True, explicit_match


def _exact_sidecar_matches_recording(doc: dict[str, Any], eff_rec: Path, check_folder: Path, stem: str) -> bool:
    """Accept exact {stem}.json unless its declared source_file(s) point to a different recording, extension, or subject."""
    top_vals = _extract_source_files_from_mapping(doc)
    if top_vals and not _source_file_matches_recording(top_vals, eff_rec, base_dir=check_folder):
        return False

    prov_vals = _extract_source_files_from_mapping(doc.get("provenance"))
    if not prov_vals or _source_file_matches_recording(prov_vals, eff_rec, base_dir=check_folder):
        return True

    eff_ext = ".fif.gz" if eff_rec.name.lower().endswith(".fif.gz") else eff_rec.suffix.lower()
    eff_bids_toks = {_norm_subject_token(s) for s in _bids_subject_ids_from_path(eff_rec, None)}
    eff_bids_toks.discard(None)
    other_rec_names: set[str] = set()
    if check_folder.is_dir():
        for entry in check_folder.iterdir():
            if not is_recording_file(entry) or entry.name.lower() == eff_rec.name.lower():
                continue
            if recording_stem(entry).lower() == stem.lower():
                return False
            other_rec_names.add(entry.name.lower())
    for pv in prov_vals:
        pv_norm = pv.replace("\\", "/")
        if "/" in pv_norm or pv_norm.lower() in other_rec_names:
            return False
        pv_ext = ".fif.gz" if pv_norm.lower().endswith(".fif.gz") else Path(pv_norm).suffix.lower()
        if pv_ext and pv_ext != eff_ext:
            return False
        if eff_bids_toks:
            for m in re.finditer(r"(?:^|[_\-\s])(sub-[A-Za-z0-9]+)(?:[_\-.\s]|$)", pv_norm, re.IGNORECASE):
                pv_tok = _norm_subject_token(m.group(1))
                if pv_tok and pv_tok not in eff_bids_toks:
                    return False
    return True


def _discover_recording_sidecars(
    rec_path: Path,
    *,
    logical_rec_path: Path | None = None,
    orig_folder: Path | None = None,
) -> dict[str, Any]:
    """Pair BrainLens (and legacy BrainToolkit) sidecar JSON, BIDS JSON, BIDS channels.tsv, and published bipolar pairs."""
    folder = rec_path.parent
    eff_rec = logical_rec_path if logical_rec_path is not None else rec_path
    check_folder = orig_folder if orig_folder is not None else eff_rec.parent
    stem = recording_stem(eff_rec)

    sidecar: dict[str, Any] = {}
    sidecar_path: Path | None = None
    sidecar_exact = False
    sidecar_fuzzy = False
    bids_json: dict[str, Any] = {}

    exact_json = folder / f"{stem}.json"
    if exact_json.is_file():
        doc = read_json(exact_json)
        if isinstance(doc, dict) and _exact_sidecar_matches_recording(doc, eff_rec, check_folder, stem):
            has_bids_keys = any(k in doc for k in ("PowerLineFrequency", "SamplingFrequency", "iEEGReference"))
            has_native_keys = any(k in doc for k in ("reference", "spatial", "channels", "markers", "provenance", "synthetic", "line_freq"))
            if has_bids_keys:
                bids_json = doc
            if has_native_keys or not has_bids_keys:
                sidecar = doc
                sidecar_path = exact_json
                sidecar_exact = True

    sidecar_fuzzy_candidates: list[tuple[Path, dict[str, Any]]] = []
    if not sidecar:
        stem_tokens = {tok.lower() for tok in re.split(r"[_\s]+", stem) if tok}
        tier1_src: list[tuple[Path, dict[str, Any]]] = []
        tier2_exact: list[tuple[Path, dict[str, Any]]] = []
        tier3_fuzzy: list[tuple[int, int, Path, dict[str, Any]]] = []
        # Discover *_brainlens.json first, then legacy *_braintoolkit.json for full backward compatibility.
        sidecar_cands = sorted(folder.glob("*_brainlens.json")) + sorted(folder.glob("*_braintoolkit.json"))
        for cand in sidecar_cands:
            try:
                doc = read_json(cand)
            except Exception:
                continue
            if not isinstance(doc, dict):
                continue
            src_vals = _extract_declared_source_files(doc)
            if src_vals:
                if _source_file_matches_recording(src_vals, eff_rec, base_dir=check_folder):
                    tier1_src.append((cand, doc))
                continue
            if cand.stem.endswith("_brainlens"):
                prefix = cand.stem[: -len("_brainlens")]
                prio = 0
            elif cand.stem.endswith("_braintoolkit"):
                prefix = cand.stem[: -len("_braintoolkit")]
                prio = 1
            else:
                prefix = cand.stem
                prio = 2
            if prefix == stem:
                tier2_exact.append((cand, doc))
            else:
                pref_tokens = [tok.lower() for tok in re.split(r"[_\s]+", prefix) if tok]
                non_generic = [t for t in pref_tokens if t not in _GENERIC_FUZZY_PREFIX_TOKENS]
                if pref_tokens and non_generic and all(tok in stem_tokens for tok in pref_tokens):
                    tier3_fuzzy.append((-len(non_generic), prio, cand, doc))
        if tier1_src:
            sidecar_path, sidecar = tier1_src[0]
            sidecar_exact = True
        elif tier2_exact:
            sidecar_path, sidecar = tier2_exact[0]
            sidecar_exact = True
        elif tier3_fuzzy:
            tier3_fuzzy.sort(key=lambda item: (item[0], item[1], str(item[2])))
            sidecar_fuzzy_candidates = [(c, d) for _, _, c, d in tier3_fuzzy]
            sidecar_path, sidecar = sidecar_fuzzy_candidates[0]
            sidecar_fuzzy = True

    bids_bad_channels: list[str] = []
    bids_channel_kinds: dict[str, str] = {}
    tsv_candidates = [folder / f"{stem[: -len(sfx)]}_channels.tsv" for sfx in _BIDS_MOD_SUFFIXES if stem.endswith(sfx)]
    tsv_candidates.append(folder / f"{stem}_channels.tsv")
    for tsv_path in tsv_candidates:
        if tsv_path.is_file():
            with tsv_path.open("r", encoding="utf-8", errors="replace") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    cname = (row.get("name") or "").strip()
                    if not cname:
                        continue
                    ctype = (row.get("type") or "").strip().lower()
                    if ctype in _ALLOWED_KINDS:
                        bids_channel_kinds[cname] = ctype
                    if (row.get("status") or "").strip().lower() == "bad":
                        bids_bad_channels.append(cname)
            break

    bipolar_file: Path | None = None
    bipolar_was_explicit = False
    bipolar_candidates: list[Path] = []
    folder_resolved = folder.resolve()
    if isinstance(sidecar.get("provenance"), dict):
        rel_bp = sidecar["provenance"].get("published_bipolar_pairs_file")
        if rel_bp:
            cand_bp = (folder / str(rel_bp)).resolve()
            if not cand_bp.is_relative_to(folder_resolved):
                raise ToolkitError(
                    "unsafe_sidecar_path",
                    f"Sidecar published_bipolar_pairs_file escapes recording directory: {rel_bp!r}",
                )
            if cand_bp.is_file():
                bipolar_file = cand_bp
                bipolar_was_explicit = True
    if bipolar_file is None:
        explicit_hits: list[Path] = []
        generic_hits: list[Path] = []
        for raw_bp in sorted(folder.glob("*bipolar*pairs*.json")):
            cand_bp = raw_bp.resolve()
            if not cand_bp.is_relative_to(folder_resolved):
                continue
            ok_bp, is_exp = _bipolar_pairs_file_matches_recording(cand_bp, eff_rec, check_folder)
            if ok_bp:
                bipolar_candidates.append(cand_bp)
                (explicit_hits if is_exp else generic_hits).append(cand_bp)
        if len(explicit_hits) == 1:
            bipolar_file = explicit_hits[0]
        elif not explicit_hits and len(generic_hits) == 1:
            bipolar_file = generic_hits[0]

    return {
        "sidecar": sidecar,
        "sidecar_path": sidecar_path,
        "sidecar_exact": sidecar_exact,
        "sidecar_fuzzy": sidecar_fuzzy,
        "sidecar_fuzzy_candidates": sidecar_fuzzy_candidates,
        "bids_json": bids_json,
        "bids_bad_channels": bids_bad_channels,
        "bids_channel_kinds": bids_channel_kinds,
        "bipolar_file": bipolar_file,
        "bipolar_candidates": bipolar_candidates,
        "bipolar_was_explicit": bipolar_was_explicit,
    }


def _extract_provenance(
    rec_path: Path,
    sidecar: dict[str, Any],
    sidecar_path: Path | None,
    bids_json: dict[str, Any],
    folder_prov: dict[str, Any],
) -> ProvenanceSpec:
    if sidecar.get("synthetic") is True:
        return ProvenanceSpec(
            origin_kind="synthetic_engineering_data",
            is_synthetic=True,
            summary="Synthetic CC0 engineering data (not patient anatomy or clinical EEG).",
            dataset_name="BrainLens Synthetic SEEG Demo",
            license="CC0",
            sidecar_path=str(sidecar_path) if sidecar_path else None,
            limitations=[
                "Synthetic signals and linear coordinates do not establish patient anatomy or clinical performance."
            ],
        )

    prov_block = sidecar.get("provenance") if isinstance(sidecar.get("provenance"), dict) else {}
    ds = prov_block.get("dataset") if isinstance(prov_block.get("dataset"), dict) else {}
    if not ds and isinstance(folder_prov.get("same_subject_association"), dict):
        ds = folder_prov["same_subject_association"]

    limits: list[str] = []
    for lim_src in (prov_block.get("limitations"), folder_prov.get("limits")):
        if isinstance(lim_src, list):
            for x in lim_src:
                s = str(x).strip()
                if s and s not in limits:
                    limits.append(s)

    if ds or folder_prov.get("source"):
        name = ds.get("name") or bids_json.get("InstitutionName") or "Public iEEG Dataset"
        doi, lic = ds.get("DOI"), ds.get("license")
        participant = (
            (ds.get("participant_identity") or {}).get("participant_id")
            or _extract_prov_participant_id(folder_prov)
            or _extract_prov_participant_id(sidecar)
        )
        parts = [name]
        if participant:
            parts.append(str(participant))
        if doi:
            parts.append(f"DOI {doi}")
        if lic:
            parts.append(f"license {lic}")
        return ProvenanceSpec(
            origin_kind="public_dataset",
            is_synthetic=False,
            summary=" · ".join(parts),
            dataset_name=name,
            participant_id=participant,
            doi=doi,
            license=lic,
            public_url=ds.get("public_url") or folder_prov.get("source"),
            same_patient_basis=ds.get("same_patient_basis"),
            source_file=prov_block.get("source_file"),
            sidecar_path=str(sidecar_path) if sidecar_path else None,
            limitations=limits,
        )

    local_pid = _extract_prov_participant_id(folder_prov) or _extract_prov_participant_id(sidecar)
    local_sum = f"Local recording ({rec_path.name}); external dataset provenance not declared."
    if local_pid:
        local_sum = f"{local_sum} · {local_pid}"
    return ProvenanceSpec(
        origin_kind="local_recording",
        is_synthetic=False,
        summary=local_sum,
        dataset_name=str(folder_prov.get("dataset_name") or "").strip() or None,
        participant_id=local_pid,
        sidecar_path=str(sidecar_path) if sidecar_path else None,
        limitations=limits,
    )


def _recording_sidecar_dir_sig(
    rec_resolved: Path,
    spatial_overlay_path: Path | None = None,
    folder_prov_path: Path | None = None,
    stop_root: Path | None = None,
) -> tuple[tuple[str, str, tuple[int, int, int, int, int]], ...]:
    """Snapshot stat tuples of all sidecar/metadata files & ancestor PROVENANCE/overlay dirs."""
    folder = rec_resolved.parent
    cands: dict[str, tuple[str, Path]] = {}
    if folder.is_dir():
        for entry in folder.iterdir():
            low = entry.name.lower()
            if entry.is_file() and (low.endswith(".json") or low.endswith("_channels.tsv")):
                e_res = entry.resolve()
                cands[f"dir:{entry.name}"] = (str(e_res), e_res)
    if stop_root is not None:
        stop_res = stop_root.resolve()
        cur = folder
        chain_dirs: list[Path] = []
        while True:
            chain_dirs.append(cur)
            if cur == stop_res or cur.parent == cur:
                break
            cur = cur.parent
        if stop_res.parent != stop_res:
            chain_dirs.append(stop_res.parent)
        for d in chain_dirs:
            if not d.is_dir():
                continue
            p_cand = d / "PROVENANCE.json"
            if p_cand.is_file():
                p_res = p_cand.resolve()
                cands[f"prov:{d}"] = (str(p_res), p_res)
            if d != folder and d != stop_res.parent:
                for entry in d.iterdir():
                    if entry.is_file() and entry.name.lower().endswith(".json") and entry.name != "PROVENANCE.json":
                        e_res = entry.resolve()
                        cands[f"anc:{d}/{entry.name}"] = (str(e_res), e_res)
    for tag, extra in (("extra_sp", spatial_overlay_path), ("extra_prov", folder_prov_path)):
        if extra is not None and extra.is_file():
            er = extra.resolve()
            cands[f"{tag}:{extra}"] = (str(er), er)
    return tuple((key, cands[key][0], _stat_tuple(cands[key][1])) for key in sorted(cands))


def _scan_dir_spatial_overlays(folder: Path, *, target_subject_tok: str | None = None) -> list[dict[str, Any]]:
    overlays: list[dict[str, Any]] = []
    if not folder.is_dir():
        return overlays
    rec_stems = {s for e in folder.iterdir() if is_recording_file(e) for s in (e.stem, recording_stem(e))}
    for jf in sorted(folder.glob("*.json")):
        low = jf.name.lower()
        if jf.name == "PROVENANCE.json" or low.endswith(("_truth.json", "_brainlens.json", "_braintoolkit.json", "_ieeg.json", "_eeg.json", "_meg.json")):
            continue
        if jf.stem in rec_stems or any((folder / f"{jf.stem}{s}").is_file() for s in _REC_SUFFIXES):
            continue
        try:
            doc = read_json(jf)
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        if any(k in doc for k in _RECORDING_SIDECAR_KEYS) or _extract_declared_source_files(doc):
            continue
        if isinstance(doc.get("spatial"), dict) and "positions" in doc["spatial"]:
            ov_toks, _ = _extract_doc_subject_tokens(doc, jf.name)
            if len(ov_toks) > 1:
                continue
            ov_tok = next(iter(ov_toks), None)
            if target_subject_tok and ov_tok and ov_tok != target_subject_tok:
                continue
            sp = doc["spatial"]
            overlays.append({
                "name": jf.name,
                "path": str(jf.resolve()),
                "coordinate_system": sp.get("coordinate_system"),
                "units": sp.get("units"),
                "template": bool(sp.get("template", False)),
                "position_count": len(sp.get("positions", {})),
                "source": sp.get("source"),
            })
    return overlays


def _find_folder_provenance_and_overlays(folder: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    fprov: dict[str, Any] = {}
    for pfile in (folder / "PROVENANCE.json", folder.parent / "PROVENANCE.json"):
        if pfile.is_file():
            try:
                fprov = read_json(pfile)
                break
            except Exception:
                pass
    readme: str | None = None
    for rname in ("README.txt", "README.md"):
        rp = folder / rname
        if rp.is_file():
            readme = rp.read_text(encoding="utf-8", errors="replace").strip()[:2000]
            break
    return fprov, _scan_dir_spatial_overlays(folder), readme


def _find_recording_folder_context(
    rec_dir: Path,
    stop_root: Path,
    rec_path: Path | None = None,
) -> tuple[Path | None, list[dict[str, Any]]]:
    """Locate closest PROVENANCE.json path & spatial overlays for a recording without crossing subject/corrupt boundaries."""
    rec_res, root_res = rec_dir.resolve(), stop_root.resolve()
    chain: list[Path] = []
    cur = rec_res
    while True:
        chain.append(cur)
        if cur == root_res or cur.parent == cur:
            break
        cur = cur.parent

    bids_subs = _bids_subject_ids_from_path(rec_path, root_res) if rec_path is not None else []
    if len(bids_subs) > 1:
        raise ToolkitError(
            "conflicting_subjects",
            f"Recording path {rec_path.name!r} declares conflicting BIDS subject identities across filename and enclosing directory: {sorted(bids_subs)}.",
            subjects=sorted(bids_subs),
        )
    bids_tok = _norm_subject_token(bids_subs[0] if bids_subs else None)

    prov_path: Path | None = None
    prov_dir: Path | None = None
    matched_prov_tok: str | None = None
    blocked_boundary: Path | None = None
    for d in chain:
        if bids_tok and re.fullmatch(r"sub-[A-Za-z0-9]+", d.name, re.IGNORECASE):
            d_tok = _norm_subject_token(d.name)
            if d_tok and d_tok != bids_tok:
                blocked_boundary = d
                break
        cand = d / "PROVENANCE.json"
        if cand.is_file():
            try:
                doc = read_json(cand)
                if not isinstance(doc, dict):
                    raise ValueError("PROVENANCE.json must be a JSON object")
            except Exception as exc:
                raise ToolkitError(
                    "invalid_provenance",
                    f"Malformed PROVENANCE.json in {d.name!r}; refusing to fall back to parent provenance or coordinates: {exc}",
                ) from exc
            prov_tok = _norm_subject_token(_extract_prov_participant_id(doc))
            if bids_tok and prov_tok and bids_tok != prov_tok and d != rec_res:
                blocked_boundary = d
                break
            prov_path, prov_dir, matched_prov_tok = cand.resolve(), d, prov_tok
            break

    if prov_path is None and blocked_boundary is None:
        parent_cand = root_res.parent / "PROVENANCE.json"
        if parent_cand.is_file():
            try:
                doc = read_json(parent_cand)
                if isinstance(doc, dict):
                    prov_tok = _norm_subject_token(_extract_prov_participant_id(doc))
                    if not (bids_tok and prov_tok and bids_tok != prov_tok):
                        prov_path, matched_prov_tok = parent_cand.resolve(), prov_tok
            except Exception:
                pass

    eff_target_tok = bids_tok or matched_prov_tok
    overlays: list[dict[str, Any]] = []
    for d in chain:
        if blocked_boundary is not None and d == blocked_boundary:
            break
        found = _scan_dir_spatial_overlays(d, target_subject_tok=eff_target_tok)
        if found:
            overlays = found
            break
        if prov_dir is not None and d == prov_dir:
            break
    return prov_path, overlays


def _discover_folder_recordings(folder: Path) -> list[Path]:
    """Recursively discover all non-hidden recording files in `folder`, excluding FIFF split continuation chunks."""
    root_res = folder.resolve()
    recs: list[Path] = []
    for p in sorted(folder.rglob("*")):
        if not is_recording_file(p):
            continue
        p_entry = _canonical_entry_path(p)
        try:
            rel_parts = p_entry.relative_to(root_res).parts
        except ValueError:
            rel_parts = p_entry.parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        recs.append(p_entry)

    by_target_all: dict[Path, list[Path]] = {}
    for p in recs:
        by_target_all.setdefault(p.resolve(), []).append(p)

    backing_dir_names = {"objects", "blobs", "annex", "store"}

    def _has_local_sidecar(p: Path) -> bool:
        st = recording_stem(p)
        if (p.parent / f"{p.stem}.json").is_file() or (p.parent / f"{st}.json").is_file():
            return True
        for mod_sfx in _BIDS_MOD_SUFFIXES:
            if st.endswith(mod_sfx) and (p.parent / f"{st[: -len(mod_sfx)]}_channels.tsv").is_file():
                return True
        return (p.parent / f"{st}_channels.tsv").is_file()

    def _alias_priority(p: Path) -> tuple[int, int, int, int, int, str]:
        has_bids = 1 if _bids_subject_id_from_path(p, root_res) else 0
        has_sc = 1 if _has_local_sidecar(p) else 0
        is_sym = 1 if p.is_symlink() else 0
        try:
            rel_parents = {part.lower() for part in p.parent.relative_to(root_res).parts}
        except ValueError:
            rel_parents = set()
        not_backing = 0 if (rel_parents & backing_dir_names) else 1
        return (-has_bids, -has_sc, -is_sym, -not_backing, len(p.parts), str(p))

    chosen: list[Path] = []
    for tgt, aliases in by_target_all.items():
        if len(aliases) > 1:
            alias_sub_map: dict[str, str] = {}
            for ap in aliases:
                cand_labels: list[str] = list(_bids_subject_ids_from_path(ap, root_res))
                try:
                    ap_sc = _discover_recording_sidecars(ap).get("sidecar") or {}
                    _, sc_labels = _extract_doc_subject_tokens(ap_sc)
                    cand_labels.extend(sc_labels)
                except Exception:
                    pass
                try:
                    ap_prov_p, _ = _find_recording_folder_context(ap.parent, stop_root=root_res, rec_path=ap)
                    if ap_prov_p is not None and ap_prov_p.is_file():
                        ap_prov_pid = _extract_prov_participant_id(read_json(ap_prov_p))
                        if ap_prov_pid:
                            cand_labels.append(ap_prov_pid)
                except Exception:
                    pass
                for sub_lbl in cand_labels:
                    nt = _norm_subject_token(sub_lbl)
                    if nt and nt not in alias_sub_map:
                        alias_sub_map[nt] = sub_lbl
            if len(alias_sub_map) > 1:
                distinct_subs = sorted(alias_sub_map.values())
                raise ToolkitError(
                    "conflicting_subjects",
                    f"Backing recording {tgt.name!r} is aliased under conflicting subjects ({', '.join(distinct_subs)}); refusing ambiguous import.",
                    subjects=distinct_subs,
                )
        chosen.append(min(aliases, key=_alias_priority))
    return _filter_split_fif_continuations(sorted(chosen, key=str))


def discover_available_inputs() -> list[dict[str, Any]]:
    inputs_dir = PROJECT_ROOT / "inputs"
    items: list[dict[str, Any]] = []
    if not inputs_dir.is_dir():
        return items
    for child in sorted(inputs_dir.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_dir():
            recs = _discover_folder_recordings(child)
            if recs:
                items.append({
                    "label": child.name,
                    "target": str(child),
                    "recording_count": len(recs),
                    "files": [p.name for p in recs],
                })
        elif is_recording_file(child):
            items.append({
                "label": recording_stem(child),
                "target": str(child),
                "recording_count": 1,
                "files": [child.name],
            })
    return items
