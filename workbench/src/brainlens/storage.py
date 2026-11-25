"""Clean, transparent workspace persistence for BrainLens Workbench."""

from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path
from typing import Any

from filelock import FileLock

from .deid import remember_agent_read, restore_withheld_provenance, subject_digest
from .models import (
    RecordingSpec,
    ToolkitError,
    atomic_write_json,
    default_workspace_dir,
    read_json,
    utc_now,
)
from .storage_actions import ReportActionQueueMixin
from .storage_judgments import JudgmentStorageMixin, validate_clinical_overlay
from .storage_signatures import CaseSignatureMixin
from .storage_mvcc import (
    DEFAULT_CLAIM_LEASE_SECONDS,
    _UI_STATE_KEYS,
    _norm_interval,
    _normalize_legacy_summary,
    bundle_signature_entry,
    classify_judgment_role,
    pin_generation_read,
    release_gc_lock,
    run_analysis_signature,
    sanitize_slug,
    try_lock_generation_for_gc,
    write_csv_file,
)

_RECONCILE_LOCAL = threading.local()

__all__ = [
    "DEFAULT_CLAIM_LEASE_SECONDS",
    "WorkbenchStore",
    "_norm_interval",
    "_normalize_legacy_summary",
    "bundle_signature_entry",
    "classify_judgment_role",
    "pin_generation_read",
    "release_gc_lock",
    "run_analysis_signature",
    "sanitize_slug",
    "try_lock_generation_for_gc",
    "validate_clinical_overlay",
    "write_csv_file",
]


class WorkbenchStore(JudgmentStorageMixin, CaseSignatureMixin, ReportActionQueueMixin):
    """Human- and researcher-friendly case/recording/run/protocol store."""

    def __init__(self, root: str | Path | None = None, *, create_dirs: bool = True):
        self.root = Path(root).expanduser().resolve() if root is not None else default_workspace_dir()
        self.workspace_id = hashlib.sha256(str(self.root).encode("utf-8")).hexdigest()[:16]
        if create_dirs:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "cases").mkdir(exist_ok=True)
            (self.root / "protocols").mkdir(exist_ok=True)
            (self.root / "cohorts").mkdir(exist_ok=True)

    def _safe_child(self, base: Path, slug: str) -> Path:
        safe = sanitize_slug(slug)
        out = (base / safe).resolve()
        if not out.is_relative_to(base.resolve()):
            raise ToolkitError("unsafe_slug", f"Unsafe path slug: {slug!r}")
        return out

    def resolve_workspace_artifact_path(
        self,
        stored_path: str | Path | None,
        *,
        fallback_dir: Path | None = None,
    ) -> Path | None:
        """Resolve an internal workspace artifact path even after relocating the workspace directory."""
        if not stored_path:
            return None
        raw_str = str(stored_path)
        p = Path(raw_str)
        root_res = self.root.resolve()
        try:
            p_res = p.resolve()
            if p_res.is_file() and p_res.is_relative_to(root_res):
                return p_res
        except OSError:
            pass
        norm = raw_str.replace("\\", "/")
        for marker in ("/cases/", "/cohorts/", "/protocols/"):
            search_end = len(norm)
            while search_end > 0:
                idx = norm.rfind(marker, 0, search_end)
                if idx < 0:
                    break
                cand = (root_res / norm[idx + 1 :]).resolve()
                if cand.is_file() and cand.is_relative_to(root_res):
                    return cand
                search_end = idx + len(marker) - 1
        if fallback_dir is not None:
            cand_fb = (fallback_dir / p.name).resolve()
            if cand_fb.is_file() and cand_fb.is_relative_to(root_res):
                return cand_fb
        return p if p.is_file() else None

    # --- Cases ---

    def case_dir(self, case_slug: str) -> Path:
        return self._safe_child(self.root / "cases", case_slug)

    def case_json_path(self, case_slug: str) -> Path:
        return self.case_dir(case_slug) / "case.json"

    def has_case(self, case_slug: str) -> bool:
        return self.case_json_path(case_slug).is_file()

    def list_cases(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        cases_root = self.root / "cases"
        if not cases_root.is_dir():
            return out
        for d in sorted(cases_root.iterdir()):
            cpath = d / "case.json"
            if d.is_dir() and cpath.is_file():
                try:
                    out.append(self.load_case(d.name))
                except Exception:
                    continue
        return out

    def load_case(self, case_slug: str) -> dict[str, Any]:
        cpath = self.case_json_path(case_slug)
        if not cpath.is_file():
            raise ToolkitError("case_not_found", f"Case {case_slug!r} not found in {self.root}.")
        doc = read_json(cpath)
        for brief in doc.get("recordings") or ():
            prov = brief.get("provenance") or {}
            if prov.get("participant_id_source") == "withheld":
                rpath = self.recording_path(case_slug, brief["recording"])
                if rpath.is_file():
                    brief["provenance"] = restore_withheld_provenance(prov, rpath)
        return doc

    def save_case(self, case_slug: str, doc: dict[str, Any], *, _locked: bool = False) -> dict[str, Any]:
        cdir = self.case_dir(case_slug)
        cdir.mkdir(parents=True, exist_ok=True)
        payload = dict(doc)
        payload["case"] = sanitize_slug(case_slug)
        payload["updated_at"] = utc_now()
        if _locked:
            atomic_write_json(self.case_json_path(case_slug), payload)
        else:
            with FileLock(str(cdir / ".case.lock")):
                atomic_write_json(self.case_json_path(case_slug), payload)
        return payload

    def add_open_question(self, case_slug: str, question: str, *, prepend: bool = False) -> list[str]:
        """Atomically append/prepend a clinical open question to case.json under .case.lock."""
        q = str(question or "").strip()
        cdir = self.case_dir(case_slug)
        cdir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(cdir / ".case.lock")):
            cdoc = self.load_case(case_slug)
            qs = list(cdoc.get("open_questions") or [])
            if q and q not in qs:
                qs.insert(0, q) if prepend else qs.append(q)
                cdoc["open_questions"] = qs
                cdoc["case"] = sanitize_slug(case_slug)
                cdoc["updated_at"] = utc_now()
                atomic_write_json(self.case_json_path(case_slug), cdoc)
            return qs

    # --- Recordings ---

    def recording_dir(self, case_slug: str, rec_slug: str) -> Path:
        return self._safe_child(self.case_dir(case_slug) / "recordings", rec_slug)

    def recording_path(self, case_slug: str, rec_slug: str) -> Path:
        return self.recording_dir(case_slug, rec_slug) / "recording.json"

    def list_recordings(self, case_slug: str) -> list[str]:
        rroot = self.case_dir(case_slug) / "recordings"
        if not rroot.is_dir():
            return []
        return sorted(d.name for d in rroot.iterdir() if d.is_dir() and (d / "recording.json").is_file())

    def save_recording(
        self,
        case_slug: str,
        rec_slug: str,
        spec: RecordingSpec,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rdir = self.recording_dir(case_slug, rec_slug)
        rdir.mkdir(parents=True, exist_ok=True)
        doc = {
            "case": sanitize_slug(case_slug),
            "recording": sanitize_slug(rec_slug),
            "duration_seconds": round(spec.duration, 4),
            "spec": spec.model_dump(mode="json"),
            "subject_comparison_digest": spec._subject_digest,
            "native_header_terms": spec._header_terms,
            **({"header_clock": spec._header_clock} if spec._header_clock is not None else {}),
            **(extra or {}),
            "updated_at": utc_now(),
        }
        atomic_write_json(self.recording_path(case_slug, rec_slug), doc)
        return doc

    def load_recording_doc(self, case_slug: str, rec_slug: str) -> dict[str, Any]:
        p = self.recording_path(case_slug, rec_slug)
        if not p.is_file():
            raise ToolkitError("recording_not_found", f"Recording {rec_slug!r} not found in case {case_slug!r}.")
        doc = read_json(p)
        spec = doc["spec"]
        spec["provenance"] = restore_withheld_provenance(spec["provenance"], p)
        return doc

    def load_recording(self, case_slug: str, rec_slug: str) -> RecordingSpec:
        doc = self.load_recording_doc(case_slug, rec_slug)
        spec = RecordingSpec.model_validate(doc["spec"])
        spec._subject_digest = doc.get("subject_comparison_digest") or subject_digest(spec.provenance.participant_id)
        if doc.get("native_header_terms") is not None:
            spec._header_terms = tuple(doc["native_header_terms"])
        spec._registration_path = self.recording_path(case_slug, rec_slug)
        spec._header_clock = doc.get("header_clock")
        remember_agent_read(sanitize_slug(case_slug), spec)
        return spec

    # --- Runs (MVCC) ---

    def run_dir(self, case_slug: str, rec_slug: str, run_name: str) -> Path:
        return self._safe_child(self.recording_dir(case_slug, rec_slug) / "runs", run_name)

    def resolve_run_gen_dir(self, case_slug: str, rec_slug: str, run_name: str) -> tuple[Path, str | None] | None:
        """Resolve a run's active immutable generation directory via atomic `current` symlink or CURRENT.json."""
        rdir = self.run_dir(case_slug, rec_slug, run_name)

        def _validate_gen_dir(gdir: Path, expected_gen: str) -> tuple[Path, str] | None:
            mpath, cpath, spath = gdir / "run_meta.json", gdir / "config.json", gdir / "summary.json"
            if not (mpath.is_file() and cpath.is_file() and spath.is_file()):
                return None
            meta, cfg, summ = read_json(mpath), read_json(cpath), read_json(spath)
            if (
                isinstance(meta, dict)
                and meta.get("status") == "complete"
                and meta.get("generation_id") == expected_gen
                and cfg.get("generation_id") == expected_gen
                and summ.get("generation_id") == expected_gen
            ):
                return gdir, expected_gen
            return None

        cur_link = rdir / "current"
        if cur_link.is_symlink():
            for _attempt in range(10):
                try:
                    gdir = cur_link.resolve()
                    gen_root = (rdir / "generations").resolve()
                    gen_id = gdir.name
                    if not (gdir.is_relative_to(gen_root) and re.fullmatch(r"[A-Za-z0-9_-]{4,64}", gen_id)):
                        return None
                    with pin_generation_read(gdir):
                        if not gdir.is_dir():
                            continue
                        res = _validate_gen_dir(gdir, gen_id)
                        if res is not None:
                            return res
                    try:
                        if cur_link.resolve() != gdir:
                            continue
                    except OSError:
                        continue
                    return None
                except FileNotFoundError:
                    continue
                except Exception:
                    try:
                        if cur_link.is_symlink():
                            continue
                    except OSError:
                        pass
                    return None
            return None

        cur_p = rdir / "CURRENT.json"
        if cur_p.is_file():
            for _attempt in range(8):
                try:
                    cur = read_json(cur_p)
                    if not isinstance(cur, dict) or cur.get("status") != "complete":
                        return None
                    gen_id = str(cur.get("generation_id") or "").strip()
                    if not gen_id or not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", gen_id):
                        return None
                    gdir = rdir / "generations" / gen_id
                    with pin_generation_read(gdir):
                        if not gdir.is_dir():
                            continue
                        res = _validate_gen_dir(gdir, gen_id)
                        if res is not None:
                            return res
                    return None
                except FileNotFoundError:
                    continue
                except Exception:
                    return None
            return None

        mpath, cpath, spath = rdir / "run_meta.json", rdir / "config.json", rdir / "summary.json"
        if not (spath.is_file() and cpath.is_file() and mpath.is_file()):
            return None
        try:
            meta = read_json(mpath)
            if not (isinstance(meta, dict) and meta.get("status") == "complete"):
                return None
            gen = meta.get("generation_id")
            if gen:
                cfg, summ = read_json(cpath), read_json(spath)
                if cfg.get("generation_id") != gen or summ.get("generation_id") != gen:
                    return None
            return rdir, gen
        except Exception:
            return None

    def run_generation_dir(self, case_slug: str, rec_slug: str, run_name: str) -> Path:
        resolved = self.resolve_run_gen_dir(case_slug, rec_slug, run_name)
        return resolved[0] if resolved is not None else self.run_dir(case_slug, rec_slug, run_name)

    def cache_dir(self, case_slug: str, rec_slug: str, run_name: str) -> Path:
        d = self.run_dir(case_slug, rec_slug, run_name) / "_cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def stage_cache_dir(self, case_slug: str, rec_slug: str, stage: str, fp: str, *, nonce: str | None = None) -> Path:
        """Content-addressed immutable stage cache directory so overwriting a run never mutates another run's donor."""
        base = self.recording_dir(case_slug, rec_slug) / "_stage_cache" / sanitize_slug(stage)
        leaf = f"{fp[:24]}_{sanitize_slug(nonce)[:12]}" if nonce else fp[:24]
        d = self._safe_child(base, leaf)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def has_run(self, case_slug: str, rec_slug: str, run_name: str) -> bool:
        return self.resolve_run_gen_dir(case_slug, rec_slug, run_name) is not None

    def list_runs(self, case_slug: str, rec_slug: str, *, reconcile: bool = True) -> list[dict[str, Any]]:
        rroot = self.recording_dir(case_slug, rec_slug) / "runs"
        if not rroot.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for d in sorted(rroot.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            try:
                out.append(self.load_run(case_slug, rec_slug, d.name, reconcile=reconcile)["meta"])
            except Exception:
                continue
        return out

    def load_run(self, case_slug: str, rec_slug: str, run_name: str, *, reconcile: bool = True) -> dict[str, Any]:
        last_exc: Exception | None = None
        rdir = self.run_dir(case_slug, rec_slug, run_name)
        for _attempt in range(10):
            resolved = self.resolve_run_gen_dir(case_slug, rec_slug, run_name)
            if resolved is None:
                if (rdir / "current").is_symlink() or (rdir / "CURRENT.json").is_file():
                    continue
                raise ToolkitError("run_not_found", f"Complete run {run_name!r} not found for {case_slug}/{rec_slug}.")
            gdir, _ = resolved
            is_mvcc = gdir.parent.name == "generations"
            try:
                with pin_generation_read(gdir):
                    if not gdir.is_dir():
                        raise FileNotFoundError(f"MVCC generation removed before pin: {gdir}")
                    cfg = read_json(gdir / "config.json")
                    meta = read_json(gdir / "run_meta.json")
                    if isinstance(meta, dict):
                        for pkey in ("qc_json_path", "prepared_json_path", "detection_json_path", "classification_json_path"):
                            if meta.get(pkey):
                                resolved_p = self.resolve_workspace_artifact_path(meta[pkey])
                                if resolved_p is not None:
                                    meta[pkey] = str(resolved_p)
                    summ = _normalize_legacy_summary(read_json(gdir / "summary.json"))
                    if isinstance(meta.get("counts"), dict) and isinstance(summ.get("counts"), dict):
                        for k in ("non_artifact_events", "spike_hfo_events", "ehfo_events"):
                            if k in summ["counts"] and summ["counts"][k] is None:
                                meta["counts"][k] = None
                    if is_mvcc:
                        qc_doc = read_json(gdir / "qc_summary.json")
                        ev_doc = read_json(gdir / "events.json")
                    else:
                        qc_doc = read_json(gdir / "qc_summary.json") if (gdir / "qc_summary.json").is_file() else {}
                        ev_doc = read_json(gdir / "events.json") if (gdir / "events.json").is_file() else []
                    dirty = self._reconcile_bundle_with_judgments(case_slug, rec_slug, run_name, ev_doc, summ, meta)
                    bundle = {
                        "run_dir": str(rdir),
                        "generation_dir": str(gdir),
                        "config": cfg,
                        "meta": meta,
                        "qc": qc_doc,
                        "summary": summ,
                        "events": ev_doc,
                    }
                if dirty and reconcile and not getattr(_RECONCILE_LOCAL, "active", False):
                    _RECONCILE_LOCAL.active = True
                    try:
                        from .pipeline import refresh_verdict_tallies

                        refresh_verdict_tallies(self, case_slug, rec_slug, run_name)
                        return self.load_run(case_slug, rec_slug, run_name, reconcile=False)
                    except Exception:
                        pass
                    finally:
                        _RECONCILE_LOCAL.active = False
                return bundle
            except FileNotFoundError as exc:
                last_exc = exc
                continue
        raise ToolkitError(
            "run_not_found",
            f"Run {run_name!r} generation vanished during concurrent read for {case_slug}/{rec_slug}: {last_exc}",
        )

    def resolve_default_comparison_run(self, case_slug: str, primary_run: str) -> str | None:
        cslug = sanitize_slug(case_slug)
        p_clean = sanitize_slug(primary_run)
        for r in self.list_recordings(cslug):
            if not self.has_run(cslug, r, p_clean):
                continue
            runs = self.list_runs(cslug, r)
            p_fp = next(
                (m.get("recording_fingerprint") for m in runs if m.get("run_name") and sanitize_slug(str(m["run_name"])) == p_clean),
                None,
            )
            others = [
                m["run_name"]
                for m in runs
                if m.get("run_name")
                and sanitize_slug(str(m["run_name"])) != p_clean
                and (not p_fp or not m.get("recording_fingerprint") or m.get("recording_fingerprint") == p_fp)
            ]
            if others:
                return str(others[0])
        return None

    def primary_run_signature(self, case_slug: str, primary_run: str) -> dict[str, Any]:
        cslug = sanitize_slug(case_slug)
        rname = sanitize_slug(primary_run)
        return {
            r: bundle_signature_entry(self.load_run(cslug, r, rname))
            for r in sorted(self.list_recordings(cslug))
            if self.has_run(cslug, r, rname)
        }

    # --- Cohorts & Live UI Focus State ---

    def cohort_dir(self, cohort_slug: str) -> Path:
        return self._safe_child(self.root / "cohorts", cohort_slug)

    def list_cohorts(self) -> list[dict[str, Any]]:
        croot = self.root / "cohorts"
        if not croot.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for d in sorted(croot.iterdir()):
            p = d / "cohort_summary.json"
            if d.is_dir() and p.is_file():
                try:
                    out.append(read_json(p))
                except Exception:
                    continue
        return out

    def ui_state_path(self) -> Path:
        return self.root / "ui_state.json"

    def load_ui_state(self) -> dict[str, Any]:
        p = self.ui_state_path()
        if not p.is_file():
            return {
                "seq": 0,
                "case": None,
                "recording": None,
                "run_name": "baseline",
                "compare_run": None,
                "view": "overview",
                "channel": None,
                "shaft": None,
                "start_seconds": 0.0,
                "window_seconds": 8.0,
                "event_id": None,
                "queue_mode": "focal",
                "updated_by": "init",
                "updated_at": utc_now(),
            }
        try:
            return read_json(p)
        except Exception:
            return {"seq": 0, "case": None}

    def update_ui_state(self, patch: dict[str, Any], updated_by: str = "workbench") -> dict[str, Any]:
        with FileLock(str(self.root / ".ui_state.lock")):
            cur = {k: v for k, v in self.load_ui_state().items() if k in _UI_STATE_KEYS or k in {"seq", "updated_by", "updated_at"}}
            clean_patch = {
                k: v
                for k, v in patch.items()
                if k in _UI_STATE_KEYS and (v is not None or k in {"compare_run", "event_id", "shaft"})
            }
            merged = {
                **cur,
                **clean_patch,
                "seq": int(cur.get("seq", 0)) + 1,
                "updated_by": updated_by,
                "updated_at": utc_now(),
            }
            atomic_write_json(self.ui_state_path(), merged)
            return merged
