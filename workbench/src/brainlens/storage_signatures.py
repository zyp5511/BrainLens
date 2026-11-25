"""Active-snapshot binding, clipped evidence target projection, and MDT case-sheet signature digests."""

from __future__ import annotations

from typing import Any

from .anatomy import load_anatomy_labels
from .models import canonical_digest
from .storage_mvcc import _norm_interval, classify_judgment_role, run_analysis_signature, sanitize_slug

_ROLE_CLIP_LABELS = {
    "clinician": "Clinician-clipped evidence",
    "researcher": "Researcher-clipped evidence",
    "rehearsal": "Rehearsal-clipped evidence",
    "agent": "Agent-clipped candidate",
}


class CaseSignatureMixin:
    """Case-level snapshot binding, active clips, clinical notes digests, and MDT sheet cache signatures."""

    def case_judgment_binding_context(self, case_slug: str) -> dict[str, Any]:
        """Collect current recording fingerprints, replacement WAL floors, and active run analysis signatures & floors."""
        cslug = sanitize_slug(case_slug)
        rec_fps: dict[str, str] = {}
        rec_wal_floors: dict[str, int] = {}
        run_asigs: dict[tuple[str, str], str] = {}
        run_wal_floors: dict[tuple[str, str], int] = {}
        for r in sorted(self.list_recordings(cslug)):
            try:
                rdoc = self.load_recording_doc(cslug, r)
                rec_fps[r] = str((rdoc.get("spec") or {}).get("fingerprint") or "")
                rec_wal_floors[r] = int(rdoc.get("replaced_at_wal_index") or 0)
            except Exception:
                rec_fps[r] = ""
                rec_wal_floors[r] = 0
            for rm in self.list_runs(cslug, r):
                rn = rm.get("run_name")
                if not rn:
                    continue
                rn_s = sanitize_slug(str(rn))
                run_asigs[(r, rn_s)] = str(rm.get("analysis_signature") or run_analysis_signature(rm))
                run_wal_floors[(r, rn_s)] = int(rm.get("analysis_superseded_wal_index") or 0)
        return {
            "rec_fps": rec_fps,
            "rec_wal_floors": rec_wal_floors,
            "run_asigs": run_asigs,
            "run_wal_floors": run_wal_floors,
        }

    def judgment_matches_active_state(
        self,
        j: dict[str, Any],
        wal_idx: int,
        ctx: dict[str, Any],
        *,
        default_run: str = "baseline",
    ) -> bool:
        """Return True iff WAL row `j` at 0-based index `wal_idx` is still valid for current recording & analysis."""
        if not j.get("recording"):
            return True
        j_rec = sanitize_slug(str(j["recording"]))
        if wal_idx < int((ctx.get("rec_wal_floors") or {}).get(j_rec, 0)):
            return False
        cur_fp = (ctx.get("rec_fps") or {}).get(j_rec)
        j_fp = str(j.get("recording_fingerprint") or "").strip()
        if j_fp and cur_fp and j_fp != cur_fp:
            return False

        j_run = sanitize_slug(str(j.get("run_name") or default_run))
        if wal_idx < int((ctx.get("run_wal_floors") or {}).get((j_rec, j_run), 0)):
            return False
        cur_asig = (ctx.get("run_asigs") or {}).get((j_rec, j_run))
        j_asig = str(j.get("analysis_signature") or "").strip()
        if j_asig and cur_asig and j_asig != cur_asig:
            return False
        return True

    def active_clipped_targets(
        self,
        case_slug: str,
        default_run: str = "baseline",
        *,
        judgments: list[dict[str, Any]] | None = None,
        rec_fps: dict[str, str] | None = None,
        binding_ctx: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return deduplicated currently-clipped evidence targets preserving per-role clip independence & binding."""
        rows = judgments if judgments is not None else self.load_judgments(case_slug)
        ctx = binding_ctx if binding_ctx is not None else self.case_judgment_binding_context(case_slug)
        if rec_fps is not None:
            ctx = {**ctx, "rec_fps": rec_fps}
        by_role_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        order_targets: list[tuple[str, str, str]] = []
        seen_targets: set[tuple[str, str, str]] = set()
        for wal_idx, j in enumerate(rows):
            if not j.get("recording") or j.get("clipped") is None:
                continue
            if not self.judgment_matches_active_state(j, wal_idx, ctx, default_run=default_run):
                continue
            j_rec = sanitize_slug(str(j["recording"]))
            j_run = sanitize_slug(str(j.get("run_name") or default_run))
            iv = _norm_interval(j.get("interval_seconds"))
            if j.get("event_id"):
                j_target, eff_center = str(j["event_id"]), j.get("center_seconds")
            elif iv is not None:
                j_target, eff_center = f"iv:{iv[0]:.4f}-{iv[1]:.4f}:{j.get('channel') or ''}", round(0.5 * (iv[0] + iv[1]), 5)
            else:
                eff_center = j.get("center_seconds")
                j_target = f"t:{round(float(eff_center or 0.0), 2)}:{j.get('channel') or ''}"
            rbucket = classify_judgment_role(j.get("role"))
            tkey = (j_rec, j_run, j_target)
            rkey = (*tkey, rbucket)
            if bool(j.get("clipped")):
                by_role_key[rkey] = {
                    "recording": j_rec,
                    "run_name": j_run,
                    "channel": j.get("channel"),
                    "event_id": j.get("event_id"),
                    "center_seconds": eff_center,
                    "interval_seconds": iv,
                    "role_bucket": rbucket,
                }
                if tkey not in seen_targets:
                    seen_targets.add(tkey)
                    order_targets.append(tkey)
            else:
                by_role_key.pop(rkey, None)

        role_priority = ("clinician", "researcher", "rehearsal", "agent")
        out: list[dict[str, Any]] = []
        for tkey in order_targets:
            active_roles = [rb for rb in role_priority if (*tkey, rb) in by_role_key]
            if not active_roles:
                continue
            best_role = active_roles[0]
            item = dict(by_role_key[(*tkey, best_role)])
            item["active_roles"] = active_roles
            item["source_label"] = _ROLE_CLIP_LABELS[best_role]
            out.append(item)
        return out

    def case_clip_signature(
        self,
        case_slug: str,
        default_run: str = "baseline",
        *,
        judgments: list[dict[str, Any]] | None = None,
        rec_fps: dict[str, str] | None = None,
        binding_ctx: dict[str, Any] | None = None,
    ) -> str:
        clips = self.active_clipped_targets(
            case_slug, default_run=default_run, judgments=judgments, rec_fps=rec_fps, binding_ctx=binding_ctx
        )
        return canonical_digest([
            {
                "recording": c["recording"],
                "run_name": c["run_name"],
                "channel": c["channel"],
                "event_id": c["event_id"],
                "center_seconds": c["center_seconds"],
                "interval_seconds": c.get("interval_seconds"),
                "active_roles": c["active_roles"],
            }
            for c in clips
        ])

    def case_clinical_notes_digest(
        self,
        case_slug: str,
        *,
        judgments: list[dict[str, Any]] | None = None,
        rec_fps: dict[str, str] | None = None,
        binding_ctx: dict[str, Any] | None = None,
    ) -> str:
        """Digest active clinician/reviewer notes baked into MDT sheet HTML."""
        rows = judgments if judgments is not None else self.load_judgments(case_slug)
        ctx = binding_ctx if binding_ctx is not None else self.case_judgment_binding_context(case_slug)
        if rec_fps is not None:
            ctx = {**ctx, "rec_fps": rec_fps}
        items = [
            {
                "id": r.get("id"),
                "kind": r.get("kind"),
                "recording": r.get("recording"),
                "run_name": r.get("run_name"),
                "channel": r.get("channel"),
                "event_id": r.get("event_id"),
                "verdict": r.get("verdict"),
                "author": r.get("author"),
                "role": r.get("role"),
                "note": str(r.get("note") or "").strip(),
            }
            for wal_idx, r in enumerate(rows)
            if str(r.get("note") or "").strip()
            and r.get("kind") != "report_verdict"
            and self.judgment_matches_active_state(r, wal_idx, ctx)
        ]
        return canonical_digest(items)

    def case_recording_spatial_digest(self, case_slug: str, recordings: list[str] | None = None) -> dict[str, str]:
        cslug = sanitize_slug(case_slug)
        recs = sorted(recordings) if recordings is not None else sorted(self.list_recordings(cslug))
        out: dict[str, str] = {}
        for r in recs:
            try:
                rm = self.load_recording(cslug, r)
                out[r] = canonical_digest(rm.spatial.model_dump(mode="json") if rm.spatial is not None else None)
            except Exception:
                out[r] = canonical_digest(None)
        return out

    def case_recording_fingerprints(self, case_slug: str, recordings: list[str] | None = None) -> dict[str, str]:
        cslug = sanitize_slug(case_slug)
        recs = sorted(recordings) if recordings is not None else sorted(self.list_recordings(cslug))
        out: dict[str, str] = {}
        for r in recs:
            try:
                out[r] = str(self.load_recording(cslug, r).fingerprint or "")
            except Exception:
                out[r] = ""
        return out

    def case_sheet_run_signature(
        self,
        case_slug: str,
        primary_run: str,
        comparison_run: str | None = None,
        *,
        referenced_runs: list[str] | set[str] | None = None,
    ) -> dict[str, Any]:
        cslug = sanitize_slug(case_slug)
        p_clean = sanitize_slug(primary_run)
        eff_cmp = sanitize_slug(comparison_run) if comparison_run else self.resolve_default_comparison_run(cslug, p_clean)
        primary_sig = self.primary_run_signature(cslug, p_clean)
        cmp_sig = self.primary_run_signature(cslug, eff_cmp) if eff_cmp else {}
        ref_sigs = {
            rn: self.primary_run_signature(cslug, rn)
            for rn in sorted({sanitize_slug(str(x)) for x in (referenced_runs or []) if x})
            if rn not in {p_clean, eff_cmp}
        }
        judgments = self.load_judgments(cslug)
        overlay = self.load_clinical_overlay(cslug)
        try:
            open_questions = list(self.load_case(cslug).get("open_questions") or [])
        except Exception:
            open_questions = []
        rec_keys = sorted(primary_sig.keys())
        binding_ctx = self.case_judgment_binding_context(cslug)
        all_rec_fps = binding_ctx["rec_fps"]
        signature = {
            "primary_run": p_clean,
            "primary": primary_sig,
            "comparison_run": eff_cmp,
            "comparison": cmp_sig,
            "referenced_runs": ref_sigs,
            "clinical_overlay_digest": canonical_digest(overlay),
            "anatomy_labels_digest": canonical_digest(load_anatomy_labels(self, cslug)),
            "clip_signature": self.case_clip_signature(cslug, default_run=p_clean, judgments=judgments, binding_ctx=binding_ctx),
            "open_questions_digest": canonical_digest(open_questions),
            "clinical_notes_digest": self.case_clinical_notes_digest(cslug, judgments=judgments, binding_ctx=binding_ctx),
            "recording_spatial_digest": self.case_recording_spatial_digest(cslug, recordings=rec_keys),
            "recording_fingerprints": {r: all_rec_fps.get(r, "") for r in rec_keys},
            "judgment_revision": len(judgments),
        }
        from .evidence_map import get_evidence_map

        maps = {}
        for r in rec_keys:
            doc = get_evidence_map(self, cslug, recording=r, run_name=p_clean, cached_only=True)
            if doc is not None:
                maps[r] = doc["evidence_digest"]
        if maps:
            signature["evidence_map_digests"] = maps
        return signature
