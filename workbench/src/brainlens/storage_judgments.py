"""Clinician judgment WAL, reconciliation, verdict projections, and clinical overlays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from filelock import FileLock

from .models import ToolkitError, atomic_write_json, canonical_digest, read_json, utc_now
from .storage_mvcc import (
    _norm_interval,
    _normalize_legacy_summary,
    append_jsonl_line,
    classify_judgment_role,
    pin_generation_read,
    run_analysis_signature,
    sanitize_slug,
)

_ROLE_FIELDS = {
    "clinician": ("verdict", "note", "clipped"),
    "researcher": ("researcher_verdict", "researcher_note", "researcher_clipped"),
    "rehearsal": ("rehearsal_verdict", "rehearsal_note", "rehearsal_clipped"),
    "agent": ("agent_verdict", "agent_note", "agent_clipped"),
}


def validate_clinical_overlay(overlay: Any) -> dict[str, Any]:
    """Validate and normalize a clinical_overlay payload before any disk write."""
    if not isinstance(overlay, dict):
        raise ToolkitError("invalid_overlay", "clinical_overlay must be a JSON object.")
    raw_source = overlay.get("source")
    if raw_source is not None and not isinstance(raw_source, str):
        raise ToolkitError("invalid_overlay", "clinical_overlay.source must be a string.")
    source = str(raw_source or "").strip()
    if not source:
        raise ToolkitError(
            "missing_overlay_source",
            "External clinical overlays (SOZ / Resected channels) require an explicit provenance `source` citation.",
        )
    raw_notes = overlay.get("notes")
    if raw_notes is not None and not isinstance(raw_notes, str):
        raise ToolkitError("invalid_overlay", "clinical_overlay.notes must be a string.")

    def _norm_chs(field_name: str, val: Any) -> list[str]:
        if val is None:
            return []
        if isinstance(val, str):
            items = [x.strip() for x in val.replace(";", ",").split(",") if x.strip()]
            return list(dict.fromkeys(items))
        if isinstance(val, (list, tuple, set)):
            items: list[str] = []
            for entry in val:
                if not isinstance(entry, str):
                    raise ToolkitError(
                        "invalid_overlay",
                        f"clinical_overlay.{field_name} entries must be strings, got {type(entry).__name__}.",
                    )
                s = entry.strip()
                if s:
                    items.append(s)
            return list(dict.fromkeys(items))
        raise ToolkitError(
            "invalid_overlay",
            f"clinical_overlay.{field_name} must be a list of channel names or comma-separated string, got {type(val).__name__}.",
        )

    return {
        "source": source,
        "author": str(overlay.get("author") or "clinician"),
        "notes": str(raw_notes or "").strip(),
        "soz_channels": _norm_chs("soz_channels", overlay.get("soz_channels")),
        "resected_channels": _norm_chs("resected_channels", overlay.get("resected_channels")),
        "irritative_channels": _norm_chs("irritative_channels", overlay.get("irritative_channels")),
    }


def _fold_judgment_into_verdict_state(cur: dict[str, Any], row: dict[str, Any]) -> None:
    """Apply a single WAL row onto an event's accumulated per-role verdict/clip/note state."""
    if row.get("channel"):
        cur["channel"] = row.get("channel")
    cur["updated_at"] = row.get("created_at") or cur["updated_at"]
    bucket = classify_judgment_role(row.get("role"))
    v_key, n_key, c_key = _ROLE_FIELDS[bucket]
    if row.get("verdict") is not None:
        cur[v_key] = row["verdict"]
        if bucket == "clinician":
            cur["author"] = row.get("author")
            cur["role"] = row.get("role") or "clinician"
    if row.get("note"):
        cur[n_key] = str(row["note"])
    if row.get("clipped") is not None:
        cur[c_key] = bool(row["clipped"])
        if bucket == "clinician":
            cur["clipped_role"] = (row.get("role") or "clinician") if cur[c_key] else None


class JudgmentStorageMixin:
    """WAL-backed clinician judgments, active-snapshot binding, verdict projection, and clinical overlays."""

    def _reconcile_bundle_with_judgments(
        self,
        case_slug: str,
        rec_slug: str,
        run_name: str,
        events: list[dict[str, Any]],
        summary: dict[str, Any],
        meta: dict[str, Any],
    ) -> bool:
        """Overlay durable judgments.jsonl WAL onto loaded events, summary, & meta; returns True if disk lagged."""
        run_fp = meta.get("recording_fingerprint") or summary.get("recording_fingerprint")
        run_asig = meta.get("analysis_signature") or run_analysis_signature({"meta": meta, "summary": summary})
        run_floor = int(meta.get("analysis_superseded_wal_index") or 0)
        verdicts = self.latest_event_verdicts(
            case_slug,
            rec_slug,
            run_name,
            recording_fingerprint=run_fp,
            analysis_signature=run_asig,
            analysis_wal_floor=run_floor,
        )
        if not verdicts:
            return False
        dirty = False
        by_ch_conf: dict[str, int] = {}
        by_ch_art: dict[str, int] = {}
        by_ch_unc: dict[str, int] = {}
        for ev in events:
            eid = ev.get("id")
            v = verdicts.get(eid) if eid else None
            if v is not None:
                persisted_updates = {
                    "clinician_verdict": v.get("verdict"),
                    "clipped": bool(v.get("clipped", False)),
                    "clinician_note": v.get("note") or "",
                    "verdict_role": v.get("role"),
                    "researcher_verdict": v.get("researcher_verdict"),
                    "researcher_clipped": bool(v.get("researcher_clipped", False)),
                    "rehearsal_verdict": v.get("rehearsal_verdict"),
                    "rehearsal_clipped": bool(v.get("rehearsal_clipped", False)),
                    "agent_verdict": v.get("agent_verdict"),
                    "agent_note": v.get("agent_note") or "",
                    "agent_clipped": bool(v.get("agent_clipped", False)),
                }
                for fk, fv in persisted_updates.items():
                    if ev.get(fk) != fv:
                        dirty = True
                    ev[fk] = fv
                ev["clinician_author"] = v.get("author")
                ev["clinician_role"] = v.get("role")
                ev["clipped_role"] = v.get("clipped_role")
                ev["researcher_note"] = v.get("researcher_note") or ""
                ev["rehearsal_note"] = v.get("rehearsal_note") or ""
            cid = str(ev.get("channel_id") or "")
            cv = ev.get("clinician_verdict")
            if cv == "confirm":
                by_ch_conf[cid] = by_ch_conf.get(cid, 0) + 1
            elif cv == "artifact":
                by_ch_art[cid] = by_ch_art.get(cid, 0) + 1
            elif cv == "uncertain":
                by_ch_unc[cid] = by_ch_unc.get(cid, 0) + 1

        for list_key in ("channels", "leading_focal_channels", "leading_raw_channels"):
            for row in summary.get(list_key) or []:
                if not isinstance(row, dict):
                    continue
                cid = str(row.get("channel_id") or "")
                new_conf, new_art, new_unc = by_ch_conf.get(cid, 0), by_ch_art.get(cid, 0), by_ch_unc.get(cid, 0)
                if (
                    row.get("clinician_confirmed") != new_conf
                    or row.get("clinician_artifact") != new_art
                    or row.get("clinician_uncertain") != new_unc
                ):
                    dirty = True
                row["clinician_confirmed"] = new_conf
                row["clinician_artifact"] = new_art
                row["clinician_uncertain"] = new_unc

        valid_v = {"confirm", "artifact", "uncertain"}
        reconciled = {
            "reviewed_focal_events": sum(
                1 for e in events if e.get("synchrony_class") != "widespread_burst" and e.get("clinician_verdict") in valid_v
            ),
            "clinician_confirmed": sum(1 for e in events if e.get("clinician_verdict") == "confirm"),
            "clinician_artifact": sum(1 for e in events if e.get("clinician_verdict") == "artifact"),
            "clinician_uncertain": sum(1 for e in events if e.get("clinician_verdict") == "uncertain"),
            "researcher_confirmed": sum(1 for e in events if e.get("researcher_verdict") == "confirm"),
            "researcher_reviewed": sum(1 for e in events if e.get("researcher_verdict") in valid_v),
            "rehearsal_confirmed": sum(1 for e in events if e.get("rehearsal_verdict") == "confirm"),
            "rehearsal_reviewed": sum(1 for e in events if e.get("rehearsal_verdict") in valid_v),
            "agent_confirmed": sum(1 for e in events if e.get("agent_verdict") == "confirm"),
            "agent_artifact": sum(1 for e in events if e.get("agent_verdict") == "artifact"),
            "agent_uncertain": sum(1 for e in events if e.get("agent_verdict") == "uncertain"),
            "agent_reviewed": sum(1 for e in events if e.get("agent_verdict") in valid_v),
            "clipped_for_discussion": sum(1 for e in events if e.get("clipped")),
        }
        for target_doc in (summary, meta):
            if isinstance(target_doc.get("counts"), dict):
                if any(target_doc["counts"].get(rk) != rv for rk, rv in reconciled.items()):
                    dirty = True
                target_doc["counts"].update(reconciled)
        return dirty

    def reconcile_pending_judgments(self, case_slug: str | None = None) -> int:
        """Heal any runs/cases whose on-disk MVCC JSON/CSV lagged behind durable judgments.jsonl."""
        targets = [sanitize_slug(case_slug)] if case_slug else [str(c.get("case")) for c in self.list_cases() if c.get("case")]
        healed = 0
        for cslug in targets:
            rows = self.load_judgments(cslug)
            if rows:
                for row in rows:
                    q = str(row.get("clinical_question") or "").strip()
                    if q:
                        try:
                            self.add_open_question(cslug, q)
                        except Exception:
                            pass
                pairs = {
                    (sanitize_slug(str(row["recording"])), sanitize_slug(str(row.get("run_name") or "baseline")))
                    for row in rows
                    if row.get("event_id") and row.get("recording")
                }
                for rec_s, rn_s in sorted(pairs):
                    if not self.has_run(cslug, rec_s, rn_s):
                        continue
                    try:
                        resolved = self.resolve_run_gen_dir(cslug, rec_s, rn_s)
                        if resolved is None:
                            continue
                        gdir, _ = resolved
                        with pin_generation_read(gdir):
                            if not gdir.is_dir():
                                continue
                            meta = read_json(gdir / "run_meta.json")
                            summ = _normalize_legacy_summary(read_json(gdir / "summary.json"))
                            ev_doc = read_json(gdir / "events.json") if (gdir / "events.json").is_file() else []
                            dirty = self._reconcile_bundle_with_judgments(cslug, rec_s, rn_s, ev_doc, summ, meta)
                        if dirty:
                            from .pipeline import refresh_verdict_tallies

                            refresh_verdict_tallies(self, cslug, rec_s, rn_s)
                            healed += 1
                    except Exception:
                        continue
            for rec_s in self.list_recordings(cslug):
                try:
                    rmodel = self.load_recording(cslug, rec_s)
                except Exception:
                    continue
                cur_sp_digest = canonical_digest(
                    rmodel.spatial.model_dump(mode="json") if rmodel.spatial is not None else None
                )
                for rm in self.list_runs(cslug, rec_s):
                    rn_s = str(rm.get("run_name") or "")
                    if not rn_s or rm.get("recording_fingerprint") != rmodel.fingerprint:
                        continue
                    if rm.get("recording_spatial_digest") != cur_sp_digest:
                        try:
                            from .pipeline import refresh_verdict_tallies

                            refresh_verdict_tallies(self, cslug, rec_s, rn_s)
                            healed += 1
                        except Exception:
                            continue
        return healed

    # --- Append-only Clinician Judgments WAL ---

    def judgments_path(self, case_slug: str) -> Path:
        return self.case_dir(case_slug) / "judgments.jsonl"

    def load_judgments(
        self,
        case_slug: str,
        recording: str | None = None,
        run_name: str | None = None,
    ) -> list[dict[str, Any]]:
        jpath = self.judgments_path(case_slug)
        if not jpath.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with jpath.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        if recording is not None:
            r_clean = sanitize_slug(recording)
            rows = [r for r in rows if r.get("recording") in {None, recording, r_clean}]
        if run_name is not None:
            n_clean = sanitize_slug(run_name)
            rows = [r for r in rows if r.get("run_name") in {None, run_name, n_clean}]
        return rows

    def record_judgment_ex(
        self,
        case_slug: str,
        item: dict[str, Any],
        *,
        validate_event_ownership: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        """Append a validated judgment entry to the WAL. Returns (entry, created_new)."""
        cdir = self.case_dir(case_slug)
        cdir.mkdir(parents=True, exist_ok=True)
        verdict = item.get("verdict")
        if verdict is not None and verdict not in {"confirm", "artifact", "uncertain"}:
            raise ToolkitError("invalid_verdict", "verdict must be 'confirm', 'artifact', or 'uncertain'.")
        op_id = str(item.get("operation_id") or "").strip() or None

        with FileLock(str(cdir / ".case.lock")):
            with FileLock(str(cdir / ".judgments.lock")):
                existing = self.load_judgments(case_slug)
                if op_id:
                    for prev in reversed(existing):
                        if prev.get("operation_id") == op_id:
                            return prev, False

                eff_item = dict(item)
                wal_floor = 0
                rec_val = eff_item.get("recording")
                if rec_val:
                    rec_clean = sanitize_slug(str(rec_val))
                    run_clean = sanitize_slug(str(eff_item.get("run_name") or "baseline"))
                    if self.recording_path(case_slug, rec_clean).is_file():
                        rdoc = self.load_recording_doc(case_slug, rec_clean)
                        cur_rec = self.load_recording(case_slug, rec_clean)
                        wal_floor = max(wal_floor, int(rdoc.get("replaced_at_wal_index") or 0))
                        req_fp = str(eff_item.get("recording_fingerprint") or "").strip() or None
                        if req_fp and req_fp != cur_rec.fingerprint:
                            raise ToolkitError(
                                "run_recording_mismatch",
                                f"Recording {rec_clean!r} changed since this review view loaded ({req_fp[:12]} vs {cur_rec.fingerprint[:12]}); "
                                "reload and re-verify on the current recording before saving.",
                                recording=rec_clean,
                                run_name=run_clean,
                            )
                        if self.has_run(case_slug, rec_clean, run_clean):
                            run_bundle = self.load_run(case_slug, rec_clean, run_clean, reconcile=False)
                            rmeta = run_bundle.get("meta") or {}
                            run_fp = rmeta.get("recording_fingerprint") or (run_bundle.get("summary") or {}).get("recording_fingerprint")
                            if run_fp and run_fp != cur_rec.fingerprint:
                                raise ToolkitError(
                                    "run_recording_mismatch",
                                    f"Run {run_clean!r} for {rec_clean!r} belongs to superseded recording snapshot {run_fp[:12]} "
                                    f"(current registered recording is {cur_rec.fingerprint[:12]}); re-analyze before recording verdicts or clips.",
                                    recording=rec_clean,
                                    run_name=run_clean,
                                )
                            cur_asig = str(rmeta.get("analysis_signature") or run_analysis_signature(run_bundle))
                            cur_agen = str(rmeta.get("analysis_generation_id") or rmeta.get("generation_id") or "")
                            wal_floor = max(wal_floor, int(rmeta.get("analysis_superseded_wal_index") or 0))
                            req_asig = str(eff_item.get("analysis_signature") or "").strip() or None
                            req_agen = str(eff_item.get("analysis_generation_id") or "").strip() or None
                            if (req_asig and req_asig != cur_asig) or (req_agen and cur_agen and req_agen != cur_agen):
                                raise ToolkitError(
                                    "stale_run_generation",
                                    f"Run {run_clean!r} for {rec_clean!r} was overwritten with a newer analysis generation; "
                                    "reload and re-verify before recording verdicts or clips.",
                                    recording=rec_clean,
                                    run_name=run_clean,
                                )
                            ev_id = eff_item.get("event_id")
                            if validate_event_ownership and ev_id:
                                ev_match = next((e for e in run_bundle.get("events", []) if e.get("id") == ev_id), None)
                                if ev_match is None:
                                    raise ToolkitError(
                                        "event_not_in_run",
                                        f"Event {ev_id!r} does not belong to {sanitize_slug(case_slug)}/{rec_clean}/{run_clean}.",
                                    )
                                ch_req = eff_item.get("channel")
                                if ch_req and ev_match.get("channel_name") and ch_req != ev_match.get("channel_name"):
                                    raise ToolkitError(
                                        "event_channel_mismatch",
                                        f"Event {ev_id!r} belongs to channel {ev_match['channel_name']!r}, not {ch_req!r}.",
                                    )
                            eff_item["analysis_signature"] = cur_asig
                            if cur_agen:
                                eff_item["analysis_generation_id"] = cur_agen
                        eff_item["recording_fingerprint"] = cur_rec.fingerprint

                if not op_id:
                    target_rec = eff_item.get("recording")
                    target_run = eff_item.get("run_name")
                    target_ev = eff_item.get("event_id")
                    target_ch = eff_item.get("channel")
                    target_iv = _norm_interval(eff_item.get("interval_seconds"))
                    target_center = eff_item.get("center_seconds")
                    target_fp = eff_item.get("recording_fingerprint")
                    target_asig = eff_item.get("analysis_signature")
                    last_for_target = None
                    for idx_back in range(len(existing) - 1, wal_floor - 1, -1):
                        prev = existing[idx_back]
                        if (
                            prev.get("recording") == target_rec
                            and prev.get("run_name") == target_run
                            and prev.get("recording_fingerprint") == target_fp
                            and prev.get("analysis_signature") == target_asig
                            and prev.get("event_id") == target_ev
                            and prev.get("channel") == target_ch
                            and _norm_interval(prev.get("interval_seconds")) == target_iv
                            and prev.get("center_seconds") == target_center
                        ):
                            last_for_target = prev
                            break
                    if (
                        last_for_target is not None
                        and last_for_target.get("kind") == eff_item.get("kind")
                        and last_for_target.get("verdict") == eff_item.get("verdict")
                        and last_for_target.get("clipped") == eff_item.get("clipped")
                        and last_for_target.get("clinical_question") == eff_item.get("clinical_question")
                        and last_for_target.get("note") == eff_item.get("note")
                        and last_for_target.get("author") == eff_item.get("author")
                        and last_for_target.get("role") == eff_item.get("role")
                    ):
                        return last_for_target, False

                max_seq = len(existing)
                for row in existing:
                    rid = str(row.get("id") or "")
                    if rid.startswith("jdg_") and rid[4:].isdigit():
                        max_seq = max(max_seq, int(rid[4:]))
                entry = {
                    "id": f"jdg_{max_seq + 1:04d}",
                    "case": sanitize_slug(case_slug),
                    "created_at": utc_now(),
                    **eff_item,
                }
                if op_id:
                    entry["operation_id"] = op_id
                append_jsonl_line(self.judgments_path(case_slug), entry)
                return entry, True

    def record_judgment(
        self,
        case_slug: str,
        item: dict[str, Any],
        *,
        validate_event_ownership: bool = False,
    ) -> dict[str, Any]:
        entry, _ = self.record_judgment_ex(case_slug, item, validate_event_ownership=validate_event_ownership)
        return entry

    def latest_event_verdicts(
        self,
        case_slug: str,
        recording: str,
        run_name: str,
        *,
        recording_fingerprint: str | None = None,
        analysis_signature: str | None = None,
        analysis_wal_floor: int = 0,
        rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        all_rows = rows if rows is not None else self.load_judgments(case_slug)
        r_clean = sanitize_slug(recording)
        rn_clean = sanitize_slug(str(run_name or "baseline"))
        rec_floor = 0
        try:
            rdoc = self.load_recording_doc(case_slug, r_clean)
            cur_rec_fp = str((rdoc.get("spec") or {}).get("fingerprint") or "")
            if not recording_fingerprint or not cur_rec_fp or str(recording_fingerprint) == cur_rec_fp:
                rec_floor = int(rdoc.get("replaced_at_wal_index") or 0)
        except Exception:
            rec_floor = 0
        eff_floor = max(rec_floor, int(analysis_wal_floor or 0))
        by_ev: dict[str, dict[str, Any]] = {}
        for wal_idx, r in enumerate(all_rows):
            if wal_idx < eff_floor:
                continue
            eid = r.get("event_id")
            if not eid or not r.get("recording"):
                continue
            if sanitize_slug(str(r["recording"])) != r_clean:
                continue
            if sanitize_slug(str(r.get("run_name") or "baseline")) != rn_clean:
                continue
            r_fp = str(r.get("recording_fingerprint") or "").strip()
            if r_fp and recording_fingerprint and r_fp != str(recording_fingerprint):
                continue
            r_asig = str(r.get("analysis_signature") or "").strip()
            if r_asig and analysis_signature and r_asig != str(analysis_signature):
                continue
            cur = by_ev.setdefault(
                eid,
                {
                    "event_id": str(eid),
                    "channel": r.get("channel"),
                    "verdict": None,
                    "note": "",
                    "author": None,
                    "role": None,
                    "clipped": False,
                    "clipped_role": None,
                    "researcher_verdict": None,
                    "researcher_note": "",
                    "researcher_clipped": False,
                    "rehearsal_verdict": None,
                    "rehearsal_note": "",
                    "rehearsal_clipped": False,
                    "agent_verdict": None,
                    "agent_note": "",
                    "agent_clipped": False,
                    "updated_at": None,
                },
            )
            _fold_judgment_into_verdict_state(cur, r)
        return by_ev

    def case_verdicts_projection(self, case_slug: str) -> dict[str, Any]:
        """Return active-snapshot verdict projection keyed by 'recording|run_name|event_id' + review tallies."""
        cslug = sanitize_slug(case_slug)
        rows = self.load_judgments(cslug)
        by_key: dict[str, dict[str, Any]] = {}
        review_tallies: dict[str, dict[str, Any]] = {}
        covered_pairs: set[tuple[str, str]] = set()

        def _store_projected_verdicts(rec_s: str, rn_s: str, verdicts: dict[str, dict[str, Any]]) -> None:
            for eid, v in verdicts.items():
                by_key[f"{rec_s}|{rn_s}|{eid}"] = {
                    "recording": rec_s,
                    "run_name": rn_s,
                    "event_id": str(eid),
                    "channel": v.get("channel"),
                    "verdict": v.get("verdict"),
                    "researcher_verdict": v.get("researcher_verdict"),
                    "rehearsal_verdict": v.get("rehearsal_verdict"),
                    "agent_verdict": v.get("agent_verdict"),
                    "role": v.get("role"),
                    "clipped": bool(v.get("clipped", False)),
                    "researcher_clipped": bool(v.get("researcher_clipped", False)),
                    "rehearsal_clipped": bool(v.get("rehearsal_clipped", False)),
                    "agent_clipped": bool(v.get("agent_clipped", False)),
                    "updated_at": v.get("updated_at"),
                }

        for r in self.list_recordings(cslug):
            rroot = self.recording_dir(cslug, r) / "runs"
            if not rroot.is_dir():
                continue
            for run_dir_path in sorted(rroot.iterdir()):
                if not run_dir_path.is_dir() or run_dir_path.name.startswith("."):
                    continue
                rn_s = sanitize_slug(run_dir_path.name)
                covered_pairs.add((r, rn_s))
                try:
                    b = self.load_run(cslug, r, run_dir_path.name, reconcile=False)
                    rmeta = b.get("meta") or {}
                    rsumm = b.get("summary") or {}
                    cnt = rsumm.get("counts") or {}
                    run_fp = rmeta.get("recording_fingerprint") or rsumm.get("recording_fingerprint")
                    run_asig = rmeta.get("analysis_signature") or run_analysis_signature(b)
                    run_floor = int(rmeta.get("analysis_superseded_wal_index") or 0)
                    verdicts = self.latest_event_verdicts(
                        cslug,
                        r,
                        rn_s,
                        recording_fingerprint=run_fp,
                        analysis_signature=run_asig,
                        analysis_wal_floor=run_floor,
                        rows=rows,
                    )
                    _store_projected_verdicts(r, rn_s, verdicts)
                    review_tallies[f"{r}|{rn_s}"] = {
                        "non_burst_events": int(cnt.get("non_burst_events", 0) or 0),
                        "reviewed_focal_events": int(
                            cnt.get(
                                "reviewed_focal_events",
                                (cnt.get("clinician_confirmed", 0) or 0)
                                + (cnt.get("clinician_artifact", 0) or 0)
                                + (cnt.get("clinician_uncertain", 0) or 0),
                            )
                            or 0
                        ),
                        "clinician_confirmed": int(cnt.get("clinician_confirmed", 0) or 0),
                        "clinician_artifact": int(cnt.get("clinician_artifact", 0) or 0),
                        "clinician_uncertain": int(cnt.get("clinician_uncertain", 0) or 0),
                        "researcher_reviewed": int(cnt.get("researcher_reviewed", 0) or 0),
                        "researcher_confirmed": int(cnt.get("researcher_confirmed", 0) or 0),
                        "rehearsal_reviewed": int(cnt.get("rehearsal_reviewed", 0) or 0),
                        "rehearsal_confirmed": int(cnt.get("rehearsal_confirmed", 0) or 0),
                        "agent_reviewed": int(cnt.get("agent_reviewed", 0) or 0),
                    }
                except Exception:
                    continue

        # Also project any WAL pairs whose run is not yet committed on disk (respecting recording floor & fingerprint).
        extra_pairs = {
            (sanitize_slug(str(row["recording"])), sanitize_slug(str(row.get("run_name") or "baseline")))
            for row in rows
            if row.get("event_id") and row.get("recording")
        } - covered_pairs
        for rec_s, rn_s in sorted(extra_pairs):
            cur_fp = None
            try:
                cur_fp = self.load_recording(cslug, rec_s).fingerprint
            except Exception:
                cur_fp = None
            verdicts = self.latest_event_verdicts(cslug, rec_s, rn_s, recording_fingerprint=cur_fp, rows=rows)
            _store_projected_verdicts(rec_s, rn_s, verdicts)

        return {
            "case": cslug,
            "judgment_revision": len(rows),
            "verdicts": by_key,
            "review_tallies": review_tallies,
        }

    # --- External Clinical Annotation Overlay (SOZ / Resected — Strictly Attributed) ---

    def clinical_overlay_path(self, case_slug: str) -> Path:
        return self.case_dir(case_slug) / "clinical_overlay.json"

    def load_clinical_overlay(self, case_slug: str) -> dict[str, Any] | None:
        p = self.clinical_overlay_path(case_slug)
        if not p.is_file():
            return None
        try:
            return read_json(p)
        except Exception:
            return None

    def save_clinical_overlay(
        self,
        case_slug: str,
        overlay: dict[str, Any],
        *,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        clean = validate_clinical_overlay(overlay)
        cdir = self.case_dir(case_slug)
        cdir.mkdir(parents=True, exist_ok=True)
        op_id = str(operation_id or "").strip() or None
        with FileLock(str(cdir / ".case.lock")):
            prev = self.load_clinical_overlay(case_slug) or {}
            applied_ops = list(prev.get("applied_operation_ids") or [])
            if op_id and op_id in applied_ops:
                return prev
            if op_id:
                applied_ops.append(op_id)
                applied_ops = applied_ops[-128:]
            doc = {
                "case": sanitize_slug(case_slug),
                **clean,
                "external_only_disclaimer": (
                    "External clinician/dataset reference annotation — NOT inferred or validated by BrainLens signal processing."
                ),
                "updated_at": utc_now(),
            }
            if applied_ops:
                doc["applied_operation_ids"] = applied_ops
            atomic_write_json(self.clinical_overlay_path(case_slug), doc)
            return doc

    def delete_clinical_overlay(self, case_slug: str) -> bool:
        cdir = self.case_dir(case_slug)
        if not cdir.is_dir():
            return False
        with FileLock(str(cdir / ".case.lock")):
            p = self.clinical_overlay_path(case_slug)
            if p.is_file():
                p.unlink()
                return True
            return False
