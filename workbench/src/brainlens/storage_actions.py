"""Live report & clinician review action queue persistence mixin for WorkbenchStore."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from filelock import FileLock

from .models import ToolkitError, atomic_write_json, read_json, utc_now
from .storage_mvcc import DEFAULT_CLAIM_LEASE_SECONDS, append_jsonl_line, sanitize_slug

_VALID_ACTION_KINDS = frozenset({
    "ask_agent",
    "annotate",
    "verdict",
    "submit_batch",
    "exclude_request",
    "question_toggle",
})
NAVIGATION_QUESTION_SOURCES = ("waveform_reader", "admission_timeline")
QUESTION_CASE_REF = re.compile(r"case:[a-f0-9]{32}")


class ReportActionQueueMixin:
    """Append-only report action inbox with per-case lifecycle (pending -> claimed -> resolved)."""

    root: Path

    def report_actions_path(self) -> Path:
        return self.root / "report_actions.jsonl"

    def report_actions_state_path(self) -> Path:
        return self.root / "report_actions_state.json"

    def report_actions_cursor_path(self) -> Path:
        return self.root / "report_actions_cursor.json"

    def _load_action_state(self) -> dict[str, Any]:
        p = self.report_actions_state_path()
        if p.is_file():
            try:
                doc = read_json(p)
                return {
                    "actions": dict(doc.get("actions") or {}),
                    "consumer_cursors": dict(doc.get("consumer_cursors") or {}),
                    "legacy_consumed_seq": int(doc.get("legacy_consumed_seq", 0)),
                    "updated_at": doc.get("updated_at"),
                }
            except Exception:
                pass
        lp = self.report_actions_cursor_path()
        legacy_seq = 0
        if lp.is_file():
            try:
                legacy_seq = int(read_json(lp).get("consumed_seq", 0))
            except Exception:
                legacy_seq = 0
        return {
            "actions": {},
            "consumer_cursors": {},
            "legacy_consumed_seq": legacy_seq,
            "updated_at": None,
        }

    def _save_action_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        atomic_write_json(self.report_actions_state_path(), state)

    def _read_all_report_actions(self) -> list[dict[str, Any]]:
        apath = self.report_actions_path()
        if not apath.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with apath.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows

    def append_report_action_ex(self, item: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Append a validated clinician intent action. Returns (entry, created_new)."""
        self.root.mkdir(parents=True, exist_ok=True)
        verdict = item.get("verdict")
        if verdict is not None:
            if not isinstance(verdict, str) or verdict not in {"confirm", "artifact", "uncertain"}:
                raise ToolkitError("invalid_verdict", "verdict must be 'confirm', 'artifact', or 'uncertain'.")
        act_kind = str(item.get("action") or "annotate").strip()
        if act_kind not in _VALID_ACTION_KINDS:
            act_kind = "annotate"
        c_clean = sanitize_slug(str(item["case"])) if item.get("case") else None
        r_clean = sanitize_slug(str(item["recording"])) if item.get("recording") else None
        rn_clean = sanitize_slug(str(item["run_name"])) if item.get("run_name") else None
        comment_clean = str(item.get("comment") or "").strip()[:1200]
        op_id = str(item.get("operation_id") or "").strip() or None

        with FileLock(str(self.root / ".report_actions.lock")):
            existing = self._read_all_report_actions()
            next_seq = (int(existing[-1].get("seq", len(existing))) + 1) if existing else 1
            entry: dict[str, Any] = {
                "seq": next_seq,
                "id": f"act_{next_seq:04d}",
                "operation_id": op_id,
                "created_at": utc_now(),
                "case": c_clean,
                "recording": r_clean,
                "run_name": rn_clean,
                "source": str(item.get("source") or "case_sheet"),
                "action": act_kind,
                "anchor_type": str(item.get("anchor_type") or "general"),
                "section": str(item.get("section") or ""),
                "channel": str(item["channel"]) if item.get("channel") else None,
                "event_id": str(item["event_id"]) if item.get("event_id") else None,
                "quoted_text": str(item.get("quoted_text") or "")[:400],
                "verdict": verdict,
                "comment": comment_clean,
                "author": str(item.get("author") or "clinician").strip() or "clinician",
                "trigger_agent": bool(
                    item.get("trigger_agent")
                    or act_kind in {"ask_agent", "submit_batch", "exclude_request"}
                ),
            }
            if entry["source"] in NAVIGATION_QUESTION_SOURCES:
                # A native identifier can occur in a case slug. Keep a random, workspace-local
                # reply address stable across questions without sharing any part of that slug.
                previous_ref = next((row["case_ref"] for row in reversed(existing)
                                     if row.get("case") == c_clean and row.get("case_ref")), None)
                entry["case_ref"] = previous_ref or f"case:{uuid4().hex}"
                entry.update({k: item[k] for k in ("role", "recording_id", "start_seconds", "end_seconds", "marker")
                              if item.get(k) is not None})
            if op_id:
                for prev in reversed(existing):
                    if prev.get("operation_id") != op_id:
                        continue
                    if entry["source"] in NAVIGATION_QUESTION_SOURCES or prev.get("source") in NAVIGATION_QUESTION_SOURCES:
                        fields = (entry.keys() | prev.keys()) - {"id", "seq", "created_at"}
                        if any(prev.get(k) != entry.get(k) for k in fields):
                            raise ToolkitError("operation_conflict", "This operation_id already identifies a different question.")
                    return prev, False
            append_jsonl_line(self.report_actions_path(), entry)
            return entry, True

    def case_for_question_ref(self, ref: str) -> str:
        for row in self._read_all_report_actions():
            if row.get("case_ref") == ref and row.get("source") in NAVIGATION_QUESTION_SOURCES:
                return row["case"]
        raise ToolkitError("case_not_found", "This question's case reference is unknown in the current workspace.")

    def append_report_action(self, item: dict[str, Any]) -> dict[str, Any]:
        entry, _ = self.append_report_action_ex(item)
        return entry

    @staticmethod
    def _effective_action_status(
        st_meta: dict[str, Any],
        rseq: int,
        legacy_seq: int,
        now_ts: float,
    ) -> str:
        raw = st_meta.get("status")
        if raw == "resolved":
            return "resolved"
        if raw == "claimed":
            exp = st_meta.get("claim_expires_ts")
            if exp is not None:
                return "pending" if float(exp) <= now_ts else "claimed"
            claimed_at = st_meta.get("claimed_at")
            if isinstance(claimed_at, str) and claimed_at:
                try:
                    c_ts = datetime.fromisoformat(claimed_at.replace("Z", "+00:00")).timestamp()
                    return "pending" if c_ts + DEFAULT_CLAIM_LEASE_SECONDS <= now_ts else "claimed"
                except ValueError:
                    return "pending"
            return "pending"
        if raw == "pending":
            return "pending"
        return "resolved" if rseq <= legacy_seq else "pending"

    @classmethod
    def _contiguous_cursor(
        cls,
        rows: list[dict[str, Any]],
        act_states: dict[str, Any],
        legacy_seq: int,
        case_key: str,
        now_ts: float,
    ) -> int:
        """Return highest non-pending seq strictly before the earliest remaining pending item."""
        last_safe = 0
        for r in rows:
            if r.get("action") == "inspect_waveform":
                continue
            if case_key != "*" and r.get("case") != case_key:
                continue
            rseq = int(r.get("seq", 0))
            aid = str(r.get("id") or "")
            if cls._effective_action_status(act_states.get(aid) or {}, rseq, legacy_seq, now_ts) == "pending":
                break
            last_safe = max(last_safe, rseq)
        return last_safe

    def project_report_actions(
        self,
        rows: list[dict[str, Any]],
        state: dict[str, Any],
        *,
        case_slug: str | None = None,
        since_seq: int = 0,
        only_pending: bool = False,
        include_claimed: bool = True,
        trigger_only: bool = False,
        limit: int = 100,
        consumer: str = "default",
        now_ts: float | None = None,
    ) -> dict[str, Any]:
        """Pure projection of report actions + per-action lifecycle state with lease expiration."""
        c_clean = sanitize_slug(case_slug) if case_slug else None
        act_states = state.get("actions") or {}
        legacy_seq = int(state.get("legacy_consumed_seq", 0))
        latest_seq = int(rows[-1].get("seq", len(rows))) if rows else 0
        case_latest_seq = 0
        ts_now = time.time() if now_ts is None else float(now_ts)

        pending_total = claimed_total = resolved_total = 0
        filtered: list[dict[str, Any]] = []

        for r in rows:
            if r.get("action") == "inspect_waveform":
                continue
            if c_clean is not None and r.get("case") != c_clean:
                continue
            rseq = int(r.get("seq", 0))
            case_latest_seq = max(case_latest_seq, rseq)
            aid = str(r.get("id") or "")
            st_meta = act_states.get(aid) or {}
            status = self._effective_action_status(st_meta, rseq, legacy_seq, ts_now)

            if status == "pending":
                pending_total += 1
            elif status == "claimed":
                claimed_total += 1
            else:
                resolved_total += 1

            if rseq <= since_seq:
                continue
            if only_pending and (status == "resolved" or (status == "claimed" and not include_claimed)):
                continue
            if trigger_only and not r.get("trigger_agent"):
                continue

            filtered.append({
                **r,
                "status": status,
                "claimed_by": st_meta.get("claimed_by") if status == "claimed" else None,
                "claimed_at": st_meta.get("claimed_at") if status == "claimed" else None,
                "claim_expires_ts": st_meta.get("claim_expires_ts") if status == "claimed" else None,
                "resolved_by": st_meta.get("resolved_by"),
                "resolved_at": st_meta.get("resolved_at"),
                "outcome": st_meta.get("outcome"),
                "agent_reply": st_meta.get("agent_reply") or "",
            })

        if limit > 0 and len(filtered) > limit:
            filtered = filtered[:limit] if only_pending else filtered[-limit:]

        cursors = (state.get("consumer_cursors") or {}).get(consumer) or {}
        stored_cursor = int(cursors.get(c_clean or "*", 0))
        safe_watermark = self._contiguous_cursor(rows, act_states, legacy_seq, c_clean or "*", ts_now)
        return {
            "case": c_clean,
            "actions": filtered,
            "latest_seq": latest_seq,
            "case_latest_seq": case_latest_seq,
            "consumer_cursor": min(stored_cursor, safe_watermark),
            "pending_count": pending_total,
            "claimed_count": claimed_total,
            "resolved_count": resolved_total,
            "unaddressed_count": pending_total + claimed_total,
        }

    def load_report_actions(
        self,
        case_slug: str | None = None,
        since_seq: int = 0,
        only_pending: bool = False,
        include_claimed: bool = True,
        trigger_only: bool = False,
        limit: int = 100,
        consumer: str = "default",
    ) -> dict[str, Any]:
        return self.project_report_actions(
            self._read_all_report_actions(),
            self._load_action_state(),
            case_slug=case_slug,
            since_seq=since_seq,
            only_pending=only_pending,
            include_claimed=include_claimed,
            trigger_only=trigger_only,
            limit=limit,
            consumer=consumer,
        )

    def claim_report_actions(
        self,
        case_slug: str | None,
        action_ids: list[str],
        consumer: str = "agent",
        lease_seconds: float = DEFAULT_CLAIM_LEASE_SECONDS,
    ) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        c_clean = sanitize_slug(case_slug) if case_slug else None
        id_set = {str(x).strip() for x in action_ids if str(x).strip()}
        if not id_set:
            return {"ok": True, "claimed_ids": [], "already_claimed_ids": []}
        lease = max(5.0, float(lease_seconds))
        with FileLock(str(self.root / ".report_actions.lock")):
            rows = self._read_all_report_actions()
            state = self._load_action_state()
            act_states = state.setdefault("actions", {})
            legacy_seq = int(state.get("legacy_consumed_seq", 0))
            cursors = state.setdefault("consumer_cursors", {}).setdefault(consumer, {})
            claimed: list[str] = []
            already_claimed: list[str] = []
            touched_cases: set[str] = set()
            now, now_ts = utc_now(), time.time()
            for r in rows:
                aid = str(r.get("id") or "")
                if aid not in id_set:
                    continue
                rcase = r.get("case")
                if c_clean is not None and rcase != c_clean:
                    continue
                rseq = int(r.get("seq", 0))
                prev_meta = act_states.get(aid) or {}
                eff_st = self._effective_action_status(prev_meta, rseq, legacy_seq, now_ts)
                if eff_st == "resolved":
                    continue
                if eff_st == "claimed" and prev_meta.get("claimed_by") != consumer:
                    already_claimed.append(aid)
                    continue
                act_states[aid] = {
                    **prev_meta,
                    "status": "claimed",
                    "claimed_by": consumer,
                    "claimed_at": now,
                    "claim_expires_ts": now_ts + lease,
                }
                claimed.append(aid)
                touched_cases.add(str(rcase or "*"))
            for k in touched_cases | {"*"}:
                cursors[k] = self._contiguous_cursor(rows, act_states, legacy_seq, k, now_ts)
            self._save_action_state(state)
            proj = self.project_report_actions(
                rows, state, case_slug=c_clean, only_pending=True, consumer=consumer, now_ts=now_ts
            )
            return {"ok": True, "claimed_ids": claimed, "already_claimed_ids": already_claimed, **proj}

    def release_report_actions(
        self,
        *,
        case_slug: str,
        action_ids: list[str],
        consumer: str = "agent",
    ) -> dict[str, Any]:
        c_clean = sanitize_slug(str(case_slug))
        id_set = {str(x).strip() for x in (action_ids or []) if str(x).strip()}
        if not id_set:
            raise ToolkitError("missing_action_ids", "release_report_actions requires non-empty `action_ids`.")
        with FileLock(str(self.root / ".report_actions.lock")):
            rows = self._read_all_report_actions()
            state = self._load_action_state()
            act_states = state.setdefault("actions", {})
            legacy_seq = int(state.get("legacy_consumed_seq", 0))
            by_id = {str(r.get("id")): r for r in rows}
            released: list[str] = []
            now_ts = time.time()
            for aid in id_set:
                row = by_id.get(aid)
                if row is None or row.get("case") != c_clean:
                    continue
                prev = act_states.get(aid) or {}
                if prev.get("status") == "claimed" and (not consumer or prev.get("claimed_by") == consumer):
                    act_states[aid] = {
                        **prev,
                        "status": "pending",
                        "claimed_by": None,
                        "claimed_at": None,
                        "claim_expires_ts": None,
                    }
                    released.append(aid)
            cursors = state.setdefault("consumer_cursors", {}).setdefault(consumer, {})
            cursors[c_clean] = self._contiguous_cursor(rows, act_states, legacy_seq, c_clean, now_ts)
            self._save_action_state(state)
            proj = self.project_report_actions(
                rows, state, case_slug=c_clean, only_pending=True, consumer=consumer, now_ts=now_ts
            )
            return {"ok": True, "released_ids": released, **proj}

    def resolve_report_actions(
        self,
        *,
        case_slug: str,
        action_ids: list[str],
        agent_reply: str = "",
        outcome: str = "resolved",
        resolved_by: str = "agent",
    ) -> dict[str, Any]:
        if not case_slug or not str(case_slug).strip():
            raise ToolkitError(
                "missing_case",
                "resolve_report_actions requires an explicit `case` slug (cross-case bare ack is disallowed).",
            )
        id_list = [str(x).strip() for x in (action_ids or []) if str(x).strip()]
        if not id_list:
            raise ToolkitError("missing_action_ids", "resolve_report_actions requires explicit non-empty `action_ids`.")
        c_clean = sanitize_slug(str(case_slug))
        reply_clean = str(agent_reply or "").strip()[:1200]
        outcome_clean = str(outcome or "resolved").strip()[:60] or "resolved"
        self.root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.root / ".report_actions.lock")):
            rows = self._read_all_report_actions()
            state = self._load_action_state()
            act_states = state.setdefault("actions", {})
            legacy_seq = int(state.get("legacy_consumed_seq", 0))
            cursors = state.setdefault("consumer_cursors", {}).setdefault(resolved_by, {})
            by_id = {str(r.get("id")): r for r in rows}

            invalid_ids = [aid for aid in id_list if aid not in by_id or by_id[aid].get("case") != c_clean]
            if invalid_ids:
                raise ToolkitError(
                    "action_not_found",
                    f"Action ID(s) {invalid_ids} were not found in case {c_clean!r}.",
                    invalid_ids=invalid_ids,
                )

            resolved_ids: list[str] = []
            now, now_ts = utc_now(), time.time()
            for aid in id_list:
                prev_meta = act_states.get(aid) or {}
                final_reply = reply_clean if reply_clean else str(prev_meta.get("agent_reply") or "")
                act_states[aid] = {
                    **prev_meta,
                    "status": "resolved",
                    "resolved_by": resolved_by,
                    "resolved_at": now,
                    "outcome": outcome_clean,
                    "agent_reply": final_reply,
                    "claim_expires_ts": None,
                }
                resolved_ids.append(aid)
            cursors[c_clean] = self._contiguous_cursor(rows, act_states, legacy_seq, c_clean, now_ts)
            self._save_action_state(state)
            proj = self.project_report_actions(
                rows, state, case_slug=c_clean, only_pending=True, consumer=resolved_by, now_ts=now_ts
            )
            return {
                "ok": True,
                "case": c_clean,
                "resolved_ids": resolved_ids,
                "agent_reply": reply_clean,
                "pending_count": proj["pending_count"],
                "claimed_count": proj["claimed_count"],
                "latest_seq": proj["latest_seq"],
            }
