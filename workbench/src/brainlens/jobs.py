"""Non-blocking background job manager with live stage progress for long SEEG runs & cohorts."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from typing import Any

from .models import ToolkitError, utc_now


class JobManager:
    """In-process background worker registry keeping the Clinical Workbench responsive."""

    def __init__(self, max_history: int = 50):
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._max_history = max_history

    def submit(
        self,
        kind: str,
        label: str,
        fn: Callable[[Callable[[str, float, str], None]], dict[str, Any]],
        on_success: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        jid = f"job_{uuid.uuid4().hex[:10]}"
        rec: dict[str, Any] = {
            "job_id": jid,
            "kind": kind,
            "label": label,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "detail": f"Queued {label}",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "finished_at": None,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[jid] = rec
            self._order.append(jid)
            while len(self._order) > self._max_history:
                old = self._order.pop(0)
                if self._jobs.get(old, {}).get("status") in {"complete", "failed"}:
                    self._jobs.pop(old, None)

        def _update(stage: str, pct: float, detail: str) -> None:
            with self._lock:
                j = self._jobs.get(jid)
                if j is None:
                    return
                j["status"] = "running"
                j["stage"] = stage
                j["progress"] = round(max(0.0, min(1.0, float(pct))), 4)
                j["detail"] = detail
                j["updated_at"] = utc_now()

        def _worker() -> None:
            _update("starting", 0.02, f"Starting {label}...")
            try:
                res = fn(_update)
                if on_success is not None:
                    try:
                        on_success(res)
                    except Exception:
                        pass
                with self._lock:
                    j = self._jobs[jid]
                    j["status"] = "complete"
                    j["stage"] = "complete"
                    j["progress"] = 1.0
                    j["detail"] = f"Completed {label}"
                    j["result"] = res
                    j["updated_at"] = utc_now()
                    j["finished_at"] = utc_now()
            except ToolkitError as exc:
                with self._lock:
                    j = self._jobs[jid]
                    j["status"] = "failed"
                    j["stage"] = "failed"
                    j["detail"] = exc.message
                    j["error"] = exc.to_dict()
                    j["updated_at"] = utc_now()
                    j["finished_at"] = utc_now()
            except Exception as exc:
                with self._lock:
                    j = self._jobs[jid]
                    j["status"] = "failed"
                    j["stage"] = "failed"
                    j["detail"] = str(exc)
                    j["error"] = {"code": "job_error", "message": str(exc)}
                    j["updated_at"] = utc_now()
                    j["finished_at"] = utc_now()

        t = threading.Thread(target=_worker, daemon=True, name=f"brainlens-{jid}")
        t.start()
        return self.get(jid)  # type: ignore[return-value]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            j = self._jobs.get(job_id)
            return dict(j) if j is not None else None

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(reversed(self._order[-limit:]))
            return [dict(self._jobs[i]) for i in ids if i in self._jobs]
