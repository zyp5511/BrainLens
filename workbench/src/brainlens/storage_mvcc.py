"""MVCC generation read-pinning, GC locks, and shared storage helper functions."""

from __future__ import annotations

import csv
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

from .models import canonical_digest

_SLUG_RE = re.compile(r"[^A-Za-z0-9._\-]+")
DEFAULT_CLAIM_LEASE_SECONDS = 180.0
_CLINICIAN_ROLES = frozenset({
    "clinician",
    "physician",
    "md",
    "doctor",
    "epileptologist",
    "neurologist",
    "neurosurgeon",
    "attending",
    "fellow",
})
_AGENT_ROLES = frozenset({"agent", "ai", "assistant", "bot"})
_REHEARSAL_ROLES = frozenset({"engineering_rehearsal", "rehearsal"})
_RESEARCHER_ROLES = frozenset({"researcher", "reviewer", "analyst", "scientist"})
_UI_STATE_KEYS = frozenset({
    "case",
    "recording",
    "run_name",
    "compare_run",
    "view",
    "channel",
    "shaft",
    "start_seconds",
    "window_seconds",
    "event_id",
    "queue_mode",
})


def _norm_interval(iv: Any) -> list[float] | None:
    if isinstance(iv, (list, tuple)) and len(iv) == 2:
        try:
            a, b = float(iv[0]), float(iv[1])
            if a > b:
                a, b = b, a
            if b > a:
                return [round(a, 5), round(b, 5)]
        except (TypeError, ValueError):
            return None
    return None


def classify_judgment_role(role: str | None) -> str:
    """Bucket judgment author role strictly into 'clinician', 'researcher', 'rehearsal', or 'agent' (fail-closed to 'agent')."""
    if role is None:
        return "clinician"  # Legacy pre-v2 WAL rows omitted role
    norm = str(role).strip().lower()
    if norm in _CLINICIAN_ROLES:
        return "clinician"
    if norm in _REHEARSAL_ROLES:
        return "rehearsal"
    if norm in _RESEARCHER_ROLES:
        return "researcher"
    return "agent"


def append_jsonl_line(path: Path, entry: dict[str, Any]) -> None:
    """Append a JSONL record durably, healing an unterminated trailing line left by a prior crash."""
    payload = (json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as fh:
        fh.seek(0, os.SEEK_END)
        if fh.tell() > 0:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                fh.write(b"\n")
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())


def _normalize_legacy_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Ensure pre-v2 summaries with unavailable ONNX stages never report 0 instead of null."""
    stages = (summary.get("classification") or {}).get("stages") or {}
    if not stages:
        return summary
    art_ok = (stages.get("artifact") or {}).get("status") == "complete"
    spike_ok = (stages.get("spike") or {}).get("status") == "complete"
    ehfo_ok = (stages.get("ehfo") or {}).get("status") == "complete"
    counts = summary.get("counts")
    if isinstance(counts, dict):
        if not art_ok and counts.get("non_artifact_events") is not None:
            counts["non_artifact_events"] = None
        if not spike_ok and counts.get("spike_hfo_events") is not None:
            counts["spike_hfo_events"] = None
        if not ehfo_ok and counts.get("ehfo_events") is not None:
            counts["ehfo_events"] = None
    for row in summary.get("channels") or []:
        if not isinstance(row, dict):
            continue
        if not art_ok:
            row["model_non_artifact_count"] = None
            row["model_non_artifact_rate_per_min"] = None
        if not spike_ok:
            row["spike_hfo_count"] = None
        if not ehfo_ok:
            row["ehfo_count"] = None
            row["ehfo_rate_per_min"] = None
    return summary


@contextmanager
def pin_generation_read(gdir: Path) -> Iterator[None]:
    """Acquire a shared OS-level advisory lock pinning an MVCC generation against concurrent GC."""
    if fcntl is None or gdir.parent.name != "generations":
        yield
        return
    pin_path = gdir.parent / f".{gdir.name}.pin"
    fd: int | None = None
    try:
        fd = os.open(str(pin_path), os.O_CREAT | os.O_RDONLY, 0o644)
        fcntl.flock(fd, fcntl.LOCK_SH)
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass


def try_lock_generation_for_gc(gdir: Path) -> int | None:
    """Try non-blocking exclusive lock on a generation's pin file; returns open fd if safe to delete, else None."""
    if fcntl is None:
        return -1
    pin_path = gdir.parent / f".{gdir.name}.pin"
    fd = os.open(str(pin_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except (BlockingIOError, OSError):
        os.close(fd)
        return None


def release_gc_lock(gdir: Path, fd: int | None) -> None:
    if fd is None or fd < 0:
        return
    pin_path = gdir.parent / f".{gdir.name}.pin"
    try:
        if not gdir.exists():
            pin_path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


def run_analysis_signature(meta_or_bundle: dict[str, Any]) -> str:
    """Stable digest of a run's recording snapshot + analysis configuration."""
    if "meta" in meta_or_bundle and isinstance(meta_or_bundle.get("meta"), dict):
        meta = meta_or_bundle["meta"]
        summ = meta_or_bundle.get("summary") or {}
    else:
        meta = meta_or_bundle
        summ = {}
    return canonical_digest({
        "recording_fingerprint": meta.get("recording_fingerprint") or summ.get("recording_fingerprint"),
        "config_fingerprint": meta.get("config_fingerprint"),
    })


def bundle_signature_entry(bundle: dict[str, Any]) -> dict[str, Any]:
    meta = bundle.get("meta") or {}
    summ = bundle.get("summary") or {}
    cnt = summ.get("counts") or {}
    return {
        "analysis_generation_id": meta.get("analysis_generation_id") or meta.get("generation_id"),
        "recording_fingerprint": meta.get("recording_fingerprint"),
        "config_fingerprint": meta.get("config_fingerprint"),
        "created_at": meta.get("created_at"),
        "total_raw_events": cnt.get("total_raw_events"),
        "spatial_digest": canonical_digest(summ.get("spatial")),
    }


def sanitize_slug(value: str, fallback: str = "item") -> str:
    cleaned = _SLUG_RE.sub("_", (value or "").strip()).strip("._-")
    if not cleaned or cleaned in {".", ".."}:
        cleaned = fallback
    return cleaned[:80]


def write_csv_file(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (
                    json.dumps(row[k], ensure_ascii=False)
                    if isinstance(row.get(k), (dict, list))
                    else ("" if row.get(k) is None else row.get(k, ""))
                )
                for k in fieldnames
            })
    return str(path)
