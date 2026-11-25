"""Reader-specific views over the judgment WAL used by blinded HTTP requests."""

from __future__ import annotations

from typing import Any

from .models import ToolkitError
from .storage import WorkbenchStore, classify_judgment_role, run_analysis_signature, sanitize_slug
from .storage_judgments import _fold_judgment_into_verdict_state


def reader_projection(
    store: WorkbenchStore,
    case: str,
    recording: str,
    run_name: str,
    bundle: dict[str, Any],
    reader: str,
    role: str | None,
    *,
    blinded: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Replay only this reader's live rows, retaining the original case-wide WAL indices."""
    recording, run_name = sanitize_slug(recording), sanitize_slug(run_name)
    key = (recording, run_name)
    doc = store.load_recording_doc(case, recording)
    fp = (doc.get("spec") or {}).get("fingerprint")
    meta = bundle["meta"]
    run_fp = meta.get("recording_fingerprint") or bundle["summary"].get("recording_fingerprint")
    if run_fp and fp != run_fp:
        raise ToolkitError("run_recording_mismatch", "The run belongs to an earlier recording snapshot.")
    ctx = {
        "rec_fps": {recording: fp},
        "rec_wal_floors": {recording: doc.get("replaced_at_wal_index") or 0},
        "run_asigs": {key: meta.get("analysis_signature") or run_analysis_signature(bundle)},
        "run_wal_floors": {key: meta.get("analysis_superseded_wal_index") or 0},
    }
    selected_role = classify_judgment_role(role) if role else None
    own_rows = []
    by_event: dict[str, dict[str, Any]] = {}
    seen_roles: set[str] = set()
    for index, row in enumerate(store.load_judgments(case)):
        if (
            str(row.get("author") or "").strip() != reader
            or not row.get("recording")
            or sanitize_slug(row["recording"]) != recording
            or sanitize_slug(row.get("run_name") or "baseline") != run_name
        ):
            continue
        bucket = classify_judgment_role(row.get("role"))
        if (selected_role and selected_role != bucket) or (blinded and bucket == "agent"):
            continue
        if not store.judgment_matches_active_state(row, index, ctx):
            continue
        own_rows.append(row)
        eid = row.get("event_id")
        if eid:
            seen_roles.add(bucket)
            _fold_judgment_into_verdict_state(by_event.setdefault(eid, {"updated_at": None}), row)
    if not role and len(seen_roles) > 1:
        raise ToolkitError("ambiguous_reader_role", "Select a reader role for this author.")
    return own_rows, by_event


def without_shared_review(value: Any) -> Any:
    """Strip event annotations and aggregate role tallies, including nested ranking tables."""
    if isinstance(value, list):
        return [without_shared_review(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        k: without_shared_review(v) for k, v in value.items()
        if not k.startswith(("clinician_", "researcher_", "rehearsal_", "agent_", "verdict_", "clipped_", "reviewed_"))
        and k not in {"verdict", "clipped", "note", "notes", "role", "author"}
    }


def blind_event(event: dict[str, Any], own: dict[str, dict[str, Any]], *, marker: bool = False) -> dict[str, Any]:
    """Return the detector evidence plus only this reader's actual annotations."""
    out = without_shared_review(event)
    state = own.get(event["id"], {})
    for key, val in state.items():
        if key in {"role", "author", "updated_at", "channel"}:
            continue
        if key == "verdict":
            key = "verdict" if marker else "clinician_verdict"
        elif key == "note":
            key = "clinician_note"
        out[key] = val
    return out


def blind_signal_payload(
    store: WorkbenchStore, data: dict[str, Any], reader: str, role: str | None
) -> dict[str, Any]:
    """Redact triad and multichannel overlays without traversing waveform or spectral arrays."""
    own: dict[str, dict[str, Any]] = {}
    if data.get("analysis_signature"):
        case, recording, run = data["case"], data["recording"], data["run_name"]
        bundle = store.load_run(case, recording, run, reconcile=False)
        meta = bundle["meta"]
        gen = meta.get("analysis_generation_id") or meta.get("generation_id")
        if (meta.get("analysis_signature") or run_analysis_signature(bundle)) != data["analysis_signature"] or (
            data.get("analysis_generation_id") and gen != data["analysis_generation_id"]
        ):
            raise ToolkitError("stale_run_generation", "Analysis changed while loading the reader view. Reload the event.")
        _, own = reader_projection(store, case, recording, run, bundle, reader, role, blinded=True)
    if data.get("focus_event"):
        data["focus_event"] = blind_event(data["focus_event"], own)
    if "overlapping_events" in data:
        data["overlapping_events"] = [blind_event(event, own) for event in data["overlapping_events"]]
    if "events" in data:
        data["events"] = [blind_event(event, own, marker=True) for event in data["events"]]
    return data
