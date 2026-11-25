"""Frozen event samples and reproducible presentation order for one analysis."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from filelock import FileLock

from .models import ToolkitError, atomic_write_json, canonical_digest, read_json, utc_now
from .storage import WorkbenchStore, run_analysis_signature, sanitize_slug


def _directory(store: WorkbenchStore, case: str) -> Path:
    folder = store.case_dir(case) / "review_sets"
    if folder.is_symlink():
        raise ToolkitError("unsafe_review_set", "Review sets must be stored inside the case directory.")
    return folder


def _set_path(store: WorkbenchStore, case: str, name: str) -> Path:
    if not isinstance(name, str) or not name or name != sanitize_slug(name):
        raise ToolkitError("invalid_review_set_name", "Use a review set slug of up to 80 letters, digits, dots, underscores or hyphens.")
    path = _directory(store, case) / f"{name}.json"
    if path.is_symlink():
        raise ToolkitError("unsafe_review_set", "A review set must be a regular case file.")
    return path


def _sample(events: list[dict[str, Any]], size: int, seed: int) -> list[str]:
    # Sample without replacement, one event per nonempty stratum in each round.
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for event in events:
        sync = str(event.get("synchrony_class") or "unknown")
        band = str(event.get("detector_band") or event.get("dominant_band") or "unknown")
        groups[(sync, band)].append(event["id"])
    rng = random.Random(seed)
    strata = []
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        strata.append(ids)
    rng.shuffle(strata)
    active = deque(strata)
    selected = []
    while len(selected) < size:
        ids = active.popleft()
        selected.append(ids.pop())
        if ids:
            active.append(ids)
    return sorted(selected)


def _binding(store: WorkbenchStore, case: str, recording: str, bundle: dict[str, Any]) -> tuple[str, str]:
    meta = bundle["meta"]
    fingerprint = store.load_recording(case, recording).fingerprint
    run_fp = meta.get("recording_fingerprint") or bundle["summary"].get("recording_fingerprint")
    if run_fp and run_fp != fingerprint:
        raise ToolkitError("run_recording_mismatch", "The run belongs to an earlier recording snapshot.")
    return fingerprint, meta.get("analysis_signature") or run_analysis_signature(bundle)


def create_review_set(
    store: WorkbenchStore, case: str, recording: str, run_name: str, name: str,
    sample_size: int | None = None, seed: int | None = None,
) -> dict[str, Any]:
    case, recording, run_name = map(sanitize_slug, (case, recording, run_name))
    path = _set_path(store, case, name)
    bundle = store.load_run(case, recording, run_name, reconcile=False)
    fingerprint, signature = _binding(store, case, recording, bundle)
    events = bundle["events"]
    ids = [event["id"] for event in events]
    if len(set(ids)) != len(ids):
        raise ToolkitError("duplicate_event_id", "A frozen review set requires unique event IDs.")
    if sample_size is not None and (type(sample_size) is not int or not 1 <= sample_size <= len(ids)):
        raise ToolkitError("invalid_sample_size", f"Sample size must be a positive integer at most {len(ids)}.")
    if seed is None:
        seed = 0
    if type(seed) is not int:
        raise ToolkitError("invalid_seed", "The sampling seed must be an integer.")
    selected = sorted(ids) if sample_size is None else _sample(events, sample_size, seed)
    doc = {
        "name": name, "case": case, "recording": recording, "run_name": run_name,
        "recording_fingerprint": fingerprint, "analysis_signature": signature,
        "created_at": utc_now(), "seed": seed, "sample_size": sample_size,
        "sampling_rule": "all" if sample_size is None else "balanced_round_robin_by_synchrony_class_and_band",
        "population_size": len(ids), "n_events": len(selected), "event_ids": selected,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.parent / ".create.lock")):
        if path.exists():
            raise ToolkitError("review_set_exists", f"Review set {name!r} already exists. Choose a new name.")
        atomic_write_json(path, doc)
    return doc


def load_review_set(
    store: WorkbenchStore, case: str, recording: str, run_name: str, name: str, *,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case, recording, run_name = map(sanitize_slug, (case, recording, run_name))
    path = _set_path(store, case, name)
    if not path.is_file():
        raise ToolkitError("review_set_not_found", f"No review set named {name!r} in case {case!r}.")
    doc = read_json(path)
    if (doc["case"], doc["recording"], doc["run_name"]) != (case, recording, run_name):
        raise ToolkitError("review_set_mismatch", "The selected review set belongs to another recording or run.")
    if bundle is None:
        bundle = store.load_run(case, recording, run_name, reconcile=False)
    fingerprint, signature = _binding(store, case, recording, bundle)
    if (doc["recording_fingerprint"], doc["analysis_signature"]) != (fingerprint, signature):
        raise ToolkitError("stale_review_set", "The recording or analysis changed. Create a new review set.")
    if not set(doc["event_ids"]).issubset(event["id"] for event in bundle["events"]):
        raise ToolkitError("stale_review_set", "The current run no longer contains every frozen event ID.")
    return doc


def list_review_sets(store: WorkbenchStore, case: str, recording: str, run_name: str) -> list[dict[str, Any]]:
    case, recording, run_name = map(sanitize_slug, (case, recording, run_name))
    result = []
    for path in sorted(_directory(store, case).glob("*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        doc = read_json(path)
        if (doc["case"], doc["recording"], doc["run_name"]) == (case, recording, run_name):
            result.append({key: value for key, value in doc.items() if key != "event_ids"})
    return result


def reader_event_order(review_set: dict[str, Any], reader: str) -> list[str]:
    if not isinstance(reader, str) or not reader.strip():
        raise ToolkitError("missing_reader", "Provide a reader name for the review queue.")
    key = [review_set[k] for k in ("case", "recording", "run_name", "analysis_signature", "name", "seed")]
    order = list(review_set["event_ids"])
    random.Random(canonical_digest([*key, reader.strip()])).shuffle(order)
    return order
