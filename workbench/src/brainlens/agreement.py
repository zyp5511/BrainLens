"""Independent reader agreement on one recording, run and optionally frozen event set."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import numpy as np
from filelock import FileLock

from .models import ToolkitError, atomic_write_json, utc_now
from .storage import WorkbenchStore, classify_judgment_role, run_analysis_signature, sanitize_slug

_CATEGORIES = ("confirm", "artifact", "uncertain")
_HUMAN = {"clinician", "researcher"}
_ROLES = _HUMAN | {"agent", "engineering_rehearsal"}
_Z95 = 1.959963984540054


def _undefined(reason: str) -> dict[str, Any]:
    return {"value": None, "ci95": None, "reason": reason}


def _interval(value: float, variance: float) -> dict[str, Any]:
    margin = _Z95 * math.sqrt(max(variance, 0.0))
    return {"value": value, "ci95": [max(-1.0, value - margin), min(1.0, value + margin)], "reason": None}


def _cohen(matrix: np.ndarray) -> dict[str, Any]:
    n = int(matrix.sum())
    if n < 2:
        return _undefined("Fewer than two shared events.")
    # Exact integer denominator detects a constant rating by both readers.
    rows, cols = matrix.sum(axis=1), matrix.sum(axis=0)
    chance = int(rows @ cols)
    if chance == n * n:
        return _undefined("All ratings are in one category.")
    observed = float(matrix.trace()) / n
    expected = chance / (n * n)
    value = (int(matrix.trace()) * n - chance) / (n * n - chance)
    # Influence on kappa of one event rated (i,j), including both estimated marginals.
    marginal_change = (cols[:, None] + rows[None, :]) / n - 2 * expected
    influence = (np.eye(len(rows)) - observed - (1 - value) * marginal_change) / (1 - expected)
    variance = float(np.sum(matrix * influence**2)) / (n * (n - 1))
    return _interval(value, variance)


def _pairwise(rows: list[dict[str, Any]], readers: list[str]) -> list[dict[str, Any]]:
    pairs = []
    for a, b in combinations(readers, 2):
        matrix = np.zeros((3, 3), dtype=np.int64)
        for row in rows:
            first, second = row["verdicts"][a], row["verdicts"][b]
            if first is not None and second is not None:
                matrix[_CATEGORIES.index(first), _CATEGORIES.index(second)] += 1
        n = int(matrix.sum())
        binary = np.array([[matrix[0, 0], matrix[0, 1:].sum()],
                           [matrix[1:, 0].sum(), matrix[1:, 1:].sum()]])
        pairs.append({
            "readers": [a, b],
            "n_shared": n,
            "percent_agreement": 100 * int(matrix.trace()) / n if n else None,
            "confusion_matrix": matrix.tolist(),
            "kappa": _cohen(matrix),
            "binary_kappa": _cohen(binary),
            "binary_confusion_matrix": binary.tolist(),
            "prevalence_index": abs(int(binary[0, 0] - binary[1, 1])) / n if n else None,
            "bias_index": abs(int(binary[0, 1] - binary[1, 0])) / n if n else None,
            "reason": None if n else "No shared events.",
        })
    return pairs


def _fleiss(rows: list[dict[str, Any]], readers: list[str]) -> dict[str, Any]:
    m = len(readers)
    complete = [list(row["verdicts"].values()) for row in rows if m and None not in row["verdicts"].values()]
    n = len(complete)
    if m < 3:
        return {"n_shared": n, **_undefined("Fleiss' kappa requires at least three distinct readers.")}
    if n < 2:
        return {"n_shared": n, **_undefined("Fewer than two events judged by every selected reader.")}
    counts = np.array([[verdicts.count(cat) for cat in _CATEGORIES] for verdicts in complete])
    if np.count_nonzero(counts.sum(axis=0)) == 1:
        return {"n_shared": n, **_undefined("All ratings are in one category.")}
    proportions = counts / m
    marginals = proportions.mean(axis=0)
    event_agreement = (counts * (counts - 1)).sum(axis=1) / (m * (m - 1))
    observed, expected = float(event_agreement.mean()), float(marginals @ marginals)
    value = (observed - expected) / (1 - expected)
    chance_change = 2 * (proportions @ marginals - expected)
    influence = (event_agreement - observed - (1 - value) * chance_change) / (1 - expected)
    return {"n_shared": n, **_interval(value, float(influence.var(ddof=1)) / n)}


def _select_readers(
    votes: dict[tuple[str, str], dict[str, str]], requested: list[str] | None,
) -> tuple[dict[str, tuple[str, str | None]], dict[str, list[str]]]:
    roles: dict[str, set[str]] = defaultdict(set)
    for author, role in votes:
        roles[author].add(role)
    excluded: dict[str, list[str]] = {}
    selected: dict[str, tuple[str, str | None]] = {}
    if requested is None:
        for author in sorted(roles):
            eligible = roles[author] & _HUMAN
            if len(eligible) == 1:
                selected[author] = (author, next(iter(eligible)))
            elif len(eligible) > 1:
                excluded[author] = sorted(eligible)
        return selected, excluded

    for raw in requested:
        label = raw.strip()
        if not label:
            raise ToolkitError("invalid_reader", "Reader names must be nonempty.")
        author = label
        role: str | None = None
        prefix, colon, suffix = label.partition(":")
        if label not in roles and colon and prefix in _ROLES and suffix.strip():
            author, role = suffix.strip(), prefix
        else:
            eligible = roles.get(author, set())
            # An unqualified name denotes its human identity when present.
            eligible = eligible & _HUMAN or eligible
            if len(eligible) > 1:
                raise ToolkitError("ambiguous_reader", f"Choose one role for {author!r} using role:author.",
                                   author=author, roles=sorted(eligible))
            if eligible:
                role = next(iter(eligible))
        if label not in selected and any(a == author for a, _ in selected.values()):
            raise ToolkitError("duplicate_reader", f"{author!r} cannot count as two independent readers.")
        selected[label] = (author, role)
    return selected, excluded


def agreement_from_snapshot(
    store: WorkbenchStore, case: str, recording: str, run_name: str,
    bundle: dict[str, Any], judgments: list[dict[str, Any]], binding_ctx: dict[str, Any],
    *, readers: list[str] | None = None, set_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the report entirely from supplied data, including for a signed case sheet."""
    meta = bundle["meta"]
    signature = meta.get("analysis_signature") or run_analysis_signature(bundle)
    fingerprint = meta.get("recording_fingerprint") or bundle["summary"].get("recording_fingerprint")
    active_fp = binding_ctx["rec_fps"].get(recording)
    if fingerprint and active_fp and fingerprint != active_fp:
        raise ToolkitError("run_recording_mismatch", "This run belongs to an earlier recording snapshot.")
    bound_signature = binding_ctx["run_asigs"].get((recording, run_name))
    if bound_signature and signature != bound_signature:
        raise ToolkitError("run_changed", "The analysis changed while reading the agreement snapshot.")

    events = {ev["id"]: ev for ev in bundle["events"]}
    ids = set_doc["event_ids"] if set_doc is not None else list(events)
    eligible_ids = set(ids)
    if not eligible_ids.issubset(events):
        raise ToolkitError("stale_review_set", "Review set events are missing from the active run.")

    votes: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for index, row in enumerate(judgments):
        verdict = row.get("verdict")
        author = str(row.get("author") or "").strip()
        eid = row.get("event_id")
        if not author or eid not in eligible_ids or verdict not in _CATEGORIES:
            continue
        if (not row.get("recording") or sanitize_slug(row["recording"]) != recording
                or sanitize_slug(row.get("run_name") or "baseline") != run_name):
            continue
        if not store.judgment_matches_active_state(row, index, binding_ctx):
            continue
        role = classify_judgment_role(row.get("role"))
        role = "engineering_rehearsal" if role == "rehearsal" else role
        votes[author, role][eid] = verdict

    selected, excluded = _select_readers(votes, readers)
    names = list(selected)
    matrix = []
    disagreements = []
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for eid in ids:
        ev = events[eid]
        verdicts = {name: votes.get(key, {}).get(eid) for name, key in selected.items()}
        entry = {
            "event_id": eid, "channel": ev.get("channel_name"), "time_seconds": ev.get("start_seconds"),
            "synchrony_class": ev.get("synchrony_class") or "unknown", "verdicts": verdicts,
        }
        matrix.append(entry)
        strata[entry["synchrony_class"]].append(entry)
        if len(set(verdicts.values()) - {None}) > 1:
            disagreements.append(entry)

    return {
        "case": case, "recording": recording, "run_name": run_name,
        "recording_fingerprint": fingerprint or active_fp, "analysis_signature": signature,
        "analysis_generation_id": meta.get("analysis_generation_id") or meta.get("generation_id"),
        "judgment_revision": len(judgments),
        "review_set": set_doc["name"] if set_doc is not None else None,
        "sampling": {k: v for k, v in set_doc.items() if k != "event_ids"} if set_doc is not None else {"sampling_rule": "all"},
        "categories": list(_CATEGORIES), "binary_categories": ["confirm", "not_confirm"],
        "readers": names, "reader_roles": {name: role for name, (_, role) in selected.items()},
        "excluded_ambiguous_readers": excluded,
        "reader_selection_note": "Specify role:author to select an identity listed in excluded_ambiguous_readers." if excluded else None,
        "n_events": len(ids), "pairwise": _pairwise(matrix, names),
        "pairwise_reason": "At least two distinct readers are required." if len(names) < 2 else None,
        "fleiss": _fleiss(matrix, names),
        "by_synchrony_class": {
            cls: {"n_events": len(group), "pairwise": _pairwise(group, names), "fleiss": _fleiss(group, names)}
            for cls, group in sorted(strata.items())
        },
        "ci_method": "95% event-level delta-method Wald interval, clipped to [-1, 1]. Events are treated as independent.",
        "index_definition": "Binary confirm vs all other verdicts. For cells a,b,c,d: prevalence = |a-d|/N, bias = |b-c|/N.",
        "scope": "Research use only. BrainLens is not a medical device. This report is not a standalone surgical resection boundary.",
        "disagreements": disagreements, "event_matrix": matrix,
    }


def reader_agreement(
    store: WorkbenchStore, case: str, recording: str, run_name: str = "baseline",
    readers: list[str] | None = None, review_set: str | None = None,
) -> dict[str, Any]:
    from .review_sets import load_review_set

    if not recording:
        raise ToolkitError("missing_recording", "Select a recording for reader agreement.")
    case, recording, run_name = map(sanitize_slug, (case, recording, run_name))
    cdir = store.case_dir(case)
    if not cdir.is_dir():
        raise ToolkitError("case_not_found", f"Case {case!r} does not exist.")
    # The existing write locks freeze replacement/re-analysis and WAL append for this short read.
    with FileLock(str(cdir / ".case.lock")), FileLock(str(cdir / ".judgments.lock")):
        rec_doc = store.load_recording_doc(case, recording)
        bundle = store.load_run(case, recording, run_name, reconcile=False)
        judgments = store.load_judgments(case)
    meta = bundle["meta"]
    binding_ctx = {
        "rec_fps": {recording: rec_doc["spec"]["fingerprint"]},
        "rec_wal_floors": {recording: rec_doc.get("replaced_at_wal_index", 0)},
        "run_asigs": {(recording, run_name): meta.get("analysis_signature") or run_analysis_signature(bundle)},
        "run_wal_floors": {(recording, run_name): meta.get("analysis_superseded_wal_index", 0)},
    }
    set_doc = load_review_set(store, case, recording, run_name, review_set, bundle=bundle) if review_set else None
    report = agreement_from_snapshot(store, case, recording, run_name, bundle, judgments, binding_ctx,
                                     readers=readers, set_doc=set_doc)
    report["recording_provenance"] = rec_doc["spec"].get("provenance")
    return report


def export_agreement_report(
    store: WorkbenchStore, case: str, recording: str, run_name: str = "baseline",
    readers: list[str] | None = None, review_set: str | None = None,
) -> dict[str, Any]:
    report = reader_agreement(store, case, recording, run_name, readers, review_set)
    report["exported_at"] = utc_now()
    exports = store.case_dir(report["case"]) / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    target = exports / f"agreement_{report['recording']}_{report['run_name']}_{uuid4().hex[:12]}"
    csv_fields = ["event_id", "channel", "time_seconds", "synchrony_class"]
    names = report["readers"]
    with TemporaryDirectory(prefix=".agreement-", dir=exports) as tmp:
        stage = Path(tmp)
        atomic_write_json(stage / "agreement.json", report)
        for key in ("disagreements", "event_matrix"):
            with (stage / f"{key}.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(csv_fields + [f"verdict:{name}" for name in names])
                for row in report[key]:
                    writer.writerow([row[k] for k in csv_fields] + [row["verdicts"][name] for name in names])
        stage.rename(target)
    return {
        "case": report["case"], "recording": report["recording"], "run_name": report["run_name"],
        "review_set": report["review_set"], "json_path": str(target / "agreement.json"),
        "disagreements_csv_path": str(target / "disagreements.csv"), "matrix_csv_path": str(target / "event_matrix.csv"),
    }
