"""Multi-patient Cohort batch runner applying a reusable Group Protocol across cases."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .analytics import annotate_summary_anatomy
from .anatomy import load_anatomy_labels
from .io_readers import PROJECT_ROOT, discover_available_inputs, inspect_case, is_recording_file, resolve_case_target
from .models import ToolkitError, atomic_write_json, utc_now
from .pipeline import analyze_case
from .protocols import load_protocol_config
from .storage import WorkbenchStore, sanitize_slug, write_csv_file


def _resolve_cohort_targets(targets: list[str] | str | None, store: WorkbenchStore) -> list[str]:
    if targets is None or targets == "*" or targets == ["*"]:
        reg = [c["case"] for c in store.list_cases()]
        if reg:
            return reg
        return [it["target"] for it in discover_available_inputs()]

    raw_list = [targets] if isinstance(targets, str) else list(targets)
    out: list[str] = []
    for item in raw_list:
        tok = str(item).strip()
        if not tok:
            continue
        p = Path(tok).expanduser()
        if not p.exists():
            alt = PROJECT_ROOT / tok
            if alt.exists():
                p = alt
        if p.exists() and p.is_dir():
            # If directory contains subdirectories with recordings (e.g. inputs/), expand each child case
            direct_recs = [c for c in p.iterdir() if is_recording_file(c)]
            if direct_recs:
                out.append(str(p.resolve()))
            else:
                for child in sorted(p.iterdir()):
                    if child.name.startswith("."):
                        continue
                    if is_recording_file(child):
                        out.append(str(child.resolve()))
                    elif child.is_dir() and any(is_recording_file(x) for x in child.rglob("*")):
                        out.append(str(child.resolve()))
        else:
            out.append(tok)
    if not out:
        raise ToolkitError("empty_cohort", "No valid cases or recordings found for cohort execution.")
    return out


def run_cohort(
    targets: list[str] | str | None = None,
    protocol: str | None = None,
    cohort_name: str | None = None,
    run_name: str | None = None,
    overrides: dict[str, Any] | None = None,
    workspace_dir: str | Path | None = None,
    overwrite: bool = False,
    progress_cb: Callable[[str, float, str], None] | None = None,
) -> dict[str, Any]:
    """Run a saved Group Protocol (or baseline config) across a multi-patient cohort.

    Per-recording incompatibilities (e.g. bipolar re-referencing on an 'unknown'-reference recording)
    are recorded explicitly as `skipped_incompatible` instead of aborting the cohort.
    """
    store = WorkbenchStore(workspace_dir)
    if protocol:
        load_protocol_config(protocol, store)  # validate protocol exists before starting

    resolved = _resolve_cohort_targets(targets, store)
    rslug = sanitize_slug(run_name or (f"proto_{protocol}" if protocol else "baseline"))
    cslug = sanitize_slug(cohort_name or f"cohort_{protocol or rslug}")
    cdir = store.cohort_dir(cslug)
    cdir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    total = len(resolved)

    for idx, tgt in enumerate(resolved):
        if progress_cb:
            progress_cb("cohort", idx / max(1, total), f"Cohort ({idx + 1}/{total}): {tgt}")
        try:
            spec = resolve_case_target(tgt, store)
            if spec["kind"] == "registered_recording":
                case_name = spec["case"]
                recs = [spec["recording"]]
            else:
                def inspect_progress(stage: str, fraction: float, detail: str) -> None:
                    if progress_cb:
                        progress_cb(stage, (idx + 0.05 * fraction) / total, detail)

                insp = inspect_case(tgt, workspace_dir=store.root, persist=True,
                                    progress_cb=inspect_progress if progress_cb else None)
                case_name = insp["case"]
                recs = [r["recording"] for r in insp["recordings"]]
        except Exception as exc:
            summary_rows.append({
                "case": str(tgt),
                "recording": "",
                "run_name": rslug,
                "status": "failed_inspect",
                "origin_kind": "",
                "sfreq_hz": None,
                "acquired_reference": "",
                "effective_reference": "",
                "analyzed_seconds": None,
                "measured_channels": None,
                "focal_events": None,
                "widespread_burst_events": None,
                "widespread_burst_fraction": None,
                "fast_ripple_focal_events": None,
                "ehfo_events": None,
                "top_focal_channel": "",
                "top_focal_rate_per_min": None,
                "note": str(exc),
            })
            continue

        for r_i, rec in enumerate(recs):
            sub_target = f"{case_name}/{rec}"
            try:
                def _sub_cb(stage: str, frac: float, detail: str) -> None:
                    if progress_cb:
                        overall = (idx + (r_i + frac) / max(1, len(recs))) / max(1, total)
                        progress_cb(stage, round(overall, 4), f"[{case_name}/{rec}] {detail}")

                analyze_case(
                    target=sub_target,
                    workspace_dir=store.root,
                    run_name=rslug,
                    overrides=overrides,
                    protocol=protocol,
                    reason=f"Cohort '{cslug}' batch run ({protocol or 'default'})",
                    overwrite=overwrite,
                    progress_cb=_sub_cb,
                )
                bundle = store.load_run(case_name, rec, rslug)
                anatomy = load_anatomy_labels(store, case_name)
                rmodel = store.load_recording(case_name, rec)
                s = annotate_summary_anatomy(bundle["summary"], anatomy, recording=rmodel)
                cnt = s["counts"]
                lead0 = s["leading_focal_channels"][0] if s["leading_focal_channels"] else None
                measured_secs = [
                    r["detector_analyzed_seconds"]
                    for r in s["channels"]
                    if r["status"] == "measured" and r["detector_analyzed_seconds"] > 0
                ]
                med_sec = round(float(sorted(measured_secs)[len(measured_secs) // 2]), 2) if measured_secs else 0.0
                summary_rows.append({
                    "case": case_name,
                    "recording": rec,
                    "run_name": rslug,
                    "status": "complete",
                    "origin_kind": rmodel.provenance.origin_kind,
                    "sfreq_hz": s["selection"]["sfreq_hz"],
                    "acquired_reference": s["acquired_reference"],
                    "effective_reference": s.get("effective_reference_label") or s["effective_reference"]["mode"],
                    "analyzed_seconds": med_sec,
                    "measured_channels": cnt["measured_channels"],
                    "focal_events": cnt["non_burst_events"],
                    "widespread_burst_events": cnt["widespread_burst_events"],
                    "widespread_burst_fraction": cnt["widespread_burst_fraction"],
                    "fast_ripple_focal_events": cnt.get("fast_ripple_focal_events"),
                    "ehfo_events": cnt.get("ehfo_events"),
                    "top_focal_channel": lead0["channel_name"] if lead0 else "",
                    "top_focal_anatomy": ((lead0 or {}).get("anatomy") or {}).get("display_label"),
                    "anatomy_source": (anatomy or {}).get("source"),
                    "top_focal_rate_per_min": lead0["focal_rate_per_min"] if lead0 else None,
                    "note": s["warnings"][0] if s["warnings"] else "",
                })
                for row in s["channels"]:
                    if row["status"] == "measured" and ((row.get("focal_count") or 0) > 0 or (row.get("raw_count") or 0) > 0):
                        channel_rows.append({
                            "case": case_name,
                            "recording": rec,
                            "run_name": rslug,
                            "origin_kind": rmodel.provenance.origin_kind,
                            "channel_name": row["channel_name"],
                            "anatomy": (row["anatomy"] or {}).get("display_label"),
                            "anatomy_source": (anatomy or {}).get("source"),
                            "shaft": row["shaft"],
                            "focal_rank": row["focal_rank"],
                            "focal_count": row["focal_count"],
                            "focal_rate_per_min": row["focal_rate_per_min"],
                            "fast_ripple_count": row.get("fast_ripple_count"),
                            "raw_count": row["raw_count"],
                            "raw_rate_per_min": row["raw_rate_per_min"],
                            "ehfo_count": row.get("ehfo_count"),
                            "analyzed_seconds": row["detector_analyzed_seconds"],
                        })
            except ToolkitError as exc:
                rmodel = store.load_recording(case_name, rec)
                summary_rows.append({
                    "case": case_name,
                    "recording": rec,
                    "run_name": rslug,
                    "status": f"skipped_{exc.code}",
                    "origin_kind": rmodel.provenance.origin_kind,
                    "sfreq_hz": rmodel.sfreq,
                    "acquired_reference": rmodel.reference,
                    "effective_reference": "",
                    "analyzed_seconds": None,
                    "measured_channels": None,
                    "focal_events": None,
                    "widespread_burst_events": None,
                    "widespread_burst_fraction": None,
                    "fast_ripple_focal_events": None,
                    "ehfo_events": None,
                    "top_focal_channel": "",
                    "top_focal_rate_per_min": None,
                    "note": exc.message,
                })

    sum_fields = [
        "case",
        "recording",
        "run_name",
        "status",
        "origin_kind",
        "sfreq_hz",
        "acquired_reference",
        "effective_reference",
        "analyzed_seconds",
        "measured_channels",
        "focal_events",
        "widespread_burst_events",
        "widespread_burst_fraction",
        "fast_ripple_focal_events",
        "ehfo_events",
        "top_focal_channel",
        "top_focal_rate_per_min",
        "note",
        "top_focal_anatomy",
        "anatomy_source",
    ]
    ch_fields = [
        "case",
        "recording",
        "run_name",
        "origin_kind",
        "channel_name",
        "shaft",
        "focal_rank",
        "focal_count",
        "focal_rate_per_min",
        "fast_ripple_count",
        "raw_count",
        "raw_rate_per_min",
        "ehfo_count",
        "analyzed_seconds",
        "anatomy",
        "anatomy_source",
    ]
    sum_csv = write_csv_file(cdir / "cohort_summary.csv", summary_rows, sum_fields)
    ch_csv = write_csv_file(cdir / "cohort_channels.csv", channel_rows, ch_fields)

    doc = {
        "cohort_slug": cslug,
        "protocol": protocol,
        "run_name": rslug,
        "created_at": utc_now(),
        "total_recordings": len(summary_rows),
        "completed_recordings": sum(1 for r in summary_rows if r["status"] == "complete"),
        "skipped_recordings": sum(1 for r in summary_rows if r["status"] != "complete"),
        "summary_csv_path": sum_csv,
        "channels_csv_path": ch_csv,
        "recordings": summary_rows,
        "note": "Rates are reported per recording without pooling across separate recordings or mixing synthetic and real clinical datasets.",
    }
    atomic_write_json(cdir / "cohort_summary.json", doc)
    if progress_cb:
        progress_cb("complete", 1.0, f"Cohort '{cslug}' complete ({doc['completed_recordings']}/{doc['total_recordings']} recordings).")
    return doc


def list_cohorts(workspace_dir: str | Path | None = None) -> list[dict[str, Any]]:
    return WorkbenchStore(workspace_dir).list_cohorts()
