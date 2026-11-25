"""MDT-ready Presurgical Case Discussion Sheet HTML generator (focal-stratified, ONNX, 3D spatial, print-ready)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cached_property
import hashlib
import html
import json
import os
import secrets
from itertools import chain
from pathlib import Path
from typing import Any

from .analytics import annotate_summary_anatomy, compare_recordings, compare_runs, compute_overlay_concordance
from . import deid
from .anatomy import load_anatomy_labels, match_anatomy
from .clinical_display import model_score_labels
from .models import RecordingSpec, ToolkitError, canonical_digest, record_verified_sheet, utc_now
from .storage import WorkbenchStore, bundle_signature_entry, classify_judgment_role, run_analysis_signature, sanitize_slug
from .triad_views import render_evidence_card, render_spatial_figure

SHEET_DISPLAY_META = '<meta name="brainlens-sheet-layout" content="2" />'


def _e(val: Any) -> str:
    return html.escape("" if val is None else str(val))


def _anatomy_suffix(row: dict[str, Any]) -> str:
    label = (row.get("anatomy") or {}).get("display_label")
    return f" · {label}" if label else ""


def _shaft_badge(row: dict[str, Any]) -> str:
    return f" <span class='sh'>{_e(row['display_shaft'])}</span>" if row.get("display_shaft") else ""


def _fmt_rate_cell(count: Any, secs: Any, rate: Any) -> str:
    if rate is None or count is None or secs is None:
        return "<span class='muted'>unassessed</span>"
    seconds = float(secs)
    return f"{int(count)} focal in {int(seconds) if seconds.is_integer() else seconds} s ({float(rate):.2f}/min)"


def _fmt_delta_cell(delta: Any) -> str:
    if delta is None:
        return "<span class='muted'>N/A</span>"
    return f"<strong>{float(delta):+.2f}/min</strong>"


def reader_agreement_reports(
    store: WorkbenchStore, case: str, run_name: str, bundles: dict[str, dict[str, Any]],
    judgments: list[dict[str, Any]], binding_ctx: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return one report per recording with at least one shared human rating."""
    from .agreement import agreement_from_snapshot

    human_authors = {
        j.get("author") for j in judgments if j.get("verdict")
        and classify_judgment_role(j.get("role")) in {"clinician", "researcher"}
    }
    if len(human_authors) < 2:
        return []
    reports = []
    for recording, bundle in bundles.items():
        report = agreement_from_snapshot(store, case, recording, run_name, bundle, judgments, binding_ctx)
        if any(pair["n_shared"] for pair in report["pairwise"]):
            reports.append(report)
    return reports


def _reader_agreement_section(
    store: WorkbenchStore, case: str, run_name: str, bundles: dict[str, dict[str, Any]],
    judgments: list[dict[str, Any]], binding_ctx: dict[str, Any],
) -> str:
    rows = []
    for report in reader_agreement_reports(store, case, run_name, bundles, judgments, binding_ctx):
        recording = report["recording"]
        for pair in report["pairwise"]:
            if not pair["n_shared"]:
                continue
            kappa = pair["kappa"]
            val = kappa["value"]
            estimate = f"{val:.3f}" if val is not None else f"Unavailable ({_e(kappa['reason'])})"
            if kappa["ci95"] is not None:
                lo, hi = kappa["ci95"]
                estimate += f" ({lo:.3f} to {hi:.3f})"
            readers = " / ".join(
                f"{_e(r)} ({_e(report['reader_roles'][r])})" for r in pair["readers"]
            )
            rows.append(
                f"<tr><td>{_e(recording)}</td><td>{readers}</td><td>{pair['n_shared']}</td>"
                f"<td>{pair['percent_agreement']:.1f}%</td><td>{estimate}</td></tr>"
            )
    if not rows:
        return ""
    return (
        "<section><h2>Inter-rater agreement</h2><p>Latest ratings for events judged by both readers "
        "in the primary run. Ratings use confirm, artifact and uncertain.</p>"
        "<table><thead><tr><th>Recording</th><th>Readers</th><th>Shared events</th>"
        "<th>Agreement</th><th>Cohen’s κ (95% CI)</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def _evidence_map_section(evidence: dict[str, Any]) -> str:
    """Render the captured recording-level result with its original units and missing-data reasons."""
    columns = evidence["columns"]
    headers = "".join(f"<th>{_e(col['label'])}<br>{_e(col['unit'])}</th>" for col in columns)
    has_overlay = evidence["external_overlay_available"]
    tag_header = "<th>External tags</th>" if has_overlay else ""
    overlay_note = (f"External SOZ / resected annotations · {_e(evidence['external_overlay_source'])}"
                    if has_overlay else "No external SOZ / resected overlay is attached.")
    reasons: dict[str, int] = {}
    rows = []
    for row in evidence["rows"]:
        cells = []
        for col in columns:
            cell = row["biomarkers"][col["key"]]
            if cell["value"] is None:
                reason = cell.get("reason") or cell["state"]
                number = reasons.setdefault(reason, len(reasons) + 1)
                cells.append(f"<td class='muted'>{_e(cell['state'])}<sup>{number}</sup></td>")
            else:
                cells.append(f"<td>{cell['value']:.3g} <small>(P{cell['percentile']:.3g})</small></td>")
        tags = ", ".join(row["external_tags"]) or "—"
        tag_cell = f"<td>{_e(tags)}</td>" if has_overlay else ""
        count = row["convergence_count"]
        convergence = "—" if count is None else f"{count} of {row['available_count']}"
        rows.append(f"<tr><td>{_e(row['channel'])}<br><small>{_e(row['state'])}</small></td>"
                    f"{''.join(cells)}<td>{_e(convergence)}</td>{tag_cell}</tr>")
    notes = " · ".join(f"<sup>{num}</sup> {_e(reason)}" for reason, num in reasons.items())
    source = evidence.get("ei_source")
    source_note = (f"Ictal EI comes from recording <strong>{_e(source['recording_id'])}</strong> "
                   f"at onset {_e(source['onset_seconds'])} s. {_e(source['provenance']['summary'])}"
                   if source else "Ictal EI is unavailable.")
    return (
        f"<section><h2>Evidence map · {_e(evidence['recording_id'])} / {_e(evidence['run_name'])}</h2>"
        f"<p>HFO and IED source · {_e(evidence['provenance']['summary'])}<br>{source_note}<br>{overlay_note}</p>"
        "<p>Each cell shows the measured value and its within-recording percentile (P).</p>"
        f"<table><thead><tr><th>Channel</th>{headers}<th>Convergence count</th>"
        f"{tag_header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        f"<p class='sub'>{notes}</p><p>{_e(evidence['top_tier_rule'])} "
        "Convergence count is descriptive, not a validated predictor. "
        "IED candidates — require visual verification.</p>"
        f"<p class='sub'>{_e(evidence['scope'])}</p></section>"
    )


@dataclass
class CaseEvidence:
    """Inputs captured for a single MDT export. Signatures precede display annotations."""

    store: WorkbenchStore
    case_slug: str
    case_label: str
    recording_labels: dict[str, str]
    display_text: Callable[[str], str] | None
    primary_run: str
    comparison_run: str | None
    recs: list[str]
    bundles: dict[str, dict[str, Any]]
    rec_models: dict[str, RecordingSpec]
    cross_rec: dict[str, Any] | None
    branch_cmps: list[dict[str, Any]]
    cards: list[dict[str, Any]]
    spatial_fig: dict[str, Any] | None
    overlay_conc: dict[str, Any]
    overlay_doc: dict[str, Any] | None
    anatomy_labels: dict[str, Any] | None
    judgments: list[dict[str, Any]]
    binding_ctx: dict[str, Any]
    open_questions: list[str]
    evidence_maps: dict[str, dict[str, Any]]
    rendered_signature: dict[str, Any]

    @cached_property
    def review_roles(self) -> dict[str, str]:
        """Show optional roles if they have a verdict anywhere in the current case."""
        rated = {
            classify_judgment_role(row.get("role"))
            for idx, row in enumerate(self.judgments)
            if row.get("verdict") and self.store.judgment_matches_active_state(
                row, idx, self.binding_ctx, default_run=self.primary_run
            )
        }
        roles = {"clinician": "Clinician"}
        for key, title in (("researcher", "Researcher"), ("rehearsal", "Engineering rehearsal")):
            if key in rated or any(b["summary"]["counts"].get(f"{key}_reviewed") for b in self.bundles.values()):
                roles[key] = title
        roles["agent"] = "Agent"
        return roles


def _control_targets(
    recs: list[str], bundles: dict[str, dict[str, Any]], run_name: str, search_all: bool,
) -> Iterator[dict[str, Any]]:
    """Try the sheet's leading channel first, then other measured channels for slides."""
    for recording in recs if search_all else recs[:1]:
        summary = bundles[recording]["summary"]
        leaders = summary.get("leading_focal_channels") or summary.get("leading_raw_channels") or []
        first = leaders[0]["channel_name"] if leaders else None
        channels = [first] if first else []
        if search_all and summary.get("detector_status") == "complete":
            channels.extend(
                row["channel_name"] for row in summary["channels"]
                if row["status"] == "measured" and row["channel_name"] != first
            )
        for channel in channels:
            yield {
                "recording": recording, "run_name": run_name, "channel": channel,
                "event_id": "control", "center_seconds": None, "source_label": "Background control",
            }


def gather_case_evidence(
    case_slug: str,
    workspace_dir: str | Path | None = None,
    primary_run: str = "baseline",
    comparison_run: str | None = None,
    clinical_question: str | None = None,
    max_cards: int = 4,
    *,
    deidentify: bool = False,
    pseudonym: str | None = None,
    reserve_control: bool = False,
    first_available_spatial: bool = False,
) -> CaseEvidence:
    """Gather the same evidence for slide and HTML exports, preserving per-run bindings."""
    store = WorkbenchStore(workspace_dir)
    cslug = sanitize_slug(case_slug)
    cdoc = store.load_case(cslug)
    all_case_recs = sorted(store.list_recordings(cslug))
    if deidentify:
        case_label, recording_labels = deid.share_labels(cslug, all_case_recs, pseudonym)
    else:
        case_label, recording_labels = cslug, {r: r for r in all_case_recs}
    recs = [r for r in all_case_recs if store.has_run(cslug, r, primary_run)]
    if not recs:
        raise ToolkitError("no_runs", f"Case {cslug!r} has no saved {primary_run!r} runs.")

    if comparison_run is None:
        comparison_run = store.resolve_default_comparison_run(cslug, primary_run)

    # Atomically register any new clinical_question BEFORE snapshotting so rendered_signature binds the exact list!
    if clinical_question and clinical_question.strip():
        open_questions = store.add_open_question(cslug, clinical_question.strip(), prepend=True)
    else:
        open_questions = list(cdoc.get("open_questions", []))

    # Load registered recordings & unified binding context FIRST so clips, notes, & run validations bind to current source snapshots & analyses!
    rec_models_all: dict[str, RecordingSpec] = {}
    for r in all_case_recs:
        try:
            rec_models_all[r] = store.load_recording(cslug, r)
        except Exception:
            continue
    binding_ctx = store.case_judgment_binding_context(cslug)

    # Snapshot judgments, per-role active clips, & clinical overlay FIRST so EVERY rendered dependency
    # (primary, comparison, 3rd-branch clipped runs, clinical overlay, clips, open questions, judgments)
    # is bound into rendered_signature!
    p_slug = sanitize_slug(primary_run)
    cmp_slug = sanitize_slug(comparison_run) if comparison_run else None
    judgments = store.load_judgments(cslug)
    active_clips = store.active_clipped_targets(cslug, default_run=p_slug, judgments=judgments, binding_ctx=binding_ctx)
    overlay_doc = store.load_clinical_overlay(cslug)
    anatomy_labels = load_anatomy_labels(store, cslug)

    # Verify recording fingerprints, derived-run stage artifacts, AND auto-heal lagged spatial summaries!
    from .triad_views import _load_run_channel_pool

    for r in recs:
        rmodel = rec_models_all.get(r)
        if rmodel is None:
            continue
        cur_sp_d = canonical_digest(rmodel.spatial.model_dump(mode="json") if rmodel.spatial is not None else None)
        for rn_check in ([primary_run] + ([comparison_run] if comparison_run else [])):
            if not store.has_run(cslug, r, rn_check):
                continue
            b_chk = store.load_run(cslug, r, rn_check, reconcile=False)
            m_chk = b_chk["meta"]
            if m_chk.get("recording_fingerprint") and m_chk.get("recording_fingerprint") != rmodel.fingerprint:
                raise ToolkitError(
                    "run_recording_mismatch",
                    f"Run {rn_check!r} for {cslug}/{r} was computed on an earlier version of the recording; "
                    "re-run analysis on the updated recording before exporting an MDT Case Discussion Sheet.",
                    recording=r,
                    run_name=rn_check,
                )
            # Fail loudly if a derived/re-referenced run is missing prepared.json (NEVER publish a 0-card fresh sheet!)
            _load_run_channel_pool(rmodel, b_chk, store)
            if (
                m_chk.get("recording_fingerprint") == rmodel.fingerprint
                and m_chk.get("recording_spatial_digest") != cur_sp_d
            ):
                from .pipeline import refresh_verdict_tallies

                try:
                    refresh_verdict_tallies(store, cslug, r, rn_check)
                except Exception as exc:
                    raise ToolkitError(
                        "incomplete_run_artifacts",
                        f"Cannot refresh outdated electrode coordinates for {cslug}/{r}:{rn_check} because required run stage artifacts are missing ({exc}); "
                        "restore the complete workspace or re-run analysis before exporting an MDT Case Discussion Sheet.",
                        recording=r,
                        run_name=rn_check,
                    ) from exc

    bundles = {r: store.load_run(cslug, r, primary_run) for r in recs}
    cmp_bundles: dict[str, dict[str, Any]] = {}
    if comparison_run:
        for r in all_case_recs:
            if store.has_run(cslug, r, comparison_run):
                b_cmp = store.load_run(cslug, r, comparison_run)
                rm_cmp = rec_models_all.get(r)
                if rm_cmp is not None:
                    m_cmp = b_cmp["meta"]
                    if m_cmp.get("recording_fingerprint") and m_cmp.get("recording_fingerprint") != rm_cmp.fingerprint:
                        raise ToolkitError(
                            "run_recording_mismatch",
                            f"Comparison run {comparison_run!r} for {cslug}/{r} belongs to an earlier recording snapshot; "
                            "cannot compare different source recordings in an MDT Case Discussion Sheet.",
                            recording=r,
                            run_name=comparison_run,
                        )
                    _load_run_channel_pool(rm_cmp, b_cmp, store)
                cmp_bundles[r] = b_cmp

    ref_run_names = sorted(
        {
            sanitize_slug(str(c["run_name"]))
            for c in active_clips
            if c.get("run_name")
        }
        - {p_slug, *([cmp_slug] if cmp_slug else [])}
    )
    ref_bundles: dict[str, dict[str, Any]] = {}
    for rn in ref_run_names:
        rb: dict[str, Any] = {}
        for r in all_case_recs:
            if store.has_run(cslug, r, rn):
                b_ref = store.load_run(cslug, r, rn)
                rm_ref = rec_models_all.get(r)
                if rm_ref is not None:
                    m_ref = b_ref["meta"]
                    if m_ref.get("recording_fingerprint") and m_ref.get("recording_fingerprint") != rm_ref.fingerprint:
                        raise ToolkitError(
                            "run_recording_mismatch",
                            f"Clipped reference run {rn!r} for {cslug}/{r} belongs to an earlier recording snapshot.",
                            recording=r,
                            run_name=rn,
                        )
                    _load_run_channel_pool(rm_ref, b_ref, store)
                rb[r] = b_ref
        ref_bundles[rn] = rb

    rec_models = {r: store.load_recording(cslug, r) for r in recs}
    redact = (
        deid.recording_redactor(
            list(rec_models_all.values()), aliases={cslug: case_label, **recording_labels}, pseudonym=pseudonym,
        )
        if deidentify else None
    )
    rendered_signature: dict[str, Any] = {
        "primary_run": p_slug,
        "primary": {r: bundle_signature_entry(bundles[r]) for r in sorted(bundles)},
        "comparison_run": cmp_slug,
        "comparison": {r: bundle_signature_entry(cmp_bundles[r]) for r in sorted(cmp_bundles)},
        "referenced_runs": {
            rn: {r: bundle_signature_entry(b) for r, b in sorted(rb.items())}
            for rn, rb in sorted(ref_bundles.items())
        },
        "clinical_overlay_digest": canonical_digest(overlay_doc),
        "anatomy_labels_digest": canonical_digest(anatomy_labels),
        "clip_signature": store.case_clip_signature(cslug, default_run=p_slug, judgments=judgments, binding_ctx=binding_ctx),
        "open_questions_digest": canonical_digest(open_questions),
        "clinical_notes_digest": store.case_clinical_notes_digest(cslug, judgments=judgments, binding_ctx=binding_ctx),
        "recording_spatial_digest": {
            r: canonical_digest(rec_models[r].spatial.model_dump(mode="json") if rec_models[r].spatial is not None else None)
            for r in sorted(rec_models)
        },
        "recording_fingerprints": {
            r: str(rec_models[r].fingerprint or "")
            for r in sorted(rec_models)
        },
        "judgment_revision": len(judgments),
    }

    from .evidence_map import get_evidence_map

    maps = {}
    for r, bundle in bundles.items():
        evidence = get_evidence_map(store, cslug, recording=r, run_name=p_slug, cached_only=True)
        if evidence is None:
            continue
        expected_gen = rendered_signature["primary"][r]["analysis_generation_id"]
        source = evidence.get("ei_source")
        if (evidence["analysis_generation_id"] != expected_gen
                or evidence["analysis_signature"] != run_analysis_signature(bundle)
                or evidence["clinical_overlay_digest"] != rendered_signature["clinical_overlay_digest"]
                or (source and source["recording_fingerprint"] != binding_ctx["rec_fps"].get(source["recording_id"]))):
            raise ToolkitError("evidence_map_changed", "Evidence changed during case sheet export. Retry the export.")
        maps[r] = evidence
    if maps:
        rendered_signature["evidence_map_digests"] = {r: doc["evidence_digest"] for r, doc in maps.items()}

    cross_rec = compare_recordings(
        store, cslug, run_name=primary_run, top_k=12, bundles=bundles, anatomy_labels=anatomy_labels,
    ) if len(recs) > 1 else None

    branch_cmps: list[dict[str, Any]] = []
    if comparison_run:
        for r in recs:
            if r in cmp_bundles:
                branch_cmps.append(
                    compare_runs(
                        store,
                        cslug,
                        r,
                        primary_run,
                        comparison_run,
                        top_k=12,
                        bundle_a=bundles[r],
                        bundle_b=cmp_bundles[r],
                        anatomy_labels=anatomy_labels,
                    )
                )

    overlay_conc = compute_overlay_concordance(
        store,
        cslug,
        rec_slug=recs[0],
        run_name=primary_run,
        overlay=overlay_doc,
        bundle=bundles[recs[0]],
    )

    # Curate deduplicated evidence cards:
    # 1) Currently clipped clinician/researcher/rehearsal/agent evidence first (per-role clip independence!)
    # 2) Top focal example per recording
    # 3) Widespread burst contrast (if recording had widespread bursts)
    # 4) Candidate-free background control on the leading channel
    requested_cards: list[dict[str, Any]] = list(active_clips)

    for r in recs:
        s = bundles[r]["summary"]
        lead_f = s.get("leading_focal_channels") or s.get("leading_raw_channels") or []
        if lead_f:
            requested_cards.append({
                "recording": r,
                "run_name": primary_run,
                "channel": lead_f[0]["channel_name"],
                "event_id": "top_focal",
                "center_seconds": None,
                "source_label": "Focal candidate",
            })

    first_s = bundles[recs[0]]["summary"]
    first_lead = first_s.get("leading_focal_channels") or first_s.get("leading_raw_channels") or []
    if first_s["counts"].get("widespread_burst_events", 0) > 0 and first_lead:
        requested_cards.append({
            "recording": recs[0],
            "run_name": primary_run,
            "channel": first_lead[0]["channel_name"],
            "event_id": "top_burst",
            "center_seconds": None,
            "source_label": "Synchronous burst",
        })

    def render_card(spec: dict[str, Any]) -> tuple[tuple[str, str, str, str], dict[str, Any]] | None:
        try:
            iv = spec.get("interval_seconds")
            req_win = 0.6
            req_center = spec.get("center_seconds")
            norm_iv: list[float] | None = None
            if not spec.get("event_id") and isinstance(iv, (list, tuple)) and len(iv) == 2:
                t0, t1 = float(iv[0]), float(iv[1])
                if t0 > t1:
                    t0, t1 = t1, t0
                if t1 > t0:
                    norm_iv = [t0, t1]
                    req_win = max(0.4, round(t1 - t0, 5))
                    req_center = 0.5 * (t0 + t1)
            card = render_evidence_card(
                target=f"{cslug}/{spec['recording']}",
                workspace_dir=store.root,
                recording=spec["recording"],
                run_name=spec["run_name"],
                channel=spec["channel"],
                event_id=spec["event_id"],
                center_seconds=req_center,
                window_seconds=req_win,
                return_base64=True,
                highlight_interval=norm_iv,
                display_text=redact,
                display_names=(case_label, recording_labels[spec["recording"]]) if deidentify else None,
            )
            if (card.get("focus_event") or {}).get("id"):
                ev_id_key = str(card["focus_event"]["id"])
            elif norm_iv is not None:
                ev_id_key = f"iv:{round(norm_iv[0], 4)}:{round(norm_iv[1], 4)}"
            else:
                w0, w1 = float(card["window_seconds"][0]), float(card["window_seconds"][1])
                ev_id_key = f"win:{round(w0, 3)}:{round(w1, 3)}"
            sig_key = (card["recording"], str(spec["run_name"]), card["channel"], ev_id_key)
            card["source_label"] = spec["source_label"]
            card["run_name"] = spec["run_name"]
            card["interval_seconds"] = norm_iv
            card_run = sanitize_slug(spec["run_name"])
            card_bundles = bundles if card_run == p_slug else (
                cmp_bundles if card_run == cmp_slug else ref_bundles.get(card_run, {})
            )
            card_rows = card_bundles.get(card["recording"], {}).get("summary", {}).get("channels", [])
            card_row = next((r for r in card_rows if r["channel_name"] == card["channel"]), {})
            card["anatomy"] = match_anatomy(anatomy_labels, card["channel"], card_row.get("contacts") or [])
            return sig_key, card
        except ToolkitError as exc:
            if exc.code in {
                "incomplete_run_artifacts",
                "run_recording_mismatch",
                "source_recording_missing",
                "source_recording_modified",
                "non_voltage_channel",
                "channel_not_in_run",
            }:
                raise
            return None
        except Exception:
            return None

    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    remaining = iter(requested_cards if max_cards > 0 else [])
    for spec in remaining:
        result = render_card(spec)
        if result is not None:
            key, card = result
            selected.setdefault(key, card)
        if len(selected) >= max_cards:
            break

    need_control = reserve_control and not any(c["view_mode"] == "background_control" for c in selected.values())
    if max_cards > 0 and (need_control or (not reserve_control and len(selected) < max_cards)):
        # An explicitly clipped control (including one on a comparison run) takes priority.
        deferred_controls = (spec for spec in remaining if spec.get("event_id") == "control")
        for spec in chain(deferred_controls, _control_targets(recs, bundles, primary_run, search_all=reserve_control)):
            result = render_card(spec)
            if result is None:
                continue
            key, control = result
            if key in selected:
                if need_control:
                    # Preserve the clip attribution if the same window also qualifies as a control.
                    control["source_label"] = selected[key]["source_label"]
                    selected[key] = control
            else:
                if len(selected) == max_cards:
                    selected.popitem()
                selected[key] = control
            break

    spatial_fig: dict[str, Any] | None = None
    for recording in recs if first_available_spatial else recs[:1]:
        if not (bundles[recording]["summary"].get("spatial") or {}).get("available"):
            continue
        try:
            spatial_fig = render_spatial_figure(
                store,
                cslug,
                recording,
                run_name=primary_run,
                return_base64=True,
                overlay=overlay_doc,
                bundle=bundles[recording],
                display_text=redact,
                display_names=(case_label, recording_labels[recording]) if deidentify else None,
            )
            break
        except Exception:
            continue

    return CaseEvidence(
        store=store,
        case_slug=cslug,
        case_label=case_label,
        recording_labels=recording_labels,
        display_text=redact,
        primary_run=primary_run,
        comparison_run=comparison_run,
        recs=recs,
        bundles={
            r: {**b, "summary": annotate_summary_anatomy(b["summary"], anatomy_labels, recording=rec_models[r])}
            for r, b in bundles.items()
        },
        rec_models={**rec_models_all, **rec_models},
        cross_rec=cross_rec,
        branch_cmps=branch_cmps,
        cards=list(selected.values()),
        spatial_fig=spatial_fig,
        overlay_conc=overlay_conc,
        overlay_doc=overlay_doc,
        anatomy_labels=anatomy_labels,
        judgments=judgments,
        binding_ctx=binding_ctx,
        open_questions=open_questions,
        evidence_maps=maps,
        rendered_signature=rendered_signature,
    )


def export_case_sheet(
    case_slug: str,
    workspace_dir: str | Path | None = None,
    primary_run: str = "baseline",
    comparison_run: str | None = None,
    clinical_question: str | None = None,
    max_cards: int = 4,
    *,
    deidentify: bool = False,
    pseudonym: str | None = None,
) -> dict[str, Any]:
    evidence = gather_case_evidence(
        case_slug, workspace_dir, primary_run, comparison_run, clinical_question, max_cards,
        deidentify=deidentify, pseudonym=pseudonym,
    )
    store = evidence.store
    cslug = evidence.case_slug
    case_label = evidence.case_label
    redact = evidence.display_text
    comparison_run = evidence.comparison_run
    gen_ts = utc_now()
    export_hash = hashlib.sha256(
        f"{store.workspace_id}:{cslug}:{primary_run}:{gen_ts}:{secrets.token_hex(4)}".encode("utf-8")
    ).hexdigest()[:16]

    html_str = _render_sheet_html(
        store=store, case_slug=cslug, case_label=case_label, recording_labels=evidence.recording_labels,
        display_text=redact, primary_run=primary_run, workspace_id=store.workspace_id, export_hash=export_hash,
        recs=evidence.recs, bundles=evidence.bundles,
        rec_models={r: evidence.rec_models[r] for r in evidence.recs},
        cross_rec=evidence.cross_rec, branch_cmps=evidence.branch_cmps, cards=evidence.cards,
        spatial_fig=evidence.spatial_fig, overlay_conc=evidence.overlay_conc, anatomy_labels=evidence.anatomy_labels,
        judgments=evidence.judgments, binding_ctx=evidence.binding_ctx, open_questions=evidence.open_questions,
        evidence_maps=evidence.evidence_maps, review_roles=evidence.review_roles,
    )

    exp_dir = store.case_dir(cslug) / "exports"
    exp_dir.mkdir(parents=True, exist_ok=True)
    run_slug = sanitize_slug(redact(primary_run) if redact else primary_run)
    ending = "_case_sheet.shared.html" if deidentify else "_case_sheet.html"
    out_path = exp_dir / f"{case_label}_{run_slug}{ending}"
    html_bytes = html_str.encode("utf-8")
    tmp_path = exp_dir / f".{out_path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    tmp_path.write_bytes(html_bytes)
    tmp_path.replace(out_path)

    # Record out-of-workspace HMAC-SHA256 trust entry bound strictly to the EXACT `rendered_signature`
    # loaded at the start of export_case_sheet (NEVER re-read disk after rendering -> zero TOCTOU race!).
    record_verified_sheet(
        store.workspace_id,
        cslug,
        out_path.name,
        html_bytes,
        primary_run,
        run_signature=evidence.rendered_signature,
    )

    return {
        "case": case_label,
        "deidentified": deidentify,
        "report_path": str(out_path),
        "export_filename": out_path.name,
        "export_hash": export_hash,
        "primary_run": redact(primary_run) if redact else primary_run,
        "comparison_run": redact(comparison_run) if redact and comparison_run else comparison_run,
        "embedded_cards": len(evidence.cards),
        "has_spatial_figure": evidence.spatial_fig is not None,
        "open_questions": [redact(q) for q in evidence.open_questions] if redact else evidence.open_questions,
    }


def _render_sheet_html(
    *,
    store: WorkbenchStore,
    case_slug: str,
    case_label: str,
    recording_labels: dict[str, str],
    display_text: Callable[[str], str] | None,
    primary_run: str,
    workspace_id: str,
    export_hash: str,
    recs: list[str],
    bundles: dict[str, dict[str, Any]],
    rec_models: dict[str, Any],
    cross_rec: dict[str, Any] | None,
    branch_cmps: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    spatial_fig: dict[str, Any] | None,
    overlay_conc: dict[str, Any],
    anatomy_labels: dict[str, Any] | None,
    judgments: list[dict[str, Any]],
    binding_ctx: dict[str, Any],
    open_questions: list[str],
    review_roles: dict[str, str],
    evidence_maps: dict[str, dict[str, Any]] | None = None,
) -> str:
    def _e(val: Any) -> str:
        text = "" if val is None else str(val)
        return html.escape(display_text(text) if display_text else text)

    def recording_label(name: str | None) -> str:
        if not name:
            return ""
        return recording_labels.get(name, "Recording withheld" if display_text else name)

    def _rec(name: str) -> str:
        return html.escape(recording_label(name))

    def _shared(fragment: str) -> str:
        # Sections rendered by other modules (evidence map, reader agreement) print recording ids and
        # provenance text directly; a shared sheet must pass them through the same disclosure rules.
        return display_text(fragment) if display_text else fragment

    prov0 = rec_models[recs[0]].provenance
    badge_cls = "badge-warn" if prov0.is_synthetic else "badge-info"
    badge_txt = "SYNTHETIC DEMO DATA (CC0)" if prov0.is_synthetic else _e(prov0.summary)
    anatomy_provenance = ""
    if anatomy_labels:
        anatomy_provenance = (
            f"<p class='sub'>Anatomy labels: external, source={_e(anatomy_labels['source'])}, "
            f"patient-native={'yes' if anatomy_labels['patient_native'] else 'no'}, "
            f"atlas={_e(anatomy_labels.get('atlas') or 'unspecified')}</p>"
        )

    q_block = ""
    if open_questions:
        lis = "".join(
            f"<li class='q-item' data-anchor-type='question' data-section='Clinical Discussion Questions' data-quoted='{_e(q)}'>"
            f"<span>{_e(q)}</span> <span class='pill'>OPEN</span>"
            f"<button type='button' class='row-chip live-ui' data-act='ask_anchor' title='Comment or ask Agent about this question'>💬 Ask</button>"
            f"</li>"
            for q in open_questions
        )
        q_block = f"<div class='qbox'><strong>Clinical Discussion Question(s):</strong><ul>{lis}</ul></div>"

    # KPI Cards separating Focal vs Widespread Synchrony, Reference Derivation, ONNX & Review Coverage
    kpis = ""
    for r in recs:
        s = bundles[r]["summary"]
        cnt = s["counts"]
        participant = deid.trusted_participant_label(rec_models[r]) if display_text else None
        participant_tag = f" · Participant <code>{html.escape(participant)}</code>" if participant else ""
        eff_ref = s.get("effective_reference_label") or s["effective_reference"]["mode"]
        ref_str = (
            f"Acquired: <code>{_e(s['acquired_reference'])}</code> &rarr; Analyzed: <code>{_e(eff_ref)}</code>"
            if s["effective_reference"]["mode"] != "original"
            else f"Ref: <code>{_e(eff_ref)}</code>"
        )
        state_str = f" · State: <code>{_e(s['selection']['state'])}</code>" if s["selection"].get("state") else ""
        if segments := s["selection"].get("segments"):
            state_str += " · Analyzed segments: " + ", ".join(
                f"{'Segment ' + str(i) if display_text else _e(seg['label'])} "
                f"({seg['start_seconds']:g}–{seg['end_seconds']:g} s)" for i, seg in enumerate(segments, 1)
            )
        focal_quiet = cnt.get("focal_quiet_measured_channels", cnt["measured_channels"] - cnt["active_focal_channels"])
        raw_quiet = cnt["quiet_measured_channels"]
        quiet_lbl = (
            f"{focal_quiet} focal-quiet"
            + (f", {raw_quiet} raw-quiet" if raw_quiet != focal_quiet else "")
        )
        lead_str = ", ".join(
            f"{row['channel_name']}{_anatomy_suffix(row)} ({row['focal_count']} focal, {row['focal_rate_per_min']:.2f}/m)"
            for row in s["leading_focal_channels"][:3]
        ) or "None (>0 focal events)"
        burst_note = (
            f"<div class='warn-line'>&#9888; {cnt['widespread_burst_events']}/{cnt['total_raw_events']} "
            f"({100*cnt['widespread_burst_fraction']:.1f}%) raw detections occurred in "
            f"{cnt['widespread_burst_episodes']} multi-shaft synchronous burst(s).</div>"
            if cnt["widespread_burst_events"] > 0
            else "<div class='ok-line'>&#10003; 0 multi-shaft synchronous artifact bursts detected.</div>"
        )

        # ONNX Neural Classification Summary strip
        onnx_line = ""
        if s.get("classifier_status") in {"complete", "partial"}:
            st_map = (s.get("classification") or {}).get("stages") or {}
            st_pills = []
            for st_k in ("artifact", "spike", "ehfo"):
                if st_k in st_map:
                    st_v = st_map[st_k]
                    ok = st_v.get("status") == "complete"
                    st_pills.append(
                        f"<span class='{'pill-ok' if ok else 'pill-na'}'>{_e(st_k)}: {_e(st_v.get('status'))}</span>"
                    )
            na_c = cnt.get("non_artifact_events")
            sp_c = cnt.get("spike_hfo_events")
            eh_c = cnt.get("ehfo_events")
            onnx_line = (
                f"<div class='onnx-line'><strong>PyHFO ONNX:</strong> {' '.join(st_pills)} · "
                f"Non-artifact: <strong>{na_c if na_c is not None else 'N/A'}</strong> · "
                f"Spike-HFO: <strong>{sp_c if sp_c is not None else 'N/A'}</strong> · "
                f"eHFO: <strong>{eh_c if eh_c is not None else 'N/A'}</strong></div>"
            )

        outcomes = {v: cnt.get(f"clinician_{v}", 0) for v in ("confirmed", "artifact", "uncertain")}
        reviewed = cnt.get("reviewed_focal_events", sum(outcomes.values()))
        breakdown = ", ".join(f"{n} {v}" for v, n in outcomes.items() if n)
        review = f"{reviewed} of {cnt['non_burst_events']} focal candidates"
        if breakdown:
            if reviewed == sum(outcomes.values()):
                review += f" ({breakdown})"
            else:
                review += f". Across all detections: {breakdown}"
        extra_roles = [
            f"{review_roles[k]}: {cnt.get(k + '_reviewed', 0)} ({cnt.get(k + '_confirmed', 0)} confirmed)."
            for k in ("researcher", "rehearsal") if k in review_roles
        ]
        extra_roles.append(f"Agent verdicts: {cnt.get('agent_reviewed', 0)}, kept separate.")
        rev_line = (
            f"<div class='kpi-lead kpi-review' data-rev-rec='{_rec(r)}' data-rev-run='{_e(sanitize_slug(primary_run))}'>"
            f"<strong>Clinician review:</strong> {review}. {' '.join(extra_roles)}</div>"
        )
        fr_n = cnt.get("fast_ripple_focal_events")
        fr_str = f"<div><span class='big'>{fr_n}</span> <span class='lbl'>FR-Dom</span></div>" if fr_n is not None else ""

        kpis += f"""
        <div class="kpi">
          <div class="kpi-head"><strong>{_rec(r)}</strong>{participant_tag} · <code>{_e(primary_run)}</code> · fs={s['selection']['sfreq_hz']:g} Hz · {ref_str}{state_str}</div>
          <div class="kpi-nums">
            <div><span class="big accent">{cnt['non_burst_events']}</span> <span class="lbl">Focal Events</span></div>
            <div><span class="big muted">{cnt['total_raw_events']}</span> <span class="lbl">Total Raw</span></div>
            {fr_str}
            <div><span class="big">{cnt['measured_channels']}</span> <span class="lbl">Measured Ch<span class="lbl-sub">{_e(quiet_lbl)}</span></span></div>
          </div>
          {burst_note}
          {onnx_line}
          <div class="kpi-lead"><strong>Focal Leaders:</strong> {_e(lead_str)}</div>
          {rev_line}
        </div>
        """

    # External Clinical Overlay Concordance Banner (only if explicitly supplied)
    overlay_sec = ""
    if overlay_conc.get("available"):
        ov = overlay_conc["overlay"]
        mt = overlay_conc["metrics"]
        p_at_k = f"{100*mt['soz_precision_at_top_k']:.0f}%" if mt["soz_precision_at_top_k"] is not None else "N/A"
        rec_soz = f"{100*mt['soz_recall_all_focal']:.0f}%" if mt["soz_recall_all_focal"] is not None else "N/A"
        jac = f"{mt['soz_focal_jaccard']:.2f}" if mt["soz_focal_jaccard"] is not None else "N/A"
        overlay_sec = f"""
        <section class="overlay-box">
          <h2>External Clinical Annotation Reference Overlay <span class="pill-na">EXTERNAL REFERENCE — NOT INFERRED</span></h2>
          <p class="sub">
            <strong>Source:</strong> {_e(ov['source'])} ·
            <strong>SOZ ({len(ov.get('soz_channels', []))}):</strong> {_e(', '.join(ov.get('soz_channels', [])) or 'none')} ·
            <strong>Resected ({len(ov.get('resected_channels', []))}):</strong> {_e(', '.join(ov.get('resected_channels', [])) or 'none')}
          </p>
          <p class="sub">
            Concordance on <code>{_rec(overlay_conc['recording'])}</code> ({_e(overlay_conc['run_name'])}):
            Top-{mt['top_k']} Focal in SOZ: <strong>{_e(', '.join(mt['top_k_in_soz']) or 'none')}</strong> (Precision@{mt['top_k']}: <strong>{p_at_k}</strong>) ·
            SOZ Focal Recall: <strong>{rec_soz}</strong> · Focal-SOZ Jaccard: <strong>{jac}</strong>
          </p>
        </section>
        """

    ov_tags = overlay_conc.get("per_channel_tags", {}) if overlay_conc.get("available") else {}

    def _tag_html(ch_name: str) -> str:
        tags = ov_tags.get(ch_name) or []
        return "".join(f" <span class='ext-tag'>{_e(t)}</span>" for t in tags)

    # Active Channel Table (ZERO 0-event channels polluted into Top rankings!)
    if cross_rec is not None:
        pw_notes = []
        for pw in cross_rec["pairwise"]:
            f_sh = ", ".join(pw["focal_top5_shared"]) or "none"
            pw_notes.append(
                f"<strong>{_rec(pw['recording_a'])} vs {_rec(pw['recording_b'])}:</strong> "
                f"Shared Focal Top-5: <strong>{_e(f_sh)}</strong> · Raw Spearman &rho; = {pw['raw_spearman_rho'] if pw['raw_spearman_rho'] is not None else 'N/A'}"
            )
        ths = "".join(f"<th>{_rec(r)}</th>" for r in recs)
        trows = ""
        for row in cross_rec["aligned_active_channels"]:
            tds = ""
            for r in recs:
                c = row["by_recording"].get(r)
                if not c or c.get("focal_rate_per_min") is None:
                    tds += "<td class='muted'>unassessed</td>"
                else:
                    rank = f"<strong>#{c['focal_rank']}</strong> · " if c["focal_rank"] else ""
                    raw_rate = f" ({float(c['raw_rate_per_min']):.2f}/min)" if c.get("raw_rate_per_min") is not None else ""
                    detections = "detection" if c["raw_count"] == 1 else "detections"
                    tds += (
                        f"<td>{rank}{_fmt_rate_cell(c['focal_count'], c['analyzed_seconds'], c['focal_rate_per_min'])} · "
                        f"<span class='muted'>{c['raw_count']} {detections} total{raw_rate}</span>{_e(_anatomy_suffix(c))}</td>"
                    )
            trows += (
                f"<tr class='ch-row' data-anchor-type='channel_row' data-section='Active Channel Consistency' "
                f"data-channel='{_e(row['channel'])}' data-recording='{_rec(recs[0])}'>"
                f"<td><div class='ch-cell'><span><strong>{_e(row['channel'])}</strong>{_shaft_badge(row)}{_tag_html(row['channel'])}</span>"
                f"<span class='row-tools live-ui'>"
                f"<button type='button' class='row-chip' data-act='jump_waveform' title='Inspect waveforms in Reader'>↗ Wave</button>"
                f"<button type='button' class='row-chip' data-act='ask_anchor' title='Comment or ask Agent about this channel'>💬 Ask</button>"
                f"</span></div></td>{tds}</tr>"
            )
        table_sec = f"""
        <section>
          <h2>Consistency across recordings</h2>
          <p class="sub">Channels with zero events are excluded from the rankings.</p>
          <p class="sub">{' &nbsp;|&nbsp; '.join(pw_notes)}</p>
          <table><thead><tr><th>Channel</th>{ths}</tr></thead><tbody>{trows}</tbody></table>
        </section>
        """
    else:
        r0 = recs[0]
        s0 = bundles[r0]["summary"]
        has_onnx = s0.get("classifier_status") in {"complete", "partial"}
        leaders = s0["leading_focal_channels"] + s0["leading_raw_channels"]
        active_names = set(list(dict.fromkeys(row["channel_name"] for row in leaders))[:14])
        active_rows = [row for row in s0["channels"] if row["channel_name"] in active_names]
        onnx_hdr = "<th>Non-artifact</th><th>Spike-HFO</th><th>eHFO (/min)</th>" if has_onnx else ""
        trows = ""
        for row in active_rows:
            onnx_cells = ""
            if has_onnx:
                na_v = row.get("model_non_artifact_count")
                sp_v = row.get("spike_hfo_count")
                eh_v = row.get("ehfo_count")
                eh_r = row.get("ehfo_rate_per_min")
                onnx_cells = (
                    f"<td class='num'>{na_v if na_v is not None else '—'}</td>"
                    f"<td class='num'>{sp_v if sp_v is not None else '—'}</td>"
                    f"<td class='num'><strong>{eh_v if eh_v is not None else '—'}</strong>"
                    + (f" ({eh_r:.2f}/min)" if eh_r is not None else "")
                    + "</td>"
                )
            fr_c = row.get("fast_ripple_count")
            trows += (
                f"<tr class='ch-row' data-anchor-type='channel_row' data-section='Active Candidate Channels' "
                f"data-channel='{_e(row['channel_name'])}' data-recording='{_rec(r0)}'>"
                f"<td class='num'>{('#' + str(row['focal_rank'])) if row['focal_rank'] else '—'}</td>"
                f"<td><div class='ch-cell'><span><strong>{_e(row['channel_name'])}</strong>{_e(_anatomy_suffix(row))}{_shaft_badge(row)}{_tag_html(row['channel_name'])}</span>"
                f"<span class='row-tools live-ui'>"
                f"<button type='button' class='row-chip' data-act='jump_waveform' title='Inspect waveforms in Reader'>↗ Wave</button>"
                f"<button type='button' class='row-chip' data-act='ask_anchor' title='Comment or ask Agent about this channel'>💬 Ask</button>"
                f"</span></div></td>"
                f"<td class='num'><strong>{row['focal_count']}</strong> ({row['focal_rate_per_min']:.2f}/min)</td>"
                f"<td class='num'>{fr_c if fr_c is not None else '—'}</td>"
                f"<td class='num'>{row['raw_count']} ({row['raw_rate_per_min']:.2f}/min)</td>"
                f"{onnx_cells}"
                f"<td class='num'>{row['detector_analyzed_seconds']}s</td></tr>"
            )
        fq = s0["counts"].get("focal_quiet_measured_channels", s0["counts"]["quiet_measured_channels"])
        anatomy_by_name = {row["channel_name"]: _anatomy_suffix(row) for row in s0["channels"]}
        unassessed = [name + anatomy_by_name.get(name, "") for name in s0["unassessed_channel_names"]]
        table_sec = f"""
        <section>
          <h2>Active Candidate Channels ({_rec(r0)} · {_e(primary_run)})</h2>
          <p class="sub">Focal-quiet measured channels: {fq} / {s0['counts']['measured_channels']} · Unassessed/excluded: {_e(', '.join(unassessed) or 'none')}</p>
          <table><thead><tr><th>Rank</th><th>Channel</th><th>Focal (per min)</th><th>Fast ripple</th><th>Total (per min)</th>{onnx_hdr}<th>Time analyzed</th></tr></thead><tbody>{trows}</tbody></table>
        </section>
        """

    # Symmetric Sensitivity Comparison Section (including L7 side-by-side non-subtracted table!)
    sens_sec = ""
    for cmp_d in branch_cmps:
        if not cmp_d.get("directly_comparable_rates"):
            la = cmp_d.get("run_a_leading") or []
            lb = cmp_d.get("run_b_leading") or []
            n_max = max(len(la[:8]), len(lb[:8]), 1)
            side_rows = ""
            for i in range(n_max):
                ca = (
                    f"#{la[i]['focal_rank']} <strong>{_e(la[i]['channel_name'])}</strong>{_e(_anatomy_suffix(la[i]))} ({la[i]['focal_count']} / {la[i]['focal_rate_per_min']:.2f}/m)"
                    if i < len(la[:8]) else "<span class='muted'>—</span>"
                )
                cb = (
                    f"#{lb[i]['focal_rank']} <strong>{_e(lb[i]['channel_name'])}</strong>{_e(_anatomy_suffix(lb[i]))} ({lb[i]['focal_count']} / {lb[i]['focal_rate_per_min']:.2f}/m)"
                    if i < len(lb[:8]) else "<span class='muted'>—</span>"
                )
                side_rows += f"<tr><td>{i + 1}</td><td>{ca}</td><td>{cb}</td></tr>"
            ref_a_m = (cmp_d.get("run_a_reference") or {}).get("mode", "a")
            ref_b_m = (cmp_d.get("run_b_reference") or {}).get("mode", "b")
            sens_sec += f"""
            <section>
              <h2>Cross-Derivation Comparison — {_rec(cmp_d['recording'])}: <code>{_e(cmp_d['run_a'])}</code> ({_e(ref_a_m)}) vs <code>{_e(cmp_d['run_b'])}</code> ({_e(ref_b_m)})</h2>
              <p class="sub">{_e(cmp_d.get('reason'))}</p>
              <table>
                <thead><tr><th>#</th><th>{_e(cmp_d['run_a'])} ({_e(ref_a_m)}) Focal Leaders</th><th>{_e(cmp_d['run_b'])} ({_e(ref_b_m)}) Focal Leaders</th></tr></thead>
                <tbody>{side_rows}</tbody>
              </table>
            </section>
            """
            continue

        sm = cmp_d["summary_metrics"]
        ef = cmp_d["event_fate"]
        srows = "".join(
            f"<tr><td><strong>{_e(c['channel'])}</strong>{_e(_anatomy_suffix(c))}{_shaft_badge(c)}</td>"
            f"<td>{('#' + str(c['rank_a'])) if c.get('rank_a') else ('unassessed' if c.get('rate_a') is None else 'quiet')} &rarr; "
            f"{('#' + str(c['rank_b'])) if c.get('rank_b') else ('unassessed' if c.get('rate_b') is None else 'quiet')}</td>"
            f"<td>{_fmt_rate_cell(c.get('count_a'), c.get('analyzed_seconds_a'), c.get('rate_a'))}</td>"
            f"<td>{_fmt_rate_cell(c.get('count_b'), c.get('analyzed_seconds_b'), c.get('rate_b'))}</td>"
            f"<td>{_fmt_delta_cell(c.get('rate_delta'))}</td></tr>"
            for c in cmp_d["channels"][:10]
        )
        sens_sec += f"""
        <section>
          <h2>Sensitivity Comparison — {_rec(cmp_d['recording'])}: <code>{_e(cmp_d['run_a'])}</code> vs <code>{_e(cmp_d['run_b'])}</code> (Symmetric Top Union)</h2>
          <p class="sub">
            Focal events: {sm['focal_events_a']} &rarr; {sm['focal_events_b']} (Raw: {sm['raw_events_a']} &rarr; {sm['raw_events_b']}) ·
            Focal Top-5 shared: <strong>{_e(', '.join(sm['top5_shared']) or 'none')}</strong> ·
            Excluded by mask: {ef['excluded_by_changed_time_or_channel_mask']} · Disappeared in analyzed span: {ef['disappeared_within_analyzed_coverage']}
          </p>
          <table>
            <thead><tr><th>Channel</th><th>Focal Rank Shift</th><th>{_e(cmp_d['run_a'])}</th><th>{_e(cmp_d['run_b'])}</th><th>Focal &Delta;</th></tr></thead>
            <tbody>{srows}</tbody>
          </table>
        </section>
        """

    # Spatial Distribution Projection (if coordinates exist)
    spatial_sec = ""
    if spatial_fig and spatial_fig.get("png_base64"):
        tpl_warn = " (Template coordinate frame — NOT patient-native MRI anatomy)" if spatial_fig.get("template") else ""
        spatial_sec = f"""
        <section class="keep-together">
          <h2>SEEG Depth-Shaft Spatial Distribution ({_rec(spatial_fig['recording'])} · {_e(spatial_fig['run_name'])})</h2>
          <div class="fig-card">
            <img src="data:image/png;base64,{spatial_fig['png_base64']}" alt="SEEG 3D Spatial Projection" />
            <div class="fig-cap">
              Coordinate frame: <strong>{_e(spatial_fig.get('coordinate_system'))}</strong>{_e(tpl_warn)} · Mapped contacts: {spatial_fig.get('mapped_contacts')}
            </div>
          </div>
        </section>
        """

    # Deduplicated Evidence Figures in High-Legibility 2-Column Layout
    cards_sec = ""
    if cards:
        c_html = ""
        for c in cards:
            ev = c.get("focus_event")
            ev_id = (ev.get("id") or "") if ev else ""
            card_run = str(c.get("run_name") or primary_run)
            score_html = ""
            if ev:
                btn_v = ev.get("clinician_verdict") or ""
                roles_html = f'Clinician: <span class="vtag">{_e(btn_v or "unreviewed")}</span>'
                for key, role in review_roles.items():
                    if key != "clinician":
                        roles_html += f" · {role}: {_e(ev.get(key + '_verdict') or 'unreviewed')}"
                score_html = "".join(f"<div>{_e(line)}</div>" for line in model_score_labels(ev))
            else:
                roles_html = "Control window" if c.get("view_mode") == "background_control" else "Clipped interval"
                btn_v = ""
            conf_cls = " active-ok" if btn_v == "confirm" else ""
            art_cls = " active-bad" if btn_v == "artifact" else ""
            unc_cls = " active-warn" if btn_v == "uncertain" else ""
            verdict_btns = (
                f'<button type="button" class="fig-btn{conf_cls}" data-act="verdict" data-verdict="confirm">✓ Confirm</button>'
                f'<button type="button" class="fig-btn{unc_cls}" data-act="verdict" data-verdict="uncertain">? Uncertain</button>'
                f'<button type="button" class="fig-btn{art_cls}" data-act="verdict" data-verdict="artifact">✕ Artifact</button>'
                if ev else ""
            )
            run_tag = f" · <code>{_e(card_run)}</code>" if card_run != primary_run else ""
            iv_bounds = c.get("interval_seconds")
            win_bounds = c.get("window_seconds")
            if isinstance(iv_bounds, (list, tuple)) and len(iv_bounds) == 2:
                c_t0, c_t1 = float(iv_bounds[0]), float(iv_bounds[1])
                c_ctr = 0.5 * (c_t0 + c_t1)
                start_attr = f"{c_t0:.5f}"
                end_attr = f"{c_t1:.5f}"
                center_attr = f"{c_ctr:.5f}"
                span_tag = f" · [{c_t0:.2f}–{c_t1:.2f}s]"
            elif (not ev) and isinstance(win_bounds, (list, tuple)) and len(win_bounds) == 2:
                c_t0, c_t1 = float(win_bounds[0]), float(win_bounds[1])
                c_ctr = 0.5 * (c_t0 + c_t1)
                start_attr = f"{c_t0:.5f}"
                end_attr = f"{c_t1:.5f}"
                center_attr = f"{c_ctr:.5f}"
                span_tag = f" · [{c_t0:.2f}–{c_t1:.2f}s]"
            else:
                start_attr = ""
                end_attr = ""
                center_attr = ""
                span_tag = ""
            c_html += f"""
            <div class="fig-card keep-together" data-anchor-type="figure" data-section="Signal &amp; Time-Frequency Evidence" data-recording="{_rec(c['recording'])}" data-run-name="{_e(card_run)}" data-channel="{_e(c['channel'])}" data-event-id="{_e(ev_id)}" data-start-seconds="{_e(start_attr)}" data-end-seconds="{_e(end_attr)}" data-center-seconds="{_e(center_attr)}" data-quoted="{_rec(c['recording'])} · {_e(c['channel'])} ({_e(card_run)}) · {_e(c.get('source_label'))}{_e(span_tag)}">
              <img src="data:image/png;base64,{c['png_base64']}" alt="{_e(c['channel'])}" />
              <div class="fig-cap">
                <strong>{_rec(c['recording'])} · {_e(c['channel'])}</strong>{_e(_anatomy_suffix(c))}{run_tag}{_e(span_tag)} · {_e(c.get('source_label'))}
                <div class="fig-verdicts">{roles_html}</div>{score_html}
              </div>
              <div class="fig-actions live-ui">
                <button type="button" class="fig-btn" data-act="jump_waveform">↗ Waveforms</button>
                {verdict_btns}
                <button type="button" class="fig-btn fig-ask" data-act="ask_anchor">💬 Ask Agent</button>
              </div>
            </div>
            """
        cards_sec = f"<section><h2>Signal &amp; Time-Frequency Evidence (Deduplicated, 2-Column MDT Layout)</h2><div class='fig-grid'>{c_html}</div></section>"

    # Consecutive-deduplicated Attributed Notes scoped by recording, run, recording_fingerprint, AND exact timestamp/interval
    # (NEVER drop flip-flops, two-session identical notes, or sub-10ms distinct timestamps, AND NEVER show notes from superseded recordings!)
    active_rec_fps = {r: str(rec_models[r].fingerprint or "") for r in rec_models}

    def _note_exact_time_key(j: dict[str, Any]) -> str:
        iv = j.get("interval_seconds")
        if isinstance(iv, (list, tuple)) and len(iv) == 2:
            try:
                return f"iv:{float(iv[0]):.6f}-{float(iv[1]):.6f}"
            except Exception:
                pass
        cs = j.get("center_seconds")
        if cs is not None:
            try:
                return f"t:{float(cs):.6f}"
            except Exception:
                pass
        return ""

    def _note_time_label(j: dict[str, Any], decimals: int = 2) -> str:
        iv = j.get("interval_seconds")
        if isinstance(iv, (list, tuple)) and len(iv) == 2:
            try:
                return f"[{float(iv[0]):.{decimals}f}, {float(iv[1]):.{decimals}f}]s"
            except Exception:
                pass
        cs = j.get("center_seconds")
        if cs is not None:
            try:
                return f"@{float(cs):.{decimals}f}s"
            except Exception:
                pass
        return ""

    last_note_key: tuple[str, str, str, str, str, str, str, str] | None = None
    unique_notes: list[dict[str, Any]] = []
    for wal_idx, j in enumerate(judgments):
        txt = str(j.get("note") or "").strip()
        if not txt and not j.get("verdict"):
            continue
        if not store.judgment_matches_active_state(j, wal_idx, binding_ctx, default_run=sanitize_slug(primary_run)):
            continue
        t_exact = _note_exact_time_key(j)
        key = (
            str(j.get("author", "")),
            str(j.get("role", "")),
            str(j.get("recording") or ""),
            str(j.get("run_name") or ""),
            str(j.get("event_id") or j.get("channel") or ""),
            t_exact,
            str(j.get("verdict") or ""),
            txt or str(j.get("verdict")),
        )
        if key != last_note_key:
            last_note_key = key
            unique_notes.append(j)

    notes_sec = ""
    if unique_notes:
        shown_notes = unique_notes[-10:]
        # Auto-increase decimal precision whenever two distinct displayed timestamps collide at 2 decimals (e.g. 4.151s vs 4.154s)!
        disp_counts_2: dict[tuple[str, str, str], set[str]] = {}
        for n in shown_notes:
            lbl2 = _note_time_label(n, 2)
            if lbl2:
                disp_counts_2.setdefault((str(n.get("recording") or ""), str(n.get("channel") or ""), lbl2), set()).add(_note_exact_time_key(n))
        disp_counts_3: dict[tuple[str, str, str], set[str]] = {}
        for n in shown_notes:
            lbl3 = _note_time_label(n, 3)
            if lbl3:
                disp_counts_3.setdefault((str(n.get("recording") or ""), str(n.get("channel") or ""), lbl3), set()).add(_note_exact_time_key(n))

        def _fmt_note_time(n: dict[str, Any]) -> str:
            lbl2 = _note_time_label(n, 2)
            if not lbl2:
                return ""
            k2 = (str(n.get("recording") or ""), str(n.get("channel") or ""), lbl2)
            if len(disp_counts_2.get(k2, ())) <= 1:
                return lbl2
            lbl3 = _note_time_label(n, 3)
            k3 = (str(n.get("recording") or ""), str(n.get("channel") or ""), lbl3)
            return lbl3 if len(disp_counts_3.get(k3, ())) <= 1 else _note_time_label(n, 4)

        nlis = "".join(
            f"<li><strong>[{_e(n.get('author', 'reviewer'))} · {_e(n.get('role', 'review'))}]</strong> "
            f"{_e(' / '.join(str(x) for x in [recording_label(n.get('recording')), n.get('channel'), _fmt_note_time(n), n.get('verdict')] if x))}: "
            f"{_e(n.get('note') or '')}</li>"
            for n in shown_notes
        )
        notes_sec = f"<section><h2>Attributed Review Judgments &amp; Notes</h2><ul class='nlist'>{nlis}</ul></section>"

    lim_items = "".join(f"<li>{_e(x)}</li>" for x in prov0.limitations[:5]) if prov0.limitations else "<li>Preserve original acquisition reference and per-channel coverage limits.</li>"

    if display_text:
        live_css = ".live-ui { display:none !important; }"
        live_section = ""
        share_line = (
            "<p><strong>De-identified export</strong> · Native patient names, codes and birth dates are withheld. "
            "Sharing labels replace source case and recording names.</p>"
        )
    else:
        live_css = _LIVE_REPORT_CSS
        report_meta_json = json.dumps(
            {
                "case": case_label,
                "primary_run": primary_run,
                "workspace_id": workspace_id,
                "export_hash": export_hash,
                "export_filename": f"{case_label}_{sanitize_slug(primary_run)}_case_sheet.html",
                "default_recording": recs[0] if recs else None,
                "recordings": recs,
                "is_synthetic": bool(prov0.is_synthetic),
                "review_roles": list(review_roles),
                "default_role": "engineering_rehearsal" if prov0.is_synthetic else "clinician",
            },
            ensure_ascii=False,
        ).replace("</", "<\\/")
        live_section = f"{_LIVE_REPORT_HTML}\n<script>window.__BRAINLENS_REPORT__ = {report_meta_json};</script>\n<script>{_LIVE_REPORT_SCRIPT}</script>"
        share_line = ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="brainlens-primary-run" content="{_e(primary_run)}" />
{SHEET_DISPLAY_META}
{deid.SHEET_DEIDENTIFICATION_META if display_text else ""}
<title>{html.escape(case_label)} — Presurgical SEEG/EEG Case Discussion Sheet</title>
<style>
  :root {{ --ink:#0f172a; --ink2:#374151; --muted:#6b7280; --line:rgba(15,23,42,0.09); --accent:#1d4ed8; --focal:#059669; --bg:#f5f6f8; --surface:#ffffff; --subtle:#fafbfc; --chip:#eef1f5; }}
  [data-theme="dark"] {{ --ink:#f3f4f6; --ink2:#cbd5e1; --muted:#8a94a6; --line:rgba(255,255,255,0.09); --accent:#38bdf8; --focal:#10b981; --bg:#0b0d11; --surface:#12151c; --subtle:#171b24; --chip:#1c212c; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Inter","Segoe UI",sans-serif; color:var(--ink); background:var(--bg); margin:0; padding:22px 22px 72px; font-size:12.5px; line-height:1.5; -webkit-font-smoothing:antialiased; }}
  code {{ font-family:"SF Mono",ui-monospace,Menlo,monospace; font-size:11px; background:var(--chip); padding:1px 5px; border-radius:4px; }}
  .page {{ max-width:1160px; margin:0 auto; background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:28px 32px; box-shadow:0 1px 3px rgba(15,23,42,0.04), 0 12px 32px -12px rgba(15,23,42,0.08); }}
  header {{ display:flex; justify-content:space-between; align-items:flex-start; border-bottom:1px solid var(--line); padding-bottom:14px; margin-bottom:18px; gap:16px; }}
  h1 {{ margin:0 0 4px 0; font-size:20px; font-weight:700; letter-spacing:-0.025em; }}
  h2 {{ margin:22px 0 8px 0; font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; color:var(--ink2); display:flex; align-items:center; gap:8px; }}
  h2::before {{ content:""; width:3px; height:13px; border-radius:2px; background:var(--accent); display:inline-block; }}
  .badge-info {{ background:rgba(29,78,216,0.08); color:var(--accent); padding:5px 11px; border-radius:8px; font-weight:600; font-size:11px; max-width:430px; line-height:1.38; }}
  .badge-warn {{ background:rgba(217,119,6,0.10); color:#b45309; padding:5px 11px; border-radius:8px; font-weight:600; font-size:11px; }}
  .qbox {{ background:var(--subtle); border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:9px; padding:10px 14px; margin-bottom:14px; }}
  .qbox ul {{ list-style:none; margin:5px 0 0 0; padding:0; }}
  .q-item {{ position:relative; padding-left:14px; margin:4px 0; line-height:1.45; }}
  .q-item::before {{ content:""; position:absolute; left:0; top:7px; width:5px; height:5px; border-radius:50%; background:var(--accent); }}
  .q-item .row-chip {{ margin-left:5px; vertical-align:middle; }}
  .overlay-box {{ background:var(--subtle); border:1px solid rgba(29,78,216,0.22); border-radius:10px; padding:10px 14px; margin-bottom:12px; }}
  .overlay-box h2 {{ margin-top:2px; }}
  .pill {{ background:var(--accent); color:#fff; font-size:9.5px; padding:1px 7px; border-radius:99px; font-weight:600; }}
  [data-theme="dark"] .pill {{ color:#0b0d11; }}
  .pill-ok {{ background:rgba(5,150,105,0.14); color:var(--focal); font-size:10px; padding:2px 7px; border-radius:5px; font-weight:600; }}
  .pill-na {{ background:var(--chip); color:var(--muted); font-size:10px; padding:2px 7px; border-radius:5px; font-weight:600; }}
  .ext-tag {{ background:rgba(29,78,216,0.12); color:var(--accent); font-size:9.5px; padding:1px 6px; border-radius:4px; font-weight:700; margin-left:4px; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:12px; margin-bottom:16px; }}
  .kpi {{ border:1px solid var(--line); border-radius:11px; padding:14px 16px; background:var(--subtle); }}
  .kpi-head {{ font-size:11.5px; color:var(--muted); margin-bottom:8px; }}
  .kpi-nums {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(92px, 1fr)); gap:12px; margin-bottom:10px; align-items:start; }}
  .big {{ font-size:22px; font-weight:700; letter-spacing:-0.025em; font-variant-numeric:tabular-nums; line-height:1.15; }}
  .big.accent {{ color:var(--focal); }}
  .big.muted {{ color:var(--muted); }}
  .lbl {{ display:block; font-size:10.5px; color:var(--muted); font-weight:500; margin-top:2px; }}
  .lbl-sub {{ display:block; font-size:9.8px; color:var(--muted); opacity:0.8; font-weight:400; margin-top:1px; }}
  td.num {{ white-space:nowrap; font-variant-numeric:tabular-nums; }}
  .warn-line {{ font-size:11.2px; color:#b45309; background:rgba(217,119,6,0.10); padding:5px 9px; border-radius:6px; margin:5px 0; }}
  [data-theme="dark"] .warn-line {{ color:#fbbf24; }}
  .ok-line {{ font-size:11.2px; color:var(--focal); background:rgba(5,150,105,0.10); padding:5px 9px; border-radius:6px; margin:5px 0; }}
  .onnx-line {{ font-size:11.2px; color:var(--ink2); background:var(--surface); border:1px solid var(--line); padding:5px 9px; border-radius:6px; margin:5px 0; }}
  .kpi-lead, .sub {{ font-size:11.6px; color:var(--muted); margin:4px 0; }}
  .muted {{ color:var(--muted); }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:8px; }}
  th, td {{ border-bottom:1px solid var(--line); padding:7px 10px; text-align:left; }}
  th {{ background:transparent; font-weight:600; font-size:10.8px; text-transform:uppercase; letter-spacing:0.04em; color:var(--muted); }}
  tbody tr:last-child td {{ border-bottom:none; }}
  .ch-cell {{ display:flex; align-items:center; justify-content:space-between; gap:8px; }}
  .sh {{ font-family:"SF Mono",ui-monospace,Menlo,monospace; font-size:10px; color:var(--muted); background:var(--chip); padding:1px 5px; border-radius:4px; margin-left:4px; }}
  .fig-grid {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:16px; margin-top:10px; }}
  .fig-card {{ border:1px solid var(--line); border-radius:10px; padding:10px; background:var(--surface); display:flex; flex-direction:column; }}
  .fig-card img {{ width:100%; height:auto; display:block; border-radius:6px; background:#fff; }}
  .fig-cap {{ font-size:11.2px; color:var(--muted); text-align:center; padding-top:7px; }}
  .vtag {{ background:var(--chip); padding:1px 7px; border-radius:5px; font-weight:600; color:var(--ink); }}
  .nlist, .lim-list {{ margin:5px 0; padding-left:18px; font-size:11.5px; color:var(--ink2); }}
  .lim-list li {{ margin-bottom:3px; }}
  .keep-together {{ break-inside:avoid; page-break-inside:avoid; }}
  footer {{ margin-top:22px; padding-top:14px; border-top:1px solid var(--line); font-size:11px; color:var(--muted); display:grid; grid-template-columns:1fr 1.25fr; gap:18px; }}
{live_css}
  @media print {{ :root, [data-theme="dark"] {{ --ink:#0f172a; --ink2:#374151; --muted:#6b7280; --line:rgba(15,23,42,0.12); --bg:#ffffff; --surface:#ffffff; --subtle:#fafbfc; --chip:#eef1f5; }} body {{ background:#fff; padding:0; }} .page {{ border:none; box-shadow:none; padding:6px; }} .live-ui, .offline-pill {{ display:none !important; }} }}
</style>
</head>
<body>
<div class="page">
  <header>
    <div>
      <h1>Presurgical SEEG / iEEG Evidence Sheet — {html.escape(case_label)}</h1>
      <div class="sub">Generated {_e(utc_now()[:19])}Z · Primary run: <code>{_e(primary_run)}</code> · Source recordings immutable</div>
    </div>
    <div class="{badge_cls}">{badge_txt}</div>
  </header>
  {share_line}
  {anatomy_provenance}
  {q_block}
  <div class="kpi-grid">{kpis}</div>
  {overlay_sec}
  {table_sec}{_shared(''.join(_evidence_map_section(doc) for doc in (evidence_maps or {}).values()))}
  {sens_sec}
  {spatial_sec}
  {cards_sec}
  {_shared(_reader_agreement_section(store, case_slug, sanitize_slug(primary_run), bundles, judgments, binding_ctx))}
  {notes_sec}
  <footer>
    <div>
      <strong>Quantitative &amp; Clinical Scope:</strong> Rates use exact per-channel detector coverage
      (<code>events/min = 60 &times; count / detector_analyzed_seconds</code>; unassessed = N/A, 0-event channels omitted from Top rankings).
      Multi-shaft synchronous bursts (&ge;5 shafts within &plusmn;30 ms) are separated from focal candidates. Signal counts and ONNX scores support clinician review and do not establish SOZ or resection targets.
      BrainLens is not a medical device. This sheet is not a standalone surgical resection boundary.
    </div>
    <div><strong>Provenance &amp; Dataset Limits:</strong><ul class="lim-list">{lim_items}</ul></div>
  </footer>
</div>
{live_section}
</body>
</html>
"""


_LIVE_REPORT_CSS = """
  /* Progressive Interactive Agent-Sync Layer (auto-hidden offline & when printing) */
  .live-ui { display:none; }
  body.live-connected .row-tools.live-ui,
  body.live-connected .row-chip.live-ui,
  body.live-connected .tray-pill.live-ui { display:inline-flex; }
  body.live-connected .fig-actions.live-ui { display:flex; }
  body.live-connected .tray-drawer.live-ui.open { display:flex; }
  body.live-connected .sel-pop.live-ui.open { display:inline-flex; }
  body.live-connected .composer-backdrop.live-ui.open { display:flex; }
  body.live-connected .live-toast.live-ui.open { display:block; }

  .row-tools { align-items:center; gap:4px; opacity:0; transition:opacity 0.14s ease; }
  .ch-row:hover .row-tools, .q-item:hover .row-chip { opacity:1; }
  .ch-row:hover td { background:var(--subtle); }
  .q-item .row-chip { opacity:0.35; transition:opacity 0.14s ease; }
  .row-chip {
    align-items:center; gap:3px; padding:2px 7px; border-radius:5px;
    border:1px solid var(--line); background:var(--surface); color:var(--ink2);
    font-size:10.5px; font-weight:600; cursor:pointer; font-family:inherit;
    white-space:nowrap; transition:all 0.12s ease;
  }
  .row-chip:hover { border-color:rgba(29,78,216,0.35); color:var(--accent); background:rgba(29,78,216,0.06); }

  .fig-actions {
    margin-top:8px; padding-top:8px; border-top:1px solid var(--line);
    justify-content:center; flex-wrap:wrap; gap:6px;
  }
  .fig-btn {
    display:inline-flex; align-items:center; gap:4px; padding:3px 9px; border-radius:6px;
    border:1px solid var(--line); background:var(--subtle); color:var(--ink2);
    font-size:10.8px; font-weight:600; cursor:pointer; font-family:inherit; transition:all 0.12s ease;
  }
  .fig-btn:hover { background:var(--surface); border-color:var(--accent); color:var(--ink); }
  .fig-btn.fig-ask:hover { border-color:var(--accent); color:var(--accent); }
  .fig-btn.active-ok { background:#059669; border-color:#059669; color:#fff; }
  .fig-btn.active-warn { background:#b45309; border-color:#b45309; color:#fff; }
  .fig-btn.active-bad { background:#dc2626; border-color:#dc2626; color:#fff; }
  .fig-btn:disabled { opacity:0.55; cursor:wait; }

  .offline-pill {
    display:none; position:fixed; right:18px; bottom:16px; z-index:70;
    align-items:center; gap:7px; padding:6px 12px; border-radius:99px;
    background:var(--surface); border:1px solid var(--line); color:var(--muted);
    font-size:11px; font-weight:600; text-decoration:none;
    box-shadow:0 4px 14px -4px rgba(15,23,42,0.14);
  }
  body.offline-static .offline-pill { display:inline-flex; }
  .offline-pill:hover { border-color:var(--accent); color:var(--accent); }
  .tray-dot.offline { background:#dc2626; box-shadow:0 0 0 3px rgba(220,38,38,0.18); }
  .act-st.claimed { background:rgba(217,119,6,0.14); color:#b45309; }
  [data-theme="dark"] .act-st.claimed { color:#fbbf24; }
  .act-reply {
    margin-top:4px; padding:6px 8px; border-radius:7px;
    background:rgba(5,150,105,0.08); border:1px solid rgba(5,150,105,0.22);
    font-size:11.2px; color:var(--ink); line-height:1.45;
  }
  .act-reply strong { color:var(--focal); font-size:10px; text-transform:uppercase; letter-spacing:0.04em; display:block; margin-bottom:2px; }

  /* Floating Text-Selection Comment Pill */
  .sel-pop {
    position:fixed; z-index:90; align-items:center; gap:6px;
    background:#0f172a; color:#fff; border:1px solid rgba(255,255,255,0.12);
    border-radius:99px; padding:5px 12px; font-size:11.2px; font-weight:600;
    box-shadow:0 8px 24px -6px rgba(15,23,42,0.45); cursor:pointer;
    transform:translate(-50%, -100%);
  }
  .sel-pop:hover { background:#1e293b; }

  /* Floating Bottom-Right Agent Review Tray Pill & Drawer */
  .tray-pill {
    position:fixed; right:20px; bottom:18px; z-index:80;
    align-items:center; gap:8px; padding:7px 14px; border-radius:99px;
    background:var(--surface); backdrop-filter:blur(10px);
    border:1px solid var(--line); color:var(--ink);
    font-size:11.8px; font-weight:600; cursor:pointer; font-family:inherit;
    box-shadow:0 4px 16px -4px rgba(15,23,42,0.18), 0 1px 2px rgba(15,23,42,0.08);
    transition:transform 0.12s ease, border-color 0.12s ease;
  }
  .tray-pill:hover { border-color:var(--accent); transform:translateY(-1px); }
  .tray-dot { width:7px; height:7px; border-radius:50%; background:var(--focal); box-shadow:0 0 0 3px rgba(5,150,105,0.16); }
  .tray-count {
    font-family:"SF Mono",ui-monospace,Menlo,monospace; font-size:10.5px; font-weight:700;
    padding:1px 7px; border-radius:99px; background:rgba(29,78,216,0.12); color:var(--accent);
  }
  .tray-count.zero { background:var(--chip); color:var(--muted); }

  .tray-drawer {
    position:fixed; right:20px; bottom:62px; z-index:85; width:370px; max-width:calc(100vw - 32px);
    max-height:520px; flex-direction:column;
    background:var(--surface); backdrop-filter:blur(12px);
    border:1px solid var(--line); border-radius:14px;
    box-shadow:0 18px 42px -12px rgba(15,23,42,0.32); overflow:hidden;
  }
  .tray-head {
    display:flex; align-items:center; justify-content:space-between; padding:11px 14px;
    border-bottom:1px solid var(--line); background:var(--subtle);
  }
  .tray-title { font-size:12.2px; font-weight:700; color:var(--ink); display:flex; align-items:center; gap:7px; }
  .tray-close {
    border:none; background:transparent; color:var(--muted); font-size:16px; line-height:1;
    cursor:pointer; padding:2px 6px; border-radius:6px;
  }
  .tray-close:hover { background:var(--chip); color:var(--ink); }
  .tray-list {
    flex:1; overflow-y:auto; padding:10px 12px; display:flex; flex-direction:column; gap:8px; min-height:90px;
  }
  .tray-empty { font-size:11.5px; color:var(--muted); text-align:center; padding:18px 12px; line-height:1.5; }
  .act-item {
    border:1px solid var(--line); border-radius:9px; padding:8px 10px; background:var(--subtle);
    display:flex; flex-direction:column; gap:4px;
  }
  .act-top { display:flex; align-items:center; justify-content:space-between; gap:6px; font-size:10.5px; }
  .act-anchor { font-weight:700; color:var(--ink); }
  .act-st { font-size:9.8px; font-weight:700; padding:1px 6px; border-radius:99px; }
  .act-st.pending { background:rgba(29,78,216,0.12); color:var(--accent); }
  .act-st.consumed { background:rgba(5,150,105,0.14); color:var(--focal); }
  .act-quote {
    font-size:10.8px; color:var(--muted); border-left:2px solid var(--line);
    padding-left:7px; margin:1px 0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  }
  .act-body { font-size:11.6px; color:var(--ink2); word-break:break-word; }
  .tray-compose {
    padding:10px 12px; border-top:1px solid var(--line); background:var(--subtle);
    display:flex; flex-direction:column; gap:7px;
  }
  .tray-input-row { display:flex; gap:6px; }
  .tray-input {
    flex:1; border:1px solid var(--line); border-radius:8px; padding:6px 9px;
    font-size:11.8px; font-family:inherit; outline:none; background:var(--surface); color:var(--ink);
  }
  .tray-input:focus { border-color:var(--accent); box-shadow:0 0 0 3px rgba(29,78,216,0.12); }
  .tray-foot { display:flex; align-items:center; justify-content:space-between; gap:6px; }
  .tbtn {
    border:1px solid var(--line); background:var(--surface); color:var(--ink);
    border-radius:7px; padding:5px 10px; font-size:11.2px; font-weight:600;
    cursor:pointer; font-family:inherit; transition:all 0.12s ease;
  }
  .tbtn:hover { background:var(--chip); }
  .tbtn-primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  [data-theme="dark"] .tbtn-primary { color:#0b0d11; }
  .tbtn-primary:hover { filter:brightness(0.92); }

  /* Anchored Comment Composer Modal */
  .composer-backdrop {
    position:fixed; inset:0; z-index:95; background:rgba(15,23,42,0.42);
    backdrop-filter:blur(3px); align-items:center; justify-content:center; padding:16px;
  }
  .composer-card {
    width:440px; max-width:100%; background:var(--surface); border:1px solid var(--line);
    border-radius:14px; padding:16px 18px; box-shadow:0 20px 48px -12px rgba(15,23,42,0.38);
    display:flex; flex-direction:column; gap:10px;
  }
  .composer-head { display:flex; align-items:center; justify-content:space-between; font-size:13px; font-weight:700; color:var(--ink); }
  .composer-quote {
    font-size:11.2px; color:var(--muted); background:var(--subtle); border:1px solid var(--line);
    border-left:3px solid var(--accent); border-radius:7px; padding:7px 10px; max-height:74px; overflow-y:auto;
  }
  .composer-ta {
    width:100%; min-height:76px; border:1px solid var(--line); border-radius:8px;
    padding:8px 10px; font-family:inherit; font-size:12px; background:var(--surface); color:var(--ink); resize:vertical; outline:none;
  }
  .composer-ta:focus { border-color:var(--accent); box-shadow:0 0 0 3px rgba(29,78,216,0.12); }
  .composer-foot { display:flex; justify-content:flex-end; gap:7px; }

  .live-toast {
    position:fixed; left:50%; bottom:20px; transform:translateX(-50%); z-index:100;
    background:#0f172a; color:#fff; font-size:11.5px; font-weight:600;
    padding:7px 14px; border-radius:99px; box-shadow:0 8px 24px -6px rgba(15,23,42,0.45);
  }
"""

_LIVE_REPORT_HTML = """
<a id="offlineOpenLink" class="offline-pill" href="http://127.0.0.1:8765/" target="_blank" rel="noopener">
  Offline Archive · Open Live in Workbench ↗
</a>

<button type="button" id="selPopBtn" class="sel-pop live-ui">💬 Annotate / Ask Agent</button>

<button type="button" id="agentTrayBtn" class="tray-pill live-ui" title="View report annotations synced with Agent">
  <span id="trayStatusDot" class="tray-dot"></span>
  <span id="trayStatusLbl">Agent Sync</span>
  <span id="agentTrayCount" class="tray-count zero">Synced</span>
</button>

<div id="agentTrayDrawer" class="tray-drawer live-ui" role="dialog" aria-label="Agent Review Inbox">
  <div class="tray-head">
    <div class="tray-title"><span id="drawerStatusDot" class="tray-dot"></span> Agent Review Inbox</div>
    <button type="button" id="agentTrayClose" class="tray-close" aria-label="Close">&times;</button>
  </div>
  <div id="agentTrayList" class="tray-list">
    <div class="tray-empty">Hover any channel row, click a Triad figure button, or highlight text anywhere on the sheet to annotate or ask the Agent.</div>
  </div>
  <div class="tray-compose">
    <div class="tray-input-row">
      <input id="trayQuickInput" class="tray-input" type="text" placeholder="Ask Agent about this case sheet..." />
      <button type="button" id="trayQuickSend" class="tbtn tbtn-primary">Send</button>
    </div>
    <div class="tray-foot">
      <button type="button" id="trayCopyPrompt" class="tbtn" title="Copy concise trigger prompt for chat">Copy Prompt</button>
      <button type="button" id="trayNotifyBtn" class="tbtn tbtn-primary" title="Wake standby Agent with all pending annotations">Notify Agent ↵</button>
    </div>
  </div>
</div>

<div id="composerBackdrop" class="composer-backdrop live-ui">
  <div class="composer-card">
    <div class="composer-head">
      <span id="composerTitle">Annotate / Ask Agent</span>
      <button type="button" id="composerClose" class="tray-close">&times;</button>
    </div>
    <div id="composerQuote" class="composer-quote"></div>
    <textarea id="composerTa" class="composer-ta" placeholder="Write a question or review note for the Agent (e.g. 'Check whether LB10-LB9 attenuates here or run a bipolar sensitivity branch')..."></textarea>
    <div class="composer-foot">
      <button type="button" id="composerCancel" class="tbtn">Cancel</button>
      <button type="button" id="composerAddBtn" class="tbtn">Add to Inbox</button>
      <button type="button" id="composerSendBtn" class="tbtn tbtn-primary">Send to Agent ↵</button>
    </div>
  </div>
</div>

<div id="liveToast" class="live-toast live-ui"></div>
"""

_LIVE_REPORT_SCRIPT = """
(function () {
  const META = window.__BRAINLENS_REPORT__ || window.__BTK_REPORT__ || {};
  const isHttp = window.location.protocol === "http:" || window.location.protocol === "https:";
  const tokenMeta = document.querySelector('meta[name="brainlens-token"]') || document.querySelector('meta[name="btk-token"]');
  const AUTH_TOKEN = tokenMeta ? (tokenMeta.getAttribute("content") || "").trim() : "";
  const API_BASE = isHttp ? window.location.origin : "";

  // Strictly static & read-only when opened via file:// or when underlying analysis was superseded!
  const staleMeta = document.querySelector('meta[name="brainlens-sheet-stale"]') || document.querySelector('meta[name="btk-sheet-stale"]');
  const IS_STALE = Boolean(staleMeta && (staleMeta.getAttribute("content") || "").trim() === "1");
  const offLink = document.getElementById("offlineOpenLink");
  if (offLink && META.case) {
    offLink.href = `${isHttp ? window.location.origin : "http://127.0.0.1:8765"}/?case=${encodeURIComponent(META.case)}&run=${encodeURIComponent(META.primary_run || "baseline")}&view=sheet`;
    if (IS_STALE) {
      offLink.textContent = "Outdated Snapshot · Open Live in Workbench ↗";
    }
  }
  if (IS_STALE) {
    document.body.classList.add("offline-static");
    if (isHttp && window.parent && window.parent !== window) {
      try {
        window.parent.postMessage({
          type: "brainlens_sheet_stale",
          case: META.case || "",
          primary_run: META.primary_run || "baseline",
        }, window.location.origin);
      } catch (_) {}
    }
    return;
  }
  if (!isHttp || !AUTH_TOKEN) {
    document.body.classList.add("offline-static");
    return;
  }

  let connected = false;
  let online = false;
  let sessionExpired = false;
  let actions = [];
  let pendingCount = 0;
  let claimedCount = 0;
  let activeAnchor = null;
  let toastTimer = null;
  let sse = null;
  let lastVerdictRev = -1;
  const composerDrafts = {};

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s || "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const newOpId = () => (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : ("op_" + Date.now() + "_" + Math.random().toString(36).slice(2, 10));

  function showToast(msg) {
    const el = $("liveToast");
    if (!el) return;
    el.textContent = msg;
    el.classList.add("open");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove("open"), 2600);
  }

  function setConnectionOnline(isUp) {
    online = isUp && !sessionExpired;
    ["trayStatusDot", "drawerStatusDot"].forEach((id) => {
      const el = $(id);
      if (el) el.classList.toggle("offline", !online);
    });
    const lbl = $("trayStatusLbl");
    if (lbl) lbl.textContent = sessionExpired ? "Reload Needed" : (online ? "Agent Sync" : "Reconnecting…");
    renderInbox();
  }

  async function apiGet(path) {
    const r = await fetch(API_BASE + path, { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }

  async function apiPost(path, body) {
    const r = await fetch(API_BASE + path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-BrainLens-Token": AUTH_TOKEN,
      },
      body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const msg = (data.error && data.error.message) || data.error || ("HTTP " + r.status);
      const err = new Error(msg);
      err.httpStatus = r.status;
      err.code = (data.error && data.error.code) || "";
      throw err;
    }
    return data;
  }

  function applyCardVerdictUI(card, hit, roles) {
    if (!card) return;
    const verdict = activeRole === "researcher" ? hit.researcher_verdict
      : activeRole === "engineering_rehearsal" ? hit.rehearsal_verdict
      : activeRole === "agent" ? hit.agent_verdict : hit.verdict;
    card.querySelectorAll("[data-act='verdict']").forEach((b) => {
      b.classList.remove("active-ok", "active-warn", "active-bad");
      const bv = b.getAttribute("data-verdict");
      if (bv && verdict && bv === verdict) {
        b.classList.add(bv === "confirm" ? "active-ok" : (bv === "uncertain" ? "active-warn" : "active-bad"));
      }
    });
    const line = card.querySelector(".fig-verdicts");
    if (line) {
      const labels = [`Clinician: <span class="vtag">${esc(hit.verdict || "unreviewed")}</span>`];
      for (const [key, title] of roles) labels.push(`${title}: ${esc(hit[key + "_verdict"] || "unreviewed")}`);
      labels.push(`Agent: ${esc(hit.agent_verdict || "unreviewed")}`);
      line.innerHTML = labels.join(" · ");
    }
  }

  async function refreshVerdictsProjection() {
    if (!connected || !META.case) return;
    try {
      const proj = await apiGet(`/api/case-verdicts?case=${encodeURIComponent(META.case)}`);
      const rev = Number(proj.judgment_revision ?? 0);
      if (rev < lastVerdictRev) return;
      lastVerdictRev = rev;
      const vmap = proj.verdicts || {};
      const tallies = proj.review_tallies || {};
      const optionalRoles = [["researcher", "Researcher"], ["rehearsal", "Engineering rehearsal"]].filter(([key]) =>
        META.review_roles.includes(key) || Object.values(tallies).some(t => t[key + "_reviewed"]) ||
        Object.values(vmap).some(v => v[key + "_verdict"])
      );
      document.querySelectorAll(".fig-card[data-event-id]").forEach((card) => {
        const eid = card.getAttribute("data-event-id");
        if (!eid) return;
        const rec = card.getAttribute("data-recording") || "";
        const rname = card.getAttribute("data-run-name") || META.primary_run || "baseline";
        const hit = vmap[`${rec}|${rname}|${eid}`] || {};
        applyCardVerdictUI(card, hit, optionalRoles);
      });
      document.querySelectorAll(".kpi-review[data-rev-rec]").forEach((el) => {
        const rec = el.getAttribute("data-rev-rec") || "";
        const rname = el.getAttribute("data-rev-run") || META.primary_run || "baseline";
        const t = tallies[`${rec}|${rname}`];
        if (!t) return;
        const reviewed = Number(t.reviewed_focal_events || 0);
        const outcomes = ["confirmed", "artifact", "uncertain"].map(v => [v, Number(t["clinician_" + v] || 0)]);
        const breakdown = outcomes.filter(([, n]) => n).map(([v, n]) => `${n} ${v}`).join(", ");
        let review = `${reviewed} of ${Number(t.non_burst_events || 0)} focal candidates`;
        if (breakdown) {
          review += reviewed === outcomes.reduce((sum, [, n]) => sum + n, 0)
            ? ` (${breakdown})` : `. Across all detections: ${breakdown}`;
        }
        const extras = optionalRoles.map(([key, title]) =>
          `${title}: ${Number(t[key + "_reviewed"] || 0)} (${Number(t[key + "_confirmed"] || 0)} confirmed).`
        );
        extras.push(`Agent verdicts: ${Number(t.agent_reviewed || 0)}, kept separate.`);
        el.innerHTML = `<strong>Clinician review:</strong> ${esc(review)}. ${esc(extras.join(" "))}`;
      });
    } catch (_) {}
  }

  function renderInbox() {
    const cntEl = $("agentTrayCount");
    if (cntEl) {
      if (!online) {
        cntEl.textContent = "Offline";
        cntEl.classList.add("zero");
      } else if (pendingCount > 0) {
        cntEl.textContent = `${pendingCount} pending`;
        cntEl.classList.remove("zero");
      } else if (claimedCount > 0) {
        cntEl.textContent = `${claimedCount} in review`;
        cntEl.classList.remove("zero");
      } else {
        cntEl.textContent = "Synced";
        cntEl.classList.add("zero");
      }
    }
    const list = $("agentTrayList");
    if (!list) return;
    if (!actions.length) {
      list.innerHTML = `<div class="tray-empty">Select text, a channel, or a figure to add a note for the Agent.</div>`;
      return;
    }
    list.innerHTML = [...actions].reverse().slice(0, 25).map((a) => {
      const kind = a.anchor_type === "text_selection" ? "selected text" : (a.anchor_type || "").replace(/_/g, " ");
      const anchorLbl = [a.channel, a.recording, a.run_name && a.run_name !== META.primary_run ? a.run_name : "", kind !== "general" ? kind : ""].filter(Boolean).join(" · ") || "Report note";
      const st = a.status === "resolved" ? "resolved" : (a.status === "claimed" ? "claimed" : "pending");
      const stCls = st === "resolved" ? "consumed" : st;
      const stTxt = st === "resolved" ? "✓ Resolved" : (st === "claimed" ? "● In Review" : "Pending");
      const qHtml = a.quoted_text ? `<div class="act-quote">${esc(a.quoted_text)}</div>` : "";
      const vTxt = (typeof a.verdict === "string" && a.verdict) ? `<strong>[${esc(a.verdict.toUpperCase())}]</strong> ` : "";
      const bodyTxt = a.comment ? esc(a.comment) : (a.action === "submit_batch" ? "Notified Agent to review pending annotations." : esc(a.action));
      const replyHtml = a.agent_reply ? `<div class="act-reply"><strong>Agent Reply</strong>${esc(a.agent_reply)}</div>` : "";
      return `
        <div class="act-item" data-action-id="${esc(a.id)}">
          <div class="act-top">
            <span class="act-anchor" title="${esc(a.id)}">${esc(anchorLbl)}</span>
            <span class="act-st ${stCls}">${stTxt}</span>
          </div>
          ${qHtml}
          <div class="act-body">${vTxt}${bodyTxt}</div>
          ${replyHtml}
        </div>
      `;
    }).join("");
  }

  async function refreshActions() {
    if (!connected || !META.case) return;
    try {
      const data = await apiGet(`/api/report-actions?case=${encodeURIComponent(META.case)}&limit=40`);
      actions = data.actions || [];
      pendingCount = data.pending_count || 0;
      claimedCount = data.claimed_count || 0;
      setConnectionOnline(true);
    } catch (_) {
      setConnectionOnline(false);
    }
  }

  async function sendAction(payload, toastMsg) {
    if (!connected || !isSheetContextValid()) return null;
    try {
      const res = await apiPost("/api/report-action", {
        case: META.case,
        recording: payload.recording || META.default_recording,
        run_name: payload.run_name || META.primary_run || "baseline",
        source: "case_sheet",
        operation_id: payload.operation_id || newOpId(),
        ...payload,
        workspace_id: META.workspace_id || "",
        primary_run: META.primary_run || "baseline",
        export_filename: META.export_filename || `${META.case || ""}_${META.primary_run || "baseline"}_case_sheet.html`,
        export_hash: META.export_hash || "",
      });
      await refreshActions();
      if (toastMsg) showToast(toastMsg + (res.action && res.action.id ? ` (${res.action.id})` : ""));
      return res;
    } catch (err) {
      if (err && err.httpStatus === 401) {
        sessionExpired = true;
        setConnectionOnline(false);
        showToast("Workbench server restarted — reload this sheet to reconnect");
      } else if (err && err.httpStatus === 409 && err.code === "stale_sheet") {
        markSheetOutdated();
        showToast("MDT sheet is outdated — reloading latest evidence");
      } else {
        if (!err || !err.httpStatus) setConnectionOnline(false);
        showToast(`Action failed: ${err.message || "Workbench unreachable"}`);
      }
      return null;
    }
  }

  function jumpToWaveform(anchor) {
    const rec = anchor.recording || META.default_recording || "";
    const rname = anchor.run_name || META.primary_run || "baseline";
    const ch = anchor.channel || "";
    const evId = anchor.event_id || "";
    const s0 = (anchor.start_seconds !== undefined && anchor.start_seconds !== null && anchor.start_seconds !== "") ? Number(anchor.start_seconds) : null;
    const s1 = (anchor.end_seconds !== undefined && anchor.end_seconds !== null && anchor.end_seconds !== "") ? Number(anchor.end_seconds) : null;
    const ctr = (anchor.center_seconds !== undefined && anchor.center_seconds !== null && anchor.center_seconds !== "")
      ? Number(anchor.center_seconds)
      : (s0 !== null && s1 !== null ? 0.5 * (s0 + s1) : null);
    // Telemetry only — never inflates Agent Inbox pending count
    sendAction({
      action: "inspect_waveform",
      anchor_type: anchor.anchor_type || "channel_row",
      section: anchor.section || "",
      recording: rec,
      run_name: rname,
      channel: ch || null,
      event_id: evId || null,
      start_seconds: Number.isFinite(s0) ? s0 : null,
      end_seconds: Number.isFinite(s1) ? s1 : null,
      center_seconds: Number.isFinite(ctr) ? ctr : null,
      trigger_agent: false,
    });
    if (window.parent && window.parent !== window) {
      window.parent.postMessage({
        type: "brainlens_jump_waveform",
        case: META.case,
        recording: rec,
        run_name: rname,
        channel: ch,
        event_id: evId,
        start_seconds: Number.isFinite(s0) ? s0 : null,
        end_seconds: Number.isFinite(s1) ? s1 : null,
        center_seconds: Number.isFinite(ctr) ? ctr : null,
      }, window.location.origin);
      return;
    }
    const qs = new URLSearchParams({
      case: META.case || "",
      recording: rec,
      run: rname,
      view: "reader",
    });
    if (ch) qs.set("channel", ch);
    if (evId) qs.set("event_id", evId);
    if (!evId && Number.isFinite(ctr)) {
      const span = (Number.isFinite(s0) && Number.isFinite(s1) && s1 > s0) ? (s1 - s0) : 0;
      const win = span > 8 ? Math.min(30, Math.ceil(span)) : 8;
      qs.set("start", String(Math.max(0, ctr - win * 0.5).toFixed(2)));
      qs.set("dur", String(win));
      qs.set("center_seconds", String(ctr));
      if (span > 0) qs.set("span_seconds", String(span));
    }
    window.open(`${API_BASE}/?${qs.toString()}`, "_blank", "noopener");
  }

  function anchorDraftKey(anc) {
    if (!anc) return "general";
    return [anc.anchor_type || "general", anc.recording || "", anc.run_name || "", anc.channel || "", anc.event_id || "", anc.section || ""].join("|");
  }

  function notifyParentComposer(isOpen) {
    if (window.parent && window.parent !== window) {
      try { window.parent.postMessage({ type: "brainlens_composer_state", open: Boolean(isOpen) }, window.location.origin); } catch (_) {}
    }
  }

  function openComposer(anchor) {
    activeAnchor = anchor;
    const titleBits = [anchor.channel, anchor.section || anchor.anchor_type].filter(Boolean);
    $("composerTitle").textContent = titleBits.length ? `Ask Agent · ${titleBits.join(" — ")}` : "Annotate / Ask Agent";
    const qEl = $("composerQuote");
    if (anchor.quoted_text) {
      qEl.textContent = `"${anchor.quoted_text}"`;
      qEl.style.display = "block";
    } else {
      qEl.style.display = "none";
    }
    $("composerTa").value = composerDrafts[anchorDraftKey(anchor)] || "";
    $("composerBackdrop").classList.add("open");
    notifyParentComposer(true);
    setTimeout(() => $("composerTa").focus(), 20);
  }

  function closeComposer(clearDraft = false) {
    if (activeAnchor) {
      const k = anchorDraftKey(activeAnchor);
      if (clearDraft) delete composerDrafts[k];
      else {
        const cur = $("composerTa").value;
        if (cur && cur.trim()) composerDrafts[k] = cur;
        else delete composerDrafts[k];
      }
    }
    $("composerBackdrop").classList.remove("open");
    activeAnchor = null;
    notifyParentComposer(false);
  }

  async function submitComposer(triggerAgent) {
    if (!activeAnchor) return;
    const ta = $("composerTa");
    const txt = ta.value.trim();
    if (!txt) { ta.focus(); return; }
    const anc = activeAnchor;
    const addBtn = $("composerAddBtn"), sendBtn = $("composerSendBtn");
    addBtn.disabled = true; sendBtn.disabled = true;
    try {
      const res = await sendAction({
        action: triggerAgent ? "ask_agent" : "annotate",
        anchor_type: anc.anchor_type || "general",
        section: anc.section || "",
        recording: anc.recording || META.default_recording,
        run_name: anc.run_name || META.primary_run || "baseline",
        channel: anc.channel || null,
        event_id: anc.event_id || null,
        quoted_text: anc.quoted_text || "",
        comment: txt,
        trigger_agent: triggerAgent,
      }, triggerAgent ? "Sent to Agent" : "Saved to Agent Inbox");
      // ONLY close & clear AFTER confirmed server persistence!
      if (res && res.ok) {
        ta.value = "";
        closeComposer(true);
      }
    } finally {
      addBtn.disabled = false; sendBtn.disabled = false;
    }
  }

  function extractAnchor(el) {
    const host = el.closest("[data-anchor-type]");
    if (!host) return { anchor_type: "general", run_name: META.primary_run || "baseline" };
    const rawS0 = host.getAttribute("data-start-seconds");
    const rawS1 = host.getAttribute("data-end-seconds");
    const rawCtr = host.getAttribute("data-center-seconds");
    return {
      anchor_type: host.getAttribute("data-anchor-type") || "general",
      section: host.getAttribute("data-section") || "",
      recording: host.getAttribute("data-recording") || META.default_recording || "",
      run_name: host.getAttribute("data-run-name") || META.primary_run || "baseline",
      channel: host.getAttribute("data-channel") || "",
      event_id: host.getAttribute("data-event-id") || "",
      start_seconds: rawS0 !== null && rawS0 !== "" ? Number(rawS0) : null,
      end_seconds: rawS1 !== null && rawS1 !== "" ? Number(rawS1) : null,
      center_seconds: rawCtr !== null && rawCtr !== "" ? Number(rawCtr) : null,
      quoted_text: host.getAttribute("data-quoted") || "",
    };
  }

  // Active reviewer identity & parent Workbench context lock (prevents cross-case clicks during switches!)
  let sheetLocked = false;
  let parentActiveCase = "";
  let parentActiveRun = "";
  function isSheetContextValid() {
    if (sheetLocked) return false;
    if (window.parent && window.parent !== window) {
      if (parentActiveCase && parentActiveCase !== (META.case || "")) return false;
      if (parentActiveRun && META.primary_run && parentActiveRun !== META.primary_run) return false;
    }
    return true;
  }

  const initParams = new URLSearchParams(window.location.search);
  let activeRole = initParams.get("role") || (() => {
    try {
      if (!META.is_synthetic) return localStorage.getItem("brainlens_reviewer_role") || localStorage.getItem("btk_reviewer_role") || "";
    } catch (_) {}
    return "";
  })() || META.default_role || "clinician";
  let activeAuthor = initParams.get("author") || (() => {
    try { return localStorage.getItem("brainlens_reviewer_author") || localStorage.getItem("btk_reviewer_author") || ""; } catch (_) {}
    return "";
  })() || (activeRole === "engineering_rehearsal" ? "rehearsal" : activeRole);

  // Event delegation on anchored rows, questions, and Triad figure cards
  document.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-act]");
    if (!btn || !connected || !isSheetContextValid()) return;
    const act = btn.getAttribute("data-act");
    const anc = extractAnchor(btn);
    if (act === "jump_waveform") {
      jumpToWaveform(anc);
    } else if (act === "ask_anchor") {
      openComposer(anc);
    } else if (act === "verdict") {
      const v = btn.getAttribute("data-verdict");
      const card = btn.closest(".fig-card");
      const vBtns = card ? [...card.querySelectorAll("[data-act='verdict']")] : [];
      vBtns.forEach((b) => { b.disabled = true; });
      try {
        // NEVER mutate card UI optimistically before server confirms!
        const roleVal = activeRole || META.default_role || "clinician";
        const authorVal = activeAuthor || (roleVal === "engineering_rehearsal" ? "rehearsal" : roleVal);
        const actorLbl = roleVal === "engineering_rehearsal" ? "Rehearsal reviewer" : (roleVal === "researcher" ? "Researcher" : "Clinician");
        const res = await sendAction({
          action: "verdict",
          anchor_type: anc.anchor_type,
          section: anc.section,
          recording: anc.recording,
          run_name: anc.run_name,
          channel: anc.channel,
          event_id: anc.event_id || null,
          quoted_text: anc.quoted_text,
          verdict: v,
          author: authorVal,
          role: roleVal,
          comment: `${actorLbl} marked ${anc.channel || "figure"} (${anc.run_name}) as ${v} on MDT Case Sheet`,
          trigger_agent: false,
        }, `Saved ${v} verdict (${roleVal})`);
        if (res && res.ok) {
          await refreshVerdictsProjection();
          if (window.parent && window.parent !== window) {
            window.parent.postMessage({
              type: "brainlens_report_verdict",
              case: META.case,
              recording: anc.recording,
              run_name: anc.run_name,
              channel: anc.channel,
              event_id: anc.event_id || null,
              verdict: v,
              role: roleVal,
            }, window.location.origin);
          }
        }
      } finally {
        vBtns.forEach((b) => { b.disabled = false; });
      }
    }
  });

  let sheetBecameStale = false;
  function markSheetOutdated() {
    if (sheetBecameStale) return;
    sheetBecameStale = true;
    connected = false;
    online = false;
    document.body.classList.remove("live-connected");
    document.body.classList.add("offline-static");
    document.querySelectorAll("[data-act='verdict']").forEach((b) => { b.disabled = true; });
    if (offLink) offLink.textContent = "Outdated Snapshot · Open Live in Workbench ↗";
    if (sse) { try { sse.close(); } catch (_) {} sse = null; }
    if (window.parent && window.parent !== window) {
      try {
        window.parent.postMessage({
          type: "brainlens_sheet_stale",
          case: META.case || "",
          primary_run: META.primary_run || "baseline",
        }, window.location.origin);
      } catch (_) {}
    }
  }

  async function checkSheetFreshness() {
    if (!connected || sheetBecameStale || !META.case) return;
    try {
      const qs = new URLSearchParams({
        case: META.case,
        file: META.export_filename || `${META.case}_${META.primary_run || "baseline"}_case_sheet.html`,
      });
      if (META.export_hash) qs.set("export_hash", META.export_hash);
      const res = await apiGet(`/api/sheet-freshness?${qs.toString()}`);
      if (!res || res.fresh !== true) {
        markSheetOutdated();
      }
    } catch (_) {}
  }

  // Authenticated Theme & Reviewer sync (URL ?theme=dark or exact-origin parent Workbench postMessage)
  if (initParams.get("theme") === "dark") document.documentElement.setAttribute("data-theme", "dark");
  window.addEventListener("message", (ev) => {
    if (ev.origin !== window.location.origin || ev.source !== window.parent) return;
    const msg = ev.data;
    if (!msg || typeof msg !== "object") return;
    if (msg.type === "brainlens_theme") {
      document.documentElement.setAttribute("data-theme", msg.theme === "dark" ? "dark" : "light");
    } else if (msg.type === "brainlens_sheet_context") {
      sheetLocked = Boolean(msg.locked);
      if (msg.case !== undefined) parentActiveCase = String(msg.case || "");
      if (msg.run_name !== undefined) parentActiveRun = String(msg.run_name || "");
    } else if (msg.type === "brainlens_reviewer") {
      if (msg.role) activeRole = String(msg.role);
      if (msg.author) activeAuthor = String(msg.author);
      refreshVerdictsProjection();
    } else if (msg.type === "brainlens_sse" && connected) {
      if (msg.event === "status") {
        if (msg.sessionExpired) sessionExpired = true;
        setConnectionOnline(Boolean(msg.online));
        if (msg.online && !sessionExpired) {
          checkSheetFreshness();
          refreshActions();
          refreshVerdictsProjection();
        }
      } else if (msg.event === "resync") {
        if (msg.data && msg.data.reason === "server_restarted") {
          sessionExpired = true;
          setConnectionOnline(false);
          showToast("Workbench server restarted — reload this sheet to reconnect");
        }
        checkSheetFreshness();
        refreshActions();
        refreshVerdictsProjection();
      } else if (msg.event === "case_updated") {
        if (!msg.data || !msg.data.case || msg.data.case === META.case) {
          checkSheetFreshness();
        }
      } else if (msg.event === "report_actions") {
        refreshActions();
      } else if (msg.event === "verdict") {
        if (!msg.data || !msg.data.case || msg.data.case === META.case) {
          checkSheetFreshness();
          refreshVerdictsProjection();
        }
      }
    }
  });

  // Google-Docs / Notion style text selection floating comment trigger
  const selPop = $("selPopBtn");
  let selSnapshot = null;
  window.addEventListener("scroll", () => selPop.classList.remove("open"), { passive: true });
  document.addEventListener("keydown", (e) => {
    if (e.isComposing || e.keyCode === 229) return;
    if (e.key === "Escape") {
      selPop.classList.remove("open");
      closeComposer(false);
      $("agentTrayDrawer").classList.remove("open");
    }
  });
  $("composerBackdrop").addEventListener("click", (e) => {
    if (e.target === $("composerBackdrop")) closeComposer(false);
  });
  document.addEventListener("mouseup", (e) => {
    if (!connected) return;
    if (e.target.closest(".live-ui")) return;
    const sel = window.getSelection();
    const txt = sel ? sel.toString().trim() : "";
    if (txt.length >= 3 && sel.rangeCount > 0) {
      const rect = sel.getRangeAt(0).getBoundingClientRect();
      if (rect.width > 0 || rect.height > 0) {
        const node = sel.anchorNode && (sel.anchorNode.nodeType === 1 ? sel.anchorNode : sel.anchorNode.parentElement);
        const secEl = node && node.closest("section");
        const secTitle = secEl && secEl.querySelector("h2") ? secEl.querySelector("h2").textContent.trim() : "Report Excerpt";
        const anc = node ? extractAnchor(node) : {};
        selSnapshot = {
          anchor_type: "text_selection",
          section: anc.section || secTitle,
          recording: anc.recording || META.default_recording || "",
          run_name: anc.run_name || META.primary_run || "baseline",
          channel: anc.channel || "",
          event_id: anc.event_id || "",
          quoted_text: txt.slice(0, 260),
        };
        selPop.style.left = Math.max(90, Math.min(window.innerWidth - 90, rect.left + rect.width * 0.5)) + "px";
        selPop.style.top = Math.max(40, rect.top - 8) + "px";
        selPop.classList.add("open");
        return;
      }
    }
    selPop.classList.remove("open");
  });

  selPop.addEventListener("mousedown", (e) => e.preventDefault());
  selPop.addEventListener("click", () => {
    selPop.classList.remove("open");
    if (selSnapshot) openComposer(selSnapshot);
  });

  // Tray & Composer controls (with IME composition guard for Chinese/Japanese input!)
  $("agentTrayBtn").addEventListener("click", () => {
    $("agentTrayDrawer").classList.toggle("open");
    if ($("agentTrayDrawer").classList.contains("open")) refreshActions();
  });
  $("agentTrayClose").addEventListener("click", () => $("agentTrayDrawer").classList.remove("open"));
  $("composerClose").addEventListener("click", closeComposer);
  $("composerCancel").addEventListener("click", closeComposer);
  $("composerAddBtn").addEventListener("click", () => submitComposer(false));
  $("composerSendBtn").addEventListener("click", () => submitComposer(true));
  $("composerTa").addEventListener("keydown", (e) => {
    if (e.isComposing || e.keyCode === 229) return;
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") submitComposer(true);
    if (e.key === "Escape") closeComposer();
  });

  async function sendQuick() {
    const inp = $("trayQuickInput");
    const btn = $("trayQuickSend");
    const txt = inp.value.trim();
    if (!txt) return;
    btn.disabled = true;
    try {
      const res = await sendAction({
        action: "ask_agent",
        anchor_type: "general",
        section: "Case Discussion Sheet",
        comment: txt,
        trigger_agent: true,
      }, "Sent to Agent");
      // ONLY clear input AFTER confirmed persistence!
      if (res && res.ok) inp.value = "";
    } finally {
      btn.disabled = false;
    }
  }
  $("trayQuickSend").addEventListener("click", sendQuick);
  $("trayQuickInput").addEventListener("keydown", (e) => {
    if (e.isComposing || e.keyCode === 229) return;
    if (e.key === "Enter") { e.preventDefault(); sendQuick(); }
  });

  $("trayNotifyBtn").addEventListener("click", async () => {
    await sendAction({
      action: "submit_batch",
      anchor_type: "general",
      section: "Case Discussion Sheet",
      comment: `Clinician requested Agent review of ${pendingCount} pending report action(s) on ${META.case}.`,
      trigger_agent: true,
    }, "Notified Agent");
  });

  $("trayCopyPrompt").addEventListener("click", async () => {
    const pend = actions.filter((a) => a.status !== "resolved");
    const ids = pend.map((a) => a.id).join(", ");
    const txt = pend.length
      ? `Please check my ${pend.length} open annotation(s) on the ${META.case} MDT Case Sheet (${ids}) via open_workbench(action="status", case="${META.case}") and resolve them with your reply.`
      : `Please sync with my current view on the ${META.case} MDT Case Sheet via open_workbench(action="status", case="${META.case}").`;
    try {
      await navigator.clipboard.writeText(txt);
      showToast("Copied Agent trigger prompt");
    } catch (_) {
      showToast(txt.slice(0, 72));
    }
  });

  let sseListenersBound = false;
  function bindPageVisibilitySync() {
    if (sseListenersBound) return;
    sseListenersBound = true;
    window.addEventListener("pagehide", () => {
      if (sse) { try { sse.close(); } catch (_) {} sse = null; }
    });
    window.addEventListener("pageshow", () => {
      if (!connected || sessionExpired) return;
      checkSheetFreshness();
      refreshActions();
      refreshVerdictsProjection();
      if (!(window.parent && window.parent !== window) && !sse) connectSSE();
    });
  }

  function connectSSE() {
    bindPageVisibilitySync();
    // When embedded inside Workbench View 3 iframe, share parent's single SSE connection via postMessage
    // instead of consuming a second HTTP/1.1 browser socket per tab!
    if (window.parent && window.parent !== window) {
      setConnectionOnline(false);
      try {
        window.parent.postMessage({ type: "brainlens_request_sse_status" }, window.location.origin);
        window.parent.postMessage({ type: "brainlens_request_sheet_context" }, window.location.origin);
      } catch (_) {}
      return;
    }
    if (sse) { try { sse.close(); } catch (_) {} }
    try {
      sse = new EventSource(API_BASE + "/api/events");
      sse.addEventListener("open", () => {
        setConnectionOnline(true);
        checkSheetFreshness();
        refreshActions();
        refreshVerdictsProjection();
      });
      sse.addEventListener("resync", (ev) => {
        try {
          const d = JSON.parse(ev.data || "{}");
          if (d.reason === "server_restarted") {
            sessionExpired = true;
            setConnectionOnline(false);
            showToast("Workbench server restarted — reload this sheet to reconnect");
          }
        } catch (_) {}
        checkSheetFreshness();
        refreshActions();
        refreshVerdictsProjection();
      });
      sse.addEventListener("case_updated", (ev) => {
        try {
          const d = JSON.parse(ev.data || "{}");
          if (!d.case || d.case === META.case) checkSheetFreshness();
        } catch (_) {
          checkSheetFreshness();
        }
      });
      sse.addEventListener("report_actions", () => { refreshActions(); });
      sse.addEventListener("verdict", (ev) => {
        try {
          const d = JSON.parse(ev.data || "{}");
          if (!d.case || d.case === META.case) {
            checkSheetFreshness();
            refreshVerdictsProjection();
          }
        } catch (_) {
          checkSheetFreshness();
          refreshVerdictsProjection();
        }
      });
      sse.onerror = () => { setConnectionOnline(false); };
    } catch (_) {
      setConnectionOnline(false);
    }
  }

  async function initLive() {
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 1200);
      const r = await fetch(API_BASE + "/api/health", { signal: ctrl.signal, cache: "no-store" });
      clearTimeout(t);
      if (!r.ok) { document.body.classList.add("offline-static"); return; }
      const h = await r.json();
      if (!h || !h.ok) { document.body.classList.add("offline-static"); return; }
      // Prevent cross-workspace writes when an archived sheet collides on case slug!
      if (META.workspace_id && h.workspace_id && h.workspace_id !== META.workspace_id) {
        document.body.classList.add("offline-static");
        return;
      }
      connected = true;
      document.body.classList.add("live-connected");
      await Promise.all([refreshActions(), refreshVerdictsProjection()]);
      if (initParams.get("inbox") === "1") {
        $("agentTrayDrawer").classList.add("open");
      }
      connectSSE();
    } catch (_) {
      document.body.classList.add("offline-static");
    }
  }
  initLive();
})();
"""

