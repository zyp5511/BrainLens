"""Recording-specific MDT presentation built from the case sheet's verified evidence."""
from __future__ import annotations

import base64
import io
import tempfile
import textwrap
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import __version__
from .analytics import compute_overlay_concordance
from .case_sheet import CaseEvidence, gather_case_evidence, reader_agreement_reports
from .clinical_display import model_score_labels
from .models import ToolkitError
from .storage import sanitize_slug
from .triad_views import render_evidence_card


_SCOPE = "Quantitative neurophysiology review support — not a medical device / not a standalone surgical resection boundary"
_VERDICTS = ("confirm", "artifact", "uncertain")
_INK, _TEAL, _MUTED = "172D43", "146E71", "435B69"


def _channel(row: dict[str, Any]) -> str:
    name = row.get("channel_name") or row["channel"]
    label = (row.get("anatomy") or {}).get("display_label")
    return f"{name} · {label}" if label else name


def _anatomy_source(e: CaseEvidence) -> str:
    labels = e.anatomy_labels
    if not labels:
        return ""
    return (f"External anatomy labels · source={labels['source']} · "
            f"patient-native={'yes' if labels['patient_native'] else 'no'} · atlas={labels.get('atlas') or 'unspecified'}")


class _Deck:
    """Use one disclosure boundary for slide text, table cells, and speaker notes."""

    def __init__(self) -> None:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.enum.text import MSO_AUTO_SIZE
        from pptx.util import Inches, Pt

        self.inches, self.pt, self.rgb = Inches, Pt, RGBColor.from_string
        self.autofit = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(40 / 3), Inches(7.5)
        self.prs.core_properties.author = "BrainLens"
        self.prs.core_properties.title = "MDT case discussion"
        self.prs.core_properties.comments = ""
        self.display: Callable[[str], str] = str
        self.recording_labels: dict[str, str] = {}
        self.image_hashes: set[str] = set()

    def context(self, recording: str, run: str = "") -> str:
        label = self.recording_labels[recording]
        return f"{label}   /   {self.display(run)}" if run else label

    def text(self, slide, value: str, x: float, y: float, w: float, h: float,
             size: int, *, bold: bool = False, color: str = _INK, disclosed: bool = False) -> None:
        frame = slide.shapes.add_textbox(*(self.inches(v) for v in (x, y, w, h))).text_frame
        frame.margin_top = frame.margin_bottom = frame.margin_left = frame.margin_right = 0
        frame.word_wrap = True
        frame.auto_size = self.autofit
        frame.text = value if disclosed else self.display(value)
        for p in frame.paragraphs:
            p.font.name = "Arial"
            p.font.size, p.font.bold, p.font.color.rgb = self.pt(size), bold, self.rgb(color)

    def slide(self, title: str, context: str = "", *, anatomy: str = ""):
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.text(slide, context or "BRAINLENS   /   MDT REVIEW", 0.58, 0.3, 12.2, 0.35, 15, color=_TEAL, disclosed=True)
        self.text(slide, title, 0.56, 0.82, 12.25, 0.65, 33, bold=True)
        if anatomy:
            self.text(slide, anatomy, 0.58, 6.88, 11.6, 0.47, 14, color=_MUTED)
        self.text(slide, str(len(self.prs.slides)), 12.52, 6.96, 0.65, 0.3, 13, color=_MUTED)
        return slide

    def notes(self, slide, text: str) -> None:
        slide.notes_slide.notes_text_frame.text = f"{slide.shapes[0].text}\n{self.display(text)}"

    def picture(self, slide, image: dict, x: float, y: float, w: float, h: float) -> None:
        # Embed the exact rendered bytes. Local card paths can be overwritten by later requests.
        stream = io.BytesIO(base64.b64decode(image["png_base64"], validate=True))
        pic = slide.shapes.add_picture(stream, self.inches(x), self.inches(y))
        scale = min(self.inches(w) / pic.width, self.inches(h) / pic.height)
        pic.width, pic.height = int(pic.width * scale), int(pic.height * scale)
        pic.left += (self.inches(w) - pic.width) // 2
        pic.top += (self.inches(h) - pic.height) // 2
        self.image_hashes.add(pic.image.sha1)

    def table(self, slide, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
        t = slide.shapes.add_table(len(rows) + 1, len(headers), self.inches(0.55), self.inches(1.83),
                                   self.inches(12.2), self.inches(0.66 * (1 + len(rows)))).table
        for col, width in zip(t.columns, widths):
            col.width = self.inches(12.2 * width / sum(widths))
        for i, row in enumerate([headers, *rows]):
            for j, value in enumerate(row):
                cell = t.cell(i, j)
                cell.fill.solid()
                cell.fill.fore_color.rgb = self.rgb(_INK if i == 0 else ("EAF1F2" if i % 2 else "F8FAFA"))
                cell.margin_left = cell.margin_right = self.inches(0.07)
                cell.margin_top = cell.margin_bottom = self.inches(0.04)
                cell.text = self.display(value)
                frame = cell.text_frame
                frame.word_wrap, frame.auto_size = True, self.autofit
                for p in frame.paragraphs:
                    p.font.name = "Arial"
                    p.font.size = self.pt(15 if i == 0 else 16)
                    p.font.color.rgb = self.rgb("FFFFFF" if i == 0 else _INK)
                    p.font.bold = i == 0

    def paragraphs(self, title: str, lines: list[str], context: str = "", anatomy: str = "",
                   detail: str = "", *, disclosed: bool = False) -> None:
        # Paginate long question lists and QC warnings instead of reducing body type.
        pages: list[list[str]] = [[]]
        for line in lines:
            # Apply disclosure before splitting a name across lines or pages.
            wrapped = textwrap.wrap(line if disclosed else self.display(line), width=85) or [""]
            if pages[-1] and len(pages[-1]) + len(wrapped) > 12:
                pages.append([])
            for fragment in wrapped:
                if len(pages[-1]) == 12:
                    pages.append([])
                pages[-1].append(fragment)
        for page in pages:
            body = "\n".join(page)
            slide = self.slide(title, context, anatomy=anatomy)
            self.text(slide, body, 0.6, 1.68, 12.08, 4.98, 21, disclosed=True)
            slide.notes_slide.notes_text_frame.text = "\n".join(filter(None, [
                context, body, self.display(anatomy), self.display(detail)]))


def _title_and_questions(deck: _Deck, e: CaseEvidence) -> None:
    dates = sorted({str(b["meta"].get("created_at") or "unavailable")[:10] for b in e.bundles.values()})
    date_label = dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"
    slide = deck.slide("Presurgical MDT case discussion")
    deck.text(slide, e.case_label, 0.55, 2.0, 12.2, 1.0, 48, bold=True, disclosed=True)
    deck.text(slide, f"Primary run   {e.primary_run}\nAnalysis date   {date_label}", 0.6, 3.35, 12, 1.0, 24)
    deck.text(slide, _SCOPE, 0.6, 5.7, 11.8, 0.95, 21, color=_TEAL, disclosed=True)
    slide.notes_slide.notes_text_frame.text = f"{len(e.recs)} recording(s) in {deck.display(e.primary_run)}.\n" + "\n".join(
        f"{deck.context(r)} · {deck.display(e.bundles[r]['meta'].get('created_at') or 'Analysis date unavailable')}"
        for r in e.recs)

    findings = []
    for r in e.recs:
        s = e.bundles[r]["summary"]
        leaders = s.get("leading_focal_channels") or []
        if leaders:
            lead = leaders[0]
            tied = [row for row in leaders if row['focal_rank'] == lead['focal_rank']]
            names = ", ".join(_channel(row) for row in tied[:3])
            if len(tied) > 3:
                names += f" and {len(tied) - 3} others"
            result = f"{names} led at {lead['focal_rate_per_min']:.2f} focal candidates/min"
        else:
            result = "no focal leader was measured"
        c = s["counts"]
        bursts = (f"{c['widespread_burst_events']}/{c['total_raw_events']} "
                  f"({c['widespread_burst_fraction']:.1%}) raw candidates occurred in widespread bursts"
                  if c["total_raw_events"] else "there were no raw candidates to classify for synchrony")
        findings.append(f"In {deck.context(r)}, {deck.display(result)}. {bursts[0].upper() + bursts[1:]}.")
    for pair in (e.cross_rec or {}).get("pairwise", []):
        findings.append(f"{deck.context(pair['recording_a'])} and {deck.context(pair['recording_b'])} shared "
                        f"{len(pair['focal_top5_shared'])} top-five focal leaders across "
                        f"{pair['shared_measured_channels']} jointly measured channels. Rates are reported separately.")
    for cmp in e.branch_cmps:
        sm = cmp.get("summary_metrics")
        if cmp["directly_comparable_rates"]:
            findings.append(f"For {deck.context(cmp['recording'])}, switching from {deck.display(cmp['run_a'])} "
                            f"to {deck.display(cmp['run_b'])} changed "
                            f"the non-burst count from {sm['focal_events_a']} to {sm['focal_events_b']}, "
                            f"with {len(sm['top5_shared'])} shared top-five leaders.")
        else:
            a, b = cmp.get("run_a_leading"), cmp.get("run_b_leading")
            findings.append(f"For {deck.context(cmp['recording'])}, {deck.display(cmp['run_a'])} and "
                            f"{deck.display(cmp['run_b'])} use different derivations. Their respective focal leaders "
                            f"are {deck.display(_channel(a[0])) if a else 'none'} and "
                            f"{deck.display(_channel(b[0])) if b else 'none'}. Per-channel rate subtraction is undefined.")
    if e.comparison_run and not e.branch_cmps:
        findings.append(f"No primary recording has a paired {deck.display(e.comparison_run)} run.")
    questions = [deck.display(q) for q in e.open_questions] or ["No clinical question recorded."]
    deck.paragraphs("Clinical questions and bottom line", [*questions, "", " ".join(findings)],
                    anatomy=_anatomy_source(e), disclosed=True)


def _spatial(deck: _Deck, e: CaseEvidence) -> None:
    fig = e.spatial_fig
    if not fig:
        return
    r = fig["recording"]
    spatial = e.bundles[r]["summary"]["spatial"]
    caption = f"{fig.get('coordinate_system')} · {fig.get('mapped_contacts')} mapped contacts"
    if fig.get("template"):
        caption += " · TEMPLATE coordinates (non-patient-native)"
    regions: dict[str, list[str]] = {}
    for row in spatial.get("contacts", []):
        if row.get("anatomy"):
            regions.setdefault(row["anatomy"]["display_label"], []).append(row["channel_name"])
    legend = [f"{region}   {', '.join(names)}" for region, names in regions.items()]
    slide = deck.slide("Spatial distribution of focal candidates", deck.context(r, e.primary_run),
                       anatomy=_anatomy_source(e))
    deck.picture(slide, fig, 0.55, 1.52, 8.3 if legend else 12.2, 4.42)
    deck.text(slide, caption, 0.6, 6.03, 12.1, 0.37, 17, bold=True)
    if legend:
        deck.text(slide, "\n\n".join(legend[:4]), 9.0, 1.68, 3.8, 4.12, 17)
    details = []
    for row in spatial.get("contacts", []):
        rate = row.get("focal_rate_per_min")
        measure = f"{rate:.2f} focal/min" if rate is not None else "unassessed"
        details.append(f"{_channel(row)} · {row['xyz']} {spatial.get('units') or ''} · {measure}")
    deck.notes(slide, "\n".join([caption, str(spatial.get("source") or ""), _anatomy_source(e), *details]))
    if len(legend) > 4:
        deck.paragraphs("Spatial contact labels", legend[4:], deck.context(r, e.primary_run), _anatomy_source(e))


def _leaders(deck: _Deck, e: CaseEvidence, recording: str) -> None:
    bundle = e.bundles[recording]
    rows = bundle["summary"].get("leading_focal_channels", [])[:5]
    overlay = e.overlay_conc if recording == e.recs[0] else compute_overlay_concordance(
        e.store, e.case_slug, recording, e.primary_run, overlay=e.overlay_doc, bundle=bundle)
    tallies = Counter((ev.get("channel_name"), key, ev.get(key + "_verdict"))
                      for ev in bundle["events"] for key in e.review_roles)
    headers = ["Rank", "Channel / anatomy", "Focal\nper min", *e.review_roles.values()]
    widths = [0.66, 2.7, 1.0, *(1.6 if k == "rehearsal" else 1.15 for k in e.review_roles)]
    if overlay.get("available"):
        headers.append("External\nreference")
        widths.append(1.35)
    table_rows, details = [], []
    for row in rows:
        name = row["channel_name"]
        cells = [str(row["focal_rank"]), _channel(row), f"{row['focal_rate_per_min']:.2f}"]
        for key in e.review_roles:
            cells.append(" / ".join(str(tallies[name, key, v]) for v in _VERDICTS))
        if overlay.get("available"):
            cells.append(", ".join(overlay["per_channel_tags"].get(name, [])) or "—")
        table_rows.append(cells)
        details.append(f"{' | '.join(cells)} · {row['focal_count']} focal / {row['detector_analyzed_seconds']} s")
    slide = deck.slide("Leading focal contacts", deck.context(recording, e.primary_run), anatomy=_anatomy_source(e))
    if table_rows:
        deck.table(slide, headers, table_rows, widths)
    else:
        deck.text(slide, "No channel had a positive measured focal rate.", 0.6, 2.1, 12.0, 0.8, 24)
    legend = "Role tallies count confirm / artifact / uncertain verdicts across all candidates on each channel."
    if overlay.get("available"):
        legend += f" External reference from {e.overlay_doc['source']}."
    deck.text(slide, legend, 0.6, 6.05, 12.1, 0.77, 16)
    deck.notes(slide, "\n".join([" | ".join(headers), *details, legend, _anatomy_source(e),
                                  f"{len(table_rows)} active leaders shown."]))


def _triad(deck: _Deck, e: CaseEvidence, card: dict) -> None:
    ev = card.get("focus_event") or {}
    peak = card.get("peak_hfo_frequency_hz")
    start, end = card["window_seconds"]
    caption = [_channel(card), f"Window   {start:.3f}–{end:.3f} s",
               f"Peak frequency   {peak:g} Hz" if peak is not None else "Peak frequency   unavailable"]
    if card.get("interval_seconds"):
        t0, t1 = card["interval_seconds"]
        caption.append(f"Clipped interval   {t0:.3f}–{t1:.3f} s")
    if ev:
        if ev.get("start_seconds") is not None:
            caption.append(f"Event onset   {ev['start_seconds']:.3f} s")
        caption.extend(model_score_labels(ev))
        caption.extend(f"{role}: {ev.get(key + '_verdict') or 'unreviewed'}" for key, role in e.review_roles.items())
    else:
        caption.append("Control window · no event verdict" if card.get("view_mode") == "background_control"
                       else "Selected interval · no event verdict")
    context = "Clinical Triad   /   " + deck.context(card["recording"], card["run_name"])
    slide = deck.slide(card.get("source_label") or "Clinical Triad", context, anatomy=_anatomy_source(e))
    deck.picture(slide, card, 0.45, 1.5, 7.88, 5.27)
    deck.text(slide, "\n".join(caption), 8.51, 1.77, 4.24, 4.96, 17)
    deck.notes(slide, "\n".join([*caption, _anatomy_source(e), f"Event ID   {ev.get('id') or 'none'}"]))


def _maps(deck: _Deck, e: CaseEvidence) -> None:
    for recording, evidence in e.evidence_maps.items():
        channels = {row["channel_name"]: row for row in e.bundles[recording]["summary"]["channels"]}
        rows = evidence["rows"]
        for start in range(0, max(len(rows), 1), 12):
            block, notes = [], []
            for row in rows[start:start + 12]:
                channel = _channel(channels.get(row["channel"], row))
                block.append({**row, "channel": channel})
                cells = []
                for col in evidence["columns"]:
                    cell = row["biomarkers"][col["key"]]
                    value = cell["reason"] if cell["value"] is None else f"{cell['value']:.4g} {col['unit']}"
                    cells.append(f"{col['label']} {value}")
                notes.append(f"{channel} · " + " · ".join(cells))
            image = render_evidence_card(
                f"{e.case_slug}/{recording}", workspace_dir=e.store.root, recording=recording,
                run_name=e.primary_run, event_id="evidence_map", return_base64=True,
                display_text=e.display_text, evidence_snapshot={**evidence, "rows": block},
                recording_labels=e.recording_labels,
            )
            caption = "Convergence count is descriptive, not a validated predictor. IED candidates require visual verification."
            slide = deck.slide("Multi-biomarker evidence map", deck.context(recording, e.primary_run),
                               anatomy=_anatomy_source(e))
            deck.picture(slide, image, 0.55, 1.45, 12.2, 4.74)
            deck.text(slide, caption, 0.6, 6.22, 12.1, 0.55, 17)
            deck.notes(slide, "\n".join([evidence["provenance"]["summary"], *notes,
                                          evidence["top_tier_rule"], caption, _anatomy_source(e)]))
            source = evidence.get("ei_source")
            if source:
                slide.notes_slide.notes_text_frame.text += (
                    f"\nIctal EI · {deck.context(source['recording_id'])} at {source['onset_seconds']} s · "
                    f"{deck.display(source['provenance']['summary'])}"
                )


def _agreement(deck: _Deck, e: CaseEvidence) -> None:
    reports = reader_agreement_reports(e.store, e.case_slug, sanitize_slug(e.primary_run),
                                       e.bundles, e.judgments, e.binding_ctx)
    for report in reports:
        rows, details = [], []
        for pair in report["pairwise"]:
            if not pair["n_shared"]:
                continue
            k = pair["kappa"]
            ci = k["ci95"]
            readers = " / ".join(f"{r} ({report['reader_roles'][r]})" for r in pair["readers"])
            estimate = f"{k['value']:.3f}" if k["value"] is not None else "Unavailable"
            interval = f"{ci[0]:.3f} to {ci[1]:.3f}" if ci is not None else (k.get("reason") or "Unavailable")
            rows.append([readers, str(pair["n_shared"]), f"{pair['percent_agreement']:.1f}%", estimate, interval])
            details.append(f"{readers} · confusion matrix {pair['confusion_matrix']}")
        headers = ["Readers", "Shared events", "Agreement", "Cohen’s κ", "95% CI"]
        for start in range(0, len(rows), 5):
            page = rows[start:start + 5]
            slide = deck.slide("Inter-rater agreement", deck.context(report["recording"], report["run_name"]))
            deck.table(slide, headers, page, [4.1, 1.5, 1.5, 1.45, 2.5])
            deck.text(slide, "Latest human ratings on the same events. Categories are confirm, artifact and uncertain.",
                      0.6, 6.14, 12, 0.65, 17)
            deck.notes(slide, "\n".join([report["ci_method"], " | ".join(headers),
                                          *(" | ".join(row) for row in page), *details[start:start + 5]]))


def _limits_and_provenance(deck: _Deck, e: CaseEvidence) -> list[str]:
    warnings = []
    for r in e.recs:
        s = e.bundles[r]["summary"]
        channels = {row["channel_name"]: row for row in s["channels"]}
        unassessed = [_channel(channels.get(ch, {"channel": ch})) for ch in s.get("unassessed_channel_names", [])]
        ref = s.get("effective_reference_label") or (s.get("effective_reference") or {}).get("mode", "unavailable")
        sel = s.get("selection") or {}
        spatial = s.get("spatial") or {}
        frame = (f"{spatial.get('coordinate_system')} · {spatial.get('mapped_count')} mapped contacts"
                 if spatial.get("available") else "No mapped electrode coordinates.")
        if spatial.get("template"):
            frame += " · TEMPLATE (non-patient-native)"
        elif spatial.get("available") and spatial.get("source"):
            frame += f" · {spatial['source']}"
        coverage = (f"in {len(sel['segments'])} selected segments" if sel.get("segments") else
                    f"over {sel.get('start_seconds', '—')}–{sel.get('end_seconds', '—')} s")
        lines = [f"Coverage   {s['counts']['measured_channels']} of {len(channels)} channels measured {coverage}.",
                 f"Unassessed channels   {', '.join(unassessed) or 'none'}.",
                 f"Reference   acquired {s.get('acquired_reference', 'unknown')}, analyzed {ref}.",
                 f"Spatial coverage   {frame}",
                 f"Detector   {s.get('detector_status', 'unknown')}   /   classifier   {s.get('classifier_status', 'unknown')}"]
        for i, seg in enumerate(sel.get("segments", []), 1):
            label = f"Segment {i}" if e.display_text else seg["label"]
            lines.append(f"{label}   {seg['start_seconds']:g}–{seg['end_seconds']:g} s")
        qc = s.get("qc_overview") or {}
        for key, label in (("high_line_noise_channels", "Elevated line noise"),
                           ("flat_window_channels", "Flat windows"), ("amplitude_warning_channels", "Amplitude warnings")):
            if qc.get(key):
                names = [_channel(channels.get(ch, {"channel": ch})) for ch in qc[key]]
                lines.append(f"{label}   {', '.join(names)}")
        for msg in s.get("warnings", []):
            lines.append(str(msg))
            warnings.append(f"{deck.context(r)} · {deck.display(str(msg))}")
        lines.extend(e.rec_models[r].provenance.limitations)
        deck.paragraphs("QC, coverage and limitations", lines, deck.context(r, e.primary_run), _anatomy_source(e))

    sig = e.rendered_signature
    sources = [doc["ei_source"]["recording_id"] for doc in e.evidence_maps.values() if doc.get("ei_source")]
    for r in dict.fromkeys([*e.recs, *(card["recording"] for card in e.cards), *sources]):
        rec = e.rec_models[r]
        p = rec.provenance
        origin = {"public_dataset": "Real public dataset", "synthetic_engineering_data": "Synthetic engineering data",
                  "local_recording": "Local recording"}.get(p.origin_kind, p.origin_kind)
        lines = [f"Origin   {origin}", p.summary]
        if p.dataset_name and p.dataset_name not in p.summary:
            lines.append(f"Dataset   {p.dataset_name}")
        if p.participant_id and p.participant_id not in p.summary:
            lines.append(f"Participant   {p.participant_id}")
        source_ids = [x for x in (p.doi, p.license) if x and x not in p.summary]
        if source_ids:
            lines.append(" · ".join(source_ids))
        lines += [f"Recording fingerprint   {rec.fingerprint[:16]}", f"BrainLens version   {__version__}"]
        for run, entries in [(e.primary_run, sig["primary"]), (e.comparison_run, sig["comparison"]),
                             *sig["referenced_runs"].items()]:
            if run and r in entries:
                lines.append(f"{run}   /   analysis generation   {entries[r].get('analysis_generation_id') or 'unavailable'}")
        deck.paragraphs("Recording provenance", lines, deck.context(r), detail=f"Full recording fingerprint   {rec.fingerprint}")
    return warnings


def export_mdt_slides(
    case_slug: str, workspace_dir: str | Path | None = None, primary_run: str = "baseline",
    comparison_run: str | None = None, clinical_question: str | None = None, max_cards: int = 6,
    *, deidentify: bool = False, pseudonym: str | None = None,
) -> dict[str, Any]:
    """Export an editable 16:9 PowerPoint deck from the same evidence as the case sheet."""
    try:
        deck = _Deck()
    except ImportError as exc:
        raise ToolkitError("slides_dependency_missing", "Install python-pptx>=1.0,<2 to export MDT slides.") from exc
    if isinstance(max_cards, bool) or not isinstance(max_cards, int) or max_cards < 0:
        raise ToolkitError("invalid_max_cards", "max_cards must be a non-negative integer.")
    e = gather_case_evidence(case_slug, workspace_dir, primary_run, comparison_run, clinical_question, max_cards,
                             deidentify=deidentify, pseudonym=pseudonym, reserve_control=True, first_available_spatial=True)
    deck.display = e.display_text or str
    deck.recording_labels = e.recording_labels
    _title_and_questions(deck, e)
    _spatial(deck, e)
    for recording in e.recs:
        _leaders(deck, e, recording)
    for card in e.cards:
        _triad(deck, e, card)
    _maps(deck, e)
    _agreement(deck, e)
    warnings = _limits_and_provenance(deck, e)
    if max_cards and not any(c.get("view_mode") == "background_control" for c in e.cards):
        warnings.append("No candidate-free control window was available.")
    if not e.spatial_fig and any(b["summary"].get("spatial", {}).get("available") for b in e.bundles.values()):
        warnings.append("The spatial figure could not be rendered.")

    exp_dir = e.store.case_dir(e.case_slug) / "exports"
    exp_dir.mkdir(parents=True, exist_ok=True)
    ending = "_mdt_slides.shared.pptx" if deidentify else "_mdt_slides.pptx"
    out = exp_dir / f"{e.case_label}_{sanitize_slug(deck.display(primary_run))}{ending}"
    with tempfile.NamedTemporaryFile(dir=exp_dir, prefix=".mdt-", suffix=".tmp", delete=False) as f:
        temp = Path(f.name)
    try:
        deck.prs.save(temp)
        # The triad renderer reads its run on demand. Reject a concurrent overwrite
        # before publishing slides with tables and figures from different generations.
        current = e.store.case_sheet_run_signature(e.case_slug, primary_run, e.comparison_run,
                                                   referenced_runs=list(e.rendered_signature["referenced_runs"]))
        if current != e.rendered_signature:
            raise ToolkitError("case_evidence_changed", "Case evidence changed during slide export. Retry the export.")
        temp.replace(out)
    finally:
        temp.unlink(missing_ok=True)
    return {"case": e.case_label, "slides_path": str(out), "slide_count": len(deck.prs.slides),
            "embedded_images": len(deck.image_hashes), "primary_run": deck.display(primary_run),
            "comparison_run": deck.display(e.comparison_run) if e.comparison_run else None, "warnings": warnings}
