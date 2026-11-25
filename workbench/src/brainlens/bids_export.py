"""Standard BIDS *_events.tsv & *_events.json exporter for clinician-verified & screened HFOs."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from . import deid
from .anatomy import AnatomyMatcher, load_anatomy_labels
from .models import ToolkitError, atomic_write_json, utc_now
from .storage import WorkbenchStore, sanitize_slug


def export_bids_events(
    case_slug: str,
    recording: str | None = None,
    run_name: str = "baseline",
    only_confirmed: bool = False,
    exclude_widespread_bursts: bool = True,
    workspace_dir: str | Path | None = None,
    *,
    deidentify: bool = False,
    pseudonym: str | None = None,
) -> dict[str, Any]:
    """Export HFO candidates/verdicts as BIDS-compliant events.tsv + events.json sidecar."""
    store = WorkbenchStore(workspace_dir)
    cslug = sanitize_slug(case_slug)
    recs = store.list_recordings(cslug)
    if not recs:
        raise ToolkitError("no_recordings", f"Case {cslug!r} has no recordings.")
    rslug = sanitize_slug(recording) if recording else recs[0]
    if not store.has_run(cslug, rslug, run_name):
        raise ToolkitError("run_not_found", f"Run {run_name!r} not found for {cslug}/{rslug}.")

    bundle = store.load_run(cslug, rslug, run_name)
    rec_model = store.load_recording(cslug, rslug)
    meta = bundle.get("meta") or {}
    summary = bundle["summary"]
    run_fp = meta.get("recording_fingerprint") or summary.get("recording_fingerprint")
    if run_fp and run_fp != rec_model.fingerprint:
        raise ToolkitError(
            "run_recording_mismatch",
            f"Run {run_name!r} for {cslug}/{rslug} belongs to superseded recording snapshot {str(run_fp)[:12]} "
            f"(current registered recording is {rec_model.fingerprint[:12]}); re-analyze before exporting BIDS events.",
            recording=rslug,
            run_name=sanitize_slug(run_name),
        )
    events = bundle["events"]
    anatomy_doc = load_anatomy_labels(store, cslug)
    anatomy = AnatomyMatcher(anatomy_doc)
    channel_anatomy = {
        row["channel_id"]: anatomy.match(row["channel_name"], row.get("contacts", []))
        for row in summary.get("channels", [])
    }

    selected: list[dict[str, Any]] = []
    for ev in events:
        if exclude_widespread_bursts and ev.get("synchrony_class") == "widespread_burst":
            continue
        if only_confirmed and ev.get("clinician_verdict") != "confirm":
            continue
        selected.append(ev)

    exp_dir = store.case_dir(cslug) / "exports"
    exp_dir.mkdir(parents=True, exist_ok=True)
    if only_confirmed:
        suffix = "_confirmed_focal" if exclude_widespread_bursts else "_confirmed_all"
    else:
        suffix = "_focal" if exclude_widespread_bursts else "_all"
    case_label, rec_label = cslug, rslug
    display_run = sanitize_slug(run_name)
    redact = None
    if deidentify:
        case_label, labels = deid.share_labels(cslug, recs, pseudonym)
        rec_label = labels[rslug]
        redact = deid.recording_redactor([rec_model], aliases={cslug: case_label, rslug: rec_label}, pseudonym=pseudonym)
        display_run = sanitize_slug(redact(display_run))
        # Both pieces are already bounded. Preserve the entire user-supplied pseudonym.
        base_name = f"{rec_label}_{display_run}{suffix}_desc-deidentified"
    else:
        raw_stem = sanitize_slug(f"{rslug}_{display_run}")
        max_prefix = max(16, 80 - len(suffix))
        if len(raw_stem) > max_prefix:
            h8 = hashlib.sha256(f"{rslug}|{run_name}".encode("utf-8")).hexdigest()[:8]
            raw_stem = f"{raw_stem[:max(1, max_prefix - 9)].rstrip('_')}_{h8}"
        base_name = sanitize_slug(f"{raw_stem}{suffix}")
    tsv_path = exp_dir / f"{base_name}_events.tsv"
    json_path = exp_dir / f"{base_name}_events.json"

    fieldnames = [
        "onset",
        "duration",
        "trial_type",
        "channel",
        "shaft",
        "synchrony_class",
        "coactive_shaft_count",
        "dominant_band",
        "peak_freq_hz",
        "peak_to_peak_uv",
        "artifact_score",
        "spike_score",
        "ehfo_score",
        "clinician_verdict",
        "clipped",
        "event_id",
        "provenance_origin",
    ]
    if anatomy_doc:
        fieldnames.append("anatomy")

    with tsv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for ev in selected:
            onset = float(ev["start_seconds"])
            dur = max(0.0, float(ev["end_seconds"]) - onset)
            scores = ev.get("scores") or {}
            labels = ev.get("labels") or {}
            verdict = ev.get("clinician_verdict")
            band = ev.get("dominant_band", "ripple")
            if verdict == "artifact":
                ttype = "hfo_artifact_rejected"
            elif verdict == "confirm":
                ttype = f"hfo_confirmed_{band}"
            elif verdict == "uncertain":
                ttype = f"hfo_uncertain_{band}"
            elif labels.get("ehfo") == 1:
                ttype = f"hfo_ehfo_{band}"
            elif ev.get("synchrony_class") == "widespread_burst":
                ttype = "hfo_widespread_burst"
            else:
                ttype = f"hfo_candidate_{band}"

            def _na(val: Any) -> Any:
                return "n/a" if val is None or val == "" else val

            row = {
                "onset": f"{onset:.6f}",
                "duration": f"{dur:.6f}",
                "trial_type": ttype,
                "channel": ev["channel_name"],
                "shaft": _na(ev.get("shaft")),
                "synchrony_class": ev.get("synchrony_class", "focal"),
                "coactive_shaft_count": ev.get("coactive_shaft_count", 1),
                "dominant_band": _na(ev.get("dominant_band")),
                "peak_freq_hz": _na(ev.get("peak_freq_hz")),
                "peak_to_peak_uv": _na(ev.get("peak_to_peak_uv")),
                "artifact_score": _na(scores.get("artifact")),
                "spike_score": _na(scores.get("spike")),
                "ehfo_score": _na(scores.get("ehfo")),
                "clinician_verdict": _na(ev.get("clinician_verdict")),
                "clipped": "true" if ev.get("clipped") else "false",
                "event_id": ev["id"],
                "provenance_origin": rec_model.provenance.origin_kind,
            }
            if redact:
                for key in ("channel", "shaft", "event_id"):
                    row[key] = redact(row[key])
            if anatomy_doc:
                matched = channel_anatomy.get(ev["channel_id"]) or {}
                row["anatomy"] = _na(matched.get("display_label"))
            writer.writerow(row)

    sidecar = {
        "GeneratedBy": [{"Name": "BrainLens Neurophysiology Workbench", "Version": "1.0.0"}],
        "GeneratedAt": utc_now(),
        "Case": case_label,
        "Recording": rec_label,
        "RecordingFingerprint": rec_model.fingerprint,
        "RunName": display_run,
        "AnalysisGenerationId": meta.get("generation_id"),
        "AcquiredReference": summary["acquired_reference"],
        "AnalyzedReference": summary.get("effective_reference_label") or summary["effective_reference"]["mode"],
        "FilterSelection": {
            "only_confirmed": only_confirmed,
            "exclude_widespread_bursts": exclude_widespread_bursts,
        },
        "onset": {"Description": "Event onset relative to original recording start", "Units": "s"},
        "duration": {"Description": "Candidate HFO duration", "Units": "s"},
        "trial_type": {"Description": "Stratified HFO event label (candidate, ONNX eHFO, clinician-confirmed, or widespread burst)"},
        "channel": {"Description": "Analyzed channel or bipolar derivation name"},
        "synchrony_class": {"Description": "Cross-shaft synchrony class within +-30 ms (focal, regional, widespread_burst)"},
        "dominant_band": {"Description": "Dominant HFO spectral band (ripple 80-250 Hz vs fast_ripple 250-450 Hz)"},
        "peak_freq_hz": {"Description": "Peak spectral power frequency inside HFO band", "Units": "Hz"},
        "peak_to_peak_uv": {"Description": "Peak-to-peak amplitude of prepared snippet", "Units": "uV"},
        "artifact_score": {"Description": "Artifact model probability that the event is a real HFO (non-artifact)"},
        "spike_score": {"Description": "Spike model probability that a spike accompanies the HFO"},
        "ehfo_score": {"Description": "eHFO model probability that the event is an epileptogenic HFO"},
        "clinician_verdict": {"Description": "Human visual verification verdict (confirm, artifact, uncertain, or n/a)"},
    }
    if redact:
        sidecar["AcquiredReference"] = redact(sidecar["AcquiredReference"])
        sidecar["AnalyzedReference"] = redact(sidecar["AnalyzedReference"])
        participant = deid.trusted_participant_label(rec_model)
        if participant:
            sidecar["ParticipantID"] = participant
        sidecar["Deidentified"] = True
        sidecar["Deidentification"] = (
            "De-identified export. Native patient names, codes and birth dates are withheld. "
            "Sharing labels replace source case and recording names."
        )
    if anatomy_doc:
        sidecar["anatomy"] = {
            "Description": "Anatomical label supplied by the external localization source",
            "Source": anatomy_doc["source"],
            "PatientNative": anatomy_doc["patient_native"],
            "Atlas": anatomy_doc.get("atlas"),
        }
    atomic_write_json(json_path, sidecar)

    return {
        "case": case_label,
        "recording": rec_label,
        "run_name": display_run,
        "deidentified": deidentify,
        "events_exported": len(selected),
        "tsv_path": str(tsv_path),
        "json_path": str(json_path),
        "only_confirmed": only_confirmed,
        "exclude_widespread_bursts": exclude_widespread_bursts,
    }
