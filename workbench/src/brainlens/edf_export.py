"""Export one run's review annotations and, for continuous EDF, its native digital signals."""

from __future__ import annotations

import csv
import re
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import edfio
import numpy as np

from . import __version__
from .deid import recording_redactor, share_labels, trusted_participant_label
from .io_integrity import verified_source_snapshot, verify_source_recording_integrity
from .io_signals import _check_edf_bdf_continuity
from .models import RecordingSpec, ToolkitError, atomic_write_json, canonical_digest
from .storage import WorkbenchStore, classify_judgment_role, sanitize_slug


def _run_annotations(
    bundle: dict[str, Any],
    run_name: str,
    *,
    only_confirmed: bool,
    exclude_widespread_bursts: bool,
    include_burst_episodes: bool,
    channel_name: Callable[[str], str] | None = None,
    recording_label: str | None = None,
) -> list[edfio.EdfAnnotation]:
    marker_suffix = " export (not a medical device)"
    space = 80 - len("BrainLens ") - len(marker_suffix)
    marker_label = run_name[:space]
    if recording_label is not None:
        # Keep both labels recognizable when a long pseudonym or run hits the EDF text limit.
        rec_space = space - 1 - min(10, len(run_name))
        short_rec = _abbreviate(recording_label, rec_space)
        marker_label = f"{short_rec} {_abbreviate(run_name, space - len(short_rec) - 1)}"
    marker = edfio.EdfAnnotation(0.0, 0.0, f"BrainLens {marker_label}{marker_suffix}")
    findings: list[edfio.EdfAnnotation] = []
    for event in bundle["events"]:
        role = event.get("verdict_role") or event.get("clinician_role")
        verdict = None
        if role and classify_judgment_role(role) == "clinician":
            verdict = event.get("clinician_verdict")
        if only_confirmed and verdict != "confirm":
            continue
        synchrony = event.get("synchrony_class")
        if synchrony == "widespread_burst" and exclude_widespread_bursts:
            continue
        label = {"widespread_burst": "burst", "regional": "regional", "focal": "focal"}.get(
            synchrony, "unclassified"
        )
        prefix = f"BL HFO {label} "
        suffix = {"confirm": " conf", "artifact": " art", "uncertain": " unc"}.get(verdict, "")
        name = event["channel_name"]
        channel = re.sub(r"[^ -~]", "?", channel_name(name) if channel_name else name)[:80 - len(prefix) - len(suffix)]
        onset = event["start_seconds"]
        duration = round(event["end_seconds"] - onset, 12)
        findings.append(edfio.EdfAnnotation(onset, duration, f"{prefix}{channel}{suffix}"))

    if include_burst_episodes and not exclude_widespread_bursts:
        for episode in bundle["summary"].get("synchronous_burst_windows", []):
            onset = episode["start_seconds"]
            duration = round(episode["end_seconds"] - onset, 12)
            text = f"BL burst episode n_shafts={episode['shaft_count']}"
            findings.append(edfio.EdfAnnotation(onset, duration, text))
    return [marker, *sorted(findings)]


def _abbreviate(label: str, width: int) -> str:
    if len(label) <= width:
        return label
    return f"{label[:width - 5]}~{canonical_digest(label)[:4]}"


def _identity_header(path: Path) -> bytes:
    """Read the EDF patient, recording, start-date and start-time fields verbatim."""
    with path.open("rb") as fh:
        fh.seek(8)  # immediately after the version field
        return fh.read(80 + 80 + 8 + 8)


def _verify_edf(
    snapshot: Path, written: Path, expected: list[edfio.EdfAnnotation],
    expected_header: bytes, expected_labels: tuple[str, ...],
) -> None:
    source = edfio.read_edf(snapshot)
    result = edfio.read_edf(written)
    if (
        result.reserved != "EDF+C"
        or _identity_header(written) != expected_header
        or result.starttime != source.starttime
    ):
        raise ToolkitError("edf_verification_failed", "EDF+ identification or recording clock check failed.")
    if (
        result.labels != expected_labels
        or result.num_data_records != source.num_data_records
        or result.data_record_duration != source.data_record_duration
        or not result.is_continuous
    ):
        raise ToolkitError("edf_verification_failed", "EDF+ channel or data record layout changed during export.")
    for original, copy in zip(source.signals, result.signals, strict=True):
        if (
            original.physical_range != copy.physical_range
            or original.digital_range != copy.digital_range
            or original.physical_dimension != copy.physical_dimension
            or original.sampling_frequency != copy.sampling_frequency
            or not np.array_equal(
                original.get_digital_slice(0, source.duration), copy.get_digital_slice(0, result.duration)
            )
        ):
            raise ToolkitError("edf_verification_failed", "EDF+ signal calibration or digital samples changed during export.")
    actual = result.annotations
    if len(actual) != len(expected) or any(
        abs(a.onset - b.onset) > 1e-9 or a.duration != b.duration or a.text != b.text
        for a, b in zip(actual, sorted(expected))
    ):
        raise ToolkitError("edf_verification_failed", "EDF+ annotations changed during export.")


def _normalize_record_clock(edf: edfio.Edf) -> None:
    """EDF+C readers require exact decimal increments between record timestamps."""
    if edf.is_continuous:
        return
    # edfio 0.4 computes record clocks in binary floats (e.g. 3 * 0.1).
    # Its public annotations API omits the timekeeping TAL. Change only this first
    # TAL in each newly generated annotations record, retaining every event TAL.
    records = edf._timekeeping_signal.digital.reshape(edf.num_data_records, -1)
    offset = Decimal(edf.starttime.microsecond).scaleb(-6)
    step = Decimal(str(edf.data_record_duration))
    for i, record in enumerate(records):
        clock, separator, tail = record.tobytes().partition(b"\x14")
        exact = offset + i * step
        if Decimal(clock.decode("ascii")) == exact:
            continue
        timestamp = format(exact.normalize(), "+f").encode("ascii")
        payload = timestamp + separator + tail.rstrip(b"\x00") + b"\x00"
        if len(payload) > record.size:
            raise ToolkitError("edf_export_failed", "Cannot represent the EDF+ data record timestamp exactly.")
        record[:] = np.frombuffer(payload.ljust(record.size, b"\x00"), dtype=np.uint8)


def _redact_signal_headers(
    source: edfio.Edf, rec: RecordingSpec, redact: Callable[[str], str], channel_names: dict[str, str],
) -> None:
    """Redact identifying labels and descriptive fields without decoding or scaling samples."""
    for sig, ch in zip(source.signals, rec.channels, strict=True):
        if ch.name in channel_names:
            sig.label = channel_names[ch.name]
        for field in ("transducer_type", "prefiltering"):
            original = getattr(sig, field)
            masked = redact(original)
            if original != masked:
                setattr(sig, field, masked[:80])
        # This EDF field is reserved for future use and can contain private vendor text.
        sig._set_reserved("")


def _write_edf(
    rec: RecordingSpec, path: Path, annotations: list[edfio.EdfAnnotation],
    redact: Callable[[str], str] | None, channel_names: dict[str, str],
) -> dict[str, Any]:
    if rec.pieces:
        verify_source_recording_integrity(rec)
        return {
            "status": "unsupported_format",
            "reason": "A continuous EDF+ signal copy would lose the original data-record clock. "
                      "Use the annotation CSV in true recording time with the original signal.",
        }
    if not rec.path.lower().endswith(".edf"):
        return {"status": "unsupported_format", "reason": "Signal copy is available for EDF and EDF+C recordings."}
    with verified_source_snapshot(rec) as (snapshot, _, _, _):
        try:
            source = edfio.read_edf(snapshot)
            if source.reserved.startswith("EDF+D"):
                return {"status": "unsupported_format", "reason": "An EDF+D copy would lose the original data-record clock. Use the true-time annotation CSV."}
            _check_edf_bdf_continuity(snapshot)
            existing = source.annotations
            masked_annotations = 0
            identity = _identity_header(snapshot)
            if redact is not None:
                identity = b"X X X X".ljust(80) + b"Startdate X X X X".ljust(80) + b"01.01.85" + identity[-8:]
                masked = tuple(ann._replace(text=redact(ann.text)) for ann in existing)
                masked_annotations = sum(a != b for a, b in zip(existing, masked))
                existing = masked
                _redact_signal_headers(source, rec, redact, channel_names)
            labels = source.labels
            expected = [*existing, *annotations]
            # EdfSignal objects carry the original digital arrays and calibration.
            export = edfio.Edf(
                source.signals,
                starttime=source.starttime,
                data_record_duration=source.data_record_duration,
                annotations=expected,
            )
            _normalize_record_clock(export)
            export.write(path)
            if redact is None:
                # Plain EDF permits arbitrary text here. Parsing/re-encoding could change
                # those fields or their padding, so restore the original bytes from the snapshot.
                with path.open("r+b") as fh:
                    fh.seek(8)
                    fh.write(identity)
            del source, export  # release digital arrays before the independent read-back
            _verify_edf(snapshot, path, expected, identity, labels)
        except (ValueError, OSError, IndexError) as exc:
            raise ToolkitError("edf_export_failed", "Could not read, write or verify the EDF+ signal copy.") from exc
    return {
        "status": "written", "verified": True, "source_annotations_preserved": len(existing) - masked_annotations,
        "source_annotations_redacted": masked_annotations,
    }


def export_edf_annotations(
    case_slug: str,
    recording: str | None = None,
    run_name: str = "baseline",
    *,
    only_confirmed: bool = False,
    exclude_widespread_bursts: bool = False,
    include_burst_episodes: bool = True,
    deidentify: bool = False,
    pseudonym: str | None = None,
    workspace_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Export one run for an EEG reader; retain its local identity unless sharing is requested."""
    store = WorkbenchStore(workspace_dir)
    case = sanitize_slug(case_slug)
    recordings = store.list_recordings(case)
    if not recordings:
        raise ToolkitError("no_recordings", f"Case {case!r} has no recordings.")
    rslug = sanitize_slug(recording) if recording else recordings[0]
    run_name = sanitize_slug(run_name)
    bundle = store.load_run(case, rslug, run_name)
    rec = store.load_recording(case, rslug)
    meta = bundle["meta"]
    fingerprint = meta.get("recording_fingerprint") or bundle["summary"].get("recording_fingerprint")
    if fingerprint and fingerprint != rec.fingerprint:
        raise ToolkitError(
            "run_recording_mismatch",
            f"Run {run_name!r} for {case}/{rslug} belongs to a superseded recording snapshot.",
            recording=rslug,
            run_name=run_name,
        )

    filters = {
        "only_confirmed": only_confirmed,
        "exclude_widespread_bursts": exclude_widespread_bursts,
        "include_burst_episodes": include_burst_episodes,
    }
    case_display, rec_display, run_display = case, rslug, run_name
    redact = None
    channel_names: dict[str, str] = {}
    if deidentify:
        case_display, recording_labels = share_labels(case, recordings, pseudonym)
        rec_display = recording_labels[rslug]
        redact = recording_redactor([rec], aliases={case: case_display, rslug: rec_display}, pseudonym=pseudonym)
        run_display = sanitize_slug(redact(run_name))
        # Use short, distinct channel aliases when header identity appears in a label.
        # Match the label written to the EDF signal to the label used by its findings.
        reserved = {label.casefold() for ch in rec.channels for label in (ch.original_label, ch.name)}
        number = 1
        for ch in rec.channels:
            if redact(ch.original_label) != ch.original_label or redact(ch.name) != ch.name:
                while f"channel {number}" in reserved:
                    number += 1
                channel_names[ch.name] = f"Channel {number}"
                number += 1

    def display_channel(name: str) -> str:
        if name in channel_names:
            return channel_names[name]
        return redact(name) if redact else name

    annotations = _run_annotations(
        bundle, run_display, channel_name=display_channel, recording_label=rec_display if deidentify else None, **filters,
    )
    sidecar = {
        "case": case_display,
        "recording": rec_display,
        "run_name": run_display,
        "recording_fingerprint": rec.fingerprint,
        "origin_kind": rec.provenance.origin_kind,
        "analysis_generation_id": meta.get("analysis_generation_id") or meta.get("generation_id"),
        "generation_id": meta.get("generation_id"),
        "filters": filters,
        "brainlens_version": __version__,
        "deidentified": deidentify,
        "scope": "For research and review support only (not a medical device / not a standalone surgical resection boundary).",
        "time_basis": "Seconds from the start of the recording data.",
    }
    participant_label = trusted_participant_label(rec) if deidentify else rec.provenance.participant_id
    if segments := bundle["summary"].get("selection", {}).get("segments"):
        sidecar["analysis_segments"] = [
            {**seg, "label": f"Segment {i + 1}" if deidentify else seg["label"]} for i, seg in enumerate(segments)
        ]
    if participant_label:
        sidecar["participant_id"] = participant_label
    # Different run generations, selections and verdicts get stable, distinct download URLs.
    digest = canonical_digest([sidecar, annotations])[:12]
    prefix_limit = max(67, len(rec_display)) if deidentify else 67
    stem = f"{rec_display}_{run_display}"[:prefix_limit].rstrip("._-") + "_" + digest
    edf_name = f"{stem}_brainlens.edf"
    csv_name = f"{stem}_brainlens_annotations.csv"
    json_name = f"{stem}_brainlens_annotations.json"
    exports = store.case_dir(case) / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=exports, prefix=".edf-export-") as tmp:
        staging = Path(tmp)
        copy = _write_edf(rec, staging / edf_name, annotations, redact, channel_names)
        signal_written = copy["status"] == "written"
        with (staging / csv_name).open("w", encoding="ascii", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(("onset", "duration", "description"))
            writer.writerows(annotations)
        sidecar["edf_copy"] = copy
        sidecar["annotations_exported"] = len(annotations)
        if deidentify:
            sidecar["deidentification"] = (
                "De-identified export. Patient and recording identification and start date removed. "
                "Time of day retained. Header identity withheld from annotations and signal labels."
                if signal_written else "De-identified export. Header identity withheld from annotations. No signal copy exported."
            )
        atomic_write_json(staging / json_name, sidecar)
        if signal_written:
            (staging / edf_name).replace(exports / edf_name)
        (staging / csv_name).replace(exports / csv_name)
        (staging / json_name).replace(exports / json_name)
    return {
        "case": case_display,
        "recording": rec_display,
        "run_name": run_display,
        "edf_path": str(exports / edf_name) if signal_written else None,
        "csv_path": str(exports / csv_name),
        "json_path": str(exports / json_name),
        "annotations_exported": len(annotations),
        "edf_copy": copy,
        "deidentified": deidentify,
    }
