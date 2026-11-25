"""Within-case navigation on acquisition clocks. File-local signal time always starts at zero."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .models import RecordingSpec, ToolkitError
from .storage import WorkbenchStore


def header_clock(raw: Any, path: Path) -> dict[str, str | None]:
    measured = raw.info.get("meas_date")
    if measured is None:
        return {"start": None}
    # FIF first_samp and an EDF+ first-record TAL both move the first stored sample.
    try:
        first = measured + timedelta(seconds=float(raw.first_time) + getattr(raw, "_brainlens_first_tal_seconds", 0.0))
    except OverflowError:
        return {"start": None}
    if path.suffix.lower() in {".edf", ".bdf", ".vhdr"}:
        # These headers store a local wall clock without a UTC offset.
        first = first.replace(tzinfo=None)
    return {"start": first.isoformat()}


def _header_start(rec: RecordingSpec) -> datetime | None:
    clock = rec._header_clock
    if clock is None:
        if not Path(rec.path).is_file():
            return None
        # Pre-A2 registrations are already sufficient for analysis. Read their header
        # using the same verified source as the waveform reader for clock navigation.
        from .io_integrity import verified_window_source, verify_raw_filenames
        from .io_signals import open_mne_raw

        with verified_window_source(rec) as (path, directory, _, signature), open_mne_raw(path) as raw:
            verify_raw_filenames(rec, raw.filenames, directory, signature, opened_stat=getattr(raw, "_brainlens_fd_stat", None))
            clock = header_clock(raw, path)
    return datetime.fromisoformat(clock["start"]) if clock.get("start") else None


def _scan_times(directory: Path) -> dict[str, set[datetime]]:
    times: dict[str, set[datetime]] = {}
    for table in sorted(directory.glob("*scans.tsv")):
        with table.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                filename = (row.get("filename") or "").removeprefix("./")
                value = (row.get("acq_time") or "").strip()
                if not filename or not re.match(r"^\d{4}-\d\d-\d\d[T ]\d\d:\d\d", value):
                    continue
                try:
                    stamp = datetime.fromisoformat(value)
                except ValueError:
                    continue
                times.setdefault(filename, set()).add(stamp)
    return times


def _bids_start(path: Path, scans: dict[Path, dict[str, set[datetime]]]) -> datetime | None:
    # BIDS filenames are relative to their scans.tsv. Never match just a basename
    # from a different subject, session, or modality folder.
    directories = [path.parent]
    if path.parent.name in {"ieeg", "eeg", "meg"}:
        directories.append(path.parent.parent)
    directories.extend(p for p in path.parents if p.name.startswith(("ses-", "sub-")) and p != path.parent)
    for directory in directories:
        if directory not in scans:
            scans[directory] = _scan_times(directory)
        matches = scans[directory].get(path.relative_to(directory).as_posix())
        if matches:
            return next(iter(matches)) if len(matches) == 1 else None
    return None


def _clock_basis(start: datetime | None) -> str | None:
    if start is None:
        return None
    if start.year == 1985:
        return "edf_placeholder"
    day = start.date().isoformat()
    if day in {"0001-01-01", "1900-01-01", "1970-01-01"}:
        return day  # Keep distinct placeholder epochs on separate clocks.
    if start > datetime.now(start.tzinfo):
        return "future"
    return "calendar"


def _relative(seconds: float | None) -> dict[str, float | int | None]:
    return {
        "day_index": int(seconds // 86400) if seconds is not None else None,
        "seconds_of_day": seconds % 86400 if seconds is not None else None,
    }


def admission_timeline(store: WorkbenchStore, case: str, *, text: str | None = None,
                       include_absolute: bool = False) -> dict[str, Any]:
    if not store.has_case(case):
        raise ToolkitError("case_not_found", "Choose a registered case for the admission timeline.")
    records = [store.load_recording(case, slug) for slug in store.list_recordings(case)]
    records.sort(key=lambda rec: (rec.path, rec.slug))
    starts: dict[str, datetime | None] = {}
    bases: dict[str, str | None] = {}
    scans: dict[Path, dict[str, set[datetime]]] = {}
    sources = {}
    for rec in records:
        stamp = _bids_start(Path(rec.path), scans)
        source = "bids_scans"
        basis = _clock_basis(stamp)
        if basis != "calendar":
            header = _header_start(rec)
            header_basis = _clock_basis(header)
            if stamp is None or header_basis == "calendar":
                stamp, basis, source = header, header_basis, "recording_header"
        starts[rec.slug] = stamp
        bases[rec.slug] = basis
        sources[rec.slug] = source

    # A consistently shifted calendar can establish elapsed time. Never measure
    # an admission interval between a surrogate date and a real calendar date.
    available = set(bases.values()) - {None}
    comparable = available if len(available) == 1 else {"calendar"}
    frequencies = Counter(d for slug, d in starts.items() if bases[slug] in comparable)
    incompatible = len({d.tzinfo is None for d, count in frequencies.items() if count == 1}) > 1
    reasons = {}
    for slug, stamp in starts.items():
        if bases[slug] not in comparable:
            reasons[slug] = "missing_or_placeholder_start"
        elif frequencies[stamp] > 1:
            reasons[slug] = "identical_start_times"
        elif incompatible:
            reasons[slug] = "incompatible_time_zones"
    if not reasons:
        records.sort(key=lambda rec: starts[rec.slug])
    anchor = min((r.slug for r in records if r.slug not in reasons), key=starts.get, default=None)
    origin = starts[anchor] if anchor is not None else None
    rows, gaps, events = [], [], []
    covered_until = 0.0
    needle = (text or "").casefold()
    for order, rec in enumerate(records, 1):
        start = starts[rec.slug]
        offset = (start - origin).total_seconds() if rec.slug not in reasons else None
        if offset is not None:
            # An undated file may occupy an apparent hole between dated recordings.
            if not reasons and offset > covered_until:
                gaps.append({"start_seconds": covered_until, "end_seconds": offset})
            covered_until = max(covered_until, offset + rec.duration)
        pieces = [{"start_seconds": p.onset_seconds, "end_seconds": p.time_at(p.stop_sample, rec.sfreq)}
                  for p in rec.pieces] or [{"start_seconds": 0.0, "end_seconds": rec.duration}]
        rows.append({
            "recording": rec.slug, "recording_id": rec.id, "order": order,
            "clock_status": "known" if offset is not None else "unknown",
            "clock_reason": reasons.get(rec.slug),
            "clock_source": sources[rec.slug], "offset_seconds": offset, **_relative(offset),
            "duration_seconds": rec.duration, "pieces": pieces,
            "gaps": [{"start_seconds": a, "end_seconds": b} for a, b in rec.gaps()],
            **({"absolute_start": start.isoformat() if bases[rec.slug] == "calendar" else None,
                "header_date_unreliable": start is not None and bases[rec.slug] != "calendar"}
               if include_absolute else {}),
        })
        for marker in rec.markers:
            if needle and needle not in marker.label.casefold():
                continue
            elapsed = offset + marker.start if offset is not None else None
            events.append({
                "recording": rec.slug, "recording_id": rec.id, "label": marker.label,
                "source": marker.source, "channels": marker.channels,
                "start_seconds": marker.start, "end_seconds": marker.end, "admission_seconds": elapsed,
                **_relative(elapsed),
            })
    order_by_slug = {r["recording"]: r["order"] for r in rows}
    events.sort(key=lambda e: (0 if reasons else e["admission_seconds"], order_by_slug[e["recording"]], e["start_seconds"]))
    return {"mode": "timeline", "case": case, "origin_recording": anchor,
            "recordings": rows, "gaps": gaps, "events": events}


def segments_from_markers(recording: RecordingSpec, pattern: str, before_seconds: float = 0,
                          after_seconds: float = 0) -> list[dict[str, Any]]:
    if not isinstance(pattern, str) or not pattern.strip():
        raise ToolkitError("invalid_marker_pattern", "Supply a marker text pattern.")
    try:
        matcher = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ToolkitError("invalid_marker_pattern", "Invalid marker regular expression.", reason=str(exc)) from exc
    margins = (before_seconds, after_seconds)
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0 for x in margins):
        raise ToolkitError("invalid_marker_margin", "Marker margins must be finite nonnegative seconds.")
    chosen = []
    for m in recording.markers:
        if matcher.search(m.label):
            start, end = max(0.0, m.start - before_seconds), min(recording.duration, m.end + after_seconds)
            if end > start and recording.time_slices(start, end):
                chosen.append((start, end, m.label))
    # Union overlapping padded markers so shared samples contribute exactly once.
    merged = []
    for start, end, label in sorted(chosen):
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            if label not in merged[-1][2]:
                merged[-1][2].append(label)
        else:
            merged.append([start, end, [label]])
    if not merged:
        raise ToolkitError("no_marker_segments", "No matching marker window includes acquired signal. Check the pattern and margins.")
    return [{"start_seconds": a, "end_seconds": b, "label": " / ".join(labels)} for a, b, labels in merged]
