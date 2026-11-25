"""Map compact prepared arrays to the original samples and clock of one recording."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .models import AnalysisConfig, RecordingPiece, RecordingSpec, ToolkitError


@dataclass(frozen=True)
class AnalysisWindow:
    piece: RecordingPiece
    start: int
    stop: int
    array_start: int
    segment_index: int | None

    @property
    def offset(self) -> int:
        """Add this to a prepared-array index to obtain its source sample."""
        return self.start - self.array_start

    @property
    def array_stop(self) -> int:
        return self.stop - self.offset


def selected_windows(recording: RecordingSpec, config: AnalysisConfig) -> list[AnalysisWindow]:
    if config.segments:
        intervals = [(s.start_seconds, s.end_seconds, i) for i, s in enumerate(config.segments)]
    else:
        intervals = [(config.start_seconds, config.end_seconds or recording.duration, None)]
    windows = []
    cursor = 0
    previous_end = 0.0
    for start, end, index in intervals:
        if index is not None and (start < previous_end or end > recording.duration):
            raise ToolkitError("invalid_segments", "Segments must be disjoint and inside the recording.",
                               segment_index=index, duration_seconds=recording.duration)
        previous_end = end
        before = cursor
        for piece, a, b in recording.time_slices(start, end):
            if index is not None:
                # Use complete samples inside the requested half-open segment. Adjacent
                # selections with fractional endpoints must never double count a sample.
                a = max(a, piece.start_sample + math.ceil(piece.sample_offset(start, recording.sfreq)))
                b = min(b, piece.start_sample + math.floor(piece.sample_offset(end, recording.sfreq)))
                if b <= a:
                    continue
                piece = RecordingPiece(start_sample=a, stop_sample=b, onset_seconds=piece.time_at(a, recording.sfreq))
            windows.append(AnalysisWindow(piece, a, b, cursor, index))
            cursor += b - a
        if cursor == before:
            raise ToolkitError("empty_analysis_window", "The selected interval contains no recorded samples.",
                               start_seconds=start, end_seconds=end, gaps=recording.gaps())
    return windows


def prepared_windows(prepared: dict[str, Any]) -> list[AnalysisWindow]:
    """Restore the source clock for new compact arrays and existing A1 manifests."""
    start = int(prepared["start_sample"])
    stop = start + int(prepared["n_samples"])
    pieces = [RecordingPiece.model_validate(p) for p in prepared.get("pieces", [])]
    if not pieces:
        pieces = [RecordingPiece(start_sample=0, stop_sample=stop, onset_seconds=0)]
    indexes = prepared.get("segment_indexes")
    windows = []
    cursor = 0
    for i, piece in enumerate(pieces):
        if indexes is None:
            a, b, index = max(start, piece.start_sample), min(stop, piece.stop_sample), None
        else:
            a, b, index = piece.start_sample, piece.stop_sample, indexes[i]
        windows.append(AnalysisWindow(piece, a, b, cursor, index))
        cursor += b - a
    return windows
