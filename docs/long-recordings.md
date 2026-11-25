# Reviewing long and discontinuous recordings

Open the source with `inspect_case`, then use the Waveform Reader or a Clinical
Triad evidence card to inspect a window. The first registration hashes the full
source and its companions. The analysis progress callback reports bytes hashed
for large files. The registered fingerprint and existing saved runs continue
to use the full SHA-256 identity.

## Recording time

BrainLens reads the timekeeping TAL from each EDF+/BDF+ data record. The first
sample defines time zero. A forward jump starts a new contiguous piece. This
also handles files marked continuous whose TAL clock has a gap. Overlapping
records and unreadable discontinuous clocks produce an error.

Reader pages, card centers, state markers, exclusions, event onsets, controls
and exported annotations use elapsed seconds on that clock. The Reader and
Triad shade each gap and show its duration. A window entirely inside a gap has
no signal. For example, when a 12-second piece is followed by data at 30.125 s,
a burst 5.25 seconds into the second piece appears at 35.375 s.

Recording metadata stores each piece as `start_sample`, exclusive
`stop_sample` and `onset_seconds`. Sample indices count acquired samples.
Within a piece, time is `onset_seconds + (sample - start_sample) / sfreq`.
At a gap boundary the exclusive end belongs to the preceding interval. The
next piece's first sample has its own onset. Consumers of raw sample indices
can use `RecordingSpec.time_slices` to locate a true-time window.

## Analyzing one window

The existing `analyze_case` tool accepts a window in seconds. For example, use
`{"start_seconds": 86400, "end_seconds": 87000}` as `overrides` for ten minutes
starting 24 hours into the recording. QC and signal preparation read the
selected pieces independently. Each filter handles its own edges. The validity
mask and detector coverage count acquired samples only. A gap-only analysis
returns `empty_analysis_window`.

The default prepared-data budget is 4 GiB. Count nine bytes per acquired
sample per output channel, eight for float64 and one for validity. For
example, ten minutes at 2,048 Hz on 128 output channels requires 1.32 GiB of
prepared arrays. A larger request returns `analysis_window_too_large` with
both the required bytes and the budget. Deployments can set
`BRAINLENS_PREPARED_DATA_BUDGET_BYTES` before startup. Working memory, PyHFO sessions and
output artifacts use additional space.

Each PyHFO input covers at most 600 seconds, including overlap. Chunks overlap
enough to cover detector edge margins. An event must fit within a valid
interior. If two chunks report intersecting events on the same channel, the
first complete candidate in chunk order supplies the event and its classifier
session. A continuous segment that fits one chunk uses the original detection
path. The processing limit is `pyhfo_runner.MAX_CHUNK_SECONDS`.

## Source integrity and exports

For large sources, verification brackets window reads and precedes publication
of a run. A source content change raises `source_recording_modified`. EDF/BDF
readers hold the opened signal file. Large multipart inputs need Linux procfs
or a writable temporary directory on the same APFS volume on macOS. Set
`TMPDIR` before starting BrainLens when the source lives on another APFS
volume. Small recordings keep private verified snapshots. The annotated signal
export for continuous EDF also needs enough disk space for complete files.
[Deployment limits and integrity checks](deployment.md#large-recordings-and-analysis-windows)
describe the large-file threshold and the sampled-check trade-off.

For a discontinuous source, the annotated export writes the true-time CSV and
JSON and returns an explanation in `edf_copy.reason`. Open that CSV alongside
the original recording. Writing a continuous EDF+ signal copy would discard
the data-record discontinuities.

## Find a time and contact to inspect

Open **Overview → Trends** for a registered recording, including one without
a detection run. Choose a measure and an epoch length from 2 to 60 seconds
(default 10). Click the heatmap to inspect that contact and time in the Reader.
Drag along the time axis to fill a segment's start and end, give it a label,
then choose **Add segment**. The legend gives the displayed range and units.
Hatching marks acquisition gaps. The edges mark file boundaries and the
marker ticks use the file's own annotations. These quantitative trends guide
waveform navigation. They provide no seizure classification.

The seven power measures are delta (1–4 Hz), theta (4–8), alpha (8–13),
beta (13–30), gamma (30–80), ripple (80–250) and fast ripple (250–500).
BrainLens integrates the one-sided Hann periodogram of mean-centered signal
within each continuous block (up to 10 s). It averages power by sample count
before conversion to dB relative to 1 µV², with a −300 dB numerical floor.
A band's upper edge must fall below the channel's native Nyquist and at or
below its effective low-pass. RMS amplitude and line length use µV. Line
length sums absolute differences inside each contiguous piece. Empty epochs
and unavailable measures have null values. For a partially acquired epoch
the calculation uses only acquired samples. The HFO rate measure uses the
selected run's output channels and divides event onsets by minutes of valid
detector coverage in each epoch. Unassessed time has no rate.

A cold request computes all nine signal measures in one windowed pass.
Under the case's recording directory, `trends/<digest>/manifest.json` records
the source fingerprint, epoch length, measures and algorithm version. HFO rate
also binds the run and its analysis generation. Source verification applies
to cache hits. This navigation cache is independent of analysis and can be
regenerated from source.

A local manual check used a synthetic 2 h EDF with 128 channels sampled at
2 kHz (3.69 GB). With 10 s epochs, registration took 2.85 s and the first
trend request took 8.98 s for a 128 × 720 matrix. A cached compact request
took 0.017 s. The strongest ripple epoch was the planted one on LA72 at
3640 s. These timings include a cold trend cache on a recently written file.

## Navigate an admission

**Overview → Admission timeline** shows file boundaries, inter-file gaps,
within-file pieces and their gaps. Search marker text (for example, `sleep`
or `sz`) and select an event to jump to the originating file. EDF+ TAL,
BrainVision and FIF annotations share this file-local true-time convention.

An exact relative filename in a BIDS scans.tsv supplies acquisition time when
available. Otherwise the timeline uses the recorded measurement start,
including FIF first-sample offsets and the first EDF+ TAL offset. Acquisition
times with timezone offsets are normalized for comparison. When clocks are
usable, day index 0 starts at the earliest dated recording's first sample.
The MCP returns elapsed seconds, zero-based day index and seconds within that
relative day. The clinician view also shows the source dates.

Files with missing times, 1985 dates, the Unix epoch placeholder or duplicate
starts retain unknown offsets. Mixed timezone-aware and timezone-unspecified
clocks also have unknown alignment. When an offset is unknown, the timeline
uses file order and suppresses inferred inter-file gaps. Files with independently
comparable clocks still carry their measured offsets from the indicated origin
recording. Waveform navigation within each file remains available.

## Analyze selected segments

Target a single registered recording or file. In the Trends panel, add
explicit start/end/label rows, drag ranges, or use a marker regular expression
with before/after margins. **Analyze segments** submits the selection as one run.
For example, an MCP `analyze_case` request can specify these overrides.

```json
{"segments": [
  {"start_seconds": 3600, "end_seconds": 4200, "label": "Night 1 sample"},
  {"start_seconds": 90000, "end_seconds": 90600, "label": "Night 2 sample"}
]}
```

Alternatively, use `{"marker_segments":{"pattern":"sz|seizure","before_seconds":120,"after_seconds":180}}`.
The case-insensitive expression matches marker labels. Padding is measured
from marker start/end, clipped to the recording, and overlapping marker windows
are combined. Explicit windows must be disjoint and within the file. An
empty selection or a segment containing no acquired samples returns an error.

Each segment uses the same analysis configuration. Prepared arrays contain
only selected samples. The 4 GiB default budget applies to their combined
size. Filtering, detector margins and burst grouping respect segment and piece
boundaries. Events carry a zero-based segment index and label. Per-contact
counts add across this recording's segments and rates use the sum of analyzed
valid durations. The recording summary includes each segment's counts and
coverage. Controls, Triad views and EDF/CSV annotations use original recording
time. The case sheet and MDT slides identify the analyzed windows. Each
recording keeps its own analytical results.

## Interfaces

- MCP `query_evidence(mode="trends", recording=..., measure="ripple", epoch_seconds=10)`
  returns at most 32 channels × 120 time columns, with per-column maxima,
  original-epoch peak times for up to eight channels and units. For `hfo_rate`,
  select the source with `run_a`. The default selects `baseline` when present,
  or the latest matching run when analysis began with a segment selection.
- MCP `query_evidence(mode="timeline", text="sleep")` filters annotations.
  `open_workbench(action="focus", case=..., recording=..., start_seconds=..., channel=...)`
  opens the Reader on that file's true-time clock. The returned `recording_id`
  can identify a file whose name contains a withheld date.
- CLI `query` accepts `--mode trends|timeline`, `--measure`, `--epoch-seconds`,
  `--text`, and the existing `--recording` and `--run-a`. CLI `analyze --overrides`
  accepts the same selection JSON as MCP.
- HTTP `GET /api/trends` takes `case`, `recording`, `measure`, `epoch_seconds`
  and optional `run_name`. It returns a full matrix for one measure.
  `GET /api/timeline` takes `case` and optional `text`. Authenticated
  `POST /api/marker-segments` takes `case`, `recording`, `pattern` and optional
  margins. `POST /api/jobs/branch` submits the recording-local selection.
  The Reader's `/api/multichannel` and `/api/triad` accept `source_only=1`
  to inspect an acquired contact when a run uses a derived montage.
