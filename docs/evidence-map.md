# Multi-biomarker evidence map and ictal EI

In the workbench overview, select a recording and run, then choose **Show evidence map**. Click a contact to open the reader at its first IED candidate or the middle of the selected interval. The map names the recording behind each column. The convergence count is descriptive, not a validated predictor.

## Read a row

| Column | Measurement |
| --- | --- |
| Focal ripple | Non-widespread HFO candidates with ripple dominance, per minute of that channel’s detector-analyzed time. |
| Focal fast ripple | Non-widespread fast-ripple-dominant candidates per detector-analyzed minute. Availability follows the run’s detector band and spectral classification bandwidth. The latter requires an upper bound above 270 Hz after applying the native Nyquist margin (0.92) and acquisition and run low-pass limits. |
| ONNX spike-HFO | Events with `labels.spike == 1` per detector-analyzed minute. Requires a completed spike classification stage. |
| IED candidates — require visual verification | Detected transient peaks per minute of assessable prepared signal. The JSON cell includes candidate times in recording-relative seconds, count, effective band and denominator. |
| Ictal Epileptogenicity Index | EI from one explicitly analyzed seizure, normalized within the named source recording. The source and onset accompany the values. |
| External tags | SOZ and Resected annotations from the case overlay, matched to a channel name or its constituent contacts. The overlay carries its own source. |

Each biomarker ranks its assessed channels within its own recording. Rank 1 is highest. Equal values share their best competition rank (1, 1, 3). With N measured contacts and rank r, a positive measurement has percentile `100 × (N − r) / (N − 1)`. A sole positive measurement has percentile 100. Measured zeros have percentile 0.

A contact earns one convergence point for each available biomarker with a positive value and percentile ≥80. The table also gives the number of available biomarkers. Missing values and external tags contribute no points. An entirely unassessed row has a null count. A measured contact with only zero values is `quiet_measured`. An auxiliary input or channel outside valid run coverage is `unassessed`. Detector coverage and IED coverage have separate denominators, so a channel with valid prepared signal can support IED assessment even if HFO detection was skipped.

## IED candidate detector

Each channel is processed separately in contiguous, finite, valid spans. The detector uses a third-order zero-phase Butterworth band-pass (10–60 Hz, limited by the available bandwidth), followed by the Hilbert amplitude envelope. A five-second local threshold window advances at 50% overlap. The threshold is median + 5 × IQR, interpolated between window centres. Excursions lasting 8–200 ms yield a peak time, with a 100 ms refractory period. Both polarities are eligible. Spans shorter than five low-edge cycles are omitted from the denominator. A run high-pass filter above the lower edge makes this measurement unavailable. The keyword parameters of `brainlens.spikes.detect_ieds` expose the band, threshold window, IQR multiplier, duration limits and refractory period.

## Ictal Epileptogenicity Index

Select the seizure recording in the overview, optionally enter the onset in seconds from its start, and choose **Show ictal EI**. Explicit seconds take precedence. Otherwise the calculation uses the earliest matching global onset, seizure or SZ marker. Channel-local annotations and seizure-end markers are ineligible. A missing onset produces `seizure_onset_unavailable`. A registered recording suffices for EI, including one without an HFO run.

The implementation follows the energy-ratio and recruitment-delay construction of Bartolomei, Chauvel and Wendling, *Brain* (2008), “Epileptogenicity of brain structures in human temporal lobe epilepsy: a quantified study from intracerebral EEG”. The default analysis window starts 30 seconds before onset and ends 60 seconds after onset, clipped to the source. It uses the acquired montage in a verified snapshot. Global and channel-specific BAD intervals, nonfinite samples and auxiliary channels are masked.

1. Compute power from demeaned, two-second Hann windows every 0.25 s. ER is (beta + gamma power) / (theta + alpha power). Band boundaries are 3.5, 7.4, 12.4, 24 and 97 Hz. Gamma ends at the lowest of 97 Hz, native Nyquist and the acquisition low-pass. Each channel reports its effective gamma ceiling.
2. Initialize the running ER mean on wholly pre-onset frames. From onset, the upward Page–Hinkley sum accumulates the deviation from the running mean in pre-onset standard-deviation units, less a bias of 2 per step. The scale has a floor of 5% of the baseline mean. A rise of 30 units marks recruitment at the first frame of the rise. Gaps reset the sum. The detection time is the recording-relative window centre.
3. Set N₀ to the earliest detection. For each detected channel, sum ER over [Nᵢ, Nᵢ + 5 s] and divide by (Nᵢ − N₀ + 1 s). Divide scores by the highest score in this recording. A measured channel without a detection has EI 0.
4. The detector requires at least five hop-seconds of baseline observations, 80% finite frame coverage on both sides of onset, and sufficient contiguous post-onset data. A detection also requires a complete five-second scoring interval for an assessed EI. The JSON reports per-channel reasons for missing values.

`brainlens.ictal.compute_ei` exposes the spectral window and step, Page–Hinkley bias and threshold, delay offset and integration horizon. The recording-level `get_epileptogenicity` accepts these overrides in `parameters` plus `before_seconds` and `after_seconds`. Defaults are engineering choices verified with synthetic recruitment.

## Source selection and caching

- The HFO columns reuse the selected run’s counts, spectral classifications and actual HFO coverage. IED reads the run’s prepared arrays through read-only memory maps. If arrays have been evicted, the saved montage, filters, QC and time selection are reapplied to a verified source snapshot in a temporary directory.
- The EI column uses the latest selected EI analysis on its source recording and preserves the rank calculated across **all assessed contacts in that ictal recording**. The join uses exact channel names. Bipolar channels require a matching acquired bipolar source. Conflicting contact definitions leave EI unavailable. There is no positional contact matching, summation of recording durations or averaging of seizures.
- When several recordings have saved EI, the map prefers its own recording and otherwise takes the first eligible slug in lexical order. JSON names the chosen recording and lists the alternatives in `available_ei_recordings`. On one recording with several analyzed onsets, requesting an onset again selects its saved result.
- The EI card can show a separate HFO leaderboard from the case. It names both sources and prefers an explicitly marked interictal run. A source with seizure markers or an active EI analysis is eligible only if the HFO run selected an explicit interictal state. Otherwise the comparison can use another recording and labels an unverified interictal state.
- Rate caches live in the run’s `evidence/map/` directory. EI caches live in the recording’s `evidence/ei/` directory. Cache keys include recording identity, the prepare fingerprint or source identity, numerical module versions and parameters. HFO values also bind to the run analysis generation. Existing pipeline stages and judgment bindings are untouched. Repeated requests reuse the numeric results. A case sheet includes only maps already requested for its primary run.

## Interfaces

The existing seven MCP tools support `query_evidence(mode="evidence_map", ...)` and `query_evidence(mode="epileptogenicity", ..., onset_seconds=...)`. For PNGs use `render_evidence_card(event_id="evidence_map", ...)` or `event_id="epileptogenicity"`. The EI card displays the selected cached onset when one exists.

```sh
brainlens --workspace ./review evidence CASE --recording INTERICTAL --run-name baseline
brainlens --workspace ./review ei CASE --recording ICTAL --onset-seconds 42
brainlens --workspace ./review card CASE --recording ICTAL --event epileptogenicity
```

`GET /api/evidence-map?case=CASE&recording=INTERICTAL&run_name=baseline` and `GET /api/epileptogenicity?case=CASE&recording=ICTAL&onset_seconds=42` return HTTP 202 with a job ID for a cold computation. Poll `GET /api/jobs/<job_id>` for the result. A cache hit returns HTTP 200 with the data.

Research and review only — not a medical device / not a standalone surgical resection boundary.
