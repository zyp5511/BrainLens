# Independent reader review and agreement

A review set freezes the same event IDs for every reader in one case, recording and analysis run. Each reader sees a reproducible shuffle and can resume with the same name and role. The progress counter counts distinct events in the set with a current verdict from that reader. Notes and repeated ratings do not increase it.

## Run a session

1. Open the recording and choose **Reader → Review set**. Under **Create review set**, enter a unique name. Leave the sample size empty to include all events, or supply a size and seed for a stratified sample.
2. Give each reviewer a distinct author name. Each reviewer selects their own name, role and the same review set, then enables **Blinded** before inspecting events. The preference persists in that browser. Synthetic sessions use the existing engineering rehearsal role convention.
3. Record **confirm**, **artifact** or **uncertain** for each event. A later verdict by the same reader replaces their earlier verdict for agreement. An observation or clip alone leaves the rating intact.
4. After independent review, turn off **Blinded** and open **Overview → Inter-rater agreement**. Select the review set or the whole run. The default includes clinicians and researchers. Selecting a disagreement opens its event in the signal reader.

Blinding controls presentation bias. The server projects the requesting author's own active judgments onto queue events and waveform views. Shared review counts, case sheets, agreement and agent activity are hidden during blinded review. Identity is self-declared, and reviewers with workspace access can turn blinding off. The report cannot certify that the readers remained blinded or independent. A review set fixes event membership. Ratings already present in the run count toward progress and agreement, and overlapping sets share those ratings.

## Sampling and provenance

Sets are stored in `cases/<case>/review_sets/<name>.json`. A set records its creation time, recording fingerprint, run analysis signature, sampling rule, seed, population size and selected IDs. Names are unique within a case. To change membership, create a new set. Loading a set checks the recording and analysis binding and the presence of every selected event.

Size-limited sampling uses the cross of synchrony class and event band. The band is the event's detector band when supplied, otherwise BrainLens's dominant spectral band (ripple or fast ripple). Missing labels form an unknown stratum. Within each stratum, event IDs are shuffled from a sorted list. A seeded round-robin over shuffled strata draws without replacement until the requested count is reached. The default seed is zero. Rare strata can therefore represent a larger share of a set than of the run. Report the sampling rule alongside the estimate. Reader order also depends on the set identity and exact reader name.

## Commands and API

The following commands operate on an existing baseline analysis.

```sh
brainlens --workspace build/study review-set create CASE --recording REC --name round1 --sample-size 40 --seed 17
brainlens --workspace build/study review-set list CASE --recording REC
brainlens --workspace build/study agreement CASE --recording REC --review-set round1 --readers Alice,Bob
```

Use `--run-name` to select another analysis. The same operations are available through the existing MCP tools.

| MCP entry | Parameters |
| --- | --- |
| `export_or_protocol(action="create_review_set")` | `case`, `recording`, `review_set` (name), optional `primary_run`, `sample_size`, `seed` |
| `query_evidence(mode="reader_agreement")` | `target` as case/recording, or `target` and `recording`, optional `run_a`, `review_set`, `readers` |
| `export_or_protocol(action="export_agreement_report")` | `case`, `recording`, optional `primary_run`, `review_set`, `readers` |

The HTTP API accepts `case`, `recording` and `run_name` (default `baseline`). Create with `POST /api/review-sets` using a JSON body containing `name` and optional `sample_size` and `seed`. List with `GET /api/review-sets`. For a reader queue, call `GET /api/queue` with `mode=review_set`, `review_set` and `reader`. Supply `blind_for=<author>` on reader GET requests, and `blind_role` when selecting a specific role. Fetch the unblinded report with `GET /api/agreement` and optional `review_set` and comma-separated `readers`.

## Reading the report

- **Population.** Each event matrix cell contains the latest valid journal verdict for one author and role, restricted to the selected recording, run and event population. A missing rating is JSON `null` or an empty CSV cell. The complete matrix includes unrated events. A disagreement requires at least two different observed ratings on the event.
- **Reader roles.** Researcher and clinician ratings retain their attribution. An author with both human roles appears under `excluded_ambiguous_readers` until a role is selected explicitly, such as `researcher:Alice`. Agent and engineering rehearsal readers require explicit selection. The same author cannot count twice in a comparison.
- **Two readers.** Each pair uses its own set of jointly rated events. Matrix rows follow the first reader and columns the second, in the order confirm, artifact, uncertain. Percent agreement is the matching fraction multiplied by 100. Unweighted Cohen’s κ subtracts chance agreement based on each reader's marginal category frequencies. Binary κ combines artifact and uncertain into a single category. In its two-by-two matrix with cells a, b, c, d in row order, the prevalence index is |a − d| / N and the bias index is |b − c| / N.
- **Three or more readers.** Fleiss’ κ uses events rated by every selected reader. The report gives this denominator separately from the pairwise denominators. The same statistics are available by synchrony class. Separate recordings are always reported individually.
- **Uncertainty.** The 95% intervals use an analytic event-level delta-method standard error and normal approximation bounded to [−1, 1]. They assume independent events. Within-recording temporal or channel clustering can narrow the reported intervals relative to a cluster-aware analysis. With fewer than two shared events or a zero chance-correction denominator, κ and its interval are null with a reason. Descriptive counts and observed agreement remain available whenever there is a shared rating.

Export publishes `agreement.json`, `event_matrix.csv` and `disagreements.csv` together in a new directory under `cases/<case>/exports/`. JSON includes recording provenance and fingerprint, analysis signature, journal revision, reader roles and sampling metadata. The CSVs include event ID, channel, onset in seconds, synchrony class and a verdict column for each reader. The case sheet includes recording-specific pairwise agreement for the primary run when two human readers have rated at least one shared event.

BrainLens is not a medical device. Reader agreement measures consistency of event ratings. It is not a standalone surgical resection boundary.
