# MDT presentation export

Open a registered case with a saved analysis, go to **Case Sheet**, and select **MDT slides** beside the BIDS and EDF exports. When the background job finishes, select **Download MDT slides**. The presentation uses the active primary run, comparison run, discussion questions and review evidence. Text, tables and speaker notes are editable in PowerPoint. Evidence figures are embedded PNGs.

## Identity and sharing

The default deck keeps local case and recording names and participant identity for in-hospital discussion. Select **De-identify for sharing** to mask native header identity and use a pseudonym. The default pseudonym is `shared`. For multiple recordings a pseudonym such as `P001` produces recording labels `P001-01`, `P001-02` in the same order as the case sheet. The case-sheet disclosure rules also apply to figure labels, external anatomy and overlay sources, clinical questions, reader names, notes and the export filename. Explicitly declared public participant labels follow those same rules.

Both versions are written to the case's `exports/` directory. The local filename is `<case>_<run>_mdt_slides.pptx`. The sharing filename is `<pseudonym>_<run>_mdt_slides.shared.pptx`, with the disclosure filter applied to the run label. A repeat export replaces the same file only after the new deck is complete.

## Evidence in the deck

- The title gives the analysis date and clinical scope. The discussion summary identifies the focal leader and widespread-burst fraction for each recording, with pairwise top-five overlap and branch sensitivity when available. Each recording retains its own rates and denominators.
- A spatial projection appears when mapped coordinates are available. Template coordinates carry the non-patient-native warning. The leading focal contacts table shows up to five channels per recording, their rates and separate confirm, artifact and uncertain tallies for clinicians, researchers, engineering rehearsal and agents. External SOZ and resection tags retain their source. External anatomy labels identify their source, atlas and patient-native status.
- Each selected Clinical Triad has a slide with the recording, run, channel, window, peak frequency when available and role-specific verdicts. Selection starts with clips, then focal examples and a widespread-burst example. A positive card budget reserves room for a candidate-free control when one is available. A budget of zero omits triads.
- A cached evidence map for the primary run appears with its descriptive convergence caveat. Export does not compute a map or ictal EI. Inter-rater agreement appears for human reader pairs with shared primary-run events, separately for each recording, with shared event count, agreement and Cohen’s κ with 95% CI when estimable.
- QC, coverage, reference and spatial limitations precede per-recording provenance, fingerprint, BrainLens version and analysis generation IDs. Speaker notes contain the supporting numbers. The deck is a snapshot of the saved evidence at export.

## CLI, MCP and HTTP

```sh
brainlens export-slides CASE --primary-run baseline
brainlens export-slides CASE --primary-run baseline --comparison-run sensitivity \
  --question "Which focal contacts recur across recordings?" --max-cards 6 \
  --deidentify --pseudonym P001
```

A supplied question is saved with the case's open discussion questions, as for case-sheet export. Omitting the comparison run uses the case-sheet default. The Python entry point is `brainlens.export_mdt_slides`. It takes `clinical_question`, `deidentify=False`, `pseudonym=None` and `max_cards=6`, and returns the local slide path, slide and image counts, run names, case label and warnings.

Use the existing MCP tool `export_or_protocol(action="export_mdt_slides", case="CASE", primary_run="baseline", deidentify=True, pseudonym="P001")`. For loopback HTTP clients, send the same fields (without `action`) in an authenticated JSON `POST /api/export-slides`. The response is HTTP 202 with a `job_id`. Poll `GET /api/jobs/<job_id>` until the job completes, then download the PPTX attachment from `result.download_url`.

Export requires `python-pptx>=1.0,<2`. BrainLens loads this dependency when slide export starts.
