# External anatomical labels

Attach a localization table from iEEG-recon, LeGUI, Brainstorm, 3D Slicer or another external source to a registered BrainLens case. Cite the table's origin and version. Set `patient_native` to true only when the labels come from that patient's own MRI/CT co-registration. BrainLens does not assign anatomy from coordinates. HUP134's fsaverage coordinates remain a template frame.

## Import a table

For example, a localization table for the synthetic recording could contain these rows.

```csv
contact,region,tissue,hemisphere,in_lesion
SYN_A1,Hippocampal head,gray,left,false
SYN_A2,Entorhinal cortex,gray,left,
```

```sh
brainlens --workspace /path/to/workspace anatomy synthetic_seeg contacts.csv --source "Synthetic localization example, revision 2"
brainlens --workspace /path/to/workspace anatomy synthetic_seeg --clear
```

For patient-native labels, add `--patient-native` to the import command. Each import replaces the case's previous label set. It reports rows read, distinct labels imported, rows skipped, and unmatched names for each registered recording. A run's recording summary reports coverage for its own analyzed montage.

The importer accepts UTF-8 CSV and TSV, including BIDS `*_electrodes.tsv`. Use `contact`, `name` or `channel` for the identifier. Supported region columns, in priority order, are `anatomy`, `region`, `label`, `roi`, `atlas_label`, `destrieux`, `dkt` and `aparc`. Optional fields include `hemisphere`, `tissue` or `wm_gm`, and `in_lesion` or `lesion`. Empty regions and BIDS `n/a` regions are skipped. Coordinates and other columns are ignored. Conflicting labels for the same case-insensitive name are rejected.

## Match and display

BrainLens first looks for an exact, case-insensitive channel name. If it finds none, it looks up the constituent contacts in order. A bipolar pair with the same region at both contacts shows that region once. Two different regions appear as `Hippocampal head → Entorhinal cortex`. A partially labeled pair explicitly shows the `unlabeled` end. Labels from other frames display `(non-patient-native)`.

Labels appear in overview rankings, recording and branch comparisons, the review queue, reader gutter hover, triad header, 3D contact tooltip, and exported case sheets. The overview and case sheet identify the external source, patient-native status and atlas. Newly exported BIDS event tables and cohort CSVs also carry anatomy. Committed analysis artifacts retain their original channel identifiers.

## MCP and HTTP

- `inspect_case(target=..., anatomy_sidecar=..., anatomy_source=...)` imports a table for a registered case. `anatomy_patient_native` defaults to false.
- `record_judgment(case=..., anatomy_labels=..., operation_id=...)` accepts a complete label document. Reuse the operation ID when retrying a save.
- `GET /api/anatomy?case=<slug>` returns the label document and acquisition coverage by recording. `POST /api/anatomy` accepts a JSON object with `case` and the fields below. `DELETE /api/anatomy?case=<slug>` clears it. Both writes require the Workbench session token and `Content-Type: application/json`, as for the clinical overlay.

```json
{
  "source": "Synthetic localization example, revision 2",
  "method": "Manual contact annotation",
  "atlas": "DKT",
  "patient_native": false,
  "labels": {
    "SYN_A1": {"region": "Hippocampal head", "hemisphere": "left", "tissue": "gray", "in_lesion": null}
  }
}
```

`source`, the boolean `patient_native`, and the `labels` mapping are required. Every entry requires a region. Optional tissue values are `gray`, `white` and `unknown`. Each entry can include a free-text `note`. The stored document includes its author, update time and external-source disclaimer.

The overlay lives in `cases/<slug>/anatomy_labels.json`. Saving or clearing it updates the Workbench and affects verified case-sheet freshness. Existing runs can be queried immediately with the updated labels.
