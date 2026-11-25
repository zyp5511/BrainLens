"""External, source-cited anatomical annotations for registered cases."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from filelock import FileLock

from .models import ChannelSpec, ToolkitError, atomic_write_json, parse_shaft_group, read_json, utc_now

if TYPE_CHECKING:
    from .storage import WorkbenchStore


_DISCLAIMER = (
    "Anatomy labels are supplied by an external source. BrainLens does not infer or "
    "validate anatomical regions or patient-native co-registration."
)
_REGION_COLUMNS = ("anatomy", "region", "label", "roi", "atlas_label", "destrieux", "dkt", "aparc")
_TISSUE_ALIASES = {
    "gm": "gray", "grey": "gray", "gray matter": "gray", "grey matter": "gray",
    "wm": "white", "white matter": "white",
}
_LESION_FLAGS = {
    "": None, "unknown": None, "true": True, "false": False, "yes": True, "no": False,
    "1": True, "0": False, "y": True, "n": False,
}


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ToolkitError("invalid_anatomy_labels", f"{field} must be a string.")
    return value.strip()


def _source(value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ToolkitError("missing_anatomy_source", "Anatomy labels require an explicit external source citation.")
    return _string(value, "source")


def _label(contact: str, entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict) or not isinstance(entry.get("region"), str) or not entry["region"].strip():
        raise ToolkitError("invalid_anatomy_labels", f"Label for {contact!r} requires a non-empty region.")
    clean: dict[str, Any] = {"region": entry["region"].strip()}
    for field in ("hemisphere", "note", "tissue"):
        if entry.get(field) is not None:
            value = _string(entry[field], f"{contact}.{field}")
            if field == "tissue" and value not in ("gray", "white", "unknown"):
                raise ToolkitError("invalid_anatomy_labels", f"{contact}.tissue must be gray, white, or unknown.")
            if value:
                clean[field] = value
    if "in_lesion" in entry:
        value = entry["in_lesion"]
        if value is not None and not isinstance(value, bool):
            raise ToolkitError("invalid_anatomy_labels", f"{contact}.in_lesion must be a boolean or null.")
        clean["in_lesion"] = value
    return clean


def _insert_label(
    labels: dict[str, Any], names: dict[str, str], key: str, entry: dict[str, Any]
) -> None:
    if not key:
        raise ToolkitError("invalid_anatomy_labels", "Every region must have a contact or channel name.")
    existing_key = names.setdefault(key.casefold(), key)
    if existing_key in labels and labels[existing_key] != entry:
        raise ToolkitError("invalid_anatomy_labels", f"Conflicting labels for contact or channel {key!r}.")
    labels[existing_key] = entry


def validate_anatomy_labels(payload: Any) -> dict[str, Any]:
    """Normalize external labels without modifying the supplied document."""
    if not isinstance(payload, dict):
        raise ToolkitError("invalid_anatomy_labels", "anatomy_labels must be a JSON object.")
    source = _source(payload.get("source"))
    if not isinstance(payload.get("patient_native"), bool):
        raise ToolkitError("invalid_anatomy_labels", "Anatomy labels require an explicit patient_native boolean.")
    if not isinstance(payload.get("labels"), dict):
        raise ToolkitError("invalid_anatomy_labels", "labels must be an object keyed by contact or channel name.")
    labels: dict[str, Any] = {}
    names: dict[str, str] = {}
    for raw_key, entry in payload["labels"].items():
        key = _string(raw_key, "Contact or channel name")
        _insert_label(labels, names, key, _label(key, entry))
    clean = {"source": source, "patient_native": payload["patient_native"], "labels": labels}
    for field in ("method", "tool", "atlas", "author"):
        if payload.get(field) is not None:
            value = _string(payload[field], field)
            if value:
                clean[field] = value
    return clean


def load_anatomy_labels(store: WorkbenchStore, case: str) -> dict[str, Any] | None:
    path = store.case_dir(case) / "anatomy_labels.json"
    try:
        document = read_json(path)
    except FileNotFoundError:
        return None
    except (ValueError, OSError) as exc:
        raise ToolkitError("invalid_anatomy_labels", f"Could not read anatomy labels for {case!r}: {exc}") from exc
    # Corrupt or manually edited documents must not silently appear to be absent.
    validate_anatomy_labels(document)
    return document


def save_anatomy_labels(
    store: WorkbenchStore, case: str, payload: dict[str, Any], *,
    author: str | None = None, operation_id: str | None = None,
) -> dict[str, Any]:
    clean = validate_anatomy_labels(payload)
    if not store.has_case(case):
        raise ToolkitError("case_not_found", f"Case {case!r} is not registered.")
    case_dir = store.case_dir(case)
    op_id = _string(operation_id, "operation_id") if operation_id is not None else ""
    if author is not None:
        clean["author"] = _string(author, "author")
    with FileLock(str(case_dir / ".case.lock")):
        previous = load_anatomy_labels(store, case) or {}
        applied = list(previous.get("applied_operation_ids") or [])
        if op_id and op_id in applied:
            return previous
        doc = {
            **clean, "case": case_dir.name, "author": clean.get("author") or "user",
            "updated_at": utc_now(), "external_only_disclaimer": _DISCLAIMER,
        }
        if op_id:
            applied.append(op_id)
        if applied:
            doc["applied_operation_ids"] = applied[-128:]
        atomic_write_json(case_dir / "anatomy_labels.json", doc)
    return doc


def clear_anatomy_labels(store: WorkbenchStore, case: str) -> bool:
    if not store.has_case(case):
        raise ToolkitError("case_not_found", f"Case {case!r} is not registered.")
    case_dir = store.case_dir(case)
    with FileLock(str(case_dir / ".case.lock")):
        try:
            (case_dir / "anatomy_labels.json").unlink()
        except FileNotFoundError:
            return False
    return True


def _table_cell(value: str | None) -> str:
    value = (value or "").strip()
    return "" if value.casefold() in {"n/a", "na", "null"} else value


def parse_anatomy_table(text: str) -> tuple[dict[str, Any], dict[str, int]]:
    """Parse BIDS or simple contact tables. Coordinates and unrecognized columns are ignored."""
    text = text.lstrip("\ufeff")
    first_line = text.partition("\n")[0]
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t" if "\t" in first_line else ",", strict=True)
    labels: dict[str, Any] = {}
    names: dict[str, str] = {}
    rows_read = 0
    try:
        columns = [col.strip().casefold() for col in reader.fieldnames or []]
        reader.fieldnames = columns
        contact_col = next((col for col in ("contact", "name", "channel") if col in columns), None)
        region_col = next((col for col in _REGION_COLUMNS if col in columns), None)
        if contact_col is None or region_col is None:
            raise ToolkitError("invalid_anatomy_table", "A contact or name column and a region label column are required.")
        for row in reader:
            rows_read += 1
            if None in row:
                raise ToolkitError("invalid_anatomy_table", f"Extra delimited values on row {reader.line_num}.")
            region = _table_cell(row.get(region_col))
            if not region:
                continue
            contact = _table_cell(row.get(contact_col))
            label: dict[str, Any] = {"region": region}
            for field in ("hemisphere", "note"):
                value = _table_cell(row.get(field))
                if value:
                    label[field] = value
            tissue = (_table_cell(row.get("tissue")) or _table_cell(row.get("wm_gm"))).casefold()
            if tissue:
                label["tissue"] = _TISSUE_ALIASES.get(tissue, tissue)
            if "in_lesion" in row or "lesion" in row:
                lesion = (_table_cell(row.get("in_lesion")) or _table_cell(row.get("lesion"))).casefold()
                if lesion not in _LESION_FLAGS:
                    raise ToolkitError("invalid_anatomy_table", f"Invalid lesion flag on row {reader.line_num}.")
                label["in_lesion"] = _LESION_FLAGS[lesion]
            _insert_label(labels, names, contact, _label(contact, label))
    except csv.Error as exc:
        raise ToolkitError("invalid_anatomy_table", f"Malformed table on row {reader.line_num}: {exc}") from exc
    return labels, {"rows_read": rows_read, "labels_imported": len(labels), "rows_skipped": rows_read - len(labels)}


def read_anatomy_sidecar(
    path: str | Path, source: str | None, *, patient_native: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    source = _source(source)
    path = Path(path).expanduser()
    if path.suffix.lower() not in {".csv", ".tsv"}:
        raise ToolkitError("invalid_anatomy_table", "An anatomy sidecar must be a .csv or .tsv file.")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ToolkitError("invalid_anatomy_table", f"Could not read anatomy sidecar {path}: {exc}") from exc
    labels, counts = parse_anatomy_table(text)
    return validate_anatomy_labels({"source": source, "patient_native": patient_native, "labels": labels}), counts


def _channel_contacts(channel_name: str, contacts: Iterable[str]) -> list[str]:
    supplied = list(contacts or ())
    if supplied and supplied != [channel_name]:
        return supplied
    # Older rows may provide only a bipolar display name. Require two complete,
    # distinct numbered contacts on the same shaft before using their names.
    parts = [part.strip() for part in channel_name.split("-")]
    if len(parts) == 2:
        left, right = (parse_shaft_group(part) for part in parts)
        if (left["shaft"] not in {"AUX", "OTHER"} and left["shaft"].casefold() == right["shaft"].casefold()
                and left["contact_index"] != right["contact_index"]):
            return parts
    return supplied


class AnatomyMatcher:
    """Case-insensitive exact matches for a single snapshot of the external labels."""

    def __init__(self, overlay: dict[str, Any] | None):
        self.overlay = overlay or {}
        self.labels = {key.casefold(): (key, value) for key, value in self.overlay.get("labels", {}).items()}

    def match_pair(self, left: ChannelSpec, right: ChannelSpec) -> dict[str, Any] | None:
        """Resolve both acquired channels of a temporary reader montage, including contact aliases."""
        contacts = []
        for channel in (left, right):
            if channel.name.casefold() in self.labels:
                contacts.append(channel.name)
            else:
                contacts.extend(channel.contacts or [channel.name])
        return self.match(f"{left.name}-{right.name}", contacts)

    def match(self, channel_name: str, contacts: Iterable[str] = ()) -> dict[str, Any] | None:
        if not self.labels:
            return None
        direct = self.labels.get(channel_name.strip().casefold())
        if direct:
            matches = [direct]
        else:
            matches = [self.labels.get(contact.strip().casefold()) for contact in _channel_contacts(channel_name, contacts)]
        matched = [item for item in matches if item is not None]
        if not matched:
            return None
        regions = [item[1]["region"] if item is not None else "unlabeled" for item in matches]
        region = regions[0] if len({r.casefold() for r in regions}) == 1 else " → ".join(regions)
        native = self.overlay["patient_native"] is True
        result = {
            "region": region, "display_label": region if native else f"{region} (non-patient-native)",
            "patient_native": native, "source": self.overlay["source"], "atlas": self.overlay.get("atlas"),
            "matched_labels": {key: value for key, value in matched},
        }
        # A channel-level tissue or lesion flag requires agreement across both contacts.
        if len(matched) == len(matches):
            first = matched[0][1]
            for field in ("hemisphere", "tissue", "in_lesion", "note"):
                if field in first and all(field in entry and entry[field] == first[field] for _, entry in matched):
                    result[field] = first[field]
        return result


def match_anatomy(
    overlay: dict[str, Any] | None, channel_name: str, contacts: Iterable[str] = (),
) -> dict[str, Any] | None:
    return AnatomyMatcher(overlay).match(channel_name, contacts)


def anatomy_coverage(overlay: dict[str, Any] | None, rows: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """Report coverage within the supplied recording or channel selection."""
    labels = {name.casefold(): name for name in (overlay or {}).get("labels", {})}
    matched_keys: set[str] = set()
    channels_without: list[str] = []
    for row in rows:
        name = row["channel_name"]
        tokens = {s.strip().casefold() for s in [name, *_channel_contacts(name, row.get("contacts") or [])]}
        matches = tokens.intersection(labels)
        if matches:
            matched_keys.update(matches)
        else:
            channels_without.append(name)
    return {
        "unmatched_label_keys": [key for token, key in labels.items() if token not in matched_keys],
        "channels_without_labels": channels_without,
    }


def recording_anatomy_coverage(
    store: WorkbenchStore, case: str, overlay: dict[str, Any] | None,
) -> dict[str, dict[str, list[str]]]:
    """Compare names with each registered acquisition, including cases without analysis runs."""
    coverage = {}
    for recording in store.list_recordings(case):
        channels = store.load_recording(case, recording).channels
        rows = ({"channel_name": ch.name, "contacts": ch.contacts} for ch in channels)
        coverage[recording] = anatomy_coverage(overlay, rows)
    return coverage
