"""PNG evidence cards for individual recording/run maps and ictal EI analyses."""

from __future__ import annotations

import base64
import hashlib
import io
import math
import tempfile
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .evidence_map import SCOPE
from .storage import WorkbenchStore, sanitize_slug


_COLUMN_TITLES = {
    "ripple": "Focal ripple\n/min", "fast_ripple": "Focal fast ripple\n/min",
    "spike_hfo": "Spike-HFO\n/min", "ied": "IED candidates\n/min", "ei": "Ictal EI\n0–1",
}


def _origin(result: dict[str, Any]) -> str:
    provenance = result.get("provenance", {})
    if provenance.get("is_synthetic"):
        return "synthetic engineering data"
    return provenance.get("dataset_name") or provenance.get("origin_kind", "").replace("_", " ")


def _png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    FigureCanvasAgg(fig).print_png(buf)
    return buf.getvalue()


def _save_card(
    fig: Figure, store: WorkbenchStore, case: str, recording: str, run_name: str | None,
    view_mode: str, return_base64: bool,
) -> dict[str, Any]:
    png = _png(fig)
    cards_dir = store.case_dir(case) / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(png).hexdigest()[:12]
    prefix = sanitize_slug(f"{recording}_{run_name or 'source'}_{view_mode}")
    path = cards_dir / f"{prefix}_{digest}.png"
    with tempfile.NamedTemporaryFile(dir=cards_dir, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            tmp.write(png)
            tmp.close()
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)
    result = {
        "image_path": str(path), "case": case, "recording": recording,
        "run_name": run_name, "view_mode": view_mode, "scope": SCOPE,
    }
    if return_base64:
        result["png_base64"] = base64.b64encode(png).decode("ascii")
    return result


def _map_figure(
    evidence: dict[str, Any], display_text: Callable[[str], str] = str,
    recording_labels: dict[str, str] | None = None,
) -> Figure:
    rows = evidence["rows"]
    keys = [col["key"] for col in evidence["columns"]]
    labels = [_COLUMN_TITLES[key] for key in keys] + ["Top tier /\navailable"]
    show_tags = evidence.get("external_overlay_available", False)
    if show_tags:
        labels.append("External\ntags")
    # Repeat column headings across blocks to keep large implantations readable.
    block_count = max(1, math.ceil(len(rows) / 32))
    ncols = min(2, block_count)
    nrows = math.ceil(block_count / ncols)
    block_size = max(1, math.ceil(len(rows) / block_count))
    # Leave room for the biomarker columns when rows include external anatomical
    # labels. Wrap only after disclosure, so a private name cannot straddle lines.
    row_labels = [textwrap.fill(display_text(row["channel"]), width=38, break_on_hyphens=False) for row in rows]
    label_lines = [line for name in row_labels for line in name.splitlines()]
    longest = max(map(len, label_lines), default=0)
    lines_per_row = max((name.count("\n") + 1 for name in row_labels), default=1)
    width = ncols * (8.2 + max(0, longest - 10) * 0.07)
    height = 2.4 + nrows * (0.95 + block_size * (0.26 + 0.13 * (lines_per_row - 1)))
    fig = Figure(figsize=(width, height), dpi=130, layout="constrained")
    outer = fig.add_gridspec(3, 1, height_ratios=[0.9, height - 2.37, 1.47])
    title = fig.add_subplot(outer[0])
    title.axis("off")
    title.text(0, 0.78, "Multi-biomarker evidence map", weight="bold", fontsize=15)
    def recording_label(key: str) -> str:
        return recording_labels[key] if recording_labels is not None else display_text(key)

    rec = recording_label(evidence["recording_id"])
    source = evidence.get("ei_source")
    ei_label = (f"{recording_label(source['recording_id'])}  ·  {display_text(_origin(source))}  ·  onset {source['onset_seconds']:g} s"
                if source else "unavailable")
    title.text(0, 0.37, f"HFO and IED  ·  {rec} / {display_text(evidence['run_name'])}  ·  {display_text(_origin(evidence))}",
               fontsize=9, parse_math=False, wrap=True)
    title.text(0, 0.02, f"Ictal EI  ·  {ei_label}",
               fontsize=9, parse_math=False, wrap=True)

    grid = outer[1].subgridspec(nrows, ncols, wspace=0.12)
    cmap = colormaps["viridis"]
    for block in range(block_count):
        ax = fig.add_subplot(grid[block // ncols, block % ncols])
        chunk = rows[block * block_size:(block + 1) * block_size]
        for y, row in enumerate(chunk):
            for x, key in enumerate(keys):
                cell = row["biomarkers"][key]
                value, percentile = cell["value"], cell["percentile"]
                if value is None or percentile is None:
                    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#f7f7f7",
                                           edgecolor="#a3acb1", linewidth=0.4, hatch="///"))
                else:
                    color = cmap(percentile / 100)
                    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color,
                                           edgecolor="white", linewidth=0.4))
                    ink = "#fff" if percentile < 52 else "#101b1a"
                    ax.text(x, y, f"{value:.3g}", ha="center", va="center", fontsize=8, color=ink)
            count = row["convergence_count"]
            label = "—" if count is None else f"{count} / {row['available_count']}"
            ax.text(len(keys), y, label, ha="center", va="center", fontsize=8)
            if show_tags:
                ax.text(len(keys) + 1, y, display_text(", ".join(row["external_tags"]) or "—"),
                        ha="center", va="center", fontsize=8, parse_math=False)
        ax.set(xlim=(-0.5, len(labels) - 0.5), ylim=(max(1, len(chunk)) - 0.5, -0.5))
        ax.set_xticks(range(len(labels)), labels=labels, fontsize=9)
        ax.set_yticks(range(len(chunk)), labels=row_labels[block * block_size:(block + 1) * block_size],
                      fontsize=8, parse_math=False)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)

    footer = fig.add_subplot(outer[2])
    footer.set_axis_off()
    scale = footer.inset_axes([0, 0.73, 0.34, 0.11])
    scale.imshow(np.linspace(0, 100, 101)[None, :], cmap=cmap, aspect="auto", extent=(0, 100, 0, 1))
    scale.set_yticks([])
    scale.set_xticks([0, 50, 100], labels=["0", "50", "100"], fontsize=8)
    footer.text(0.38, 0.78, "Within-recording percentile   ·   Hatched = unavailable or unassessed",
                va="center", fontsize=8)
    footer.text(0, 0.40, "Top tier requires a positive value at or above P80. Convergence count is descriptive, "
                "not a validated predictor.", fontsize=8)
    footer.text(0, 0.20, "IED candidates — require visual verification.", fontsize=8)
    footer.text(0, 0.0, SCOPE, fontsize=8)
    return fig


def render_map_card(
    store: WorkbenchStore, case_id: str, recording: str | None = None, run_name: str | None = None,
    return_base64: bool = False,
    *, snapshot: dict[str, Any] | None = None, display_text: Callable[[str], str] | None = None,
    recording_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    # A captured, verified export snapshot must never fall back to a cache lookup
    # (or a calculation on cache miss). Its display-only PNG stays in memory.
    if snapshot is not None:
        png = _png(_map_figure(snapshot, display_text or str, recording_labels))
        return {"png_base64": base64.b64encode(png).decode("ascii"), "image_path": None,
                "case": case_id, "recording": recording, "run_name": run_name, "view_mode": "evidence_map"}

    from .evidence_map import get_evidence_map

    result = get_evidence_map(store, case_id, recording=recording, run_name=run_name)
    return _save_card(_map_figure(result), store, case_id, result["recording_id"],
                      result["run_name"], "evidence_map", return_base64)


def _ei_figure(ei: dict[str, Any], interictal: dict[str, Any] | None) -> Figure:
    fig = Figure(figsize=(10.8, 8.2 if interictal else 5.4), dpi=140, layout="constrained")
    outer = fig.add_gridspec(3 if interictal else 2, 1,
                             height_ratios=([3.7, 2, 0.38] if interictal else [3.7, 0.38]))
    rec = ei["recording_id"]
    onset = ei["onset_seconds"]
    fig.suptitle(f"Epileptogenicity Index  ·  {rec}  ·  {_origin(ei)}  ·  onset {onset:g} s",
                 fontsize=13, weight="bold", parse_math=False)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.65, 1])
    er_ax, ei_ax = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
    ranked = sorted((row for row in ei["channels"] if row["ei"] is not None),
                    key=lambda row: (-row["ei"], row["channel"]))
    times = np.asarray(ei["times_seconds"])
    er_ax.axvline(onset, color="#505663", linestyle="--", linewidth=1, label="Supplied onset")
    for row in ranked[:4]:
        er = np.asarray(row["er"], dtype=float)
        line, = er_ax.plot(times, er, linewidth=1, label=row["channel"])
        detection = row["detection_seconds"]
        if detection is not None and np.isfinite(er).any():
            valid = np.isfinite(er)
            idx = np.abs(times[valid] - detection).argmin()
            er_ax.plot(detection, er[valid][idx], "o", ms=6, mfc="white", mec=line.get_color(), mew=1.7)
    er_ax.set_title("Energy ratio of leading channels", fontsize=10)
    er_ax.set_xlabel("Recording time (s)  ·  circles mark detections", fontsize=9)
    er_ax.set_ylabel("(beta + gamma) / (theta + alpha)", fontsize=9)
    er_ax.grid(alpha=0.18)
    er_ax.legend(fontsize=8, loc="best")

    bars = ranked[:10]
    ei_ax.barh(range(len(bars)), [row["ei"] for row in bars], color="#1b7971", height=0.62)
    ei_ax.set_yticks(range(len(bars)), labels=[row["channel"] for row in bars], fontsize=8, parse_math=False)
    ei_ax.invert_yaxis()
    ei_ax.set(xlim=(0, 1.05), xlabel="Normalized EI")
    ei_ax.set_title(f"EI by channel ({len(bars)} of {len(ranked)} assessed)", fontsize=10)
    if not bars:
        ei_ax.text(0.5, 0.5, "No assessed channels", ha="center", transform=ei_ax.transAxes)
    ei_ax.grid(axis="x", alpha=0.18)
    for ax in (er_ax, ei_ax):
        ax.spines[["top", "right"]].set_visible(False)
    if interictal:
        comparison = outer[1].subgridspec(1, 2)
        panels = [
            ("Interictal HFO leaders", interictal["recording_id"],
             f"Run {interictal['run_name']}  ·  focal HFO/min  ·  {interictal['state']}", interictal["leaders"]),
            ("Ictal EI leaders", rec, f"Onset {onset:g} s  ·  normalized EI",
             [{"channel": row["channel"], "value": row["ei"]} for row in ranked if row["ei"] > 0]),
        ]
        for i, (heading, source, metric, leaders) in enumerate(panels):
            ax = fig.add_subplot(comparison[0, i])
            ax.axis("off")
            ax.set_title(textwrap.fill(f"{heading} (recording {source})", width=53),
                         loc="left", fontsize=10, weight="bold", parse_math=False)
            origin = _origin(interictal if i == 0 else ei)
            ax.text(0, 0.83, f"{metric}\n{origin}", size=8, color="#465561", parse_math=False)
            if not leaders:
                ax.text(0, 0.66, "No positive measured values", fontsize=9)
            for j, row in enumerate(leaders[:5]):
                y = 0.7 - j * 0.16
                ax.text(0.02, y, row["channel"], fontsize=9, parse_math=False)
                ax.text(0.90, y, f"{row['value']:.3g}", ha="right", fontsize=9)
    foot = fig.add_subplot(outer[-1])
    foot.axis("off")
    foot.text(0, 0.38, SCOPE, fontsize=8)
    return fig


def render_ei_card(
    store: WorkbenchStore, case_id: str, recording: str | None = None, run_name: str | None = None,
    return_base64: bool = False,
) -> dict[str, Any]:
    from .evidence_map import get_epileptogenicity, get_interictal_leaders, get_latest_ei

    result = get_latest_ei(store, case_id, recording=recording)
    if result is None:
        result = get_epileptogenicity(store, case_id, recording=recording, run_name=run_name)
    source = result["recording_id"]
    interictal = get_interictal_leaders(store, case_id, source)
    card = _save_card(_ei_figure(result, interictal), store, case_id, source,
                      result.get("run_name"), "epileptogenicity", return_base64)
    card["onset_seconds"] = result["onset_seconds"]
    card["interictal_recording"] = interictal["recording_id"] if interictal else None
    return card
