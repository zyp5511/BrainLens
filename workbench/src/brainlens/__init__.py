"""BrainLens (v1.0) — Standalone SEEG/iEEG/EEG Neurophysiology Workbench & AI Analysis Engine."""

from __future__ import annotations

__version__ = "1.0.0"

from .analytics import (
    compare_recordings,
    compare_runs,
    compute_overlay_concordance,
    query_case,
    select_control_windows,
)
from .bids_export import export_bids_events
from .case_sheet import export_case_sheet
from .cohort import list_cohorts, run_cohort
from .edf_export import export_edf_annotations
from .io_readers import inspect_case
from .jobs import JobManager
from .mdt_slides import export_mdt_slides
from .models import AnalysisConfig, ToolkitError
from .pipeline import analyze_case
from .protocols import list_protocols, save_protocol
from .storage import WorkbenchStore
from .triad_views import (
    compute_multichannel_batch,
    compute_triad_view,
    render_evidence_card,
    render_spatial_figure,
)

__all__ = [
    "__version__",
    "AnalysisConfig",
    "JobManager",
    "ToolkitError",
    "WorkbenchStore",
    "inspect_case",
    "analyze_case",
    "query_case",
    "compare_runs",
    "compare_recordings",
    "compute_overlay_concordance",
    "select_control_windows",
    "compute_multichannel_batch",
    "compute_triad_view",
    "render_evidence_card",
    "render_spatial_figure",
    "export_case_sheet",
    "export_mdt_slides",
    "export_bids_events",
    "export_edf_annotations",
    "run_cohort",
    "list_cohorts",
    "save_protocol",
    "list_protocols",
]
