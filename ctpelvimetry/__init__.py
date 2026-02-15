"""
ctpelvimetry — Automated CT pelvimetry and body composition analysis.

Public API:
    run_combined_pelvimetry   — analyse existing segmentation
    run_full_pipeline         — DICOM → NIfTI → Seg → Pelvimetry
    process_single_patient    — body composition for one patient
"""

__version__ = "1.0.0"

from .pipeline import run_combined_pelvimetry as run_combined_pelvimetry  # noqa: F401
from .pipeline import run_full_pipeline as run_full_pipeline  # noqa: F401
from .body_composition import process_single_patient as process_single_patient  # noqa: F401
