"""
ctpelvimetry — Automated CT pelvimetry and body composition analysis.

Public API:
    run_combined_pelvimetry   — analyse existing segmentation
    run_full_pipeline         — DICOM → NIfTI → Seg → Pelvimetry
    process_single_patient    — body composition for one patient
"""

__version__ = "6.2.0"

from .pipeline import run_combined_pelvimetry, run_full_pipeline
from .body_composition import process_single_patient
