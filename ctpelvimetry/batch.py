"""
Batch processing for pelvimetry and body composition.
"""

import os
import gc
import traceback
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .pipeline import run_full_pipeline, run_nifti_pipeline
from .body_composition import process_single_patient


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

# Canonical column order for pelvimetry batch output. Shared between
# DICOM and NIfTI batch entry points so the CSV schema stays consistent.
_PELVIMETRY_PRIORITY_COLS: list[str] = [
    "Patient_ID", "Status", "Error_Log",
    "ISD_mm", "Inlet_AP_mm", "Outlet_AP_mm",
    "Outlet_Transverse_mm", "Outlet_Area_cm2",
    "Sacral_Length_mm", "Sacral_Depth_mm",
    "Promontory_x", "Promontory_y", "Promontory_z",
    "Upper_Symphysis_x", "Upper_Symphysis_y", "Upper_Symphysis_z",
    "Lower_Symphysis_x", "Lower_Symphysis_y", "Lower_Symphysis_z",
    "Sacral_Apex_x", "Sacral_Apex_y", "Sacral_Apex_z",
    "ISD_L_x", "ISD_L_y", "ISD_L_z",
    "ISD_R_x", "ISD_R_y", "ISD_R_z",
    "IT_L_x", "IT_L_y", "IT_L_z",
    "IT_R_x", "IT_R_y", "IT_R_z",
    "CT_NIfTI",
]

_PELVIMETRY_METRICS: list[str] = [
    "ISD_mm", "Inlet_AP_mm", "Outlet_AP_mm",
    "Outlet_Transverse_mm", "Sacral_Length_mm", "Sacral_Depth_mm",
]


def _save_and_summarize_pelvimetry_batch(
    results: list[dict], output_csv: str
) -> pd.DataFrame:
    """Aggregate per-patient results, save CSV, print failure summary.

    Used by both ``run_pelvimetry_batch`` (DICOM input) and
    ``run_pelvimetry_nifti_batch`` (NIfTI input) to keep the output
    schema and reporting identical regardless of input format.

    Parameters
    ----------
    results : list of dict
        Per-patient result dictionaries (from ``run_full_pipeline``
        or ``run_nifti_pipeline``).
    output_csv : str
        Destination CSV path.

    Returns
    -------
    pandas.DataFrame
        Aggregated results with priority columns ordered first.
    """
    df = pd.DataFrame(results)

    existing_cols = [c for c in _PELVIMETRY_PRIORITY_COLS if c in df.columns]
    other_cols = [c for c in df.columns if c not in _PELVIMETRY_PRIORITY_COLS]
    df = df[existing_cols + other_cols]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ Results saved to: {output_csv}")

    # ---- Failure Summary ----
    n_total = len(df)
    n_success = (df["Status"] == "Success").sum() if "Status" in df.columns else 0
    n_partial = (
        df["Status"].str.startswith("Partial").sum() if "Status" in df.columns else 0
    )
    n_failure = (df["Status"] == "Failure").sum() if "Status" in df.columns else 0

    print(f"\n{'='*50}")
    print(f"  FAILURE SUMMARY  ({n_total} patients)")
    print(f"{'='*50}")
    print(f"  Success : {n_success}")
    print(f"  Partial : {n_partial}")
    print(f"  Failure : {n_failure}")

    has_missing = False
    for m in _PELVIMETRY_METRICS:
        if m in df.columns:
            n_miss = df[m].isna().sum()
            if n_miss > 0:
                if not has_missing:
                    print("\n  Missing metrics:")
                    has_missing = True
                print(f"    {m:30s} {n_miss:3d}/{n_total} ({100*n_miss/n_total:.0f}%)")

    if "Error_Log" in df.columns:
        codes: Counter = Counter()
        for log in df["Error_Log"].dropna():
            for part in str(log).split("; "):
                code = part.split(" - ")[0].strip()
                if code:
                    codes[code] += 1
        if codes:
            print("\n  Top error codes:")
            for code, count in codes.most_common(10):
                print(f"    {code:45s} {count:3d}")
    print(f"{'='*50}\n")

    return df


# ------------------------------------------------------------------
# Pelvimetry batch
# ------------------------------------------------------------------

def run_pelvimetry_batch(
    dicom_root, output_root, output_csv, start=1, end=250,
    use_fast=False, skip_tissue=False,
):
    """Run pelvimetry in batch mode with per-patient error isolation.

    Iterates over patient directories numbered ``Patient_<start>`` to
    ``Patient_<end>``, running the full pipeline for each.  Results
    are saved to a CSV with a failure summary printed to stdout.

    Parameters
    ----------
    dicom_root : str
        Root directory containing per-patient DICOM folders.
    output_root : str
        Root output directory for NIfTI / segmentations / QC.
    output_csv : str
        Path for the results CSV.
    start, end : int, optional
        Patient-number range (default 1–250).
    use_fast : bool, optional
        Use TotalSegmentator ``--fast`` mode (default ``False``).
    skip_tissue : bool, optional
        Skip tissue-type segmentation (default ``False``).

    Returns
    -------
    pandas.DataFrame
        Aggregated results for all patients.
    """
    results = []

    # Find patient folders
    patient_folders = []
    for i in range(start, end + 1):
        pid = f"Patient_{i:03d}"
        dicom_path = os.path.join(dicom_root, pid)
        if os.path.isdir(dicom_path):
            patient_folders.append({"patient_id": pid, "dicom_path": dicom_path})

    print(f"🔬 Found {len(patient_folders)} patients to process")
    if not patient_folders:
        warnings.warn(
            f"No patient directories found in '{dicom_root}' "
            f"(searched Patient_{start:03d}..Patient_{end:03d}). "
            "Check that the path and patient naming convention are correct.",
            stacklevel=2,
        )

    for patient in tqdm(patient_folders, desc="Processing"):
        pid = patient["patient_id"]
        dicom_path = patient["dicom_path"]

        try:
            result = run_full_pipeline(
                pid, dicom_path, output_root, use_fast=use_fast, skip_tissue=skip_tissue
            )
            results.append(result)
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ {pid}: {error_msg}")
            traceback.print_exc()
            results.append(
                {"Patient_ID": pid, "Status": "Error", "Error_Message": error_msg}
            )

        gc.collect()

    return _save_and_summarize_pelvimetry_batch(results, output_csv)


# ------------------------------------------------------------------
# Pelvimetry batch (NIfTI input)
# ------------------------------------------------------------------

def run_pelvimetry_nifti_batch(
    nifti_root: str,
    output_root: str,
    output_csv: str,
    pattern: str = "*.nii.gz",
    use_fast: bool = False,
    skip_tissue: bool = False,
) -> pd.DataFrame:
    """Run pelvimetry batch on a directory of NIfTI files.

    Scans *nifti_root* for files matching *pattern* (non-recursive).
    Each matching file is processed as one patient via
    :func:`run_nifti_pipeline`. The patient ID is derived from the
    filename: ``case_001.nii.gz`` becomes ``case_001`` (both ``.nii``
    and ``.gz`` extensions are stripped).

    Use this entry point for public datasets distributed in NIfTI
    format (Medical Decathlon, TotalSegmentator benchmark, KiTS,
    etc.) — no DICOM directory layout required.

    Parameters
    ----------
    nifti_root : str
        Directory containing input ``.nii.gz`` (or ``.nii``) files.
    output_root : str
        Root output directory for segmentations and QC.
    output_csv : str
        Destination CSV path for aggregated results.
    pattern : str, optional
        Glob pattern for matching files (default ``"*.nii.gz"``).
        Use ``"*.nii"`` for uncompressed NIfTI.
    use_fast : bool, optional
        Use TotalSegmentator ``--fast`` mode (default ``False``).
    skip_tissue : bool, optional
        Skip tissue-type segmentation (default ``False``).

    Returns
    -------
    pandas.DataFrame
        Aggregated results for all patients with the same column
        schema as ``run_pelvimetry_batch``.
    """
    nifti_files = sorted(Path(nifti_root).glob(pattern))

    print(f"🔬 Found {len(nifti_files)} NIfTI files to process")
    if not nifti_files:
        warnings.warn(
            f"No files matching '{pattern}' found in '{nifti_root}'. "
            "Check the path and pattern.",
            stacklevel=2,
        )

    results: list[dict] = []
    for nifti_path in tqdm(nifti_files, desc="Processing"):
        # Strip .gz then .nii so case_001.nii.gz -> case_001.nii -> case_001
        pid = nifti_path.name
        for ext in (".gz", ".nii"):
            if pid.endswith(ext):
                pid = pid[: -len(ext)]

        try:
            result = run_nifti_pipeline(
                pid,
                str(nifti_path),
                output_root,
                use_fast=use_fast,
                skip_tissue=skip_tissue,
            )
            results.append(result)

        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ {pid}: {error_msg}")
            traceback.print_exc()
            results.append(
                {"Patient_ID": pid, "Status": "Error", "Error_Message": error_msg}
            )

        gc.collect()

    return _save_and_summarize_pelvimetry_batch(results, output_csv)


# ------------------------------------------------------------------
# Body composition batch
# ------------------------------------------------------------------

def run_body_composition_batch(
    seg_root, nifti_root, pelvimetry_csv, output_csv,
    qc_root=None, start=1, end=250,
):
    """Run body composition analysis in batch mode.

    Iterates over patient directories numbered ``Patient_<start>``
    to ``Patient_<end>``, computes VAT / SAT / SMA at L3 and ISD
    levels, and saves results to a CSV.

    Parameters
    ----------
    seg_root : str
        Root directory containing segmentation subfolders.
    nifti_root : str
        Root directory containing per-patient NIfTI files.
    pelvimetry_csv : str
        Path to the pelvimetry report CSV (for ISD slice).
    output_csv : str
        Path for the body-composition results CSV.
    qc_root : str or None, optional
        Root directory for QC figures (default ``None``).
    start, end : int, optional
        Patient-number range (default 1–250).

    Returns
    -------
    pandas.DataFrame
        Aggregated body-composition results.
    """
    patients = []
    for i in range(start, end + 1):
        patient_id = f"Patient_{i:03d}"
        seg_folder1 = os.path.join(seg_root, patient_id, "segmentations")
        seg_folder2 = os.path.join(seg_root, patient_id)
        if os.path.exists(seg_folder1) or os.path.exists(seg_folder2):
            patients.append(patient_id)

    print(f"🔬 Found {len(patients)} patients to process")
    if not patients:
        import warnings
        warnings.warn(
            f"No patient directories found in '{seg_root}' "
            f"(searched Patient_{start:03d}..Patient_{end:03d}). "
            "Check that the path and patient naming convention are correct.",
            stacklevel=2,
        )

    results = []

    for patient_id in tqdm(patients, desc="Processing"):
        print(f"\n{'='*50}")
        print(f"🔹 Processing: {patient_id}")
        print(f"{'='*50}")

        nifti_path = os.path.join(nifti_root, patient_id, f"{patient_id}.nii.gz")
        qc_dir = os.path.join(qc_root, patient_id) if qc_root else None

        try:
            result = process_single_patient(
                patient_id, seg_root, nifti_path, pelvimetry_csv, qc_dir
            )
            results.append(result)
        except Exception as e:
            print(f"   ❌ Error processing {patient_id}: {e}")
            traceback.print_exc()
            results.append({"Patient_ID": patient_id, "Status": "Error", "Error_Message": str(e)})

    # Save results
    df = pd.DataFrame(results)

    priority_cols = [
        "Patient_ID", "Status",
        "L3_slice", "L3_VAT_cm2", "L3_SAT_cm2", "L3_VS_ratio", "L3_SMA_cm2",
        "ISD_slice", "ISD_VAT_cm2", "ISD_SAT_cm2", "ISD_VS_ratio", "ISD_SMA_cm2",
    ]

    existing_cols = [c for c in priority_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[existing_cols + other_cols]

    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ Results saved to: {output_csv}")

    # Summary
    success_count = (df["Status"] == "Success").sum() if "Status" in df.columns else 0
    print("\n📊 Summary:")
    print(f"   Total processed: {len(df)}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {len(df) - success_count}")

    return df
