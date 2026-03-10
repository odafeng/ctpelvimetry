"""
Pipeline orchestration: run_combined_pelvimetry and run_full_pipeline.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import nibabel as nib

from .config import DEFAULT_PELVIC_CONFIG, PelvicConfig
from .io import load_mask_canonical, voxel_to_world_ras
from .conversion import convert_dicom_to_nifti
from .segmentation import setup_license, run_totalsegmentator
from .landmarks import (
    compute_midline_x,
    detect_pelvic_orientation,
    find_sacral_landmarks,
    find_symphysis_midline_sagittal,
)
from .metrics import calculate_isd, calculate_sacral_depth, find_outlet_transverse
from .qc import save_sagittal_combined_qc_figure, save_extended_qc_figure

# Canonical keys that every result dict must contain (for DataFrame safety)
_RESULT_KEYS: list[str] = [
    "Patient_ID", "Status", "Error_Log",
    "ISD_mm", "ISD_slice",
    "Inlet_AP_mm", "Outlet_AP_mm",
    "Outlet_Transverse_mm", "Outlet_Area_cm2",
    "Sacral_Length_mm", "Sacral_Depth_mm",
    "Promontory_x", "Promontory_y", "Promontory_z",
    "Coccygeal_Apex_x", "Coccygeal_Apex_y", "Coccygeal_Apex_z",
    "Upper_Symphysis_x", "Upper_Symphysis_y", "Upper_Symphysis_z",
    "Lower_Symphysis_x", "Lower_Symphysis_y", "Lower_Symphysis_z",
    "Sacral_Max_Depth_Pt_x", "Sacral_Max_Depth_Pt_y", "Sacral_Max_Depth_Pt_z",
    "ISD_L_x", "ISD_L_y", "ISD_L_z",
    "ISD_R_x", "ISD_R_y", "ISD_R_z",
    "IT_L_x", "IT_L_y", "IT_L_z",
    "IT_R_x", "IT_R_y", "IT_R_z",
    "Pelvic_Rotation_deg", "Pelvic_Tilt_deg",
    "Pelvic_Rotation_Flag", "Pelvic_Tilt_Flag",
    "Sacrum_Offset_mm", "Sacrum_Offset_Flag",
    "CT_NIfTI",
    "Seg_Sacrum", "Seg_Vertebrae_S1",
    "Seg_Hip_Left", "Seg_Hip_Right",
    "Seg_Femur_Left", "Seg_Femur_Right",
    "Seg_Torso_Fat", "Seg_Subcutaneous_Fat", "Seg_Skeletal_Muscle",
    "Seg_Vertebrae",
]


def _normalise_result(result: dict) -> dict:
    """Ensure *result* contains every canonical key (fill missing with None)."""
    for key in _RESULT_KEYS:
        result.setdefault(key, None)
    return result


# ------------------------------------------------------------------
# Main Combined Analysis Function
# ------------------------------------------------------------------

def run_combined_pelvimetry(
    patient_id: str,
    seg_folder: str,
    nifti_path: str,
    qc_dir: Optional[str] = None,
    config: Optional[PelvicConfig] = None,
) -> dict:
    """Run combined pelvimetry analysis with per-metric error isolation.

    Each of the six metrics is wrapped in its own ``try``/``except``
    so that a failure in one metric does not affect the others.

    Status semantics
    ^^^^^^^^^^^^^^^^
    * ``'Success'``       — all 6 metrics produced.
    * ``'Partial_N/6'``   — 1–5 metrics produced.
    * ``'Failure'``       — 0 metrics produced.

    Parameters
    ----------
    patient_id : str
        Patient identifier.
    seg_folder : str
        Path to the segmentation mask folder.
    nifti_path : str
        Path to the CT NIfTI file.
    qc_dir : str or None, optional
        Directory for QC image output (default ``None``).
    config : PelvicConfig or None, optional
        Detection parameters.  When ``None`` the package-level
        ``DEFAULT_PELVIC_CONFIG`` is used.

    Returns
    -------
    dict
        Measurement results, world-space landmark coordinates,
        overall ``'Status'``, and ``'Error_Log'``.  The dict always
        contains every canonical key; missing values are ``None``.
    """
    patient_id = str(patient_id)
    if config is None:
        config = DEFAULT_PELVIC_CONFIG

    result = {"Patient_ID": patient_id}
    landmarks = {}
    error_parts = []

    # --------------------------------------------------
    # Load masks — MUST use as_closest_canonical() for RAS
    # --------------------------------------------------
    hip_L, header, img_affine = load_mask_canonical(
        os.path.join(seg_folder, "hip_left.nii.gz")
    )
    hip_R, _, _ = load_mask_canonical(os.path.join(seg_folder, "hip_right.nii.gz"))
    sacrum, _, _ = load_mask_canonical(os.path.join(seg_folder, "sacrum.nii.gz"))
    femur_L, _, _ = load_mask_canonical(os.path.join(seg_folder, "femur_left.nii.gz"))
    femur_R, _, _ = load_mask_canonical(os.path.join(seg_folder, "femur_right.nii.gz"))

    # Load vertebrae_S1 for improved promontory detection
    vert_S1, _, _ = load_mask_canonical(os.path.join(seg_folder, "vertebrae_S1.nii.gz"))

    if hip_L is None or hip_R is None:
        result["Status"] = "Failure"
        result["Error_Log"] = "ALL: ISD_NO_HIP_MASK - hip_left or hip_right mask missing"
        return _normalise_result(result)

    sx, sy, sz = header.get_zooms()[:3]

    # Load CT for QC figures
    ct_vol = None
    if qc_dir and nifti_path and os.path.exists(nifti_path):
        try:
            ct_img = nib.load(nifti_path)
            ct_img = nib.as_closest_canonical(ct_img)
            ct_vol = ct_img.get_fdata()
        except Exception as e:
            print(f"      ⚠️ Failed to load CT for QC: {e}")

    # Build data dict
    isd_data = {}
    if hip_L is not None:
        isd_data["hip_L"] = hip_L
    if hip_R is not None:
        isd_data["hip_R"] = hip_R
    if sacrum is not None:
        isd_data["sacrum"] = sacrum
    if femur_L is not None:
        isd_data["femur_L"] = femur_L
    if femur_R is not None:
        isd_data["femur_R"] = femur_R

    # ========================================
    # Compute Midline X
    # ========================================
    mid_x = None
    slab_half_voxels = max(1, int(round(3.0 / sx)))
    try:
        mid_x = compute_midline_x(hip_L, hip_R)
        if mid_x is not None:
            print(
                f"      ✅ Midline X: {mid_x:.1f} voxel, "
                f"slab ± {slab_half_voxels} voxels (± {slab_half_voxels * sx:.1f} mm)"
            )
        else:
            print("      ⚠️ Midline X computation failed")
    except Exception as e:
        print(f"      ⚠️ Midline X exception: {e}")

    # ========================================
    # Quality Gate: Pelvic Orientation
    # ========================================
    orientation = None
    try:
        orientation = detect_pelvic_orientation(hip_L, hip_R, img_affine)
        if orientation is not None:
            rot = orientation["rotation_deg"]
            tlt = orientation["tilt_deg"]
            result["Pelvic_Rotation_deg"] = rot
            result["Pelvic_Tilt_deg"] = tlt
            result["Pelvic_Rotation_Flag"] = orientation["rotation_flag"]
            result["Pelvic_Tilt_Flag"] = orientation["tilt_flag"]

            rot_icon = {"ok": "✅", "warn": "⚠️", "high": "🔶"}
            print(f"      {rot_icon[orientation['rotation_flag']]} Axial rotation: {rot}°"
                  f"  ({orientation['rotation_flag']})")
            print(f"      {rot_icon[orientation['tilt_flag']]} Coronal tilt: {tlt}°"
                  f"  ({orientation['tilt_flag']})")
    except Exception as e:
        print(f"      ⚠️ Pelvic orientation detection exception: {e}")

    # ========================================
    # Pre-compute shared landmarks
    # ========================================

    # --- Sacral landmarks ---
    sacral = {}
    if sacrum is not None and np.sum(sacrum) > 0:
        try:
            sacrum_for_prom = sacrum.copy()
            if vert_S1 is not None and vert_S1.shape == sacrum.shape:
                sacrum_for_prom = np.maximum(sacrum_for_prom, vert_S1)
                print("      ✅ Merged vertebrae_S1 into sacrum for promontory detection")
            else:
                if vert_S1 is None:
                    print("      ⚠️ vertebrae_S1.nii.gz not found, using sacrum only")
                else:
                    print(
                        f"      ⚠️ vertebrae_S1 shape mismatch "
                        f"({vert_S1.shape} vs {sacrum.shape}), using sacrum only"
                    )

            sacral = find_sacral_landmarks(
                sacrum_for_prom, sacrum=sacrum, mid_x=mid_x, slab_half_voxels=slab_half_voxels
            )
        except Exception as e:
            print(f"      ⚠️ Sacral landmark detection failed: {e}")

    if "midline_x" in sacral:
        landmarks["midline_x"] = sacral["midline_x"]
    elif mid_x is not None:
        landmarks["midline_x"] = mid_x
    if "promontory" in sacral:
        landmarks["promontory"] = sacral["promontory"]
        ras = voxel_to_world_ras(sacral["promontory"], img_affine)
        if ras is not None:
            result["Promontory_x"] = round(ras[0], 2)
            result["Promontory_y"] = round(ras[1], 2)
            result["Promontory_z"] = round(ras[2], 2)
    if "coccygeal_apex" in sacral:
        landmarks["coccygeal_apex"] = sacral["coccygeal_apex"]
        ras = voxel_to_world_ras(sacral["coccygeal_apex"], img_affine)
        if ras is not None:
            result["Coccygeal_Apex_x"] = round(ras[0], 2)
            result["Coccygeal_Apex_y"] = round(ras[1], 2)
            result["Coccygeal_Apex_z"] = round(ras[2], 2)

    # --- Quality Gate: Sacrum-Symphysis midline offset ---
    sacrum_offset = None
    if mid_x is not None and sacrum is not None and np.sum(sacrum) > 0:
        try:
            sacrum_median_x = float(np.median(np.argwhere(sacrum > 0)[:, 0]))
            offset_voxels = abs(sacrum_median_x - mid_x)
            offset_mm = round(offset_voxels * sx, 1)
            result["Sacrum_Offset_mm"] = offset_mm

            def _offset_flag(val, warn_t, high_t):
                if val >= high_t:
                    return "high"
                elif val >= warn_t:
                    return "warn"
                return "ok"

            flag = _offset_flag(offset_mm,
                                config.sacrum_offset_warn_mm,
                                config.sacrum_offset_fail_mm)
            result["Sacrum_Offset_Flag"] = flag
            sacrum_offset = {"offset_mm": offset_mm, "flag": flag}

            icon = {"ok": "✅", "warn": "⚠️", "high": "🔶"}
            print(f"      {icon[flag]} Sacrum-symphysis offset: {offset_mm} mm"
                  f"  ({flag})")
        except Exception as e:
            print(f"      ⚠️ Sacrum offset detection exception: {e}")

    # --- Symphysis landmarks ---
    symph = {}
    if mid_x is not None:
        try:
            symph = find_symphysis_midline_sagittal(
                hip_L, hip_R, mid_x, slab_half_voxels, sy=sy, sz=sz
            )
        except Exception as e:
            print(f"      ⚠️ Symphysis detection failed: {e}")
    else:
        print("      ⚠️ Symphysis detection skipped: midline X not available")

    if "upper" in symph:
        landmarks["symphysis_upper"] = symph["upper"]
        ras = voxel_to_world_ras(symph["upper"], img_affine)
        if ras is not None:
            result["Upper_Symphysis_x"] = round(ras[0], 2)
            result["Upper_Symphysis_y"] = round(ras[1], 2)
            result["Upper_Symphysis_z"] = round(ras[2], 2)
    if "lower" in symph:
        landmarks["symphysis_lower"] = symph["lower"]
        ras = voxel_to_world_ras(symph["lower"], img_affine)
        if ras is not None:
            result["Lower_Symphysis_x"] = round(ras[0], 2)
            result["Lower_Symphysis_y"] = round(ras[1], 2)
            result["Lower_Symphysis_z"] = round(ras[2], 2)

    # ========================================
    # Metric 1: ISD (Inter-Spinous Distance)
    # ========================================

    isd_result = None
    try:
        isd_result = calculate_isd(isd_data, (sx, sy, sz), img_affine)

        if isd_result.get("ISD_mm") is not None and "Success" in (isd_result.get("Status") or ""):
            result["ISD_mm"] = isd_result["ISD_mm"]
            result["ISD_slice"] = isd_result["ISD_slice"]

            pt_L = isd_result.get("ISD_pt_left")
            pt_R = isd_result.get("ISD_pt_right")
            isd_slice = isd_result["ISD_slice"]

            if pt_L is not None:
                pt_L_ijk = np.array([pt_L[0], pt_L[1], isd_slice])
                ras_L = voxel_to_world_ras(pt_L_ijk, img_affine)
                if ras_L is not None:
                    result["ISD_L_x"] = round(ras_L[0], 2)
                    result["ISD_L_y"] = round(ras_L[1], 2)
                    result["ISD_L_z"] = round(ras_L[2], 2)
                    landmarks["ISD_L"] = pt_L_ijk

            if pt_R is not None:
                pt_R_ijk = np.array([pt_R[0], pt_R[1], isd_slice])
                ras_R = voxel_to_world_ras(pt_R_ijk, img_affine)
                if ras_R is not None:
                    result["ISD_R_x"] = round(ras_R[0], 2)
                    result["ISD_R_y"] = round(ras_R[1], 2)
                    result["ISD_R_z"] = round(ras_R[2], 2)
                    landmarks["ISD_R"] = pt_R_ijk

        else:
            status_msg = isd_result.get("Status", "Unknown")
            error_parts.append(f"ISD: ISD_NO_VALLEY - ISD detection failed: {status_msg}")
    except Exception as e:
        error_parts.append(f"ISD: ISD_EXCEPTION - {e}")

    # ========================================
    # Metric 3: Inlet AP (Promontory → Upper Symphysis)
    # ========================================

    try:
        if "promontory" not in landmarks:
            error_parts.append("Inlet_AP: INLET_NO_PROMONTORY - Promontory landmark not detected")
        elif "symphysis_upper" not in landmarks:
            error_parts.append("Inlet_AP: INLET_NO_SYMPHYSIS - Upper symphysis not detected")
        else:
            if mid_x is not None:
                p1 = landmarks["promontory"] * np.array([1, sy, sz])
                p2 = landmarks["symphysis_upper"] * np.array([1, sy, sz])
                inlet_val = round(np.linalg.norm(p1[1:] - p2[1:]), 1)
                result["Inlet_AP_mm"] = inlet_val
            else:
                print("      ⚠️ Inlet AP skipped: midline X not available")
                error_parts.append("Inlet_AP: INLET_NO_MIDLINE - Midline X not available")
    except Exception as e:
        error_parts.append(f"Inlet_AP: INLET_EXCEPTION - {e}")

    # ========================================
    # Metric 4: Outlet AP (Coccygeal Apex → Lower Symphysis)
    # ========================================

    try:
        if "coccygeal_apex" not in landmarks:
            error_parts.append("Outlet_AP: OUTLET_NO_APEX - Coccygeal apex not detected")
        elif "symphysis_lower" not in landmarks:
            error_parts.append("Outlet_AP: OUTLET_NO_SYMPHYSIS - Lower symphysis not detected")
        else:
            if mid_x is not None:
                p1 = landmarks["coccygeal_apex"] * np.array([1, sy, sz])
                p2 = landmarks["symphysis_lower"] * np.array([1, sy, sz])
                outlet_ap_val = round(np.linalg.norm(p1[1:] - p2[1:]), 1)
                result["Outlet_AP_mm"] = outlet_ap_val
            else:
                print("      ⚠️ Outlet AP skipped: midline X not available")
                error_parts.append("Outlet_AP: OUTLET_NO_MIDLINE - Midline X not available")
    except Exception as e:
        error_parts.append(f"Outlet_AP: OUTLET_EXCEPTION - {e}")

    # ========================================
    # Metric 5: Outlet Transverse / ITD
    # ========================================

    try:
        isd_z_for_itd = result.get("ISD_slice")
        otd_val, ot_ptL, ot_ptR, otd_z = find_outlet_transverse(
            hip_L, hip_R, img_affine, config=config,
            isd_z_voxel=isd_z_for_itd,
        )
        if otd_val is not None:
            result["Outlet_Transverse_mm"] = otd_val
            landmarks["itd_pt_left"] = ot_ptL
            landmarks["itd_pt_right"] = ot_ptR
            landmarks["_itd_z"] = otd_z

            if ot_ptL is not None and otd_z is not None:
                pt_L_3d = np.array([float(ot_ptL[0]), float(ot_ptL[1]), float(otd_z)])
                ras_L = voxel_to_world_ras(pt_L_3d, img_affine)
                if ras_L is not None:
                    result["IT_L_x"] = round(ras_L[0], 2)
                    result["IT_L_y"] = round(ras_L[1], 2)
                    result["IT_L_z"] = round(ras_L[2], 2)
            if ot_ptR is not None and otd_z is not None:
                pt_R_3d = np.array([float(ot_ptR[0]), float(ot_ptR[1]), float(otd_z)])
                ras_R = voxel_to_world_ras(pt_R_3d, img_affine)
                if ras_R is not None:
                    result["IT_R_x"] = round(ras_R[0], 2)
                    result["IT_R_y"] = round(ras_R[1], 2)
                    result["IT_R_z"] = round(ras_R[2], 2)

        else:
            error_parts.append("Outlet_Transverse: OUTLET_NO_TUBEROSITY - find_outlet_transverse returned None")
    except Exception as e:
        error_parts.append(f"Outlet_Transverse: OUTLET_EXCEPTION - {e}")

    # Outlet Area (derived, not counted as a metric)
    outlet_ap = result.get("Outlet_AP_mm")
    outlet_tr = result.get("Outlet_Transverse_mm")
    if outlet_ap and outlet_tr:
        result["Outlet_Area_cm2"] = round(np.pi / 4 * outlet_ap * outlet_tr / 100, 1)

    # ========================================
    # Metric 6: Sacral Length (Promontory → Coccygeal Apex)
    # ========================================

    try:
        if "promontory" not in landmarks or "coccygeal_apex" not in landmarks:
            reason = "Sacrum mask missing" if sacrum is None else "Promontory or coccygeal apex not detected"
            error_parts.append(f"Sacral_Length: SACRAL_NO_LANDMARKS - {reason}")
        else:
            if mid_x is not None:
                p1 = landmarks["promontory"] * np.array([1, sy, sz])
                p2 = landmarks["coccygeal_apex"] * np.array([1, sy, sz])
                saclen_val = round(np.linalg.norm(p1[1:] - p2[1:]), 1)
                result["Sacral_Length_mm"] = saclen_val
            else:
                print("      ⚠️ Sacral Length skipped: midline X not available")
                error_parts.append("Sacral_Length: SACRAL_NO_MIDLINE - Midline X not available")
    except Exception as e:
        error_parts.append(f"Sacral_Length: SACRAL_EXCEPTION - {e}")

    # ========================================
    # Metric 7: Sacral Depth (Anterior Concavity)
    # ========================================

    try:
        if sacrum_for_prom is None or np.sum(sacrum_for_prom) == 0:
            error_parts.append("Sacral_Depth: SACRAL_NO_MASK - Sacrum mask missing or empty")
        elif "promontory" not in landmarks or "coccygeal_apex" not in landmarks:
            error_parts.append("Sacral_Depth: SACRAL_NO_LANDMARKS - Promontory or coccygeal apex not detected")
        else:
            sacral_depth, depth_pt, sacral_contour = calculate_sacral_depth(
                sacrum_for_prom, landmarks["promontory"], landmarks["coccygeal_apex"], img_affine,
                mid_x=mid_x, slab_half_voxels=slab_half_voxels
            )
            if sacral_depth is not None:
                result["Sacral_Depth_mm"] = sacral_depth
                if depth_pt is not None:
                    landmarks["Sacral_Max_Depth_Pt"] = depth_pt
                    ras = voxel_to_world_ras(depth_pt, img_affine)
                    if ras is not None:
                        result["Sacral_Max_Depth_Pt_x"] = round(ras[0], 2)
                        result["Sacral_Max_Depth_Pt_y"] = round(ras[1], 2)
                        result["Sacral_Max_Depth_Pt_z"] = round(ras[2], 2)
                if sacral_contour is not None and len(sacral_contour) > 0:
                    landmarks["_sacral_contour"] = sacral_contour
                    landmarks["_depth_chord_top"] = sacral_contour[-1]
                    landmarks["_depth_chord_bot"] = sacral_contour[0]
            else:
                error_parts.append("Sacral_Depth: SACRAL_DEPTH_FAIL - calculate_sacral_depth returned None")
    except Exception as e:
        error_parts.append(f"Sacral_Depth: SACRAL_EXCEPTION - {e}")

    # ========================================
    # Status & Error Log
    # ========================================
    all_metrics = ["ISD_mm", "Inlet_AP_mm", "Outlet_AP_mm",
                   "Outlet_Transverse_mm", "Sacral_Length_mm", "Sacral_Depth_mm"]
    success_count = sum(1 for m in all_metrics if result.get(m) is not None)
    total = len(all_metrics)

    if success_count == total:
        result["Status"] = "Success"
    elif success_count > 0:
        result["Status"] = f"Partial_{success_count}/{total}"
    else:
        result["Status"] = "Failure"

    # Append orientation quality flags to error_parts
    if orientation is not None:
        if orientation["rotation_flag"] != "ok":
            error_parts.append(
                f"QUALITY: PELVIC_ROTATION_{orientation['rotation_flag'].upper()} "
                f"- Axial rotation {orientation['rotation_deg']}°"
            )
        if orientation["tilt_flag"] != "ok":
            error_parts.append(
                f"QUALITY: PELVIC_TILT_{orientation['tilt_flag'].upper()} "
                f"- Coronal tilt {orientation['tilt_deg']}°"
            )
    if sacrum_offset is not None and sacrum_offset["flag"] != "ok":
        error_parts.append(
            f"QUALITY: SACRUM_OFFSET_{sacrum_offset['flag'].upper()} "
            f"- Sacrum-symphysis offset {sacrum_offset['offset_mm']} mm"
        )
    result["Error_Log"] = "; ".join(error_parts) if error_parts else None

    # ========================================
    # File Paths
    # ========================================
    result["CT_NIfTI"] = nifti_path if os.path.exists(nifti_path) else None

    seg_files = {
        "Seg_Sacrum": "sacrum.nii.gz",
        "Seg_Vertebrae_S1": "vertebrae_S1.nii.gz",
        "Seg_Hip_Left": "hip_left.nii.gz",
        "Seg_Hip_Right": "hip_right.nii.gz",
        "Seg_Femur_Left": "femur_left.nii.gz",
        "Seg_Femur_Right": "femur_right.nii.gz",
        "Seg_Torso_Fat": "torso_fat.nii.gz",
        "Seg_Subcutaneous_Fat": "subcutaneous_fat.nii.gz",
        "Seg_Skeletal_Muscle": "skeletal_muscle.nii.gz",
    }

    for key, fname in seg_files.items():
        fpath = os.path.join(seg_folder, fname)
        result[key] = fpath if os.path.exists(fpath) else None

    vertebrae_files = [
        f
        for f in os.listdir(seg_folder)
        if "vertebra" in f.lower() and f.endswith(".nii.gz")
    ]
    if vertebrae_files:
        result["Seg_Vertebrae"] = [
            os.path.join(seg_folder, f) for f in sorted(vertebrae_files)
        ]
    else:
        result["Seg_Vertebrae"] = None

    # ========================================
    # QC Figures
    # ========================================
    if qc_dir:
        os.makedirs(qc_dir, exist_ok=True)

        if ct_vol is not None:
            # Sagittal Combined QC Figure
            try:
                sag_qc_path = os.path.join(qc_dir, f"{patient_id}_Sagittal_QC.png")
                save_sagittal_combined_qc_figure(
                    sag_qc_path, patient_id, ct_vol, result, landmarks, slab_half_voxels, sy, sz
                )
            except Exception as e:
                print(f"      ⚠️ Sagittal QC error: {e}")

            # Extended Pelvimetry QC Figure
            try:
                ext_qc_path = os.path.join(qc_dir, f"{patient_id}_Extended_QC.png")
                save_extended_qc_figure(
                    ext_qc_path, patient_id, ct_vol, result, landmarks, sx, sy, sz
                )
            except Exception as e:
                print(f"      ⚠️ Extended QC error: {e}")

    return _normalise_result(result)


# ------------------------------------------------------------------
# Full Pipeline: DICOM → NIfTI → Seg → Pelvimetry
# ------------------------------------------------------------------

def run_full_pipeline(
    patient_id: str,
    dicom_path: str,
    output_root: str,
    use_fast: bool = False,
    skip_tissue: bool = False,
    generate_qc: bool = True,
    config: Optional[PelvicConfig] = None,
) -> dict:
    """Full pipeline: DICOM → NIfTI → segmentations → pelvimetry.

    Orchestrates DICOM conversion, TotalSegmentator execution, and
    combined pelvimetry analysis as a single end-to-end call.

    Parameters
    ----------
    patient_id : str
        Patient identifier.
    dicom_path : str
        Path to the DICOM folder.
    output_root : str
        Root directory for all outputs.
    use_fast : bool, optional
        Use TotalSegmentator ``--fast`` mode (default ``False``).
    skip_tissue : bool, optional
        Skip tissue-type segmentation (default ``False``).
    generate_qc : bool, optional
        Generate QC figures (default ``True``).
    config : PelvicConfig or None, optional
        Detection parameters forwarded to ``run_combined_pelvimetry``.

    Returns
    -------
    dict
        Combined measurement result dictionary.
    """
    patient_id = str(patient_id)
    print(f"\n{'='*50}")
    print(f"🔹 Processing: {patient_id}")
    print(f"{'='*50}")

    # Ensure license is configured
    setup_license()

    nifti_dir = os.path.join(output_root, "NIfTI")
    seg_dir = os.path.join(output_root, "Segmentation")
    qc_dir = os.path.join(output_root, "QC") if generate_qc else None

    os.makedirs(nifti_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    if qc_dir:
        os.makedirs(qc_dir, exist_ok=True)

    result: dict = {"Patient_ID": patient_id}

    # Phase 1: DICOM → NIfTI
    nifti_path = convert_dicom_to_nifti(patient_id, dicom_path, nifti_dir)
    if not nifti_path:
        result["Status"] = "Fail_NIfTI"
        return _normalise_result(result)

    # Phase 2: NIfTI → Segmentations
    seg_folder = run_totalsegmentator(
        patient_id, nifti_path, seg_dir, use_fast=use_fast, skip_tissue=skip_tissue
    )
    if not seg_folder:
        result["Status"] = "Fail_Seg"
        return _normalise_result(result)

    # Phase 3: Segmentations → Pelvimetry + QC Figures
    result = run_combined_pelvimetry(
        patient_id, seg_folder, nifti_path, qc_dir=qc_dir, config=config
    )

    # Print summary
    print("\n   📊 Results:")
    for key in [
        "ISD_mm", "Inlet_AP_mm", "Outlet_AP_mm",
        "Outlet_Transverse_mm", "Outlet_Area_cm2",
        "Sacral_Length_mm", "Sacral_Depth_mm",
    ]:
        if result.get(key) is not None:
            unit = "cm²" if "cm2" in key else "mm"
            print(f"      {key}: {result[key]} {unit}")

    return result
