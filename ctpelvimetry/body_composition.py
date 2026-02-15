"""
Body composition analysis: VAT/SAT/SMA at L3 and ISD levels.
"""

import os

import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, binary_fill_holes

from .io import load_nifti_canonical


# ------------------------------------------------------------------
# Slice finding
# ------------------------------------------------------------------

def find_l3_slice(vertebrae_l3_path, _header):
    """Find the centre-of-mass slice of the L3 vertebral body.

    Parameters
    ----------
    vertebrae_l3_path : str
        Path to the L3 vertebra segmentation mask (``.nii.gz``).
    _header : nibabel.Nifti1Header
        NIfTI header (kept for API compatibility; not used).

    Returns
    -------
    int or None
        Z-axis slice index at L3 centre-of-mass, or ``None``
        on failure.
    """
    l3_mask, _, _ = load_nifti_canonical(vertebrae_l3_path)
    if l3_mask is None or np.sum(l3_mask) == 0:
        return None

    com = center_of_mass(l3_mask)
    l3_slice = int(round(com[2]))  # Z-axis index

    return l3_slice


def find_isd_slice(pelvimetry_csv, patient_id, affine):
    """Read the ISD-level voxel slice index from the pelvimetry CSV.

    Parameters
    ----------
    pelvimetry_csv : str
        Path to the pelvimetry report CSV.
    patient_id : str
        Patient identifier (must match ``Patient_ID`` column).
    affine : numpy.ndarray of shape (4, 4)
        NIfTI affine matrix (kept for API compatibility; not used).

    Returns
    -------
    int or None
        ISD slice index, or ``None`` on failure.
    """
    if not os.path.exists(pelvimetry_csv):
        return None

    df = pd.read_csv(pelvimetry_csv)
    patient_row = df[df['Patient_ID'] == patient_id]

    if len(patient_row) == 0:
        return None

    if 'ISD_slice' in patient_row.columns and pd.notna(patient_row['ISD_slice'].values[0]):
        return int(patient_row['ISD_slice'].values[0])

    return None


# ------------------------------------------------------------------
# VAT / SAT separation
# ------------------------------------------------------------------

def separate_vat_sat(adipose_mask, body_cavity_mask):
    """Separate visceral fat (VAT) and subcutaneous fat (SAT).

    Fat voxels inside the body cavity are classified as VAT; the
    remainder are SAT.

    Parameters
    ----------
    adipose_mask : numpy.ndarray
        2-D binary adipose tissue mask.
    body_cavity_mask : numpy.ndarray
        2-D binary body cavity mask.

    Returns
    -------
    vat_mask : numpy.ndarray
        2-D binary VAT mask.
    sat_mask : numpy.ndarray
        2-D binary SAT mask.
    """
    vat_mask = adipose_mask & body_cavity_mask
    sat_mask = adipose_mask & (~body_cavity_mask)
    return vat_mask, sat_mask


def create_body_cavity_mask(muscle_slice, organs_slice=None):
    """Create a body cavity mask (intra-abdominal space).

    Uses the skeletal-muscle inner boundary, optionally augmented
    with an organ mask, then fills holes to define the complete
    body interior.

    Parameters
    ----------
    muscle_slice : numpy.ndarray
        2-D binary skeletal-muscle mask.
    organs_slice : numpy.ndarray or None, optional
        2-D binary organ mask (aids cavity definition).

    Returns
    -------
    numpy.ndarray
        2-D binary cavity mask.
    """
    body_outline = muscle_slice.copy()
    if organs_slice is not None:
        body_outline = body_outline | organs_slice

    body_filled = binary_fill_holes(body_outline)
    cavity_mask = body_filled.copy()

    return cavity_mask


# ------------------------------------------------------------------
# Area calculation
# ------------------------------------------------------------------

def calculate_area_cm2(mask_2d, pixel_spacing):
    """Calculate the physical area of a 2-D mask.

    Parameters
    ----------
    mask_2d : numpy.ndarray
        2-D binary mask.
    pixel_spacing : tuple of (float, float)
        ``(spacing_x, spacing_y)`` in mm.

    Returns
    -------
    float
        Area in cm².
    """
    pixel_count = np.sum(mask_2d > 0)
    pixel_area_mm2 = pixel_spacing[0] * pixel_spacing[1]
    area_cm2 = (pixel_count * pixel_area_mm2) / 100.0  # mm² to cm²
    return area_cm2


# ------------------------------------------------------------------
# Slice composition analysis
# ------------------------------------------------------------------

def analyze_slice_composition(
    slice_idx,
    adipose_vol,
    muscle_vol,
    organs_vol,
    header,
    level_name="L3"
):
    """Analyse body composition of a single axial slice.

    Uses ``create_body_cavity_mask`` and ``separate_vat_sat`` to
    separate adipose tissue into VAT and SAT, then computes areas.

    Parameters
    ----------
    slice_idx : int or None
        Z-axis slice index.
    adipose_vol : numpy.ndarray
        3-D adipose tissue volume.
    muscle_vol : numpy.ndarray or None
        3-D skeletal muscle volume.
    organs_vol : numpy.ndarray or None
        3-D organ volume (aids cavity definition).
    header : nibabel.Nifti1Header
        NIfTI header (provides pixel spacing).
    level_name : str, optional
        Label prefix for output keys (default ``'L3'``).

    Returns
    -------
    dict
        Keys: ``'{level_name}_VAT_cm2'``, ``'{level_name}_SAT_cm2'``,
        ``'{level_name}_VS_ratio'``, ``'{level_name}_SMA_cm2'``.
    """
    result = {
        f"{level_name}_slice": slice_idx,
        f"{level_name}_VAT_cm2": None,
        f"{level_name}_SAT_cm2": None,
        f"{level_name}_VS_ratio": None,
        f"{level_name}_SMA_cm2": None,
    }

    if slice_idx is None:
        return result

    if slice_idx < 0 or slice_idx >= adipose_vol.shape[2]:
        return result

    adipose_slice = adipose_vol[:, :, slice_idx]
    muscle_slice = muscle_vol[:, :, slice_idx] if muscle_vol is not None else np.zeros_like(adipose_slice)
    organs_slice = organs_vol[:, :, slice_idx] if organs_vol is not None else None

    spacing = header.get_zooms()[:2]  # (x, y) in mm

    cavity_mask = create_body_cavity_mask(muscle_slice, organs_slice)
    vat_mask, sat_mask = separate_vat_sat(adipose_slice > 0, cavity_mask)

    vat_area = calculate_area_cm2(vat_mask, spacing)
    sat_area = calculate_area_cm2(sat_mask, spacing)
    sma_area = calculate_area_cm2(muscle_slice, spacing)

    vs_ratio = vat_area / sat_area if sat_area > 0 else None

    result[f"{level_name}_VAT_cm2"] = round(vat_area, 2)
    result[f"{level_name}_SAT_cm2"] = round(sat_area, 2)
    result[f"{level_name}_VS_ratio"] = round(vs_ratio, 3) if vs_ratio is not None else None
    result[f"{level_name}_SMA_cm2"] = round(sma_area, 2)

    return result


def analyze_slice_composition_direct(
    slice_idx,
    vat_vol,
    sat_vol,
    muscle_vol,
    header,
    level_name="L3"
):
    """Analyse body composition using pre-segmented VAT/SAT volumes.

    Unlike ``analyze_slice_composition``, this variant accepts
    pre-computed torso-fat and subcutaneous-fat masks directly.

    Parameters
    ----------
    slice_idx : int or None
        Z-axis slice index.
    vat_vol : numpy.ndarray or None
        3-D visceral fat (torso_fat) volume.
    sat_vol : numpy.ndarray or None
        3-D subcutaneous fat volume.
    muscle_vol : numpy.ndarray or None
        3-D skeletal muscle volume.
    header : nibabel.Nifti1Header
        NIfTI header (provides pixel spacing).
    level_name : str, optional
        Label prefix for output keys (default ``'L3'``).

    Returns
    -------
    dict
        Keys: ``'{level_name}_VAT_cm2'``, ``'{level_name}_SAT_cm2'``,
        ``'{level_name}_VS_ratio'``, ``'{level_name}_SMA_cm2'``.
    """
    result = {
        f"{level_name}_slice": slice_idx,
        f"{level_name}_VAT_cm2": None,
        f"{level_name}_SAT_cm2": None,
        f"{level_name}_VS_ratio": None,
        f"{level_name}_SMA_cm2": None,
    }

    if slice_idx is None:
        return result

    if vat_vol is None or slice_idx < 0 or slice_idx >= vat_vol.shape[2]:
        return result

    vat_slice = vat_vol[:, :, slice_idx]
    sat_slice = sat_vol[:, :, slice_idx] if sat_vol is not None else np.zeros_like(vat_slice)
    muscle_slice = muscle_vol[:, :, slice_idx] if muscle_vol is not None else np.zeros_like(vat_slice)

    spacing = header.get_zooms()[:2]  # (x, y) in mm

    vat_area = calculate_area_cm2(vat_slice, spacing)
    sat_area = calculate_area_cm2(sat_slice, spacing)
    sma_area = calculate_area_cm2(muscle_slice, spacing)

    vs_ratio = vat_area / sat_area if sat_area > 0 else None

    result[f"{level_name}_VAT_cm2"] = round(vat_area, 2)
    result[f"{level_name}_SAT_cm2"] = round(sat_area, 2)
    result[f"{level_name}_VS_ratio"] = round(vs_ratio, 3) if vs_ratio is not None else None
    result[f"{level_name}_SMA_cm2"] = round(sma_area, 2)

    return result


# ------------------------------------------------------------------
# Single patient processing
# ------------------------------------------------------------------

def process_single_patient(
    patient_id,
    seg_root,
    nifti_path,
    pelvimetry_csv,
    qc_dir=None
):
    """Run body composition analysis for one patient.

    Loads CT + tissue masks, identifies L3 and ISD slices, computes
    VAT / SAT / SMA areas, and optionally generates a QC figure.

    Parameters
    ----------
    patient_id : str
        Patient identifier (e.g., ``'Patient_001'``).
    seg_root : str
        Root directory containing segmentation subfolders.
    nifti_path : str
        Path to the CT NIfTI file.
    pelvimetry_csv : str
        Path to the pelvimetry report CSV (used for ISD slice).
    qc_dir : str or None, optional
        Directory for QC figures (default ``None``).

    Returns
    -------
    dict
        All measurement results, including VAT/SAT/SMA at
        L3 and ISD levels.
    """
    result = {"Patient_ID": patient_id}

    # Path setup — support two directory structures
    seg_folder = os.path.join(seg_root, patient_id, "segmentations")
    if not os.path.exists(seg_folder):
        seg_folder = os.path.join(seg_root, patient_id)

    # Check required files
    required_files = {
        "vat": os.path.join(seg_folder, "torso_fat.nii.gz"),
        "sat": os.path.join(seg_folder, "subcutaneous_fat.nii.gz"),
        "muscle": os.path.join(seg_folder, "skeletal_muscle.nii.gz"),
        "vertebrae_l3": os.path.join(seg_folder, "vertebrae_L3.nii.gz"),
    }

    for key, path in required_files.items():
        if not os.path.exists(path):
            print(f"      ⚠️ Missing: {key} ({path})")
            result["Status"] = f"Missing_{key}"
            return result

    # Load segmentation masks
    print("   🔄 Loading segmentations...")
    vat_vol, header, affine = load_nifti_canonical(required_files["vat"])
    sat_vol, _, _ = load_nifti_canonical(required_files["sat"])
    muscle_vol, _, _ = load_nifti_canonical(required_files["muscle"])

    # Load CT (for QC figures)
    ct_vol = None
    if qc_dir and nifti_path and os.path.exists(nifti_path):
        ct_vol, _, _ = load_nifti_canonical(nifti_path)

    # Find L3 slice
    print("   🔍 Finding L3 level...")
    l3_slice = find_l3_slice(required_files["vertebrae_l3"], header)
    if l3_slice is None:
        print("      ⚠️ L3 slice not found")
    else:
        print(f"      ✅ L3 slice: {l3_slice}")

    # Find ISD slice (from pelvimetry report)
    print("   🔍 Finding ISD level...")
    isd_slice = find_isd_slice(pelvimetry_csv, patient_id, affine)
    if isd_slice is None:
        print("      ⚠️ ISD slice not found in pelvimetry CSV")
    else:
        print(f"      ✅ ISD slice: {isd_slice}")

    # Analyze L3 level
    print("   📊 Analyzing L3 composition...")
    l3_results = analyze_slice_composition_direct(
        l3_slice, vat_vol, sat_vol, muscle_vol, header, level_name="L3"
    )
    result.update(l3_results)

    # Analyze ISD level
    print("   📊 Analyzing mid-pelvis composition...")
    isd_results = analyze_slice_composition_direct(
        isd_slice, vat_vol, sat_vol, muscle_vol, header, level_name="ISD"
    )
    result.update(isd_results)

    # Generate QC figure
    if qc_dir and ct_vol is not None:
        from .qc import generate_body_composition_qc
        os.makedirs(qc_dir, exist_ok=True)
        qc_path = os.path.join(qc_dir, f"{patient_id}_BodyComp_QC.png")
        adipose_vol = (vat_vol > 0) | (sat_vol > 0) if vat_vol is not None and sat_vol is not None else None
        generate_body_composition_qc(
            ct_vol, adipose_vol, muscle_vol,
            l3_slice, isd_slice, result,
            patient_id, qc_path
        )

    result["Status"] = "Success"

    # Print result summary
    print("\n   📊 Results:")
    print(f"      L3_VAT: {result.get('L3_VAT_cm2', 'N/A')} cm²")
    print(f"      L3_SAT: {result.get('L3_SAT_cm2', 'N/A')} cm²")
    print(f"      L3_VS_ratio: {result.get('L3_VS_ratio', 'N/A')}")
    print(f"      L3_SMA: {result.get('L3_SMA_cm2', 'N/A')} cm²")
    print(f"      ISD_VAT: {result.get('ISD_VAT_cm2', 'N/A')} cm²")
    print(f"      ISD_SAT: {result.get('ISD_SAT_cm2', 'N/A')} cm²")
    print(f"      ISD_SMA: {result.get('ISD_SMA_cm2', 'N/A')} cm²")

    return result
