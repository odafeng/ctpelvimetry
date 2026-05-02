"""Reproducibility (snapshot) tests for the measurement pipeline.

Goal: detect numerical drift in landmark detection and metric
calculation. A deterministic synthetic 'phantom pelvis' is built from
known voxel coordinates and fed through ``run_combined_pelvimetry``;
the resulting numbers are compared against a checked-in golden
snapshot (``tests/golden/phantom_snapshot.json``).

If a code change legitimately moves the numbers (improved landmark
detection, bug fix, etc.), regenerate the snapshot:

    pytest tests/test_reproducibility.py --update-snapshot

…then commit the updated JSON. The diff in the snapshot file is
auditable evidence of which metrics moved by how much.

Limitations:
- The phantom is geometrically crude. It currently only exercises
  sacrum-side metrics (Sacral_Length, Sacral_Depth, Promontory,
  Coccygeal_Apex). ISD / Inlet AP / Outlet AP / Outlet Transverse
  do not compute on this phantom because the geometry lacks the
  required ischial-spine and symphysis features.
- A future PR can swap the phantom for a real, license-clean CT
  fixture (e.g. a Medical Decathlon or TotalSegmentator-benchmark
  case) once one is selected and committed under tests/fixtures/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from ctpelvimetry import run_combined_pelvimetry


GOLDEN_PATH = Path(__file__).parent / "golden" / "phantom_snapshot.json"

# Phantom geometry, in voxels. 1 mm/voxel isotropic.
SHAPE = (96, 96, 96)
SPACING = (1.0, 1.0, 1.0)

# Tolerance for float comparison. 0.1 mm is well below clinical
# significance for any pelvimetric metric.
FLOAT_TOL_MM = 0.1
FLOAT_TOL_DEG = 0.1
COORD_TOL_VOX = 0.5

# Keys excluded from snapshot comparison: filesystem paths (not
# reproducible across CI runners) and metrics that don't compute
# on this phantom (always None).
EXCLUDED_KEYS = {
    "CT_NIfTI", "Seg_Vertebrae",
    "Seg_Sacrum", "Seg_Vertebrae_S1", "Seg_Hip_Left", "Seg_Hip_Right",
    "Seg_Femur_Left", "Seg_Femur_Right", "Seg_Torso_Fat",
    "Seg_Subcutaneous_Fat", "Seg_Skeletal_Muscle",
}


def _zeros() -> np.ndarray:
    return np.zeros(SHAPE, dtype=np.uint8)


def _build_phantom_masks() -> dict[str, np.ndarray]:
    """Construct deterministic mask volumes shaped vaguely like a pelvis."""
    masks = {
        "hip_left":      _zeros(),
        "hip_right":     _zeros(),
        "sacrum":        _zeros(),
        "vertebrae_S1":  _zeros(),
        "femur_left":    _zeros(),
        "femur_right":   _zeros(),
    }
    masks["hip_left"][20:40, 30:70, 30:80] = 1
    masks["hip_right"][56:76, 30:70, 30:80] = 1
    masks["sacrum"][42:54, 60:75, 25:85] = 1
    masks["vertebrae_S1"][42:54, 55:70, 78:90] = 1
    masks["femur_left"][22:36, 35:55, 10:30] = 1
    masks["femur_right"][60:74, 35:55, 10:30] = 1
    return masks


def _write_phantom_to_disk(seg_dir: Path) -> Path:
    """Write masks + a dummy CT NIfTI; return path to the CT."""
    affine = np.diag([SPACING[0], SPACING[1], SPACING[2], 1.0])
    for name, vol in _build_phantom_masks().items():
        nib.save(nib.Nifti1Image(vol, affine), str(seg_dir / f"{name}.nii.gz"))

    ct = np.full(SHAPE, 50, dtype=np.int16)
    ct_path = seg_dir / "phantom_ct.nii.gz"
    nib.save(nib.Nifti1Image(ct, affine), str(ct_path))
    return ct_path


def _filter_for_snapshot(result: dict) -> dict:
    """Drop excluded keys; convert numpy types to plain Python."""
    cleaned: dict = {}
    for k, v in result.items():
        if k in EXCLUDED_KEYS:
            continue
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        cleaned[k] = v
    return cleaned


def _compare_snapshot(actual: dict, expected: dict) -> list[str]:
    """Return a list of human-readable mismatch descriptions."""
    diffs: list[str] = []

    actual_keys = set(actual)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        added = actual_keys - expected_keys
        removed = expected_keys - actual_keys
        if added:
            diffs.append(f"Keys added: {sorted(added)}")
        if removed:
            diffs.append(f"Keys removed: {sorted(removed)}")

    for key in actual_keys & expected_keys:
        a, e = actual[key], expected[key]
        if a is None and e is None:
            continue
        if (a is None) != (e is None):
            diffs.append(f"{key}: {e!r} -> {a!r}  (None mismatch)")
            continue

        # Pick tolerance based on key suffix
        if key.endswith("_mm"):
            tol = FLOAT_TOL_MM
        elif key.endswith("_deg"):
            tol = FLOAT_TOL_DEG
        elif (key.endswith(("_x", "_y", "_z"))
              or key.endswith("_slice")):
            tol = COORD_TOL_VOX
        else:
            tol = None  # exact match

        if tol is None:
            if a != e:
                diffs.append(f"{key}: {e!r} -> {a!r}")
        else:
            try:
                af, ef = float(a), float(e)
            except (TypeError, ValueError):
                if a != e:
                    diffs.append(f"{key}: {e!r} -> {a!r}")
                continue
            if not math.isclose(af, ef, abs_tol=tol):
                diffs.append(f"{key}: {ef:.4f} -> {af:.4f}  (Δ {af - ef:+.4f}, tol ±{tol})")

    return diffs


def test_phantom_measurement_snapshot(tmp_path, request):
    """run_combined_pelvimetry on the phantom matches the golden snapshot."""
    seg_dir = tmp_path / "seg"
    seg_dir.mkdir()
    ct_path = _write_phantom_to_disk(seg_dir)

    result = run_combined_pelvimetry(
        patient_id="PHANTOM",
        seg_folder=str(seg_dir),
        nifti_path=str(ct_path),
        qc_dir=None,
    )

    actual = _filter_for_snapshot(result)

    if request.config.getoption("--update-snapshot"):
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with GOLDEN_PATH.open("w") as f:
            json.dump(actual, f, indent=2, sort_keys=True)
            f.write("\n")
        pytest.skip(f"Snapshot updated: {GOLDEN_PATH}")
        return

    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"Golden snapshot not found at {GOLDEN_PATH}. "
            "Run with --update-snapshot to generate it."
        )

    with GOLDEN_PATH.open() as f:
        expected = json.load(f)

    diffs = _compare_snapshot(actual, expected)
    if diffs:
        msg = (
            "Reproducibility snapshot mismatch — measurements drifted.\n"
            f"  golden: {GOLDEN_PATH}\n\n"
            "  Differences:\n    "
            + "\n    ".join(diffs)
            + "\n\nIf the change is intentional, regenerate the snapshot:\n"
              "    pytest tests/test_reproducibility.py --update-snapshot\n"
              "and commit the updated JSON alongside your change."
        )
        pytest.fail(msg)
