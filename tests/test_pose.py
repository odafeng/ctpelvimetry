"""Tests for pelvic pose normalization."""

import numpy as np
import pytest
from ctpelvimetry.pose import (
    compute_hip_centroids_world,
    compute_correction_angles,
    build_correction_matrix_voxel,
    correct_mask,
    correct_affine,
    normalize_pelvic_pose,
)


def _make_sphere(shape, center, radius):
    """Create a binary sphere mask for testing."""
    coords = np.mgrid[
        : shape[0], : shape[1], : shape[2]
    ].astype(float)
    dist = np.sqrt(sum((c - cen) ** 2 for c, cen in zip(coords, center)))
    return (dist <= radius).astype(np.uint8)


def _identity_affine(spacing=(1.0, 1.0, 1.0)):
    aff = np.eye(4)
    aff[0, 0], aff[1, 1], aff[2, 2] = spacing
    return aff


class TestComputeHipCentroids:
    def test_symmetric_hips(self):
        shape = (100, 100, 50)
        hip_L = _make_sphere(shape, (30, 50, 25), 10)
        hip_R = _make_sphere(shape, (70, 50, 25), 10)
        affine = _identity_affine()
        cL, cR = compute_hip_centroids_world(hip_L, hip_R, affine)
        # L hip at X~30, R hip at X~70, both at Y~50, Z~25
        assert abs(cL[0] - 30) < 1
        assert abs(cR[0] - 70) < 1
        assert abs(cL[1] - cR[1]) < 1  # same Y
        assert abs(cL[2] - cR[2]) < 1  # same Z


class TestComputeCorrectionAngles:
    def test_no_rotation(self):
        cL = np.array([30, 50, 25])
        cR = np.array([70, 50, 25])
        angles = compute_correction_angles(cL, cR)
        assert abs(angles["rotation_deg"]) < 0.1
        assert abs(angles["tilt_deg"]) < 0.1

    def test_rotated(self):
        # L hip shifted anteriorly by 10mm → rotation ~14°
        cL = np.array([30, 60, 25])
        cR = np.array([70, 50, 25])
        angles = compute_correction_angles(cL, cR)
        expected = np.degrees(np.arctan2(10, 40))  # ~14.04°
        assert abs(angles["rotation_deg"] - expected) < 0.1

    def test_tilted(self):
        # L hip shifted superiorly by 10mm → tilt ~14°
        cL = np.array([30, 50, 35])
        cR = np.array([70, 50, 25])
        angles = compute_correction_angles(cL, cR)
        expected = np.degrees(np.arctan2(10, 40))  # ~14.04°
        assert abs(angles["tilt_deg"] - expected) < 0.1


class TestNormalizePelvicPose:
    def test_no_correction_needed(self):
        """Symmetric hips should not trigger correction."""
        shape = (100, 100, 50)
        masks = {
            "hip_L": _make_sphere(shape, (30, 50, 25), 10),
            "hip_R": _make_sphere(shape, (70, 50, 25), 10),
        }
        affine = _identity_affine()
        corrected, new_affine, info = normalize_pelvic_pose(masks, affine)
        assert not info["applied"]
        assert np.array_equal(corrected["hip_L"], masks["hip_L"])

    def test_correction_applied_for_rotated_hips(self):
        """Rotated hips should trigger correction and reduce rotation."""
        shape = (100, 100, 50)
        # L hip at Y=60, R hip at Y=50 → rotation in axial plane
        masks = {
            "hip_L": _make_sphere(shape, (30, 60, 25), 8),
            "hip_R": _make_sphere(shape, (70, 50, 25), 8),
        }
        affine = _identity_affine()
        corrected, new_affine, info = normalize_pelvic_pose(
            masks, affine, correction_threshold_deg=2.0
        )
        assert info["applied"]
        assert abs(info["post_rotation_deg"]) < abs(info["pre_rotation_deg"])

    def test_correction_preserves_mask_volume(self):
        """Correction should approximately preserve total mask volume."""
        shape = (100, 100, 50)
        masks = {
            "hip_L": _make_sphere(shape, (30, 60, 25), 8),
            "hip_R": _make_sphere(shape, (70, 50, 25), 8),
            "sacrum": _make_sphere(shape, (50, 40, 25), 6),
        }
        affine = _identity_affine()
        corrected, _, info = normalize_pelvic_pose(
            masks, affine, correction_threshold_deg=2.0
        )
        for key in ["hip_L", "hip_R", "sacrum"]:
            orig_vol = int(np.sum(masks[key]))
            corr_vol = int(np.sum(corrected[key]))
            # Volume should be within 10% (nearest neighbor can change boundary)
            assert abs(corr_vol - orig_vol) / orig_vol < 0.10, (
                f"{key}: volume changed {orig_vol} → {corr_vol}"
            )

    def test_none_masks_handled(self):
        """None masks in the dict should pass through without error."""
        shape = (100, 100, 50)
        masks = {
            "hip_L": _make_sphere(shape, (30, 60, 25), 8),
            "hip_R": _make_sphere(shape, (70, 50, 25), 8),
            "sacrum": None,
            "femur_L": None,
        }
        affine = _identity_affine()
        corrected, _, info = normalize_pelvic_pose(masks, affine)
        assert corrected["sacrum"] is None
        assert corrected["femur_L"] is None

    def test_below_threshold_no_correction(self):
        """Small rotation below threshold should not trigger correction."""
        shape = (100, 100, 50)
        # Very small rotation: L hip at Y=51, R at Y=50
        masks = {
            "hip_L": _make_sphere(shape, (30, 51, 25), 8),
            "hip_R": _make_sphere(shape, (70, 50, 25), 8),
        }
        affine = _identity_affine()
        corrected, _, info = normalize_pelvic_pose(
            masks, affine, correction_threshold_deg=5.0
        )
        assert not info["applied"]
