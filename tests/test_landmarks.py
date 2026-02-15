"""Tests for ctpelvimetry.landmarks."""

import numpy as np
import pytest

from ctpelvimetry.landmarks import (
    compute_midline_x,
    _voxels_in_xslab,
    detect_pelvic_orientation,
    find_sacral_landmarks,
    find_symphysis_midline_sagittal,
)


# ------------------------------------------------------------------
# compute_midline_x
# ------------------------------------------------------------------

class TestComputeMidlineX:

    def test_symmetric_hips(self):
        """Symmetric L/R hips should produce midline near centre."""
        shape = (64, 64, 64)
        hip_L = np.zeros(shape, dtype=np.uint8)
        hip_R = np.zeros(shape, dtype=np.uint8)
        # Left hip at x≈20, right hip at x≈44
        hip_L[20:30, 25:40, 25:40] = 1
        hip_R[34:44, 25:40, 25:40] = 1

        mid = compute_midline_x(hip_L, hip_R)
        assert mid is not None
        assert 28 <= mid <= 36  # should be roughly 32

    def test_empty_hips_returns_none(self):
        shape = (64, 64, 64)
        hip_L = np.zeros(shape, dtype=np.uint8)
        hip_R = np.zeros(shape, dtype=np.uint8)
        result = compute_midline_x(hip_L, hip_R)
        assert result is None


# ------------------------------------------------------------------
# _voxels_in_xslab
# ------------------------------------------------------------------

class TestVoxelsInXSlab:

    def test_filters_correctly(self):
        coords = np.array([
            [10, 5, 5],
            [20, 5, 5],
            [30, 5, 5],
            [40, 5, 5],
        ])
        filtered = _voxels_in_xslab(coords, mid_x=25.0, half_width=6.0)
        # Only x=20 and x=30 should be within [19, 31]
        assert len(filtered) == 2
        assert 20 in filtered[:, 0]
        assert 30 in filtered[:, 0]

    def test_empty_input(self):
        coords = np.empty((0, 3))
        filtered = _voxels_in_xslab(coords, mid_x=32, half_width=5)
        assert len(filtered) == 0


# ------------------------------------------------------------------
# detect_pelvic_orientation
# ------------------------------------------------------------------

class TestDetectPelvicOrientation:

    def test_symmetric_hips_low_rotation(self):
        """Symmetric hips should produce near-zero rotation and tilt."""
        shape = (64, 64, 64)
        affine = np.eye(4)
        hip_L = np.zeros(shape, dtype=np.uint8)
        hip_R = np.zeros(shape, dtype=np.uint8)
        # Bilateral symmetric placement
        hip_L[15:25, 20:40, 20:40] = 1
        hip_R[39:49, 20:40, 20:40] = 1

        result = detect_pelvic_orientation(hip_L, hip_R, affine)
        if result is not None:
            assert abs(result["rotation_deg"]) < 15
            assert abs(result["tilt_deg"]) < 15

    def test_empty_mask_returns_none(self):
        shape = (64, 64, 64)
        affine = np.eye(4)
        hip_L = np.zeros(shape, dtype=np.uint8)
        hip_R = np.zeros(shape, dtype=np.uint8)
        result = detect_pelvic_orientation(hip_L, hip_R, affine)
        assert result is None


# ------------------------------------------------------------------
# find_sacral_landmarks
# ------------------------------------------------------------------

class TestFindSacralLandmarks:

    def test_vertical_sacrum(self):
        """Vertical column of voxels: promontory at top, apex at bottom."""
        shape = (64, 64, 64)
        sacrum = np.zeros(shape, dtype=np.uint8)
        # Vertical bar at x=32, y=45, z from 20 to 50
        sacrum[32, 45, 20:50] = 1

        result = find_sacral_landmarks(sacrum, mid_x=32, slab_half_voxels=3)
        assert "promontory" in result or "apex" in result

    def test_empty_sacrum(self):
        shape = (64, 64, 64)
        sacrum = np.zeros(shape, dtype=np.uint8)
        result = find_sacral_landmarks(sacrum)
        assert isinstance(result, dict)


# ------------------------------------------------------------------
# find_symphysis_midline_sagittal
# ------------------------------------------------------------------

class TestFindSymphysisMidlineSagittal:

    def test_with_hip_pair(self):
        """Should detect landmarks from bilateral hip masks."""
        shape = (64, 64, 64)
        hip_L = np.zeros(shape, dtype=np.uint8)
        hip_R = np.zeros(shape, dtype=np.uint8)
        # Place anterior medial edges near midline
        hip_L[30:33, 10:15, 20:40] = 1
        hip_R[31:34, 10:15, 20:40] = 1

        result = find_symphysis_midline_sagittal(hip_L, hip_R, mid_x=32, slab_half_voxels=5)
        assert isinstance(result, dict)
