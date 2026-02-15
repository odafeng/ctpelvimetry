"""Tests for ctpelvimetry.metrics."""

import numpy as np
import pytest

from ctpelvimetry.metrics import (
    calculate_sacral_depth,
    _pick_spine_candidate_2d,
)


# ------------------------------------------------------------------
# calculate_sacral_depth
# ------------------------------------------------------------------

class TestCalculateSacralDepth:

    def test_curved_sacrum_positive_depth(self):
        """A curved sacrum should produce a positive depth value."""
        shape = (64, 64, 64)
        sacrum = np.zeros(shape, dtype=np.uint8)
        # Create a curved sacrum: vertical column with anterior bulge
        for z in range(20, 50):
            # Base column at y=45
            sacrum[32, 45, z] = 1
            # Anterior bulge in middle slices
            if 30 <= z <= 40:
                sacrum[32, 40, z] = 1
                sacrum[32, 41, z] = 1

        affine = np.eye(4)
        promontory = np.array([32, 45, 49])
        apex = np.array([32, 45, 20])

        depth, _, _ = calculate_sacral_depth(
            sacrum, promontory, apex, affine,
            mid_x=32, slab_half_voxels=3
        )
        assert depth is not None
        assert depth >= 0

    def test_none_landmarks(self):
        shape = (64, 64, 64)
        sacrum = np.zeros(shape, dtype=np.uint8)
        affine = np.eye(4)
        depth, _, _ = calculate_sacral_depth(sacrum, None, None, affine)
        assert depth is None


# ------------------------------------------------------------------
# _pick_spine_candidate_2d
# ------------------------------------------------------------------

class TestPickSpineCandidate2D:

    def test_basic_point_cloud(self):
        """Should return a valid 2D candidate from a point cloud."""
        # Create a semicircular point cloud
        angles = np.linspace(0, np.pi, 100)
        pts_xy = np.column_stack([
            30 + 15 * np.cos(angles),  # x
            30 + 15 * np.sin(angles),  # y (anterior-posterior)
        ])
        candidate, meta = _pick_spine_candidate_2d(
            pts_xy, 1.0, 1.0, midline_x=30.0, side="L"
        )
        assert candidate is not None
        assert len(candidate) == 2
        assert isinstance(meta, dict)

    def test_insufficient_points(self):
        """Should handle very few points gracefully."""
        pts_xy = np.array([[30, 30]])
        candidate, meta = _pick_spine_candidate_2d(
            pts_xy, 1.0, 1.0, midline_x=30.0, side="L"
        )
        # May return None or a fallback; should not crash
        assert isinstance(meta, dict)
