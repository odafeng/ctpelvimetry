"""Tests for ctpelvimetry.io."""

import numpy as np
import nibabel as nib
import pytest

from ctpelvimetry.io import load_mask, load_nifti_canonical, load_mask_canonical, voxel_to_world_ras
from conftest import _save_nifti


# ------------------------------------------------------------------
# load_mask
# ------------------------------------------------------------------

class TestLoadMask:

    def test_missing_file_returns_none(self):
        data, header = load_mask("/nonexistent/path.nii.gz")
        assert data is None
        assert header is None

    def test_valid_file(self, tmp_path):
        arr = np.ones((4, 5, 6), dtype=np.float32)
        path = _save_nifti(tmp_path / "mask.nii.gz", arr)
        data, header = load_mask(path)
        assert data.shape == (4, 5, 6)
        assert header is not None


# ------------------------------------------------------------------
# load_nifti_canonical
# ------------------------------------------------------------------

class TestLoadNiftiCanonical:

    def test_missing_file_returns_none(self):
        data, header, affine = load_nifti_canonical("/nonexistent/ct.nii.gz")
        assert data is None
        assert header is None
        assert affine is None

    def test_valid_file_returns_tuple(self, tmp_path):
        arr = np.random.rand(8, 10, 12).astype(np.float32)
        path = _save_nifti(tmp_path / "ct.nii.gz", arr)
        data, header, affine = load_nifti_canonical(path)
        assert data is not None
        assert data.ndim == 3
        assert affine.shape == (4, 4)


# ------------------------------------------------------------------
# load_mask_canonical
# ------------------------------------------------------------------

class TestLoadMaskCanonical:

    def test_missing_file_returns_none(self):
        data, header, affine = load_mask_canonical("/nonexistent/mask.nii.gz")
        assert data is None

    def test_valid_file(self, tmp_path):
        arr = np.ones((4, 5, 6), dtype=np.float32)
        path = _save_nifti(tmp_path / "mask.nii.gz", arr)
        data, header, affine = load_mask_canonical(path)
        assert data.shape == (4, 5, 6)
        assert affine.shape == (4, 4)


# ------------------------------------------------------------------
# voxel_to_world_ras
# ------------------------------------------------------------------

class TestVoxelToWorldRas:

    def test_identity_affine(self, identity_affine):
        result = voxel_to_world_ras([10, 20, 30], identity_affine)
        np.testing.assert_array_almost_equal(result, [10.0, 20.0, 30.0])

    def test_scaled_affine(self):
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        result = voxel_to_world_ras([1, 1, 1], affine)
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])

    def test_none_inputs(self, identity_affine):
        assert voxel_to_world_ras(None, identity_affine) is None
        assert voxel_to_world_ras([1, 2, 3], None) is None
