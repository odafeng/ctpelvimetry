"""Shared pytest fixtures for ctpelvimetry tests."""

import numpy as np
import nibabel as nib
import pytest


# ------------------------------------------------------------------
# CLI options
# ------------------------------------------------------------------

def pytest_addoption(parser):
    """Register the --update-snapshot flag used by reproducibility tests."""
    parser.addoption(
        "--update-snapshot",
        action="store_true",
        default=False,
        help="Regenerate the reproducibility snapshot instead of comparing.",
    )


# ------------------------------------------------------------------
# Synthetic volume helpers
# ------------------------------------------------------------------

def _make_volume(shape=(64, 64, 64), dtype=np.float32):
    """Return an empty 3-D volume."""
    return np.zeros(shape, dtype=dtype)


def _sphere_mask(shape, centre, radius):
    """Create a binary sphere mask."""
    vol = np.zeros(shape, dtype=np.uint8)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((xx - centre[0])**2 + (yy - centre[1])**2 + (zz - centre[2])**2)
    vol[dist <= radius] = 1
    return vol


def _save_nifti(path, data, affine=None):
    """Save a numpy array as a NIfTI file."""
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, str(path))
    return str(path)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def identity_affine():
    """4x4 identity affine (1 mm isotropic, RAS+)."""
    return np.eye(4)


@pytest.fixture
def mock_header():
    """Minimal NIfTI header with 1 mm isotropic voxels."""
    hdr = nib.Nifti1Header()
    hdr.set_zooms((1.0, 1.0, 1.0))
    return hdr


@pytest.fixture
def empty_vol():
    """Empty 64³ volume."""
    return _make_volume()


@pytest.fixture
def sphere_mask_centre():
    """Sphere mask centred at (32, 32, 32) with radius 10."""
    return _sphere_mask((64, 64, 64), (32, 32, 32), 10)
