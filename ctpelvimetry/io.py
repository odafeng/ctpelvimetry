"""
I/O utilities: mask loading and coordinate transforms.
"""

import os

import numpy as np
import nibabel as nib


# ------------------------------------------------------------------
# Mask loading
# ------------------------------------------------------------------

def load_mask(path):
    """Load a NIfTI segmentation mask.

    Parameters
    ----------
    path : str
        Absolute path to the ``.nii.gz`` mask file.

    Returns
    -------
    data : numpy.ndarray or None
        3-D mask array, or ``None`` if *path* does not exist.
    header : nibabel.Nifti1Header or None
        NIfTI header, or ``None`` if *path* does not exist.
    """
    if not os.path.exists(path):
        return None, None
    img = nib.load(path)
    return img.get_fdata(), img.header


def load_nifti_canonical(path):
    """Load a NIfTI file and reorient to RAS+ canonical orientation.

    Parameters
    ----------
    path : str
        Absolute path to the ``.nii`` / ``.nii.gz`` file.

    Returns
    -------
    data : numpy.ndarray or None
        3-D volume in RAS+ orientation.
    header : nibabel.Nifti1Header or None
        NIfTI header.
    affine : numpy.ndarray (4, 4) or None
        Voxel-to-world affine matrix after reorientation.
    """
    if not os.path.exists(path):
        return None, None, None
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    header = img.header
    affine = img.affine
    return data, header, affine


def load_mask_canonical(path):
    """Load a segmentation mask and reorient to RAS+ canonical.

    Parameters
    ----------
    path : str
        Absolute path to the ``.nii.gz`` mask file.

    Returns
    -------
    data : numpy.ndarray or None
        3-D mask array in RAS+ orientation.
    header : nibabel.Nifti1Header or None
        NIfTI header.
    affine : numpy.ndarray (4, 4) or None
        Voxel-to-world affine after reorientation.
    """
    if not os.path.exists(path):
        return None, None, None
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    return img.get_fdata(), img.header, img.affine


# ------------------------------------------------------------------
# Coordinate transforms
# ------------------------------------------------------------------

def voxel_to_world_ras(pt_ijk, affine):
    """Convert voxel (i, j, k) coordinates to world RAS+ coordinates.

    Parameters
    ----------
    pt_ijk : array-like of shape (3,)
        Voxel indices ``(i, j, k)``.
    affine : numpy.ndarray of shape (4, 4)
        Voxel-to-world affine matrix.

    Returns
    -------
    numpy.ndarray of shape (3,) or None
        World coordinates in mm (RAS+), or ``None`` if inputs are
        ``None``.
    """
    if pt_ijk is None or affine is None:
        return None
    pt_ijk = np.array(pt_ijk, dtype=float)
    return nib.affines.apply_affine(affine, pt_ijk)
