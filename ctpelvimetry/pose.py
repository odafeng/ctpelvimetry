"""
Pelvic pose normalization: correct axial rotation and coronal tilt
using bilateral hip centroids before landmark extraction.
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform


def compute_hip_centroids_world(hip_L, hip_R, affine):
    """Compute bilateral hip centroids in world (RAS+) coordinates.

    Parameters
    ----------
    hip_L, hip_R : numpy.ndarray
        3-D binary masks of the left and right hip bones.
    affine : numpy.ndarray of shape (4, 4)
        Voxel-to-world affine matrix.

    Returns
    -------
    centroid_L, centroid_R : numpy.ndarray of shape (3,)
        World coordinates (RAS+) of left and right hip centroids.
    """
    coords_L = np.argwhere(hip_L > 0)
    coords_R = np.argwhere(hip_R > 0)
    centroid_L = nib.affines.apply_affine(affine, coords_L.mean(axis=0))
    centroid_R = nib.affines.apply_affine(affine, coords_R.mean(axis=0))
    return centroid_L, centroid_R


def compute_correction_angles(centroid_L, centroid_R):
    """Compute the rotation and tilt angles needed to normalize pelvic pose.

    Axial rotation is the angle between the inter-hip axis and the
    pure left-right (X) axis, projected onto the axial (XY) plane.
    Coronal tilt is the angle in the coronal (XZ) plane.

    Parameters
    ----------
    centroid_L, centroid_R : numpy.ndarray of shape (3,)
        World coordinates (RAS+) of left and right hip centroids.

    Returns
    -------
    dict
        ``'rotation_rad'``: axial rotation angle (radians, to be
        corrected by rotating *negative* this amount around Z).
        ``'tilt_rad'``: coronal tilt angle (radians, to be corrected
        by rotating *negative* this amount around Y).
        ``'rotation_deg'``, ``'tilt_deg'``: same in degrees.
    """
    delta = centroid_L - centroid_R  # vector from R to L hip
    dx = abs(delta[0])  # L-R separation (always positive)
    dy = delta[1]       # A-P offset (signed: + anterior, - posterior)
    dz = delta[2]       # S-I offset (signed: + superior, - inferior)

    if dx < 1.0:
        return {
            "rotation_rad": 0.0, "rotation_deg": 0.0,
            "tilt_rad": 0.0, "tilt_deg": 0.0,
        }

    # Axial rotation: angle of inter-hip axis in the XY plane
    # Positive = L hip is more anterior than R hip
    rotation_rad = np.arctan2(dy, dx)
    # Coronal tilt: angle of inter-hip axis in the XZ plane
    # Positive = L hip is more superior than R hip
    tilt_rad = np.arctan2(dz, dx)

    return {
        "rotation_rad": rotation_rad,
        "rotation_deg": round(np.degrees(rotation_rad), 2),
        "tilt_rad": tilt_rad,
        "tilt_deg": round(np.degrees(tilt_rad), 2),
    }


def build_correction_matrix_voxel(rotation_rad, tilt_rad, affine, center_voxel):
    """Build a 3x3 rotation matrix that corrects pose in voxel space.

    The correction is computed in world space (rotate around Z for axial,
    around Y for tilt) and then conjugated by the affine to operate
    directly on voxel coordinates.

    Parameters
    ----------
    rotation_rad : float
        Axial rotation angle to correct (radians).
    tilt_rad : float
        Coronal tilt angle to correct (radians).
    affine : numpy.ndarray of shape (4, 4)
        Voxel-to-world affine matrix.
    center_voxel : numpy.ndarray of shape (3,)
        Center of rotation in voxel space.

    Returns
    -------
    matrix : numpy.ndarray of shape (3, 3)
        Rotation matrix for ``scipy.ndimage.affine_transform``.
    offset : numpy.ndarray of shape (3,)
        Translation offset for ``scipy.ndimage.affine_transform``.
    """
    # Rotation around Z axis (corrects axial rotation)
    # Negative angle: undo the detected rotation
    cz, sz_ = np.cos(rotation_rad), np.sin(rotation_rad)
    Rz = np.array([
        [cz, -sz_, 0],
        [sz_, cz,  0],
        [0,   0,   1],
    ])

    # Rotation around Y axis (corrects coronal tilt)
    cy, sy = np.cos(tilt_rad), np.sin(tilt_rad)
    Ry = np.array([
        [cy,  0, sy],
        [0,   1,  0],
        [-sy, 0, cy],
    ])

    # Combined world-space rotation: first correct tilt, then rotation
    R_world = Rz @ Ry

    # Convert to voxel-space rotation:
    # v_world = A @ v_voxel  =>  R_voxel = A^{-1} @ R_world @ A
    A = affine[:3, :3]
    A_inv = np.linalg.inv(A)
    R_voxel = A_inv @ R_world @ A

    # scipy.ndimage.affine_transform applies: output[o] = input[A @ o + offset]
    # We want to rotate around center_voxel:
    # new_coord = R_voxel @ (old_coord - center) + center
    # => old_coord = R_voxel_inv @ new_coord + center - R_voxel_inv @ center
    # affine_transform uses the inverse, so matrix = R_voxel_inv
    R_inv = np.linalg.inv(R_voxel)
    offset = center_voxel - R_inv @ center_voxel

    return R_inv, offset


def correct_mask(mask, matrix, offset):
    """Apply affine rotation to a binary mask using nearest-neighbor.

    Parameters
    ----------
    mask : numpy.ndarray
        3-D binary mask.
    matrix : numpy.ndarray of shape (3, 3)
        Rotation matrix from ``build_correction_matrix_voxel``.
    offset : numpy.ndarray of shape (3,)
        Translation offset.

    Returns
    -------
    numpy.ndarray
        Rotated binary mask (same shape as input).
    """
    rotated = affine_transform(
        mask.astype(np.float32),
        matrix,
        offset=offset,
        order=0,  # nearest neighbor — preserves binary
        mode="constant",
        cval=0.0,
    )
    return (rotated > 0.5).astype(mask.dtype)


def correct_affine(affine, rotation_rad, tilt_rad):
    """Update the voxel-to-world affine to reflect the pose correction.

    After rotating the voxel data, the affine must be updated so that
    ``voxel_to_world_ras`` produces correct world coordinates.

    Parameters
    ----------
    affine : numpy.ndarray of shape (4, 4)
        Original voxel-to-world affine.
    rotation_rad, tilt_rad : float
        Correction angles applied.

    Returns
    -------
    numpy.ndarray of shape (4, 4)
        Updated affine matrix.
    """
    # The world-space rotation we applied to voxels
    cz, sz_ = np.cos(-rotation_rad), np.sin(-rotation_rad)
    Rz = np.array([
        [cz, -sz_, 0],
        [sz_, cz,  0],
        [0,   0,   1],
    ])
    cy, sy = np.cos(-tilt_rad), np.sin(-tilt_rad)
    Ry = np.array([
        [cy,  0, sy],
        [0,   1,  0],
        [-sy, 0, cy],
    ])
    R_world = Rz @ Ry

    # New affine: A' = R_world^{-1} @ A
    # (voxels were rotated by R, so world coords come from inverting that)
    R_inv_world = np.linalg.inv(R_world)
    new_affine = affine.copy()
    new_affine[:3, :3] = R_inv_world @ affine[:3, :3]
    new_affine[:3, 3] = R_inv_world @ affine[:3, 3]
    return new_affine


def normalize_pelvic_pose(masks, affine, correction_threshold_deg=2.0):
    """Normalize pelvic pose by correcting axial rotation and coronal tilt.

    Only applies correction when rotation or tilt exceeds the threshold.

    Parameters
    ----------
    masks : dict
        Dictionary with keys ``'hip_L'``, ``'hip_R'`` (required),
        and optionally ``'sacrum'``, ``'femur_L'``, ``'femur_R'``,
        ``'vert_S1'``.  Values are 3-D numpy arrays.
    affine : numpy.ndarray of shape (4, 4)
        Voxel-to-world affine matrix.
    correction_threshold_deg : float, optional
        Minimum angle (degrees) to trigger correction (default 2.0).

    Returns
    -------
    corrected_masks : dict
        Same keys as input, with corrected mask arrays.
    corrected_affine : numpy.ndarray of shape (4, 4)
        Updated affine matrix.
    correction_info : dict
        Correction metadata: ``'applied'`` (bool),
        ``'pre_rotation_deg'``, ``'pre_tilt_deg'``,
        ``'post_rotation_deg'``, ``'post_tilt_deg'``.
    """
    hip_L = masks["hip_L"]
    hip_R = masks["hip_R"]

    # Step 1: Compute current rotation/tilt
    centroid_L, centroid_R = compute_hip_centroids_world(hip_L, hip_R, affine)
    angles = compute_correction_angles(centroid_L, centroid_R)

    info = {
        "applied": False,
        "pre_rotation_deg": angles["rotation_deg"],
        "pre_tilt_deg": angles["tilt_deg"],
        "post_rotation_deg": angles["rotation_deg"],
        "post_tilt_deg": angles["tilt_deg"],
    }

    # Step 2: Check if correction is needed
    needs_rotation = abs(angles["rotation_deg"]) >= correction_threshold_deg
    needs_tilt = abs(angles["tilt_deg"]) >= correction_threshold_deg

    if not needs_rotation and not needs_tilt:
        return masks, affine, info

    center_voxel = np.array(hip_L.shape) / 2.0
    current_masks = dict(masks)

    # Step 3a: Correct rotation first (if needed)
    if needs_rotation:
        matrix, offset = build_correction_matrix_voxel(
            angles["rotation_rad"], 0.0, affine, center_voxel
        )
        current_masks = {
            k: correct_mask(v, matrix, offset) if v is not None else None
            for k, v in current_masks.items()
        }

    # Step 3b: Recompute tilt after rotation correction, then correct tilt
    if needs_tilt:
        cL2, cR2 = compute_hip_centroids_world(
            current_masks["hip_L"], current_masks["hip_R"], affine
        )
        tilt_angles = compute_correction_angles(cL2, cR2)

        if abs(tilt_angles["tilt_deg"]) >= correction_threshold_deg:
            matrix, offset = build_correction_matrix_voxel(
                0.0, -tilt_angles["tilt_rad"], affine, center_voxel
            )
            current_masks = {
                k: correct_mask(v, matrix, offset) if v is not None else None
                for k, v in current_masks.items()
            }

    # Step 4: Verify correction using the original affine
    post_centroid_L, post_centroid_R = compute_hip_centroids_world(
        current_masks["hip_L"], current_masks["hip_R"], affine
    )
    post_angles = compute_correction_angles(post_centroid_L, post_centroid_R)

    info["applied"] = True
    info["post_rotation_deg"] = post_angles["rotation_deg"]
    info["post_tilt_deg"] = post_angles["tilt_deg"]

    return current_masks, affine, info
