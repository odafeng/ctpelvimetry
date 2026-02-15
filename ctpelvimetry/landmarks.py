"""
Anatomical landmark detection: midline, symphysis, sacral promontory/apex,
pelvic orientation.
"""

import numpy as np
import nibabel as nib

from .config import DEFAULT_PELVIC_CONFIG


# ------------------------------------------------------------------
# Midline and slab utilities
# ------------------------------------------------------------------

def compute_midline_x(hip_L, hip_R):
    """Compute the midsagittal X coordinate from the pubic symphysis.

    Per-Z midpoint of left/right hip anterior medial edges is computed
    and the median is taken.  The result defines the axis-aligned
    midsagittal X used for all landmark detection and QC display.

    Parameters
    ----------
    hip_L : numpy.ndarray
        3-D binary mask of the left hip.
    hip_R : numpy.ndarray
        3-D binary mask of the right hip.

    Returns
    -------
    float or None
        Midline X in voxel space, or ``None`` if insufficient data.
    """
    coords_L = np.argwhere(hip_L > 0)
    coords_R = np.argwhere(hip_R > 0)

    if len(coords_L) == 0 or len(coords_R) == 0:
        return None

    # Anterior region: Y > 70th percentile (symphysis area)
    all_y = np.concatenate([coords_L[:, 1], coords_R[:, 1]])
    y_70 = np.percentile(all_y, 70)

    ant_L = coords_L[coords_L[:, 1] > y_70]
    ant_R = coords_R[coords_R[:, 1] > y_70]

    if len(ant_L) == 0 or len(ant_R) == 0:
        return None

    L_x_med = np.median(ant_L[:, 0])
    R_x_med = np.median(ant_R[:, 0])

    z_overlap = np.intersect1d(np.unique(ant_L[:, 2]), np.unique(ant_R[:, 2]))

    medial_midpoints = []
    for z in z_overlap:
        Lz = ant_L[ant_L[:, 2] == z]
        Rz = ant_R[ant_R[:, 2] == z]
        if len(Lz) == 0 or len(Rz) == 0:
            continue

        if L_x_med > R_x_med:
            l_medial_x = Lz[:, 0].min()
            r_medial_x = Rz[:, 0].max()
        else:
            l_medial_x = Lz[:, 0].max()
            r_medial_x = Rz[:, 0].min()

        medial_midpoints.append((l_medial_x + r_medial_x) / 2.0)

    if len(medial_midpoints) < 3:
        print("      ⚠️ Insufficient symphysis midline points for midline X")
        return None

    return float(np.median(medial_midpoints))


def _voxels_in_xslab(coords_voxel, mid_x, half_width):
    """Filter voxel coordinates to an axis-aligned X slab.

    Parameters
    ----------
    coords_voxel : numpy.ndarray of shape (N, 3)
        Voxel coordinates to filter.
    mid_x : float
        Centre X position in voxel space.
    half_width : float
        Half-width of the slab in voxels.

    Returns
    -------
    numpy.ndarray of shape (M, 3)
        Filtered subset of *coords_voxel*.
    """
    if len(coords_voxel) == 0:
        return coords_voxel
    return coords_voxel[np.abs(coords_voxel[:, 0] - mid_x) <= half_width]


# ------------------------------------------------------------------
# Pelvic orientation
# ------------------------------------------------------------------

def detect_pelvic_orientation(hip_L, hip_R, affine, config=None):
    """Detect pelvic axial rotation and coronal tilt.

    Analyses bilateral hip centroids in world coordinates to quantify:

    * **Axial rotation** (around SI axis) — L/R hips differ in AP (Y).
    * **Coronal tilt** (around AP axis) — L/R hips differ in SI (Z).

    Parameters
    ----------
    hip_L : numpy.ndarray
        3-D binary mask of the left hip.
    hip_R : numpy.ndarray
        3-D binary mask of the right hip.
    affine : numpy.ndarray of shape (4, 4)
        Voxel-to-world affine matrix.
    config : PelvicConfig, optional
        Threshold settings.  Defaults to ``DEFAULT_PELVIC_CONFIG``.

    Returns
    -------
    dict or None
        Dictionary with keys ``'rotation_deg'``, ``'tilt_deg'``,
        ``'rotation_flag'`` (``'ok'``/``'warn'``/``'high'``), and
        ``'tilt_flag'``.  Returns ``None`` when either mask is empty.
    """
    if config is None:
        config = DEFAULT_PELVIC_CONFIG

    coords_L = np.argwhere(hip_L > 0)
    coords_R = np.argwhere(hip_R > 0)

    if len(coords_L) == 0 or len(coords_R) == 0:
        return None

    # Centroids in world (mm) coordinates
    centroid_L = nib.affines.apply_affine(affine, coords_L.mean(axis=0))
    centroid_R = nib.affines.apply_affine(affine, coords_R.mean(axis=0))

    dx = abs(centroid_L[0] - centroid_R[0])  # L-R separation (should be dominant)
    dy = abs(centroid_L[1] - centroid_R[1])  # AP offset (rotation signal)
    dz = abs(centroid_L[2] - centroid_R[2])  # SI offset (tilt signal)

    # Avoid division by zero when hips overlap in X
    if dx < 1.0:
        return None

    rotation_deg = round(np.degrees(np.arctan2(dy, dx)), 1)
    tilt_deg = round(np.degrees(np.arctan2(dz, dx)), 1)

    def _flag(angle, warn_thresh, high_thresh):
        if angle >= high_thresh:
            return "high"
        elif angle >= warn_thresh:
            return "warn"
        return "ok"

    return {
        "rotation_deg": rotation_deg,
        "tilt_deg": tilt_deg,
        "rotation_flag": _flag(rotation_deg, config.rotation_warn_deg, config.rotation_fail_deg),
        "tilt_flag": _flag(tilt_deg, config.tilt_warn_deg, config.tilt_fail_deg),
    }


# ------------------------------------------------------------------
# Sacral landmarks
# ------------------------------------------------------------------

def find_sacral_landmarks(sacrum, mid_x=None, slab_half_voxels=None):
    """Detect sacral promontory and apex from the sacrum mask.

    When *mid_x* and *slab_half_voxels* are provided, the search is
    constrained to an X slab around the midsagittal plane.

    Parameters
    ----------
    sacrum : numpy.ndarray
        3-D binary sacrum mask.
    mid_x : float, optional
        Midline X in voxel space.
    slab_half_voxels : int, optional
        Half-width of the slab in voxels.

    Returns
    -------
    dict
        May contain ``'promontory'`` (voxel coords), ``'apex'``
        (voxel coords), and ``'midline_x'``.
    """
    result = {}
    coords = np.argwhere(sacrum > 0)
    if len(coords) == 0:
        return result

    # 0. Midline X and slab-constrained voxels
    if mid_x is not None and slab_half_voxels is not None:
        result["midline_x"] = mid_x
        coords_mid = _voxels_in_xslab(coords, mid_x, slab_half_voxels)
        if len(coords_mid) < 3:
            print("      ⚠️ Too few sacrum voxels in slab, using all")
            coords_mid = coords
    else:
        result["midline_x"] = np.median(coords[:, 0])
        coords_mid = coords

    midline_x = result["midline_x"]

    # 1. Promontory: Most anterior point at S1 superior border
    z_min_all, z_max_all = coords_mid[:, 2].min(), coords_mid[:, 2].max()
    z_range = z_max_all - z_min_all
    z_superior_threshold = z_max_all - 0.20 * z_range if z_range > 0 else z_min_all
    superior_coords = coords_mid[coords_mid[:, 2] >= z_superior_threshold]
    if len(superior_coords) == 0:
        superior_coords = coords_mid  # fallback

    # Sort: most anterior first (max Y), then closest to midline X
    candidates_prom = sorted(
        superior_coords, key=lambda p: (-p[1], abs(p[0] - midline_x))
    )
    if len(candidates_prom) > 0:
        result["promontory"] = candidates_prom[0]
    else:
        # Fallback to absolute max y
        promontory_idx = np.argmax(coords_mid[:, 1])
        result["promontory"] = coords_mid[promontory_idx].copy()

    # 2. Apex (Coccyx tip): Lowest Z
    min_z = coords_mid[:, 2].min()
    candidates_apex = coords_mid[coords_mid[:, 2] == min_z]
    candidates_apex = sorted(candidates_apex, key=lambda p: abs(p[0] - midline_x))
    if len(candidates_apex) > 0:
        result["apex"] = candidates_apex[0]
    else:
        coccyx_idx = np.argmin(coords_mid[:, 2])
        result["apex"] = coords_mid[coccyx_idx].copy()

    return result


# ------------------------------------------------------------------
# Symphysis landmarks
# ------------------------------------------------------------------

def find_symphysis_midline_sagittal(hip_L, hip_R, mid_x, slab_half_voxels, sy=1.0):
    """Locate upper and lower pubic symphysis landmarks.

    Landmarks are detected within the axis-aligned X slab defined by
    *mid_x* ± *slab_half_voxels*.  References: Baltus et al., Zhou et al.

    Parameters
    ----------
    hip_L : numpy.ndarray
        3-D binary mask of the left hip.
    hip_R : numpy.ndarray
        3-D binary mask of the right hip.
    mid_x : float
        Midline X in voxel space.
    slab_half_voxels : int
        Half-width of the X slab in voxels.
    sy : float, optional
        Voxel spacing in the Y direction (mm, default 1.0).

    Returns
    -------
    dict
        May contain ``'upper'`` (voxel coords), ``'lower'``
        (voxel coords), and ``'midline_x'``.
    """
    result = {}

    coords_L = np.argwhere(hip_L > 0)
    coords_R = np.argwhere(hip_R > 0)

    if len(coords_L) == 0 or len(coords_R) == 0:
        return result

    coords_all = np.vstack((coords_L, coords_R))

    # --- Extract midline band ---
    midline_band = _voxels_in_xslab(coords_all, mid_x, slab_half_voxels)

    if len(midline_band) == 0:
        return result

    # 4. Filter for Anterior Structure (Symphysis)
    max_y = midline_band[:, 1].max()
    anterior_threshold = max_y - int(50.0 / sy)
    anterior_cluster = midline_band[midline_band[:, 1] > anterior_threshold]

    if len(anterior_cluster) == 0:
        return result

    # 5. Upper Symphysis = Max Z
    idx_upper = np.argmax(anterior_cluster[:, 2])
    result["upper"] = anterior_cluster[idx_upper]

    # 6. Lower Symphysis = Min Z
    idx_lower = np.argmin(anterior_cluster[:, 2])
    result["lower"] = anterior_cluster[idx_lower]

    result["midline_x"] = mid_x

    return result
