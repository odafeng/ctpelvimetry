"""
Pelvimetry measurement functions: ISD, ITD (outlet transverse),
sacral depth, and helper routines.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import label as ndimage_label
from scipy.spatial import cKDTree

from .config import DEFAULT_PELVIC_CONFIG
from .landmarks import _voxels_in_xslab


# ------------------------------------------------------------------
# Sacral Depth
# ------------------------------------------------------------------

def calculate_sacral_depth(sacrum, promontory, coccygeal_apex, affine,
                           mid_x=None, slab_half_voxels=None):
    """Calculate sacral depth (anterior concavity).

    Sacral depth is the maximum perpendicular distance from the
    anterior contour of the sacrum to the chord connecting the
    promontory and the coccygeal apex, measured on the midsagittal
    plane.

    Parameters
    ----------
    sacrum : numpy.ndarray
        3-D binary sacrum mask (may be sacrum+S1 merged).
    promontory : array-like of shape (3,)
        Promontory voxel coordinates (i, j, k).
    coccygeal_apex : array-like of shape (3,)
        Coccygeal apex voxel coordinates (i, j, k).
    affine : numpy.ndarray of shape (4, 4)
        Voxel-to-world affine matrix.
    mid_x : float, optional
        Midline X in voxel space.
    slab_half_voxels : int, optional
        Half-width of the X slab in voxels.

    Returns
    -------
    depth_mm : float or None
        Maximum sacral depth in mm, rounded to 1 decimal.
    deepest_pt : numpy.ndarray of shape (3,) or None
        Voxel coordinates of the deepest contour point.
    contour_points : numpy.ndarray of shape (N, 3) or None
        Anterior contour voxel coordinates used for the
        calculation.
    """
    if promontory is None or coccygeal_apex is None:
        return None, None, None

    coords = np.argwhere(sacrum > 0)
    if len(coords) == 0:
        return None, None, None

    # Step 1: Extract anterior contour
    z_min, z_max = min(promontory[2], coccygeal_apex[2]), max(promontory[2], coccygeal_apex[2])

    if mid_x is not None and slab_half_voxels is not None:
        coords_mid = _voxels_in_xslab(coords, mid_x, slab_half_voxels)
        if len(coords_mid) < 3:
            coords_mid = coords  # fallback

        contour_points = []
        for z in range(int(z_min), int(z_max) + 1):
            z_slice = coords_mid[coords_mid[:, 2] == z]
            if len(z_slice) == 0:
                continue
            contour_points.append(z_slice[np.argmax(z_slice[:, 1])])
    else:
        # Fallback: per-Z adaptive local midline
        x_margin = 3
        contour_points = []
        for z in range(int(z_min), int(z_max) + 1):
            z_slice = coords[coords[:, 2] == z]
            if len(z_slice) == 0:
                continue
            local_mid_x = np.median(z_slice[:, 0])
            band = z_slice[
                (z_slice[:, 0] >= local_mid_x - x_margin)
                & (z_slice[:, 0] <= local_mid_x + x_margin)
            ]
            if len(band) > 0:
                contour_points.append(band[np.argmax(band[:, 1])])

    if len(contour_points) < 3:
        return None, None, None

    contour_points = np.array(contour_points)

    # Step 1b: Median-filter the contour Y values to remove outlier spikes
    from scipy.ndimage import median_filter

    raw_y = contour_points[:, 1].astype(float).copy()
    kernel_size = min(5, len(raw_y))
    if kernel_size % 2 == 0:
        kernel_size = max(kernel_size - 1, 3)
    smoothed_y = median_filter(raw_y, size=kernel_size)

    # Step 1c: Truncate contour at Y-reversal near promontory end ONLY
    n = len(smoothed_y)
    check_window = max(3, n // 4)  # only check last 25% of points
    if n > 3:
        max_y_from_top = smoothed_y[-1]
        trunc_end = n  # default: keep all
        for i in range(n - 2, max(n - 1 - check_window, -1), -1):
            if smoothed_y[i] > max_y_from_top:
                trunc_end = i + 2
                break
            max_y_from_top = max(max_y_from_top, smoothed_y[i])
        if trunc_end < n and trunc_end >= 3:
            contour_points = contour_points[:trunc_end]
            smoothed_y = smoothed_y[:trunc_end]

    # Build smoothed contour for depth calculation
    contour_smoothed = contour_points.copy().astype(float)
    contour_smoothed[:, 1] = smoothed_y

    # Step 2: Compute depth using axis-aligned sagittal projection
    if mid_x is not None:
        sy = np.abs(affine[1, 1])
        sz = np.abs(affine[2, 2])
        contour_2d = contour_smoothed[:, [1, 2]] * np.array([sy, sz])
        p1_2d = np.array([promontory[1] * sy, promontory[2] * sz])
        p2_2d = np.array([coccygeal_apex[1] * sy, coccygeal_apex[2] * sz])
    else:
        print("      ⚠️ Sacral Depth skipped: midline X not available")
        return None, None, contour_points

    chord_vec = p2_2d - p1_2d
    chord_len = np.linalg.norm(chord_vec)
    if chord_len == 0:
        return None, None, contour_points

    chord_unit = chord_vec / chord_len

    # Concavity side detection via contour midpoint
    mid_idx = len(contour_2d) // 2
    v_mid = contour_2d[mid_idx] - p1_2d
    cross_mid = v_mid[0] * chord_unit[1] - v_mid[1] * chord_unit[0]
    concavity_sign = np.sign(cross_mid) if cross_mid != 0 else -1.0

    # Step 3: Max perpendicular distance (20% margin + min 5 Z-slices from endpoints)
    margin = 0.20 * chord_len
    min_z_gap = 5  # minimum Z-slices away from promontory/apex
    prom_z = promontory[2]
    apex_z = coccygeal_apex[2]
    max_depth = 0.0
    deepest_idx = None

    for i, pt_2d in enumerate(contour_2d):
        # Skip points too close to promontory or apex in Z
        pt_z = contour_smoothed[i, 2]
        if abs(pt_z - prom_z) < min_z_gap or abs(pt_z - apex_z) < min_z_gap:
            continue

        v = pt_2d - p1_2d
        proj_len = np.dot(v, chord_unit)

        if margin < proj_len < (chord_len - margin):
            signed_dist = v[0] * chord_unit[1] - v[1] * chord_unit[0]
            if np.sign(signed_dist) == concavity_sign:
                depth = abs(signed_dist)
                if depth > max_depth:
                    max_depth = depth
                    deepest_idx = i

    if deepest_idx is None:
        return 0.0, None, contour_points

    # Return the smoothed contour point (not the noisy original) for QC display
    deepest_pt = contour_smoothed[deepest_idx].astype(int)

    return round(max_depth, 1), deepest_pt, contour_points


# ------------------------------------------------------------------
# Outlet Transverse (ITD)
# ------------------------------------------------------------------

def find_outlet_transverse(
    hip_L,
    hip_R,
    affine,
    low_quantile=0.25,
    posterior_quantile=0.30,
    tub_radius_mm=20.0,
    config=None,
    isd_z_voxel=None,
):
    """Compute the outlet transverse (intertuberous) diameter.

    Measures the shortest distance between the medial surfaces of
    the left and right ischial tuberosities at a similar Z-level.

    Parameters
    ----------
    hip_L : numpy.ndarray
        3-D binary mask of the left hip.
    hip_R : numpy.ndarray
        3-D binary mask of the right hip.
    affine : numpy.ndarray of shape (4, 4)
        Voxel-to-world affine matrix.
    low_quantile : float, optional
        Quantile for caudal Z filtering (default 0.25).
    posterior_quantile : float, optional
        Quantile for posterior Y filtering (default 0.30).
    tub_radius_mm : float, optional
        Radius around the tuberosity centroid in mm (default 20.0).
    config : PelvicConfig, optional
        Quality-gate configuration.
    isd_z_voxel : int or None, optional
        ISD Z-level; voxels at or above this level are excluded.

    Returns
    -------
    diameter_mm : float or None
        Intertuberous distance in mm.
    pt_left_ijk : numpy.ndarray of shape (2,) or None
        Left tuberosity point ``(i, j)`` in voxel space.
    pt_right_ijk : numpy.ndarray of shape (2,) or None
        Right tuberosity point ``(i, j)`` in voxel space.
    qc_z_voxel : int or None
        Axial slice index for QC visualisation.
    """

    def _mask_to_world(mask, aff):
        """Mask voxels → Nx3 world coordinates (mm)."""
        idx = np.argwhere(mask > 0)  # (i, j, k)
        if len(idx) == 0:
            return np.empty((0, 3)), np.empty((0, 3), dtype=int)
        ijk1 = np.c_[idx, np.ones(len(idx))]
        xyz = (aff @ ijk1.T).T[:, :3]
        return xyz, idx

    def _infer_tuberosity(mask, aff, low_q, post_q, radius_mm):
        """Isolate ischial tuberosity body in world coordinates."""
        pts, ijk = _mask_to_world(mask, aff)
        if pts.shape[0] < 500:
            return None, None

        # Step 1: Caudal filter — lowest Z quantile
        z_thr = np.quantile(pts[:, 2], low_q)
        caudal_mask = pts[:, 2] <= z_thr
        low = pts[caudal_mask]
        low_ijk = ijk[caudal_mask]
        if len(low) < 50:
            return None, None

        # Step 2: Posterior filter — lowest Y quantile
        y_thr = np.quantile(low[:, 1], post_q)
        post_mask = low[:, 1] <= y_thr
        low_post = low[post_mask]
        low_post_ijk = low_ijk[post_mask]
        if len(low_post) < 20:
            return None, None

        # Step 3: CC scoring — select most posterior+caudal CC
        ijk_off = np.min(low_post_ijk, axis=0)
        local = low_post_ijk - ijk_off
        shape = tuple(np.max(local, axis=0) + 1)
        sub_vol = np.zeros(shape, dtype=np.uint8)
        sub_vol[local[:, 0], local[:, 1], local[:, 2]] = 1

        labeled, n_cc = ndimage_label(sub_vol)
        if n_cc == 0:
            return low_post, low_post_ijk

        best_cc = None
        best_score = np.inf
        for cc in range(1, n_cc + 1):
            keep_local = np.argwhere(labeled == cc)
            keep_ijk_cc = keep_local + ijk_off
            if keep_ijk_cc.shape[0] < 100:
                continue
            ijk1 = np.c_[keep_ijk_cc, np.ones(len(keep_ijk_cc))]
            xyz_cc = (aff @ ijk1.T).T[:, :3]
            yc = np.median(xyz_cc[:, 1])  # posterior = low Y
            zc = np.median(xyz_cc[:, 2])  # inferior = low Z
            score = yc + zc
            if score < best_score:
                best_score = score
                best_cc = cc

        if best_cc is None:
            return low_post, low_post_ijk

        keep_local = np.argwhere(labeled == best_cc)
        keep_ijk = keep_local + ijk_off
        keep_set = set(map(tuple, keep_ijk.tolist()))
        keep_mask = np.array([tuple(r) in keep_set for r in low_post_ijk.tolist()])
        cc_pts = low_post[keep_mask]
        cc_ijk = low_post_ijk[keep_mask]

        if len(cc_pts) < 10:
            return None, None

        # Step 4: Centroid + radius — isolate tuberosity body
        centroid = np.mean(cc_pts, axis=0)
        dists_to_ctr = np.linalg.norm(cc_pts - centroid, axis=1)
        radius_mask = dists_to_ctr <= radius_mm
        tub_pts = cc_pts[radius_mask]
        tub_ijk = cc_ijk[radius_mask]

        if len(tub_pts) < 5:
            return None, None

        return tub_pts, tub_ijk

    if hip_L is None or hip_R is None or affine is None:
        return None, None, None, None

    # ISD-guided Z-mask: exclude voxels at/above the ischial spine plane
    if isd_z_voxel is not None:
        hip_L = hip_L.copy()
        hip_R = hip_R.copy()
        hip_L[:, :, isd_z_voxel:] = 0
        hip_R[:, :, isd_z_voxel:] = 0

    tub_L_xyz, tub_L_ijk = _infer_tuberosity(
        hip_L, affine, low_quantile, posterior_quantile, tub_radius_mm
    )
    tub_R_xyz, tub_R_ijk = _infer_tuberosity(
        hip_R, affine, low_quantile, posterior_quantile, tub_radius_mm
    )
    if tub_L_xyz is None or tub_R_xyz is None:
        return None, None, None, None
    if len(tub_L_xyz) < 5 or len(tub_R_xyz) < 5:
        return None, None, None, None

    # Step 5: Medial surface with posterior band constraint
    y_thr_L2 = np.quantile(tub_L_xyz[:, 1], 0.30)
    y_thr_R2 = np.quantile(tub_R_xyz[:, 1], 0.30)
    L_post2 = tub_L_xyz[:, 1] <= y_thr_L2
    R_post2 = tub_R_xyz[:, 1] <= y_thr_R2

    # RAS+: Left hip medial = high X, Right hip medial = low X
    x_thr_L = np.quantile(tub_L_xyz[:, 0], 0.95)
    x_thr_R = np.quantile(tub_R_xyz[:, 0], 0.05)
    L_med_mask = (tub_L_xyz[:, 0] >= x_thr_L) & L_post2
    R_med_mask = (tub_R_xyz[:, 0] <= x_thr_R) & R_post2
    L_med = tub_L_xyz[L_med_mask]
    R_med = tub_R_xyz[R_med_mask]
    L_med_ijk = tub_L_ijk[L_med_mask]
    R_med_ijk = tub_R_ijk[R_med_mask]

    # Fallback: if posterior band too strict, relax to X-only
    if len(L_med) == 0:
        L_med_mask = tub_L_xyz[:, 0] >= x_thr_L
        L_med = tub_L_xyz[L_med_mask]
        L_med_ijk = tub_L_ijk[L_med_mask]
    if len(R_med) == 0:
        R_med_mask = tub_R_xyz[:, 0] <= x_thr_R
        R_med = tub_R_xyz[R_med_mask]
        R_med_ijk = tub_R_ijk[R_med_mask]

    if len(L_med) == 0 or len(R_med) == 0:
        return None, None, None, None

    # Step 6: cKDTree with Z-layer AND Y-layer constraints
    _ZY_THRESHOLDS = [(2.0, 2.0), (2.0, 5.0), (5.0, 5.0), (5.0, 10.0)]

    def _find_best_pair(L_pts, R_pts, tree_r, dists_arr, idx_arr, thresholds):
        """Find closest L-R pair satisfying (|ΔZ|, |ΔY|) constraints."""
        for max_dz, max_dy in thresholds:
            candidate = None
            for i in range(L_pts.shape[0]):
                for j in range(idx_arr.shape[1]):
                    ridx = idx_arr[i, j]
                    dz = abs(L_pts[i, 2] - R_pts[ridx, 2])
                    dy = abs(L_pts[i, 1] - R_pts[ridx, 1])
                    if dz <= max_dz and dy <= max_dy:
                        d = float(dists_arr[i, j])
                        if candidate is None or d < candidate[0]:
                            candidate = (d, i, ridx)
            if candidate is not None:
                return candidate
        return None

    tree = cKDTree(R_med)
    dists, indices = tree.query(L_med, k=min(10, len(R_med)))
    # Ensure indices is 2D even if k=1
    if indices.ndim == 1:
        dists = dists.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    best = _find_best_pair(L_med, R_med, tree, dists, indices, _ZY_THRESHOLDS)

    # Fallback: retry with X-only medial surface
    if best is None:
        L_xonly = tub_L_xyz[:, 0] >= x_thr_L
        R_xonly = tub_R_xyz[:, 0] <= x_thr_R
        L_med_fb = tub_L_xyz[L_xonly]
        R_med_fb = tub_R_xyz[R_xonly]
        L_med_ijk_fb = tub_L_ijk[L_xonly]
        R_med_ijk_fb = tub_R_ijk[R_xonly]

        if len(L_med_fb) > 0 and len(R_med_fb) > 0:
            tree_fb = cKDTree(R_med_fb)
            dists_fb, idx_fb = tree_fb.query(
                L_med_fb, k=min(10, len(R_med_fb))
            )
            if idx_fb.ndim == 1:
                dists_fb = dists_fb.reshape(-1, 1)
                idx_fb = idx_fb.reshape(-1, 1)
            best = _find_best_pair(
                L_med_fb, R_med_fb, tree_fb, dists_fb, idx_fb, _ZY_THRESHOLDS
            )
            if best is not None:
                L_med = L_med_fb
                R_med = R_med_fb
                L_med_ijk = L_med_ijk_fb
                R_med_ijk = R_med_ijk_fb

    if best is None:
        return None, None, None, None

    itd, Li, Ri = best
    best_L_ijk = L_med_ijk[Li]
    best_R_ijk = R_med_ijk[Ri]

    pt_L = np.array([best_L_ijk[0], best_L_ijk[1]])
    pt_R = np.array([best_R_ijk[0], best_R_ijk[1]])
    qc_z = int(round((best_L_ijk[2] + best_R_ijk[2]) / 2.0))

    min_itd = config.min_itd_mm if config else 60.0
    max_itd = config.max_itd_mm if config else 200.0
    if min_itd <= itd <= max_itd:
        return round(itd, 1), pt_L, pt_R, qc_z
    return None, None, None, None


# ------------------------------------------------------------------
# ISD spine candidate detection
# ------------------------------------------------------------------

def _pick_spine_candidate_2d(
    pts_xy, _sx, _sy, midline_x, side="L", posterior_band_q=0.20, min_band_pts=80
):
    """Find a 2-D proxy for the ischial spine tip on an axial slice.

    Strategy: posterior gating → posterior band selection → most
    medial point → posterior tie-breaker.

    Parameters
    ----------
    pts_xy : numpy.ndarray of shape (N, 2)
        Voxel ``(x, y)`` coordinates of one hip on a single slice.
    _sx, _sy : float
        Voxel spacings in X and Y (mm).  Currently unused but kept
        for future refinement.
    midline_x : float
        Midline X in voxel space.
    side : {'L', 'R'}, optional
        Which hip side (default ``'L'``).
    posterior_band_q : float, optional
        Quantile for the posterior band threshold (default 0.20).
    min_band_pts : int, optional
        Minimum number of points in the band before fallback
        (default 80).

    Returns
    -------
    pt : numpy.ndarray of shape (2,) or None
        Best spine candidate ``(x, y)``.
    meta : dict
        Diagnostic metadata.
    """
    meta = {
        "y_centroid": np.nan,
        "n_total": 0,
        "n_posterior": 0,
        "n_band": 0,
        "midline_x": float(midline_x),
        "side": side,
        "posterior_band_q": posterior_band_q,
    }

    if pts_xy is None or len(pts_xy) == 0:
        return None, meta

    meta["n_total"] = int(len(pts_xy))

    y_centroid = float(np.mean(pts_xy[:, 1]))
    meta["y_centroid"] = round(y_centroid, 1)

    posterior_pts = pts_xy[pts_xy[:, 1] < y_centroid]
    if len(posterior_pts) == 0:
        posterior_pts = pts_xy
    meta["n_posterior"] = int(len(posterior_pts))

    y_thr = np.quantile(posterior_pts[:, 1], posterior_band_q)
    band = posterior_pts[posterior_pts[:, 1] <= y_thr]

    if len(band) < min_band_pts:
        band = posterior_pts
    meta["n_band"] = int(len(band))

    dx = np.abs(band[:, 0].astype(np.float32) - float(midline_x))
    y = band[:, 1].astype(np.float32)
    idx = np.lexsort((y, dx))
    return band[idx[0]], meta


# ------------------------------------------------------------------
# ISD core logic (valley detection + plateau fallback)
# ------------------------------------------------------------------

def _run_isd_core_logic(data, sx, sy, sz, search_range_z, config):
    """Valley-detection and plateau-fallback ISD selection.

    Per-slice spine candidates are identified, the ISD distance curve
    is smoothed, and local minima (valleys) are detected with
    ``find_peaks``.  If no valley qualifies, a plateau-based fallback
    is used.  Sacrum presence is required at the selected Z-level.

    Parameters
    ----------
    data : dict
        Pre-loaded mask arrays keyed by ROI name.
    sx, sy, sz : float
        Voxel spacings in mm.
    search_range_z : tuple of (int, int)
        ``(start_z, end_z)`` inclusive search range.
    config : PelvicConfig
        Detection parameters.

    Returns
    -------
    result : dict or None
        Contains ``'ISD_mm'``, ``'ISD_slice'``, ``'pt_L'``,
        ``'pt_R'``, ``'midline_x'`` on success.
    trace : list of dict
        Per-slice diagnostic records.
    status_msg : str
        Human-readable status string.
    valleys_debug : list of tuple
        ``(z, dist_mm, prominence)`` for each detected valley.
    """

    def _has_sacrum(z):
        if "sacrum" not in data:
            return False
        z_int = int(z)
        buffer = max(1, int(round(config.sacrum_buffer_mm / sz)))
        z_start = max(0, z_int - buffer)
        z_end = min(data["sacrum"].shape[2], z_int + buffer + 1)
        if z_start >= z_end:
            return False
        return np.sum(data["sacrum"][:, :, z_start:z_end]) > 0

    start_z, end_z = search_range_z
    start_z = int(max(0, start_z))
    end_z = int(min(data["hip_L"].shape[2], end_z))
    trace = []

    for z in range(start_z, end_z):
        rec = {
            "z": int(z),
            "status": "reject",
            "reason": None,
            "dist_mm": np.nan,
            "midline_x": np.nan,
            "pt_L_x": np.nan,
            "pt_L_y": np.nan,
            "pt_R_x": np.nan,
            "pt_R_y": np.nan,
        }

        slice_L = data["hip_L"][:, :, z]
        slice_R = data["hip_R"][:, :, z]

        if np.sum(slice_L) == 0 or np.sum(slice_R) == 0:
            rec["reason"] = "empty_mask"
            trace.append(rec)
            continue

        pts_L = np.argwhere(slice_L > 0)[:, :2]
        pts_R = np.argwhere(slice_R > 0)[:, :2]

        midline_x = float(np.mean(np.concatenate([pts_L[:, 0], pts_R[:, 0]])))
        rec["midline_x"] = int(round(midline_x))

        pt_L, _ = _pick_spine_candidate_2d(pts_L, sx, sy, midline_x, side="L")
        pt_R, _ = _pick_spine_candidate_2d(pts_R, sx, sy, midline_x, side="R")

        if pt_L is None or pt_R is None:
            rec["reason"] = "no_candidate"
            trace.append(rec)
            continue

        rec["pt_L_x"], rec["pt_L_y"] = pt_L[0], pt_L[1]
        rec["pt_R_x"], rec["pt_R_y"] = pt_R[0], pt_R[1]
        dist_mm = float(
            np.linalg.norm((pt_L - pt_R) * np.array([sx, sy], dtype=np.float32))
        )
        rec["dist_mm"] = dist_mm
        rec["midline_x"] = int(round(midline_x))
        rec["pts"] = (pt_L, pt_R)

        if dist_mm < config.min_isd_mm:
            rec["reason"] = "too_narrow"
        elif dist_mm > config.max_isd_mm:
            rec["reason"] = "too_wide"
        else:
            rec["status"] = "candidate"
        trace.append(rec)

    # Valley Detection
    if not trace:
        return None, trace, "Fail_Search_Range", []
    df_trace = pd.DataFrame(trace)
    if "dist_mm" not in df_trace.columns:
        return None, trace, "Fail_Trace_Data", []
    valid_mask = ~df_trace["dist_mm"].isna()
    if valid_mask.sum() < 5:
        return None, trace, "Fail_Ambiguous", []

    dists = (
        df_trace["dist_mm"]
        .interpolate(method="linear", limit_direction="both")
        .to_numpy()
    )
    kernel = np.ones(config.smoothing_window) / config.smoothing_window
    dists_smooth = np.convolve(dists, kernel, mode="same")
    for i, val in enumerate(dists_smooth):
        if i < len(trace):
            trace[i]["smooth_dist"] = val

    peaks, properties = find_peaks(-dists_smooth, prominence=config.prominence_mm)
    valleys = []
    for idx in peaks:
        z_val = df_trace.iloc[idx]["z"]
        dist_val = df_trace.iloc[idx]["dist_mm"]
        prominence = properties["prominences"][list(peaks).index(idx)]

        if (z_val <= start_z + config.edge_margin_slices) or (
            z_val >= end_z - config.edge_margin_slices
        ):
            continue
        if not _has_sacrum(z_val):
            continue
        if config.min_isd_mm <= dist_val <= config.max_isd_mm:
            valleys.append((z_val, dist_val, prominence))

    # Strategy 1: Valley Detection
    best_rec = None
    status_msg = None

    if valleys:
        valleys.sort(key=lambda x: x[2], reverse=True)
        best_z, _best_dist, best_prom = valleys[0]
        best_rec = next((r for r in trace if r["z"] == best_z), None)
        if best_rec:
            status_msg = f"Success_Valley (Prom={best_prom:.1f})"

    # Strategy 2: Plateau Fallback
    if best_rec is None:
        slope = np.gradient(dists_smooth)
        is_plateau = np.abs(slope) < config.plateau_slope_thresh

        df_trace["is_plateau"] = is_plateau
        df_trace["group"] = (
            df_trace["is_plateau"] != df_trace["is_plateau"].shift()
        ).cumsum()

        valid_plateaus = []
        for _, grp in df_trace[df_trace["is_plateau"]].groupby("group"):
            if len(grp) < config.plateau_min_len:
                continue
            median_idx = len(grp) // 2
            rep_row = grp.iloc[median_idx]
            z_rep = int(rep_row["z"])

            if not (
                start_z + config.edge_margin_slices
                < z_rep
                < end_z - config.edge_margin_slices
            ):
                continue
            if not _has_sacrum(z_rep):
                continue
            dist_rep = rep_row["dist_mm"]
            if pd.isna(dist_rep):
                continue
            if not (config.min_isd_mm <= dist_rep <= config.max_isd_mm):
                continue

            valid_plateaus.append(
                {
                    "z": z_rep,
                    "dist_mm": dist_rep,
                    "plateau_len": len(grp),
                    "row_idx": rep_row.name,
                }
            )

        if valid_plateaus:
            best_plateau = min(valid_plateaus, key=lambda x: x["dist_mm"])
            best_rec = next((r for r in trace if r["z"] == best_plateau["z"]), None)
            if best_rec:
                status_msg = f"Success_Plateau (Len={best_plateau['plateau_len']})"

    # Final Check
    if best_rec is None:
        return None, trace, "Fail_Ambiguous", []

    if not _has_sacrum(best_rec["z"]):
        return None, trace, f"Fail_No_Sacrum (Z={best_rec['z']})", []

    result = {
        "ISD_mm": best_rec["dist_mm"],
        "ISD_slice": best_rec["z"],
        "pt_L": best_rec["pts"][0],
        "pt_R": best_rec["pts"][1],
        "midline_x": best_rec["midline_x"],
    }
    valleys_debug = [(v[0], v[1], v[2]) for v in valleys]
    return result, trace, status_msg, valleys_debug


# ------------------------------------------------------------------
# ISD calculation
# ------------------------------------------------------------------

def calculate_isd(data, spacing, _img_affine):
    """Calculate ISD (Inter-Spinous Distance).

    Determines an optimal Z-level search range using a 3-tier
    fallback (femur → sacrum → hip), runs valley/plateau ISD
    detection.

    Parameters
    ----------
    data : dict
        Pre-loaded 3-D mask arrays keyed by ROI name
        (``'hip_L'``, ``'hip_R'``, ``'sacrum'``, ``'femur_L'``,
        ``'femur_R'``).
    spacing : tuple of (float, float, float)
        Voxel spacings ``(sx, sy, sz)`` in mm.
    _img_affine : numpy.ndarray of shape (4, 4)
        Voxel-to-world affine (kept for API compatibility).

    Returns
    -------
    dict
        Keys include ``'ISD_mm'``, ``'ISD_slice'``,
        ``'ISD_pt_left'``, ``'ISD_pt_right'``, ``'Status'``, and
        ``'midline_x'``.
    """
    result = {
        "ISD_mm": None,
        "ISD_slice": None,

        "ISD_pt_left": None,
        "ISD_pt_right": None,
        "Status": "Init",
    }

    if "hip_L" not in data or "hip_R" not in data:
        result["Status"] = "Fail_Missing_hip_mask"
        return result

    sx, sy, sz = spacing

    # Determine search range (3-tier fallback)
    config = DEFAULT_PELVIC_CONFIG
    z_min_search, z_max_search = 0, 0
    has_femur = ("femur_L" in data) or ("femur_R" in data)
    has_sacrum = "sacrum" in data

    if has_femur:
        z_indices = []
        if "femur_L" in data:
            z_indices.append(np.where(data["femur_L"] > 0)[2])
        if "femur_R" in data:
            z_indices.append(np.where(data["femur_R"] > 0)[2])
        if z_indices and len(np.concatenate(z_indices)) > 0:
            z_all = np.concatenate(z_indices)
            f_max = np.max(z_all)
            margin_sup_slice = int(config.search_margin_superior_mm / sz)
            margin_inf_slice = int(config.search_margin_inferior_mm / sz)
            z_max_search = min(f_max + margin_sup_slice, data["hip_L"].shape[2] - 1)
            z_min_search = max(f_max - margin_inf_slice, 0)
        else:
            has_femur = False

    if not has_femur and has_sacrum:
        z_idx_sac = np.where(data["sacrum"] > 0)[2]
        if len(z_idx_sac) > 0:
            s_min, s_max = np.min(z_idx_sac), np.max(z_idx_sac)
            z_max_search = int(s_min + (s_max - s_min) * 0.70)
            z_min_search = int(s_min + (s_max - s_min) * 0.10)
        else:
            has_sacrum = False

    if not has_femur and not has_sacrum:
        z_idx_hip = np.where((data["hip_L"] > 0) | (data["hip_R"] > 0))[2]
        if len(z_idx_hip) == 0:
            result["Status"] = "Fail_NoHip"
            return result
        h_min, h_max = np.min(z_idx_hip), np.max(z_idx_hip)
        h_height = h_max - h_min
        z_min_search = int(h_min + h_height * 0.15)
        z_max_search = int(h_min + h_height * 0.50)

    # Run valley detection + plateau fallback ISD logic
    isd_res, _trace, status_msg, _valleys_debug = _run_isd_core_logic(
        data, sx, sy, sz, (z_min_search, z_max_search), config
    )

    if isd_res is not None and "Success" in status_msg:
        best_z = isd_res["ISD_slice"]
        result["ISD_mm"] = round(isd_res["ISD_mm"], 1)
        result["ISD_slice"] = best_z
        result["ISD_pt_left"] = isd_res["pt_L"]
        result["ISD_pt_right"] = isd_res["pt_R"]
        result["midline_x"] = isd_res["midline_x"]
        result["Status"] = status_msg


    else:
        result["Status"] = status_msg or "Fail_NoValidISD"

    return result
