"""
QC figure generation for pelvimetry and body composition.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402


# ------------------------------------------------------------------
# Pelvimetry QC: Sagittal Combined
# ------------------------------------------------------------------

def save_sagittal_combined_qc_figure(
    out_path, patient_id, ct_vol, result, landmarks, slab_half_voxels, sy, sz
):
    """Save sagittal combined QC figure.

    Renders a standalone sagittal MIP with all pelvimetry landmarks
    and measurement annotations overlaid.  The X slab matches the
    slab used for metric computation.

    Parameters
    ----------
    out_path : str
        Destination file path for the PNG.
    patient_id : str
        Patient identifier (used in figure title).
    ct_vol : numpy.ndarray
        3-D CT volume in RAS+ orientation.
    result : dict
        Measurement result dictionary.
    landmarks : dict
        Detected landmark coordinates.
    slab_half_voxels : int
        Half-width of the midsagittal X slab.
    sy, sz : float
        Voxel spacings in Y and Z (mm).
    """
    fig = plt.figure(figsize=(12, 10), dpi=150)
    fig.suptitle(
        f"Sagittal Combined: {patient_id}", fontsize=16, fontweight="bold", y=0.98
    )

    ax = fig.add_subplot(111)

    def safe_get(key, default=np.nan):
        return result.get(key, default) if result.get(key) is not None else default

    def fmt_val(key, unit="mm"):
        """Format a metric value for display, returning 'N/A' when missing."""
        v = result.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v:.1f}{unit}"

    def sagittal_mip(ct, mid_x, half_width):
        x_lo = max(0, int(mid_x) - half_width)
        x_hi = min(ct.shape[0], int(mid_x) + half_width + 1)
        return np.max(ct[x_lo:x_hi, :, :], axis=0)

    if ct_vol is not None and "midline_x" in landmarks:
        mid_x = int(landmarks["midline_x"])
        sag_img = sagittal_mip(ct_vol, mid_x, slab_half_voxels)
        ax.imshow(
            sag_img.T, cmap="gray", origin="lower", aspect=sz / sy, vmin=-150, vmax=400
        )

        # All landmarks
        if "promontory" in landmarks:
            ax.scatter(
                [landmarks["promontory"][1]],
                [landmarks["promontory"][2]],
                c="lime", s=250, marker="o", edgecolor="white",
                linewidth=3, zorder=10, label="Promontory",
            )
        if "coccygeal_apex" in landmarks:
            ax.scatter(
                [landmarks["coccygeal_apex"][1]],
                [landmarks["coccygeal_apex"][2]],
                c="orange", s=250, marker="o", edgecolor="white",
                linewidth=3, zorder=10, label="Coccygeal Apex",
            )
        if "symphysis_upper" in landmarks:
            ax.scatter(
                [landmarks["symphysis_upper"][1]],
                [landmarks["symphysis_upper"][2]],
                c="cyan", s=220, marker="^", edgecolor="white",
                linewidth=3, zorder=10, label="Upper Symph",
            )
        if "symphysis_lower" in landmarks:
            ax.scatter(
                [landmarks["symphysis_lower"][1]],
                [landmarks["symphysis_lower"][2]],
                c="deepskyblue", s=220, marker="v", edgecolor="white",
                linewidth=3, zorder=10, label="Lower Symph",
            )

        # Sacral Length
        if "promontory" in landmarks and "coccygeal_apex" in landmarks:
            ax.plot(
                [landmarks["promontory"][1], landmarks["coccygeal_apex"][1]],
                [landmarks["promontory"][2], landmarks["coccygeal_apex"][2]],
                c="magenta", linewidth=4, linestyle="--", zorder=8,
                label=f"Sacral L: {fmt_val('Sacral_Length_mm')}",
            )

        # Inlet AP
        if "promontory" in landmarks and "symphysis_upper" in landmarks:
            ax.plot(
                [landmarks["promontory"][1], landmarks["symphysis_upper"][1]],
                [landmarks["promontory"][2], landmarks["symphysis_upper"][2]],
                c="lime", linewidth=4, linestyle="-", zorder=8,
                label=f"Inlet AP: {fmt_val('Inlet_AP_mm')}",
            )

        # Outlet AP
        if "coccygeal_apex" in landmarks and "symphysis_lower" in landmarks:
            ax.plot(
                [landmarks["coccygeal_apex"][1], landmarks["symphysis_lower"][1]],
                [landmarks["coccygeal_apex"][2], landmarks["symphysis_lower"][2]],
                c="orange", linewidth=4, linestyle="-", zorder=8,
                label=f"Outlet AP: {fmt_val('Outlet_AP_mm')}",
            )

        # Sacral Depth
        if (
            "Sacral_Max_Depth_Pt" in landmarks
            and "promontory" in landmarks
            and "coccygeal_apex" in landmarks
        ):
            d_pt = landmarks["Sacral_Max_Depth_Pt"]
            prom = landmarks["promontory"]
            apex_lm = landmarks["coccygeal_apex"]

            p1_phys = np.array([float(prom[1]) * sy, float(prom[2]) * sz])
            p2_phys = np.array([float(apex_lm[1]) * sy, float(apex_lm[2]) * sz])
            p3_phys = np.array([float(d_pt[1]) * sy, float(d_pt[2]) * sz])

            v_phys = p2_phys - p1_phys
            w_phys = p3_phys - p1_phys
            t = np.dot(w_phys, v_phys) / np.dot(v_phys, v_phys)
            proj_pt_phys = p1_phys + t * v_phys

            p3_vox = np.array([float(d_pt[1]), float(d_pt[2])])
            proj_pt_vox = np.array([proj_pt_phys[0] / sy, proj_pt_phys[1] / sz])

            ax.plot(
                [p3_vox[0], proj_pt_vox[0]],
                [p3_vox[1], proj_pt_vox[1]],
                c="cyan", linewidth=3, linestyle="--", zorder=9,
                label=f"Sacral Depth: {fmt_val('Sacral_Depth_mm')}",
            )
            ax.scatter(
                [d_pt[1]], [d_pt[2]],
                c="cyan", s=120, marker="o", edgecolor="white",
                linewidth=2, zorder=10,
            )

        ax.legend(
            loc="upper right", fontsize=10, framealpha=0.95,
            edgecolor="white", fancybox=True,
        )

    ax.set_title(
        "Sagittal MIP: Sacral + Inlet + Outlet Combined",
        fontsize=14, fontweight="bold",
    )
    ax.axis("off")

    try:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"      🖼️ Sagittal Combined QC: {os.path.basename(out_path)}")
    except Exception as e:
        print(f"      ❌ Sagittal Combined QC save failed: {e}")
    finally:
        plt.close(fig)


# ------------------------------------------------------------------
# Pelvimetry QC: Extended (3-panel)
# ------------------------------------------------------------------

def save_extended_qc_figure(
    out_path, patient_id, ct_vol, result, landmarks, sx, sy, _sz
):
    """Save extended pelvimetry QC figure (3-panel layout).

    Panel 1 — Outlet Transverse (ITD), Panel 2 — ISD axial,
    Panel 3 — Summary table.

    Parameters
    ----------
    out_path : str
        Destination file path for the PNG.
    patient_id : str
        Patient identifier (used in figure title).
    ct_vol : numpy.ndarray
        3-D CT volume in RAS+ orientation.
    result : dict
        Measurement result dictionary.
    landmarks : dict
        Detected landmark coordinates.
    sx, sy : float
        Voxel spacings in X and Y (mm).
    _sz : float
        Voxel spacing in Z (unused; kept for API consistency).
    """
    fig = plt.figure(figsize=(18, 6), dpi=150)
    gs = gridspec.GridSpec(1, 3, wspace=0.25)
    fig.suptitle(
        f"Extended Pelvimetry QC: {patient_id}", fontsize=16, fontweight="bold", y=0.98
    )

    def safe_get(key, default=np.nan):
        return result.get(key, default) if result.get(key) is not None else default

    def fmt_val(key, unit="mm"):
        """Format a metric value for display, returning 'N/A' when missing."""
        v = result.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v:.1f}{unit}"

    def fmt_tbl(key):
        """Format for summary table column (right-aligned, 16-char wide)."""
        v = result.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return f"{'N/A':>16}"
        return f"{v:>16.1f}"

    def _sagittal_mip(ct, mid_x, half_width=3):
        x_lo = max(0, int(mid_x) - half_width)
        x_hi = min(ct.shape[0], int(mid_x) + half_width + 1)
        return np.max(ct[x_lo:x_hi, :, :], axis=0)

    # Panel (a) - Outlet Transverse (ITD)
    ax1 = fig.add_subplot(gs[0, 0])
    itd_mm = result.get("Outlet_Transverse_mm")
    pt_L = landmarks.get("itd_pt_left")
    pt_R = landmarks.get("itd_pt_right")
    itd_z = landmarks.get("_itd_z")

    if ct_vol is not None and itd_z is not None:
        z_slice = int(itd_z)
        ax_slice = ct_vol[:, :, z_slice].T
        ax1.imshow(
            ax_slice, cmap="gray", origin="lower", aspect=sx / sy, vmin=-150, vmax=400
        )

    if itd_mm is not None and pt_L is not None and pt_R is not None:
        ax1.scatter([pt_L[0]], [pt_L[1]], c="red", s=150, marker="o",
                    edgecolor="white", linewidth=2, zorder=10, label="Left Tuberosity")
        ax1.scatter([pt_R[0]], [pt_R[1]], c="blue", s=150, marker="o",
                    edgecolor="white", linewidth=2, zorder=10, label="Right Tuberosity")
        ax1.plot([pt_L[0], pt_R[0]], [pt_L[1], pt_R[1]], c="lime",
                 linewidth=4, linestyle="-", alpha=0.9, zorder=9, label=f"ITD: {itd_mm:.1f}mm")
        midline_x = ct_vol.shape[0] / 2 if ct_vol is not None else (pt_L[0] + pt_R[0]) / 2
        ax1.axvline(x=midline_x, color="yellow", linewidth=0.8, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax1.set_title(
        f"(a) Outlet Transverse: {fmt_val('Outlet_Transverse_mm')}",
        fontsize=12, fontweight="bold",
    )
    ax1.axis("off")

    # Panel (b) - ISD (Axial slice at ISD level)
    ax2 = fig.add_subplot(gs[0, 1])
    isd_mm = result.get("ISD_mm")
    isd_L = landmarks.get("ISD_L")
    isd_R = landmarks.get("ISD_R")
    isd_slice = landmarks.get("ISD_slice", result.get("ISD_slice"))

    if ct_vol is not None and isd_slice is not None:
        z_isd = int(isd_slice)
        if 0 <= z_isd < ct_vol.shape[2]:
            axial_slice = ct_vol[:, :, z_isd].T
            ax2.imshow(
                axial_slice, cmap="gray", origin="lower", aspect=sy / sx,
                vmin=-150, vmax=400,
            )

    if isd_mm is not None and isd_L is not None and isd_R is not None:
        ax2.scatter([isd_L[0]], [isd_L[1]], c="red", s=150, marker="o",
                    edgecolor="white", linewidth=2, zorder=10, label="Left Spine")
        ax2.scatter([isd_R[0]], [isd_R[1]], c="blue", s=150, marker="o",
                    edgecolor="white", linewidth=2, zorder=10, label="Right Spine")
        ax2.plot([isd_L[0], isd_R[0]], [isd_L[1], isd_R[1]], c="lime",
                 linewidth=4, linestyle="-", zorder=9, label=f"ISD: {isd_mm:.1f}mm")
        midline_x = result.get("midline_x", ct_vol.shape[0] / 2 if ct_vol is not None else None)
        if midline_x is not None:
            ax2.axvline(x=midline_x, color="yellow", linewidth=1, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax2.set_title(
        f"(b) ISD: {fmt_val('ISD_mm')} (Axial @ Z={isd_slice})",
        fontsize=12, fontweight="bold",
    )
    ax2.axis("off")

    # Panel (c) - Summary Table
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")

    lines = [
        "━" * 50,
        f"{'PELVIMETRY MEASUREMENTS':<50}",
        "━" * 50,
        "",
        f"{'Mid-Pelvis':<50}",
        f"  {'ISD (mm)':<30} {fmt_tbl('ISD_mm')}",
        "",
        f"{'Sacrum':<50}",
        f"  {'Sacral Length (mm)':<30} {fmt_tbl('Sacral_Length_mm')}",
        f"  {'Sacral Depth (mm)':<30} {fmt_tbl('Sacral_Depth_mm')}",
        "",
        f"{'Inlet':<50}",
        f"  {'Inlet AP (mm)':<30} {fmt_tbl('Inlet_AP_mm')}",
        "",
        f"{'Outlet':<50}",
        f"  {'Outlet AP (mm)':<30} {fmt_tbl('Outlet_AP_mm')}",
        f"  {'Outlet Transverse (mm)':<30} {fmt_tbl('Outlet_Transverse_mm')}",
        f"  {'Outlet Area (cm²)':<30} {fmt_tbl('Outlet_Area_cm2')}",
        "",
        "━" * 50,
        f"Status: {result.get('Status', 'Unknown')}",
    ]
    ax3.text(
        0.05, 0.98, "\n".join(lines),
        transform=ax3.transAxes, fontsize=10, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray",
                  alpha=0.95, linewidth=2),
    )
    ax3.set_title("(c) Summary", fontsize=12, fontweight="bold")

    try:
        fig.subplots_adjust(top=0.94, hspace=0.3, wspace=0.25)
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"      🖼️ Extended QC: {os.path.basename(out_path)}")
    except Exception as e:
        print(f"      ❌ Extended QC save failed: {e}")
    finally:
        plt.close(fig)


# ------------------------------------------------------------------
# Body Composition QC
# ------------------------------------------------------------------

def generate_body_composition_qc(
    ct_vol,
    adipose_vol,
    muscle_vol,
    l3_slice,
    isd_slice,
    result_dict,
    patient_id,
    output_path
):
    """Generate body composition QC figure.

    Produces a 2×3 panel grid showing CT, adipose overlay, and
    muscle overlay at both L3 and ISD levels.

    Parameters
    ----------
    ct_vol : numpy.ndarray
        3-D CT volume.
    adipose_vol : numpy.ndarray
        3-D adipose tissue volume.
    muscle_vol : numpy.ndarray
        3-D skeletal muscle volume.
    l3_slice : int or None
        Z-axis slice index for L3.
    isd_slice : int or None
        Z-axis slice index for ISD.
    result_dict : dict
        Measurement results (used for panel text).
    patient_id : str
        Patient identifier (used in figure title).
    output_path : str
        Destination file path for the PNG.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Body Composition QC: {patient_id}", fontsize=16, fontweight='bold')

    levels = [
        ("L3", l3_slice, 0),
        ("ISD", isd_slice, 1)
    ]

    for level_name, slice_idx, row in levels:
        if slice_idx is None or slice_idx < 0 or slice_idx >= ct_vol.shape[2]:
            for col in range(3):
                axes[row, col].text(
                    0.5, 0.5, f"{level_name} slice not available",
                    ha='center', va='center',
                    transform=axes[row, col].transAxes,
                )
                axes[row, col].axis('off')
            continue

        ct_slice = ct_vol[:, :, slice_idx]
        adipose_slice = adipose_vol[:, :, slice_idx] if adipose_vol is not None else np.zeros_like(ct_slice)
        muscle_slice = muscle_vol[:, :, slice_idx] if muscle_vol is not None else np.zeros_like(ct_slice)

        # Column 0: CT image
        axes[row, 0].imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-200, vmax=200)
        axes[row, 0].set_title(f"{level_name} - CT (slice {slice_idx})")
        axes[row, 0].axis('off')

        # Column 1: CT + Adipose overlay
        axes[row, 1].imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-200, vmax=200)
        if np.sum(adipose_slice) > 0:
            axes[row, 1].contour(adipose_slice.T, levels=[0.5], colors='yellow', linewidths=1.5)
        vat = result_dict.get(f"{level_name}_VAT_cm2", "N/A")
        sat = result_dict.get(f"{level_name}_SAT_cm2", "N/A")
        axes[row, 1].set_title(f"{level_name} - Adipose\nVAT={vat} SAT={sat} cm²")
        axes[row, 1].axis('off')

        # Column 2: CT + Muscle overlay
        axes[row, 2].imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-200, vmax=200)
        if np.sum(muscle_slice) > 0:
            axes[row, 2].contour(muscle_slice.T, levels=[0.5], colors='red', linewidths=1.5)
        sma = result_dict.get(f"{level_name}_SMA_cm2", "N/A")
        axes[row, 2].set_title(f"{level_name} - Muscle\nSMA={sma} cm²")
        axes[row, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      🖼️ Body Composition QC: {os.path.basename(output_path)}")
