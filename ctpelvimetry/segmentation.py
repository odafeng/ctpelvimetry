"""
TotalSegmentator segmentation: license setup and mask generation.
"""

import os
import subprocess
import time

from .config import SUBPROCESS_TIMEOUT_SECONDS


# ------------------------------------------------------------------
# License Setup
# ------------------------------------------------------------------

LICENSE_ENV_VAR = "TOTALSEG_LICENSE_KEY"
LICENSE_REGISTRATION_URL = "https://backend.totalsegmentator.com/license-academic/"


def setup_license():
    """Configure the TotalSegmentator academic licence from environment.

    Reads the license key from the ``TOTALSEG_LICENSE_KEY`` environment
    variable. If the variable is unset, license setup is skipped with an
    informative message; bones/organs segmentation will still work, but
    the ``tissue_types`` task (VAT, SAT, skeletal muscle) will be
    unavailable.

    To obtain a key, register at
    https://backend.totalsegmentator.com/license-academic/, then set:

    .. code-block:: bash

        export TOTALSEG_LICENSE_KEY=aca_xxxxxxxxxxxx

    Notes
    -----
    Earlier versions of this package shipped with a hard-coded academic
    license key. That key has been removed for security and licence-
    compliance reasons (academic keys are issued per-user and are not
    redistributable).
    """
    license_key = os.environ.get(LICENSE_ENV_VAR)
    if not license_key:
        print(
            f"ℹ️  {LICENSE_ENV_VAR} not set — pelvimetry will run normally; "
            "body composition (VAT/SAT/muscle) will be skipped."
        )
        print("   To enable body composition:")
        print(f"     1. Register (free, academic) at {LICENSE_REGISTRATION_URL}")
        print(f"     2. export {LICENSE_ENV_VAR}=aca_xxxxxxxxxxxx")
        return

    print("🔑 Setting up TotalSegmentator license...")
    try:
        # Pass the key as a list arg (no shell=True) to avoid
        # shell-metacharacter injection if the env var is malformed.
        subprocess.run(
            ["totalseg_set_license", "-l", license_key],
            check=True,
            capture_output=True,
        )
        print("   ✅ License set successfully.")
    except FileNotFoundError:
        print("   ⚠️ 'totalseg_set_license' command not found")
        print("      (Install TotalSegmentator: pip install 'ctpelvimetry[seg]')")
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️ License setup failed (exit {e.returncode})")
        print("      (tissue_types task might fail without a valid license)")
    except Exception as e:
        print(f"   ⚠️ License setup failed: {e}")
        print("      (tissue_types task might fail without license)")


# ------------------------------------------------------------------
# TotalSegmentator Segmentation
# ------------------------------------------------------------------

def run_totalsegmentator(
    patient_id, nifti_path, seg_dir, use_fast=False, skip_tissue=False
):
    """Run TotalSegmentator to produce segmentation masks.

    Generates bone/organ ROIs (hip, sacrum, femur, vertebrae, colon)
    and, optionally, tissue-type masks (torso fat, subcutaneous fat,
    skeletal muscle).  Existing complete outputs are reused.

    Parameters
    ----------
    patient_id : str
        Patient identifier (used as subdirectory name).
    nifti_path : str
        Path to the input CT NIfTI file.
    seg_dir : str
        Root directory for segmentation outputs.  A subdirectory
        ``<patient_id>/`` is created under *seg_dir*.
    use_fast : bool, optional
        Use TotalSegmentator ``--fast`` mode (default ``False``).
    skip_tissue : bool, optional
        Skip the ``tissue_types`` task (default ``False``).

    Returns
    -------
    str or None
        Path to the output segmentation folder, or ``None`` on
        failure.
    """
    output_folder = os.path.join(seg_dir, patient_id)

    if not os.path.exists(nifti_path):
        print(f"   ❌ [Error] Input file not found: {nifti_path}")
        return None

    os.makedirs(output_folder, exist_ok=True)

    # Define required ROIs
    rois_main = [
        "femur_left",
        "femur_right",
        "hip_left",
        "hip_right",
        "sacrum",
        "colon",
        "vertebrae_L5",
        "vertebrae_L4",
        "vertebrae_L3",
        "vertebrae_S1",
    ]
    rois_tissue = ["torso_fat", "subcutaneous_fat", "skeletal_muscle"]
    all_target_files = [f"{roi}.nii.gz" for roi in rois_main + rois_tissue]

    # Check if complete segmentation already exists
    all_exist_and_valid = True
    for f_name in all_target_files:
        f_path = os.path.join(output_folder, f_name)
        if not os.path.exists(f_path) or os.path.getsize(f_path) == 0:
            all_exist_and_valid = False
            break

    if all_exist_and_valid:
        print("   ⏭️  [Seg] Complete segmentation files already exist")
        return output_folder

    mode_str = "🚀 fast mode" if use_fast else "🐢 standard mode"
    print(f"   🔄 [Seg] Segmenting: {patient_id} | {mode_str}")
    start_time = time.time()

    try:

        def run_command(cmd_list, task_name):
            print(f"      🔹 {task_name}...")
            cmd_str = " ".join(subprocess.list2cmdline([s]) for s in cmd_list)

            try:
                result = subprocess.run(
                    cmd_str,
                    capture_output=True,
                    text=True,
                    shell=True,
                    timeout=SUBPROCESS_TIMEOUT_SECONDS,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                print(f"      ⏰ [TIMEOUT] {task_name} TIMEOUT!")
                raise subprocess.CalledProcessError(
                    -1, cmd_list, output="TIMEOUT"
                ) from exc

            if result.returncode != 0:
                print(f"      ❌ {task_name} failed")
                raise subprocess.CalledProcessError(result.returncode, cmd_list)
            else:
                print(f"      ✅ {task_name} done")

        # Task 1: Default (Bones & Organs)
        cmd_default = [
            "TotalSegmentator",
            "-i",
            nifti_path,
            "-o",
            output_folder,
            "-rs",
        ] + rois_main
        if use_fast:
            cmd_default.append("--fast")
        run_command(cmd_default, "Bones & Organs")

        if not skip_tissue:
            if not os.environ.get("TOTALSEG_LICENSE_KEY"):
                # Fast-path: no licence set, skip tissue_types cleanly without
                # invoking the subprocess and surfacing a noisy error.
                print(
                    "      ⏭️  Tissue Types skipped "
                    "(TOTALSEG_LICENSE_KEY not set; pelvimetry only)"
                )
            else:
                # Task 2: Tissue Types (optional, requires license)
                try:
                    cmd_tissue = [
                        "TotalSegmentator",
                        "-i",
                        nifti_path,
                        "-o",
                        output_folder,
                        "-ta",
                        "tissue_types",
                    ]
                    run_command(cmd_tissue, "Tissue Types")
                    # Check if output files were actually created
                    tissue_files = [
                        "torso_fat.nii.gz",
                        "subcutaneous_fat.nii.gz",
                        "skeletal_muscle.nii.gz",
                    ]
                    tissue_ok = any(
                        os.path.exists(os.path.join(output_folder, f)) for f in tissue_files
                    )
                    if not tissue_ok:
                        print(
                            "      ⚠️ Tissue Types: Requires license "
                            "(https://backend.totalsegmentator.com/license-academic/)"
                        )
                except Exception:
                    print("      ⚠️ Tissue Types skipped (Optional, requires additional license)")
        else:
            print("      ⏭️  Tissue Types skipped (User requested skip)")

        # Verify output
        missing_files = [
            f
            for f in all_target_files
            if not os.path.exists(os.path.join(output_folder, f))
        ]
        if missing_files:
            critical_missing = [f for f in missing_files if "hip" in f or "sacrum" in f]
            if critical_missing:
                print(f"   ❌ Missing critical files: {critical_missing}")
                return None
            else:
                print(f"   ⚠️ Some files not generated: {missing_files}")

        elapsed = time.time() - start_time
        print(f"   ✅ [Seg] Segmentation complete (elapsed: {elapsed:.1f} s)")
        return output_folder

    except subprocess.CalledProcessError:
        print("   💀 [Seg] Processing failed")
        return None
    except Exception as e:
        print(f"   ❌ [Seg] Unknown error: {e}")
        return None
