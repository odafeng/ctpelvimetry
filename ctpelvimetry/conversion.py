"""
DICOM → NIfTI conversion using dcm2niix.
"""

import os
import glob
import subprocess


def convert_dicom_to_nifti(patient_id, dicom_path, nifti_dir):
    """Convert a DICOM folder to compressed NIfTI using ``dcm2niix``.

    If the target NIfTI file already exists and is non-empty the
    conversion is skipped and the existing path is returned.

    Parameters
    ----------
    patient_id : str
        Patient identifier used as the output filename stem.
    dicom_path : str
        Path to the directory containing DICOM files.
    nifti_dir : str
        Output directory for the resulting ``.nii.gz`` file.

    Returns
    -------
    str or None
        Absolute path to the converted NIfTI file, or ``None`` on
        failure.
    """
    target_gz = os.path.join(nifti_dir, f"{patient_id}.nii.gz")
    target_nii = os.path.join(nifti_dir, f"{patient_id}.nii")

    # Check if already exists
    if os.path.exists(target_gz) and os.path.getsize(target_gz) > 0:
        print(f"   ⏭️  [NIfTI] File already exists: {os.path.basename(target_gz)}")
        return target_gz

    if os.path.exists(target_nii) and os.path.getsize(target_nii) > 0:
        print(f"   ⏭️  [NIfTI] Found uncompressed file: {os.path.basename(target_nii)}")
        return target_nii

    print("   🔄 [NIfTI] Converting...")
    os.makedirs(nifti_dir, exist_ok=True)
    temp_dir = os.path.join(nifti_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Run dcm2niix
    cmd = f"dcm2niix -z y -f {patient_id} -b n -o '{temp_dir}' '{dicom_path}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print(f"   ❌ [NIfTI] dcm2niix failed: {result.stderr}")
        return None

    # Find converted output files
    converted_files = glob.glob(os.path.join(temp_dir, f"{patient_id}*.nii*"))

    if converted_files:
        largest_file = max(converted_files, key=os.path.getsize)
        extension = ".nii.gz" if largest_file.endswith(".gz") else ".nii"
        final_output = os.path.join(nifti_dir, f"{patient_id}{extension}")

        if os.path.exists(final_output):
            os.remove(final_output)
        os.rename(largest_file, final_output)

        # Clean up temp files
        for f in glob.glob(os.path.join(temp_dir, "*")):
            try:
                os.remove(f)
            except OSError:
                pass

        print(f"   ✅ [NIfTI] Conversion complete: {os.path.basename(final_output)}")
        return final_output
    else:
        print("   ❌ [NIfTI] No output files found after conversion")
        return None
