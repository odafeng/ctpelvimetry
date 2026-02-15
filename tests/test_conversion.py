"""Tests for ctpelvimetry.conversion."""

import os
from unittest.mock import patch, MagicMock

import pytest

from ctpelvimetry.conversion import convert_dicom_to_nifti


class TestConvertDicomToNifti:

    def test_skip_existing_gz(self, tmp_path):
        """Should return existing .nii.gz without calling dcm2niix."""
        nifti_dir = tmp_path / "nifti"
        nifti_dir.mkdir()
        existing = nifti_dir / "Patient_001.nii.gz"
        existing.write_bytes(b"dummy")

        result = convert_dicom_to_nifti("Patient_001", str(tmp_path), str(nifti_dir))
        assert result == str(existing)

    def test_skip_existing_nii(self, tmp_path):
        """Should return existing .nii without calling dcm2niix."""
        nifti_dir = tmp_path / "nifti"
        nifti_dir.mkdir()
        existing = nifti_dir / "Patient_001.nii"
        existing.write_bytes(b"dummy")

        result = convert_dicom_to_nifti("Patient_001", str(tmp_path), str(nifti_dir))
        assert result == str(existing)

    @patch("ctpelvimetry.conversion.subprocess.run")
    def test_subprocess_failure(self, mock_run, tmp_path):
        """Should return None when dcm2niix fails."""
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        nifti_dir = tmp_path / "nifti"
        dicom_dir = tmp_path / "dicom"
        dicom_dir.mkdir()

        result = convert_dicom_to_nifti("Patient_001", str(dicom_dir), str(nifti_dir))
        assert result is None

    @patch("ctpelvimetry.conversion.subprocess.run")
    def test_subprocess_success(self, mock_run, tmp_path):
        """Should return path when dcm2niix succeeds."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        nifti_dir = tmp_path / "nifti"
        dicom_dir = tmp_path / "dicom"
        dicom_dir.mkdir()

        # Create fake converted file in temp dir
        temp_dir = nifti_dir / "temp"
        temp_dir.mkdir(parents=True)
        fake_output = temp_dir / "Patient_001.nii.gz"
        fake_output.write_bytes(b"fake nifti data")

        result = convert_dicom_to_nifti("Patient_001", str(dicom_dir), str(nifti_dir))
        assert result is not None
        assert "Patient_001" in result

    @patch("ctpelvimetry.conversion.subprocess.run")
    def test_no_output_files(self, mock_run, tmp_path):
        """Should return None when dcm2niix produces no output."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        nifti_dir = tmp_path / "nifti"
        dicom_dir = tmp_path / "dicom"
        dicom_dir.mkdir()
        # Don't create any output files in temp
        (nifti_dir / "temp").mkdir(parents=True)

        result = convert_dicom_to_nifti("Patient_001", str(dicom_dir), str(nifti_dir))
        assert result is None
