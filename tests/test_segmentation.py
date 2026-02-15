"""Tests for ctpelvimetry.segmentation."""

import os
from unittest.mock import patch, MagicMock

import pytest

from ctpelvimetry.segmentation import setup_license, run_totalsegmentator


class TestSetupLicense:

    @patch("ctpelvimetry.segmentation.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        setup_license()  # should not raise
        mock_run.assert_called_once()

    @patch("ctpelvimetry.segmentation.subprocess.run", side_effect=Exception("fail"))
    def test_failure_prints_warning(self, mock_run, capsys):
        setup_license()
        captured = capsys.readouterr()
        assert "failed" in captured.out.lower() or "⚠" in captured.out


class TestRunTotalSegmentator:

    def test_missing_input_returns_none(self, tmp_path):
        result = run_totalsegmentator(
            "Patient_001", str(tmp_path / "nonexistent.nii.gz"), str(tmp_path)
        )
        assert result is None

    def test_skip_existing_complete(self, tmp_path):
        """Should skip if all ROI files already exist."""
        seg_dir = tmp_path / "Patient_001"
        seg_dir.mkdir(parents=True)

        # Create all expected ROI files
        rois = [
            "femur_left", "femur_right", "hip_left", "hip_right",
            "sacrum", "colon", "vertebrae_L5", "vertebrae_L4",
            "vertebrae_L3", "vertebrae_S1",
            "torso_fat", "subcutaneous_fat", "skeletal_muscle",
        ]
        for roi in rois:
            (seg_dir / f"{roi}.nii.gz").write_bytes(b"data")

        # Create a fake input NIfTI
        nifti_path = tmp_path / "ct.nii.gz"
        nifti_path.write_bytes(b"fake")

        result = run_totalsegmentator("Patient_001", str(nifti_path), str(tmp_path))
        assert result == str(seg_dir)
