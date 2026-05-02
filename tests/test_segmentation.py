"""Tests for ctpelvimetry.segmentation."""

from unittest.mock import patch, MagicMock


from ctpelvimetry.segmentation import (
    LICENSE_ENV_VAR,
    setup_license,
    run_totalsegmentator,
)


class TestSetupLicense:

    @patch.dict("os.environ", {LICENSE_ENV_VAR: "aca_TESTKEY123"}, clear=False)
    @patch("ctpelvimetry.segmentation.subprocess.run")
    def test_success_with_env_var(self, mock_run):
        """When env var is set, subprocess.run should be called with the key."""
        mock_run.return_value = MagicMock(returncode=0)
        setup_license()
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        # Should be a list (no shell=True), and key must be passed via -l
        assert isinstance(call_args, list)
        assert "aca_TESTKEY123" in call_args
        assert mock_run.call_args[1].get("check") is True

    @patch.dict("os.environ", {LICENSE_ENV_VAR: ""}, clear=False)
    @patch("ctpelvimetry.segmentation.subprocess.run")
    def test_unset_env_var_skips(self, mock_run, capsys):
        """When env var is unset/empty, subprocess should NOT be called."""
        # Explicitly unset to handle the case where the variable exists in CI
        import os
        os.environ.pop(LICENSE_ENV_VAR, None)
        setup_license()
        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert LICENSE_ENV_VAR in captured.out

    @patch.dict("os.environ", {LICENSE_ENV_VAR: "aca_TESTKEY123"}, clear=False)
    @patch("ctpelvimetry.segmentation.subprocess.run", side_effect=Exception("fail"))
    def test_failure_prints_warning(self, mock_run, capsys):
        setup_license()
        captured = capsys.readouterr()
        assert "failed" in captured.out.lower() or "⚠" in captured.out

    @patch.dict("os.environ", {LICENSE_ENV_VAR: "aca_TESTKEY123"}, clear=False)
    @patch("ctpelvimetry.segmentation.subprocess.run", side_effect=FileNotFoundError())
    def test_command_not_found_prints_install_hint(self, mock_run, capsys):
        """If totalseg_set_license isn't installed, print a helpful hint."""
        setup_license()
        captured = capsys.readouterr()
        assert "totalseg_set_license" in captured.out


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
