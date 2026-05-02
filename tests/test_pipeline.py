"""Tests for ctpelvimetry.pipeline."""

from unittest.mock import patch


from ctpelvimetry.pipeline import run_combined_pelvimetry, run_nifti_pipeline


class TestRunCombinedPelvimetry:

    @patch("ctpelvimetry.pipeline.load_mask_canonical")
    def test_missing_masks_returns_failure(self, mock_load):
        """Should return Failure status when no masks can be loaded."""
        mock_load.return_value = (None, None, None)

        result = run_combined_pelvimetry(
            "Test_001", "/fake/seg", "/fake/ct.nii.gz"
        )
        assert result["Patient_ID"] == "Test_001"
        assert "Status" in result
        assert "Error_Log" in result

    @patch("ctpelvimetry.pipeline.load_mask_canonical")
    def test_result_keys(self, mock_load):
        """Result should always contain Patient_ID, Status, Error_Log."""
        mock_load.return_value = (None, None, None)

        result = run_combined_pelvimetry("P001", "/seg", "/ct.nii.gz")
        assert "Patient_ID" in result
        assert "Status" in result
        assert "Error_Log" in result


class TestRunNiftiPipeline:

    def test_missing_nifti_file_returns_failure(self, tmp_path):
        """Non-existent NIfTI path should return Fail_NIfTI_Missing."""
        result = run_nifti_pipeline(
            "P001",
            str(tmp_path / "does_not_exist.nii.gz"),
            str(tmp_path / "out"),
        )
        assert result["Patient_ID"] == "P001"
        assert result["Status"] == "Fail_NIfTI_Missing"

    @patch("ctpelvimetry.pipeline.run_combined_pelvimetry")
    @patch("ctpelvimetry.pipeline.run_totalsegmentator")
    @patch("ctpelvimetry.pipeline.setup_license")
    def test_seg_failure_returns_fail_seg(
        self, mock_license, mock_seg, mock_combined, tmp_path
    ):
        """When TotalSegmentator returns None, status should be Fail_Seg."""
        nifti = tmp_path / "ct.nii.gz"
        nifti.write_bytes(b"")  # existence check only
        mock_seg.return_value = None

        result = run_nifti_pipeline("P001", str(nifti), str(tmp_path / "out"))

        assert result["Status"] == "Fail_Seg"
        mock_combined.assert_not_called()

    @patch("ctpelvimetry.pipeline.run_combined_pelvimetry")
    @patch("ctpelvimetry.pipeline.run_totalsegmentator")
    @patch("ctpelvimetry.pipeline.setup_license")
    def test_calls_combined_pelvimetry_with_correct_paths(
        self, mock_license, mock_seg, mock_combined, tmp_path
    ):
        """Successful seg should forward seg_folder + nifti_path to measurement."""
        nifti = tmp_path / "ct.nii.gz"
        nifti.write_bytes(b"")
        seg_folder = str(tmp_path / "out" / "Segmentation" / "P001")
        mock_seg.return_value = seg_folder
        mock_combined.return_value = {"Patient_ID": "P001", "Status": "Success"}

        run_nifti_pipeline(
            "P001", str(nifti), str(tmp_path / "out"), generate_qc=False
        )

        mock_combined.assert_called_once()
        call_args = mock_combined.call_args
        assert call_args[0][0] == "P001"
        assert call_args[0][1] == seg_folder
        assert call_args[0][2] == str(nifti)
        assert call_args[1]["qc_dir"] is None
