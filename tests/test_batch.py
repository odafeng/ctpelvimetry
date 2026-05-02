"""Tests for ctpelvimetry.batch."""

from unittest.mock import patch

import pandas as pd
import pytest

from ctpelvimetry.batch import run_pelvimetry_nifti_batch


class TestRunPelvimetryNiftiBatch:

    def test_empty_directory_warns(self, tmp_path):
        """No matching files should trigger a warning, not crash."""
        with pytest.warns(UserWarning, match="No files matching"):
            df = run_pelvimetry_nifti_batch(
                str(tmp_path),
                str(tmp_path / "out"),
                str(tmp_path / "out" / "results.csv"),
            )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @patch("ctpelvimetry.batch.run_nifti_pipeline")
    def test_patient_id_strips_nii_gz(self, mock_pipeline, tmp_path):
        """case_001.nii.gz should yield patient_id 'case_001'."""
        mock_pipeline.return_value = {"Patient_ID": "case_001", "Status": "Success"}

        nifti = tmp_path / "case_001.nii.gz"
        nifti.write_bytes(b"")

        run_pelvimetry_nifti_batch(
            str(tmp_path),
            str(tmp_path / "out"),
            str(tmp_path / "out" / "results.csv"),
        )

        assert mock_pipeline.call_count == 1
        call_args = mock_pipeline.call_args
        assert call_args[0][0] == "case_001"
        assert call_args[0][1] == str(nifti)

    @patch("ctpelvimetry.batch.run_nifti_pipeline")
    def test_patient_id_strips_nii_only(self, mock_pipeline, tmp_path):
        """case_001.nii (uncompressed) should also yield 'case_001'."""
        mock_pipeline.return_value = {"Patient_ID": "case_001", "Status": "Success"}

        nifti = tmp_path / "case_001.nii"
        nifti.write_bytes(b"")

        run_pelvimetry_nifti_batch(
            str(tmp_path),
            str(tmp_path / "out"),
            str(tmp_path / "out" / "results.csv"),
            pattern="*.nii",
        )

        assert mock_pipeline.call_count == 1
        assert mock_pipeline.call_args[0][0] == "case_001"

    @patch("ctpelvimetry.batch.run_nifti_pipeline")
    def test_processes_all_matching_files(self, mock_pipeline, tmp_path):
        """All matching files should be processed; non-matching ignored."""
        mock_pipeline.side_effect = lambda pid, *a, **kw: {
            "Patient_ID": pid, "Status": "Success",
        }

        # Create 3 matching + 1 non-matching
        for name in ["case_001.nii.gz", "case_002.nii.gz", "case_003.nii.gz", "readme.txt"]:
            (tmp_path / name).write_bytes(b"")

        df = run_pelvimetry_nifti_batch(
            str(tmp_path),
            str(tmp_path / "out"),
            str(tmp_path / "out" / "results.csv"),
        )

        assert mock_pipeline.call_count == 3
        assert len(df) == 3
        assert set(df["Patient_ID"]) == {"case_001", "case_002", "case_003"}

    @patch("ctpelvimetry.batch.run_nifti_pipeline")
    def test_per_patient_error_isolation(self, mock_pipeline, tmp_path):
        """One failing patient should not abort the batch."""
        def side_effect(pid, *args, **kwargs):
            if pid == "case_002":
                raise RuntimeError("simulated segmentation failure")
            return {"Patient_ID": pid, "Status": "Success"}

        mock_pipeline.side_effect = side_effect

        for name in ["case_001.nii.gz", "case_002.nii.gz", "case_003.nii.gz"]:
            (tmp_path / name).write_bytes(b"")

        df = run_pelvimetry_nifti_batch(
            str(tmp_path),
            str(tmp_path / "out"),
            str(tmp_path / "out" / "results.csv"),
        )

        assert len(df) == 3
        statuses = dict(zip(df["Patient_ID"], df["Status"]))
        assert statuses["case_001"] == "Success"
        assert statuses["case_002"] == "Error"
        assert statuses["case_003"] == "Success"

    @patch("ctpelvimetry.batch.run_nifti_pipeline")
    def test_csv_written_to_output_path(self, mock_pipeline, tmp_path):
        """Results CSV should land at the requested path."""
        mock_pipeline.return_value = {"Patient_ID": "case_001", "Status": "Success"}
        (tmp_path / "case_001.nii.gz").write_bytes(b"")

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        csv_path = out_dir / "results.csv"

        run_pelvimetry_nifti_batch(str(tmp_path), str(out_dir), str(csv_path))

        assert csv_path.exists()
        loaded = pd.read_csv(csv_path)
        assert "Patient_ID" in loaded.columns
        assert loaded.iloc[0]["Patient_ID"] == "case_001"
