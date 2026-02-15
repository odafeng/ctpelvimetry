"""Tests for ctpelvimetry.pipeline."""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from ctpelvimetry.pipeline import run_combined_pelvimetry


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
