"""Tests for ctpelvimetry.qc."""

import os

import numpy as np
import pytest

from ctpelvimetry.qc import (
    save_sagittal_combined_qc_figure,
    save_extended_qc_figure,
    generate_body_composition_qc,
)


# ------------------------------------------------------------------
# save_sagittal_combined_qc_figure
# ------------------------------------------------------------------

class TestSaveSagittalCombinedQC:

    def test_creates_file(self, tmp_path):
        """QC figure should be created without errors."""
        ct_vol = np.random.rand(64, 64, 64).astype(np.float32) * 1000
        result = {
            "Inlet_AP_mm": 110.0,
            "Outlet_AP_mm": 90.0,
            "Sacral_Length_mm": 100.0,
            "Sacral_Depth_mm": 15.0,
            "ISD_mm": 100.0,
            "APD_mm": 120.0,
            "ITD_mm": 110.0,
        }
        landmarks = {
            "promontory": (32, 45, 50),
            "apex": (32, 45, 20),
            "upper_symphysis": (32, 15, 45),
            "lower_symphysis": (32, 15, 25),
            "midline_x": 32,
        }
        out_path = str(tmp_path / "sagittal_qc.png")
        save_sagittal_combined_qc_figure(
            out_path, "Test_001", ct_vol, result, landmarks,
            slab_half_voxels=3, sy=1.0, sz=1.0
        )
        assert os.path.exists(out_path)


# ------------------------------------------------------------------
# save_extended_qc_figure
# ------------------------------------------------------------------

class TestSaveExtendedQC:

    def test_creates_file(self, tmp_path):
        ct_vol = np.random.rand(64, 64, 64).astype(np.float32) * 1000
        result = {
            "ISD_mm": 100.0,
            "APD_mm": 120.0,
            "ITD_mm": 110.0,
            "ISD_slice": 32,
            "ISD_pt_left": (20, 30, 32),
            "ISD_pt_right": (44, 30, 32),
            "ITD_pt_left_voxel": (20, 50, 25),
            "ITD_pt_right_voxel": (44, 50, 25),
            "ITD_z_slice": 25,
            "Inlet_AP_mm": 110.0,
            "Outlet_AP_mm": 90.0,
            "Sacral_Length_mm": 100.0,
            "Sacral_Depth_mm": 15.0,
        }
        landmarks = {"midline_x": 32}
        out_path = str(tmp_path / "extended_qc.png")
        save_extended_qc_figure(
            out_path, "Test_001", ct_vol, result, landmarks,
            sx=1.0, sy=1.0, _sz=1.0
        )
        assert os.path.exists(out_path)


# ------------------------------------------------------------------
# generate_body_composition_qc
# ------------------------------------------------------------------

class TestGenerateBodyCompositionQC:

    def test_creates_file(self, tmp_path):
        shape = (64, 64, 64)
        ct_vol = np.random.rand(*shape).astype(np.float32) * 1000
        adipose_vol = np.zeros(shape, dtype=np.uint8)
        muscle_vol = np.zeros(shape, dtype=np.uint8)
        adipose_vol[10:20, 10:20, 30:35] = 1
        muscle_vol[15:25, 15:25, 30:35] = 1

        result_dict = {
            "L3_VAT_cm2": 5.0,
            "L3_SAT_cm2": 10.0,
            "L3_SMA_cm2": 8.0,
            "ISD_VAT_cm2": 4.0,
            "ISD_SAT_cm2": 9.0,
            "ISD_SMA_cm2": 7.0,
        }
        out_path = str(tmp_path / "body_comp_qc.png")
        generate_body_composition_qc(
            ct_vol, adipose_vol, muscle_vol,
            l3_slice=32, isd_slice=28,
            result_dict=result_dict,
            patient_id="Test_001",
            output_path=out_path,
        )
        assert os.path.exists(out_path)
