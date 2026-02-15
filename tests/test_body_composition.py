"""Tests for ctpelvimetry.body_composition."""


import numpy as np
import nibabel as nib
import pandas as pd
import pytest

from ctpelvimetry.body_composition import (
    separate_vat_sat,
    create_body_cavity_mask,
    calculate_area_cm2,
    analyze_slice_composition_direct,
    find_isd_slice,
)


# ------------------------------------------------------------------
# separate_vat_sat
# ------------------------------------------------------------------

class TestSeparateVatSat:

    def test_disjoint_masks(self):
        """Fat inside cavity = VAT, outside = SAT."""
        adipose = np.zeros((32, 32), dtype=bool)
        cavity = np.zeros((32, 32), dtype=bool)
        # Fat inside cavity
        adipose[10:15, 10:15] = True
        cavity[8:18, 8:18] = True
        # Fat outside cavity
        adipose[25:30, 25:30] = True

        vat, sat = separate_vat_sat(adipose, cavity)
        assert np.sum(vat) > 0
        assert np.sum(sat) > 0
        # VAT should be subset of cavity
        assert np.all(vat <= cavity)
        # SAT should not overlap cavity
        assert np.sum(sat & cavity) == 0

    def test_no_fat(self):
        adipose = np.zeros((16, 16), dtype=bool)
        cavity = np.ones((16, 16), dtype=bool)
        vat, sat = separate_vat_sat(adipose, cavity)
        assert np.sum(vat) == 0
        assert np.sum(sat) == 0


# ------------------------------------------------------------------
# create_body_cavity_mask
# ------------------------------------------------------------------

class TestCreateBodyCavityMask:

    def test_ring_muscle_fills_interior(self):
        """A ring-shaped muscle mask should yield a filled interior."""
        muscle = np.zeros((32, 32), dtype=np.uint8)
        # Create ring
        yy, xx = np.ogrid[:32, :32]
        r = np.sqrt((xx - 16)**2 + (yy - 16)**2)
        muscle[(r >= 10) & (r <= 12)] = 1

        cavity = create_body_cavity_mask(muscle)
        assert cavity is not None
        # Interior should be filled
        assert cavity[16, 16] == 1

    def test_empty_muscle(self):
        muscle = np.zeros((16, 16), dtype=np.uint8)
        cavity = create_body_cavity_mask(muscle)
        assert cavity is not None


# ------------------------------------------------------------------
# calculate_area_cm2
# ------------------------------------------------------------------

class TestCalculateAreaCm2:

    def test_known_area(self):
        """10x10 voxels at 2mm spacing = 4 cm²."""
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5:15, 5:15] = 1  # 100 voxels
        area = calculate_area_cm2(mask, pixel_spacing=(2.0, 2.0))
        # 100 * 2 * 2 = 400 mm² = 4 cm²
        assert pytest.approx(area, rel=1e-6) == 4.0

    def test_empty_mask(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        area = calculate_area_cm2(mask, pixel_spacing=(1.0, 1.0))
        assert area == 0.0


# ------------------------------------------------------------------
# analyze_slice_composition_direct
# ------------------------------------------------------------------

class TestAnalyzeSliceCompositionDirect:

    def _make_header(self):
        """Create a valid 3-D NIfTI header with 1 mm spacing."""
        img = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
        hdr = img.header.copy()
        hdr.set_zooms((1.0, 1.0, 1.0))
        return hdr

    def test_returns_expected_keys(self):
        header = self._make_header()
        shape = (32, 32, 20)
        vat = np.zeros(shape, dtype=np.uint8)
        sat = np.zeros(shape, dtype=np.uint8)
        muscle = np.zeros(shape, dtype=np.uint8)
        vat[10:15, 10:15, 10] = 1
        sat[20:25, 20:25, 10] = 1
        muscle[12:18, 12:18, 10] = 1

        result = analyze_slice_composition_direct(
            10, vat, sat, muscle, header, level_name="Test"
        )
        assert "Test_VAT_cm2" in result
        assert "Test_SAT_cm2" in result
        assert "Test_SMA_cm2" in result

    def test_none_slice_returns_nan(self):
        header = self._make_header()
        result = analyze_slice_composition_direct(
            None, None, None, None, header, level_name="X"
        )
        assert "X_slice" in result
        assert result["X_slice"] is None


# ------------------------------------------------------------------
# find_isd_slice
# ------------------------------------------------------------------

class TestFindIsdSlice:

    def test_valid_csv(self, tmp_path):
        csv_path = tmp_path / "report.csv"
        df = pd.DataFrame({
            "Patient_ID": ["Patient_001"],
            "ISD_slice": [42],
        })
        df.to_csv(csv_path, index=False)

        result = find_isd_slice(str(csv_path), "Patient_001", np.eye(4))
        assert result == 42

    def test_missing_csv(self):
        result = find_isd_slice("/nonexistent.csv", "Patient_001", np.eye(4))
        assert result is None

    def test_missing_patient(self, tmp_path):
        csv_path = tmp_path / "report.csv"
        df = pd.DataFrame({
            "Patient_ID": ["Patient_999"],
            "ISD_slice": [50],
        })
        df.to_csv(csv_path, index=False)

        result = find_isd_slice(str(csv_path), "Patient_001", np.eye(4))
        assert result is None
