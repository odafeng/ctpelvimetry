# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.4.1] - 2026-03-10

### Fixed
- `run_body_composition_batch` crashed with `KeyError: 'Status'` when zero patients were found (empty DataFrame had no columns)

## [1.4.0] - 2026-03-10

### Added
- **`config` parameter** on `run_combined_pelvimetry` and `run_full_pipeline` — inject a custom `PelvicConfig` to tune detection thresholds without monkey-patching
- **`PelvicConfig`** exported as public API from the top-level package
- **`__all__`** defined in `__init__.py` — `from ctpelvimetry import *` now exposes only the intended public API
- **`__main__.py`** — `python -m ctpelvimetry` now works as expected
- **Batch zero-patient warning** — `run_pelvimetry_batch` and `run_body_composition_batch` emit `warnings.warn()` when no patient directories are found

### Fixed
- **Return dict schema consistency** — `run_combined_pelvimetry` now always returns the same set of keys regardless of failure mode (missing values filled with `None`), preventing `KeyError` / unexpected `NaN` when collecting batch results into a DataFrame
- **`load_mask(None)` crash** — all I/O load functions (`load_mask`, `load_mask_canonical`, `load_nifti_canonical`) now accept `None` as path and return `(None, ...)` gracefully instead of raising `TypeError`
- **QC "nanmm" display** — NaN metric values now render as "N/A" in QC legend labels, panel titles, and the summary table (previously showed `nanmm`, `nan mm`, etc.)
- **Integer `patient_id`** — automatically coerced to `str` in pipeline functions to prevent downstream type issues

### Changed
- Added type annotations to all three public API functions (`run_combined_pelvimetry`, `run_full_pipeline`, `process_single_patient`)

## [1.3.1] - 2026-03-01

### Fixed
- README QC example images now render correctly on PyPI (switched from relative paths to absolute GitHub raw URLs)

## [1.3.0] - 2026-03-01

### Removed
- **APD (Antero-Posterior Diameter)** metric removed entirely — sacrum mask frequently did not extend to the ISD Z-level, causing widespread `APD_NO_SACRUM` failures
- `calculate_isd_apd` renamed to `calculate_isd`
- APD removed from QC summary table, CSV output columns, CLI help, and batch failure summary

### Breaking Changes
- Output CSV no longer contains `APD_mm` column
- Total metric count changed from 7 to 6; status codes now use `/6` (e.g. `Partial_5/6`)
- `calculate_isd_apd()` renamed to `calculate_isd()` — update any direct imports

## [1.2.0] - 2026-02-28

### Changed
- **PCA symphysis endpoints** (V6.3): `find_symphysis_midline_sagittal` now uses PCA long-axis endpoint detection instead of pure Z extremes, handling tilted pelves
- PCA is computed in mm space (Y×sy, Z×sz) via numpy SVD — no new dependencies required
- Added `sz` parameter to `find_symphysis_midline_sagittal`
- Falls back to Z extremes if anterior cluster has ≤2 voxels

### Breaking Changes
- `find_symphysis_midline_sagittal` has a new `sz` keyword argument (default 1.0, backward compatible)
- Upper/lower symphysis coordinates may differ from v1.1.x for tilted pelves

## [1.1.1] - 2026-02-28

### Changed
- **S1 mask integration**: `find_sacral_landmarks` now accepts a merged `sacrum_s1` mask (sacrum + vertebrae_S1) for accurate promontory detection at the S1 anterior-superior border
- **Coccygeal apex naming**: Renamed `apex` → `coccygeal_apex` throughout (landmark keys, output CSV columns, QC labels) to correctly reflect TotalSegmentator anatomy
- **Sacral depth contour**: `calculate_sacral_depth` now uses the merged sacrum+S1 mask for full anterior contour coverage
- Added `Seg_Vertebrae_S1` to output file path tracking

### Breaking Changes
- Output CSV columns renamed: `Sacral_Apex_x/y/z` → `Coccygeal_Apex_x/y/z`
- `find_sacral_landmarks` signature changed: first parameter is now `sacrum_s1` (merged mask), with optional `sacrum` parameter for apex detection
- Landmark dict key `"apex"` → `"coccygeal_apex"`

## [1.0.0] - 2026-02-15

### Added
- Initial public release
- Automated CT pelvimetry with 6 measurements (ISD, ITD, Inlet AP, Outlet AP, Sacral Length, Sacral Depth)
- Body composition analysis (VAT, SAT, SMA) at L3 and ISD levels
- Per-metric error isolation for robust batch processing
- Pelvic orientation quality gates (axial rotation, coronal tilt)
- Three QC figure types: Sagittal Combined, Extended 3-panel, Body Composition
- CLI with `pelv` and `body-comp` subcommands
- Optional TotalSegmentator integration (`pip install ".[seg]"`)
- Unit tests for all 10 modules (60 tests)
- GitHub Actions CI for Python 3.10–3.13
