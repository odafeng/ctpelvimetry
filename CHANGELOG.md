# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.6.1] - 2026-05-02

### Documentation
- **Comprehensive README rewrite.** Restructured into clear sections (Quick Start → Installation → CLI Usage → Python API → Output Structure → Metrics → QC → Hardware → License → Architecture → Troubleshooting → Citation), each with copy-pasteable examples.
- **Complete CLI reference** — all 5 `pelv` input modes documented in a single overview table, plus the previously undocumented `body-comp` subcommand.
- **Complete Python API reference** — every public function shown with realistic usage including the new batch APIs (`run_pelvimetry_batch`, `run_pelvimetry_nifti_batch`).
- **Output Structure section** — directory tree of `--output_root`, full CSV column schema, and all possible `Status` values (`Success`, `Partial_N/6`, `Failure`, `Fail_NIfTI`, `Fail_Seg`, `Fail_NIfTI_Missing`, `Error`).
- **Troubleshooting section** with diagnostics for the most common failure modes.
- **Restored QC images** (`docs/images/qc_example.png`, `docs/images/qc_extended_example.png`) using GitHub raw URLs so they render on PyPI as well.
- Fixed citation: bumped to v1.6.1 / year 2026, fixed broken license link.

## [1.6.0] - 2026-05-02

NIfTI pipeline gets first-class batch processing. Drop a directory of `*.nii.gz` files in and process them all with a single command — no DICOM directory layout, no manual loops.

### Added
- **`run_pelvimetry_nifti_batch`** — new public API for batch pelvimetry on a directory of NIfTI files. Patient IDs are derived from filenames (e.g. `case_001.nii.gz` → `case_001`). Includes per-patient error isolation, progress bar, failure summary, and the same CSV schema as the DICOM batch.
- **CLI Mode 5** — `ctpelvimetry pelv --nifti_root /path/to/niftis --output_root ./output` for NIfTI batch processing
- **`--pattern`** flag — override the default `*.nii.gz` glob (e.g. `--pattern "*.nii"` for uncompressed NIfTI)

### Changed
- Refactored shared post-loop logic (CSV save, column ordering, failure summary, error code aggregation) out of `run_pelvimetry_batch` into a private `_save_and_summarize_pelvimetry_batch` helper. Both DICOM and NIfTI batch entry points now share the same output schema and reporting code.

## [1.5.0] - 2026-05-02

### Added
- **`run_nifti_pipeline`** — new public API entry point for NIfTI inputs (NIfTI → Seg → Pelvimetry), skipping the DICOM → NIfTI conversion step. Useful for public datasets distributed in NIfTI format (Medical Decathlon, TotalSegmentator benchmark, etc.)
- **CLI Mode 4** — `ctpelvimetry pelv --nifti_path <ct.nii.gz> --patient <id> --output_root <dir>` runs the full pipeline starting from a NIfTI file

### Security
- **Removed hard-coded TotalSegmentator academic license key** from `segmentation.py`. The bundled key was distributed publicly via PyPI, which (a) violated TotalSegmentator's per-user license terms and (b) exposed all users to a single point of revocation. License keys are now read from the `TOTALSEG_LICENSE_KEY` environment variable.

### Changed
- `pelv` subcommand help now documents all four input modes (DICOM single, NIfTI single, DICOM batch, existing seg)
- `setup_license()` now reads the license key from `TOTALSEG_LICENSE_KEY`. If unset, the function prints registration instructions and skips license setup; pelvimetry runs normally, only `tissue_types` (VAT/SAT/muscle) is unavailable.
- `run_totalsegmentator()` fast-paths the tissue_types task: if `TOTALSEG_LICENSE_KEY` is unset, the subprocess is skipped entirely (no error stack), keeping the no-license user experience clean.
- `subprocess.run` in `setup_license()` no longer uses `shell=True`; arguments are passed as a list to prevent shell-metacharacter injection from the env var.

### Migration
Users who were relying on the bundled license key for body composition must now register their own academic key at https://backend.totalsegmentator.com/license-academic/ and set `TOTALSEG_LICENSE_KEY`. Pelvimetry-only users are unaffected.

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
