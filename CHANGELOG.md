# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-02-15

### Added
- Initial public release
- Automated CT pelvimetry with 7 measurements (ISD, APD, ITD, Inlet AP, Outlet AP, Sacral Length, Sacral Depth)
- Body composition analysis (VAT, SAT, SMA) at L3 and ISD levels
- Per-metric error isolation for robust batch processing
- Pelvic orientation quality gates (axial rotation, coronal tilt)
- Three QC figure types: Sagittal Combined, Extended 3-panel, Body Composition
- CLI with `pelv` and `body-comp` subcommands
- Optional TotalSegmentator integration (`pip install ".[seg]"`)
- Unit tests for all 10 modules (60 tests)
- GitHub Actions CI for Python 3.10–3.13
