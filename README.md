# 📐 ctpelvimetry

[![PyPI version](https://img.shields.io/pypi/v/ctpelvimetry.svg?color=blue)](https://pypi.org/project/ctpelvimetry/)
[![Python versions](https://img.shields.io/pypi/pyversions/ctpelvimetry.svg)](https://pypi.org/project/ctpelvimetry/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **The first open-source Python package for fully automated CT pelvimetry and body composition analysis.**

`ctpelvimetry` transforms the manual, time-consuming process of measuring pelvic dimensions and body composition into a **rapid, fully automated pipeline**. Built for surgical data science and preoperative risk assessment, it integrates seamlessly with pre-trained deep learning models (e.g., [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)) to extract crucial anatomical metrics directly from raw CT scans—eliminating inter-observer variability entirely.

---

## 🚀 Quick Start

Get up and running in seconds. 

### Installation
```bash
# Basic install (if you already have segmentations)
pip install ctpelvimetry

# Full install (includes TotalSegmentator for end-to-end automation)
pip install "ctpelvimetry[seg]"
````
### ⚠️ Hardware Requirements (Important!)

Because the full end-to-end pipeline relies on deep learning models (TotalSegmentator) for 3D image segmentation, **a dedicated GPU is strongly recommended**.

* **Recommended Setup (for `ctpelvimetry[seg]`)**:
  * **GPU**: NVIDIA GPU with at least **8GB VRAM** (16GB+ recommended for high-resolution CTs, e.g., NVIDIA T4, RTX 3060/4090, or A100).
  * **RAM**: 16GB+ System RAM.
  * **Time**: With a dedicated GPU, a single patient scan takes **< 2 minutes**.
* **Without a GPU (CPU-only)**:
  * The pipeline will still run, but segmentation may take **10 to 30+ minutes** per scan, and you risk running out of memory (OOM) on large DICOM series.
* **Pro Tip for Clinical Researchers**: If you don't have a local GPU workstation, you can easily run the full pipeline in a cloud environment like **Google Colab (using a free T4 GPU)**.

*(Note: If you are only running the basic `ctpelvimetry` on pre-existing segmentations, a standard CPU is perfectly fine.)*

### End-to-End Analysis (CLI)

Go from raw DICOM to structured measurements in one command:

```bash
ctpelvimetry pelv --dicom_dir /path/to/patient_scan --output_root ./output --qc
```

*That's it. The pipeline will handle DICOM-to-NIfTI conversion, segmentation, landmark detection, metric calculation, and QC figure generation automatically.*

### Starting from NIfTI

If your data is already in NIfTI format (e.g. public datasets like the Medical Decathlon or TotalSegmentator benchmark), skip the DICOM conversion step:

```bash
ctpelvimetry pelv --nifti_path /path/to/ct.nii.gz --patient Patient_001 --output_root ./output --qc
```

The pipeline will run TotalSegmentator on the NIfTI volume directly and proceed with measurement and QC.

-----

## 💡 Why `ctpelvimetry`?

Manual measurement of the mid-pelvic workspace and body composition is tedious and highly subjective.

  * **Clinical Problem**: Measuring Inter-Spinous Distance (ISD) or Visceral Adipose Tissue (VAT) manually takes \~15 minutes per scan and suffers from significant inter-observer variability.
  * **The `ctpelvimetry` Solution**: Fully automated measurement in **\< 2 minutes**, providing standardized, reproducible data suitable for large-scale surgical data science and machine learning applications.

### Core Advantages

  - ⚡ **Fully Automated**: From raw DICOM to structured CSVs without a single manual click.
  - 📊 **High-Throughput Batching**: Process hundreds of scans sequentially with built-in failure handling and progress tracking.
  - 🛡️ **Robust Quality Control**: Automatic detection of pelvic rotation/tilt, and generation of multi-planar QC visualisations for immediate verification.
  - 🧩 **Modular Design**: Use it as a CLI tool for batch processing or import it as a Python API for custom research pipelines.

-----

## 🔬 Measured Metrics

`ctpelvimetry` extracts two major categories of surgical metrics:

### 1\. Pelvimetry (Mid-Pelvic Workspace)

| Metric | Description |
|:---|:---|
| **ISD** (mm) | Inter-Spinous Distance (Crucial for deep pelvic surgery) |
| **Inlet AP** (mm) | Promontory → Upper Symphysis |
| **Outlet AP** (mm) | Coccygeal Apex → Lower Symphysis |
| **Outlet Transverse** (mm) | Intertuberous diameter |
| **Sacral Depth & Length** (mm) | Pelvic concavity quantification |

### 2\. Body Composition

| Metric | Description |
|:---|:---|
| **VAT / SAT** (cm²) | Visceral / Subcutaneous Adipose Tissue area |
| **V/S Ratio** | VAT / SAT ratio (Indicator of visceral obesity) |
| **SMA** (cm²) | Skeletal Muscle Area (Measured at L3 and mid-pelvis levels) |

-----

## 👁️ Visual Quality Control (QC)

Trust, but verify. `ctpelvimetry` automatically generates detailed QC panels for every scan to ensure landmark accuracy.

*Sagittal QC showing automated landmark detection: sacral length (magenta), inlet AP (green), outlet AP (orange), and sacral depth (cyan).*

*Extended QC panel: (a) outlet transverse diameter, (b) interspinous distance (ISD), and (c) tabular measurement summary.*

-----

## 💻 Python API Usage

For data scientists building custom pipelines, `ctpelvimetry` provides a clean Python API:

```python
from ctpelvimetry import (
    run_combined_pelvimetry,
    run_nifti_pipeline,
    process_single_patient,
)

# 1a. Run Pelvimetry on existing segmentation
pelv_results = run_combined_pelvimetry(
    patient_id="Patient_001",
    seg_folder="/path/to/segmentation_masks",
    nifti_path="/path/to/ct.nii.gz",
)
print(f"Automated ISD: {pelv_results['ISD_mm']} mm")

# 1b. Or start from a NIfTI file (runs TotalSegmentator internally)
pelv_results = run_nifti_pipeline(
    patient_id="Patient_001",
    nifti_path="/path/to/ct.nii.gz",
    output_root="./output",
)

# 2. Run Body Composition analysis
body_comp_results = process_single_patient(
    patient_id="Patient_001",
    seg_root="/path/to/seg_root",
    nifti_path="/path/to/ct.nii.gz",
    pelvimetry_csv="/path/to/report.csv",
)
```

*(See the `batch.py` module for large-scale dataset orchestration.)*

-----

## 📂 Architecture

```text
ctpelvimetry/
├── cli.py               # Unified CLI entry point
├── pipeline.py          # Core orchestration (DICOM → Metrics)
├── segmentation.py      # TotalSegmentator integration wrapper
├── landmarks.py         # 3D geometric landmark detection
├── metrics.py           # Pelvimetric calculations (ISD, etc.)
├── body_composition.py  # Fat/Muscle area quantification
├── qc.py                # Visual reporting generation
└── batch.py             # High-throughput batch processing
```

-----

## 🤝 Contributing

We welcome contributions from both the surgical and data science communities\!

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/amazing-feature`)
3.  Commit your changes (`git commit -m "Add amazing feature"`)
4.  Push to the branch (`git push origin feature/amazing-feature`)
5.  Open a Pull Request

-----

## 📝 Citation

If you use **`ctpelvimetry`** to facilitate your research, please consider citing our work:

```bibtex
@software{huang2025ctpelvimetry,
  author    = {Huang, Shih-Feng},
  title     = {ctpelvimetry: Automated CT Pelvimetry and Body Composition Analysis},
  year      = {2025},
  url       = {[https://github.com/odafeng/ctpelvimetry](https://github.com/odafeng/ctpelvimetry)},
  version   = {1.1.0}
}
```

> *A peer-reviewed manuscript detailing the clinical validation of this pipeline is currently in preparation.*

-----

**License**: [Apache License 2.0](https://www.google.com/search?q=LICENSE) | **Author**: Shih-Feng Huang, MD
