"""
Unified CLI entry point for pelvimetry and body composition analysis.

Usage:
    ctpelvimetry pelv [options]         # Pelvimetry analysis
    ctpelvimetry body-comp [options]    # Body composition analysis
"""

import os
import argparse

import pandas as pd

from .pipeline import run_combined_pelvimetry, run_full_pipeline
from .batch import run_pelvimetry_batch, run_body_composition_batch
from .body_composition import process_single_patient


def _add_pelvimetry_parser(subparsers):
    """Register the ``pelv`` subcommand with its arguments.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparser group from the main ``ArgumentParser``.
    """
    parser = subparsers.add_parser(
        "pelv",
        help="Pelvimetry analysis (DICOM → NIfTI → Seg → Measurements)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single patient (from DICOM)
  ctpelvimetry pelv --dicom_dir /path/to/Patient_001 --output_root ./output --patient Patient_001

  # Batch processing
  ctpelvimetry pelv --dicom_root /path/to/DICOMs --output_root ./output --start 1 --end 250

  # Analysis only (existing segmentation)
  ctpelvimetry pelv --seg_folder /path/to/seg --nifti_path /path/to/ct.nii.gz --patient Demo --qc
""",
    )

    # Input modes
    parser.add_argument("--dicom_root", type=str, default=None,
                        help="Parent DICOM directory (batch mode)")
    parser.add_argument("--dicom_dir", type=str, default=None,
                        help="Single patient DICOM directory")
    parser.add_argument("--seg_folder", type=str, default=None,
                        help="Existing segmentation folder (skip preprocessing)")
    parser.add_argument("--nifti_path", type=str, default=None,
                        help="Existing NIfTI file path")

    # Output
    parser.add_argument("--output_root", type=str, default="./pelvimetry_output",
                        help="Output root directory")
    parser.add_argument("--output", type=str, default="combined_pelvimetry_report.csv",
                        help="Output CSV filename")

    # Parameters
    parser.add_argument("--patient", type=str, default=None, help="Single patient ID")
    parser.add_argument("--start", type=int, default=1, help="Start patient number")
    parser.add_argument("--end", type=int, default=250, help="End patient number")
    parser.add_argument("--fast", action="store_true", help="Use TotalSegmentator fast mode")
    parser.add_argument("--no-tissue", dest="skip_tissue", action="store_true",
                        help="Skip tissue-type segmentation")
    parser.add_argument("--qc", action="store_true", default=True,
                        help="Generate QC images (default: on)")
    parser.add_argument("--no-qc", dest="qc", action="store_false",
                        help="Disable QC image generation")


def _add_body_comp_parser(subparsers):
    """Register the ``body-comp`` subcommand with its arguments.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparser group from the main ``ArgumentParser``.
    """
    parser = subparsers.add_parser(
        "body-comp",
        help="Body composition analysis (VAT/SAT/SMA at L3 and ISD levels)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single patient
  ctpelvimetry body-comp --patient Patient_001 \\
      --seg_root ./batch_output \\
      --nifti_root ./batch_output \\
      --pelvimetry_csv ./batch_output/combined_pelvimetry_report.csv \\
      --output body_comp_001.csv --qc

  # Batch processing
  ctpelvimetry body-comp \\
      --seg_root ./batch_output \\
      --nifti_root ./batch_output \\
      --pelvimetry_csv ./batch_output/combined_pelvimetry_report.csv \\
      --output body_composition_report.csv \\
      --start 1 --end 210 --qc_root ./body_comp_qc
""",
    )

    parser.add_argument("--patient", help="Single patient ID")
    parser.add_argument("--seg_root", required=True, help="Segmentation root directory")
    parser.add_argument("--nifti_root", required=True, help="NIfTI file root directory")
    parser.add_argument("--pelvimetry_csv", required=True,
                        help="Pelvimetry report CSV path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--start", type=int, default=1, help="Start patient number")
    parser.add_argument("--end", type=int, default=250, help="End patient number")
    parser.add_argument("--qc", action="store_true", help="Generate QC images (single patient)")
    parser.add_argument("--qc_root", help="QC image root directory (batch mode)")


# ------------------------------------------------------------------
# Subcommand handlers
# ------------------------------------------------------------------

def _handle_pelvimetry(args):
    """Dispatch the pelvimetry subcommand.

    Supports three modes: analysis-only (existing segmentation),
    single-patient full pipeline, and batch processing.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """

    # Mode 1: Analysis only (existing segmentation)
    if args.seg_folder:
        if not os.path.isdir(args.seg_folder):
            print(f"❌ Segmentation folder not found: {args.seg_folder}")
            return

        nifti_path = args.nifti_path or ""
        patient_id = args.patient or os.path.basename(args.seg_folder)

        qc_dir = os.path.join(args.output_root, "QC") if args.qc else None
        if qc_dir:
            os.makedirs(qc_dir, exist_ok=True)

        result = run_combined_pelvimetry(
            patient_id, args.seg_folder, nifti_path, qc_dir=qc_dir
        )

        print(f"\n{'='*50}")
        print(f"Results for {patient_id}")
        print(f"{'='*50}")
        for key, value in result.items():
            if value is not None and key != "Seg_Vertebrae":
                print(f"  {key}: {value}")

        df = pd.DataFrame([result])
        output_path = os.path.join(args.output_root, args.output)
        os.makedirs(args.output_root, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ Saved to: {output_path}")

    # Mode 2: Single patient from DICOM
    elif args.dicom_dir and args.patient:
        result = run_full_pipeline(
            args.patient,
            args.dicom_dir,
            args.output_root,
            use_fast=args.fast,
            skip_tissue=args.skip_tissue,
            generate_qc=args.qc,
        )

        df = pd.DataFrame([result])
        output_path = os.path.join(args.output_root, args.output)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ Saved to: {output_path}")

    # Mode 3: Batch processing
    elif args.dicom_root:
        output_path = os.path.join(args.output_root, args.output)
        run_pelvimetry_batch(
            args.dicom_root,
            args.output_root,
            output_path,
            args.start,
            args.end,
            use_fast=args.fast,
            skip_tissue=args.skip_tissue,
        )

    else:
        print("❌ Please specify an input mode:")
        print("   --seg_folder (existing segmentation)")
        print("   --dicom_dir + --patient (single patient from DICOM)")
        print("   --dicom_root (batch processing)")


def _handle_body_comp(args):
    """Dispatch the body-composition subcommand.

    Supports single-patient and batch modes.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """

    # Single patient mode
    if args.patient:
        nifti_path = os.path.join(args.nifti_root, args.patient, f"{args.patient}.nii.gz")
        qc_dir = os.path.dirname(args.output) if args.qc else None

        result = process_single_patient(
            args.patient,
            args.seg_root,
            nifti_path,
            args.pelvimetry_csv,
            qc_dir
        )

        df = pd.DataFrame([result])
        df.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"\n✅ Results saved to: {args.output}")

    # Batch mode
    else:
        run_body_composition_batch(
            args.seg_root,
            args.nifti_root,
            args.pelvimetry_csv,
            args.output,
            args.qc_root,
            args.start,
            args.end,
        )


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def main():
    """Unified CLI entry point for ``ctpelvimetry``.

    Parses command-line arguments and delegates to the appropriate
    subcommand handler (``pelv`` or ``body-comp``).
    """
    parser = argparse.ArgumentParser(
        prog="ctpelvimetry",
        description="Automated Pelvimetry and Body Composition Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    _add_pelvimetry_parser(subparsers)
    _add_body_comp_parser(subparsers)

    args = parser.parse_args()

    if args.command == "pelv":
        _handle_pelvimetry(args)
    elif args.command == "body-comp":
        _handle_body_comp(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
