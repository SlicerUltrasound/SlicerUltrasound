#!/usr/bin/env python3
"""
This script anonymizes a directory of DICOM files using a pre-trained model for corner prediction.

Args:
    input_dir: The directory containing the DICOM files to anonymize.
    output_dir: The directory to save the anonymized DICOM files.
    headers_dir: The directory to save the DICOM headers.
    model_path: The path to the pre-trained model for corner prediction. default: None
    device: The device to use for the model. default: "cpu"
    skip_single_frame: Whether to skip single frame DICOM files. default: False
    no_hash_patient_id: Whether to NOT hash the patient ID. default: False
    filename_prefix: The prefix to add to the anonymized DICOM files. default: None
    preserve_directory_structure: Whether to preserve the directory structure. default: True
    no_mask_generation: Whether to NOT generate a mask. This means that only the headers will be anonymized. default: False
    top_ratio: The ratio of the top of the image to anonymize. default: 0.1
    phi_only_mode: Whether to only anonymize the PHI part of the image. default: False
    remove_phi_from_image: Whether to remove the PHI part of the image. default: True
    overwrite_files: Whether to overwrite existing output files. default: False

Masking Example:
```bash
python -m auto_anonymize --input_dir input_dicoms/ \
    --output_dir output_dicoms/ \
    --headers_dir headers_out/ \
    --model_path model_trace.pt \
    --device cpu
```

PHI removal Example:
```bash
    python -m auto_anonymize --input_dir input_dicoms/ \
        --output_dir output_dicoms/ \
        --headers_dir headers_out/ \
        --model_path model_trace.pt \
        --device cpu \
        --phi_only_mode \
        --remove_phi_from_image
```


"""
# Add the parent directory to the Python path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from typing import Optional, Dict, Any
import csv
import time

from common.dicom_file_manager import DicomFileManager
from common.dicom_processor import DicomProcessor, ProcessingConfig
from common.progress_reporter import TqdmProgressReporter
from common.overview_generator import OverviewGenerator
from common.logging import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Anonymize ultrasound DICOM files')

    # required arguments
    parser.add_argument('--input_dir',
                       help='Directory containing DICOM files to anonymize')
    parser.add_argument('--output_dir',
                       help='Directory to save anonymized DICOM files')
    parser.add_argument('--headers_dir',
                       help='Directory to save DICOM headers (and also the keys.csv)')

    # optional arguments
    parser.add_argument('--model_path',
                       help='Path to pre-trained model for corner prediction')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu',
                       help='Device to use for model inference')
    parser.add_argument('--skip_single_frame', action='store_true', default=False,
                       help='Skip single frame DICOM files')
    parser.add_argument('--no_hash_patient_id', action='store_true', default=False,
                       help='Hash patient IDs in anonymized files')
    parser.add_argument('--filename_prefix',
                       help='Prefix to add to anonymized DICOM files')
    parser.add_argument('--no_preserve_directory_structure', dest='preserve_directory_structure', action='store_false',
                       help='Do not preserve directory structure in output, saving all files to the root of the output folder.')
    parser.add_argument('--no_mask_generation', action='store_true', default=False,
                       help='Do not generate a mask. This means that only the headers will be anonymized.')
    parser.add_argument('--top_ratio', type=float, default=0.1,
                       help='Ratio of top of image to anonymize')
    parser.add_argument('--phi_only_mode', action='store_true', default=False,
                       help='Only anonymize the PHI part of the image')
    parser.add_argument('--remove_phi_from_image', action='store_true', default=True,
                       help='Remove the PHI part of the image')
    parser.add_argument('--overwrite_files', action='store_true', default=False,
                       help='Overwrite existing output files')
    args = parser.parse_args()

    start_time = time.time()

    logger, _ = setup_logging(process_name='auto_anonymize')

    logger.info(f"Starting auto-anonymize with args: {args}")

    # Create processing configuration
    config = ProcessingConfig(
        model_path=args.model_path,
        device=args.device,
        preserve_directory_structure=args.preserve_directory_structure,
        skip_single_frame=args.skip_single_frame,
        hash_patient_id=not args.no_hash_patient_id,
        no_mask_generation=args.no_mask_generation,
        top_ratio=args.top_ratio,
        phi_only_mode=args.phi_only_mode,
        remove_phi_from_image=args.remove_phi_from_image,
        overwrite_files=args.overwrite_files,
    )

    # Initialize components
    dicom_manager = DicomFileManager()
    processor = DicomProcessor(config, dicom_manager)
    progress_reporter = TqdmProgressReporter()
    overview_generator = OverviewGenerator(args.headers_dir)
    processor.initialize_model()
    overview_manifest = []

    # Scan directory
    num_files = dicom_manager.scan_directory(args.input_dir, config.skip_single_frame, config.hash_patient_id)
    logger.info(f"Found {num_files} DICOM files")

    # Save keys.csv
    if dicom_manager.dicom_df is not None and args.headers_dir:
        os.makedirs(args.headers_dir, exist_ok=True)
        dicom_manager.dicom_df.drop(columns=['DICOMDataset'], inplace=False).to_csv(
            os.path.join(args.headers_dir, 'keys.csv'), index=False
        )

    # Process files
    progress_reporter.start(num_files, f"Auto-anonymizing {args.model_path.split('/')[-1]} on {args.device} for {num_files} files...")
    success = failed = skipped = 0
    error_messages = []

    try:
        for idx, row in dicom_manager.dicom_df.iterrows():
            dicom_manager.current_index = idx

            def progress_callback(message: str):
                progress_reporter.update(idx, message)

            def overview_callback(filename: str, orig: np.ndarray, masked: np.ndarray, mask: Optional[np.ndarray], metrics: Optional[Dict[str, Any]]):
                if not args.headers_dir:
                    return
                overview_path = overview_generator.generate_overview(filename, orig, masked, mask)
                overview_manifest.append({
                    "path": overview_path,
                    "filename": filename,
                    "dice": metrics.get("dice_mean") if metrics else None,
                    "iou": metrics.get("iou_mean") if metrics else None,
                    "mean_distance_error": metrics.get("mean_distance_error") if metrics else None,
                    "upper_left_error": metrics.get("upper_left_error") if metrics else None,
                    "upper_right_error": metrics.get("upper_right_error") if metrics else None,
                    "lower_left_error": metrics.get("lower_left_error") if metrics else None,
                    "lower_right_error": metrics.get("lower_right_error") if metrics else None,
                })

            # Use headers_folder for both headers and overview
            result = processor.process_single_dicom(
                row, args.output_dir, args.headers_dir, args.headers_dir,
                progress_callback, overview_callback
            )

            # Update counters
            if result.success and not result.skipped:
                success += 1
            elif result.skipped:
                skipped += 1
            else:
                failed += 1
                if result.error_message:
                    error_messages.append(result.error_message)

        if (config.phi_only_mode and config.remove_phi_from_image):
            processor.generate_redaction_pdf()
        else:
            processor.generate_overview_pdf(overview_manifest, args.headers_dir, no_metrics_table=True)

    finally:
        progress_reporter.finish()

    logger.info(f"Complete! Success: {success}, Failed: {failed}, Skipped: {skipped}, Time: {time.time() - start_time:.2f}s")

    return {
        "status": f"Complete! Success: {success}, Failed: {failed}, Skipped: {skipped}",
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "error_messages": error_messages,
    }

if __name__ == '__main__':
    main()

