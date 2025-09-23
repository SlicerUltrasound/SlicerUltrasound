#!/usr/bin/env python3
"""
This script anonymizes a directory of DICOM files using a pre-trained model for corner prediction.

Args:
    input_folder: The directory containing the DICOM files to anonymize.
    output_folder: The directory to save the anonymized DICOM files.
    headers_folder: The directory to save the DICOM headers.
    model_path: The path to the pre-trained model for corner prediction. default: None
    device: The device to use for the model. default: "cpu"
    skip_single_frame: Whether to skip single frame DICOM files. default: False
    no_hash_patient_id: Whether to NOT hash the patient ID. default: False
    filename_prefix: The prefix to add to the anonymized DICOM files. default: None
    preserve_directory_structure: Whether to preserve the directory structure. default: True
    resume_anonymization: Whether to skip processing if the output file already exists. default: False
    no_mask_generation: Whether to NOT generate a mask. This means that only the headers will be anonymized. default: False

Example:
python -m auto_anonymize --input_folder input_dicoms/ \
    --output_folder output_dicoms/ \
    --headers_folder headers_out/ \
    --model-path model_trace.pt \
    --device cpu
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
    parser.add_argument('--input_folder',
                       help='Directory containing DICOM files to anonymize')
    parser.add_argument('--output_folder',
                       help='Directory to save anonymized DICOM files')
    parser.add_argument('--headers_folder',
                       help='Directory to save DICOM headers (and also the keys.csv)')

    # optional arguments
    parser.add_argument('--model-path',
                       help='Path to pre-trained model for corner prediction')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu',
                       help='Device to use for model inference')
    parser.add_argument('--skip-single-frame', action='store_true', default=False,
                       help='Skip single frame DICOM files')
    parser.add_argument('--no-hash-patient-id', action='store_true', default=False,
                       help='Hash patient IDs in anonymized files')
    parser.add_argument('--filename-prefix',
                       help='Prefix to add to anonymized DICOM files')
    parser.add_argument('--no-preserve-directory-structure', dest='preserve_directory_structure', action='store_false',
                       help='Do not preserve directory structure in output, saving all files to the root of the output folder.')
    parser.add_argument('--resume-anonymization', action='store_true',
                       help='Skip processing if output file already exists.')
    parser.add_argument('--no-mask-generation', action='store_true', default=False,
                       help='Do not generate a mask. This means that only the headers will be anonymized.')
    args = parser.parse_args()

    start_time = time.time()

    logger, _ = setup_logging(process_name='auto_anonymize')

    # Create processing configuration
    config = ProcessingConfig(
        model_path=args.model_path,
        device=args.device,
        preserve_directory_structure=args.preserve_directory_structure,
        resume_anonymization=args.resume_anonymization,
        skip_single_frame=args.skip_single_frame,
        hash_patient_id=not args.no_hash_patient_id,
        no_mask_generation=args.no_mask_generation,
    )

    # Initialize components
    dicom_manager = DicomFileManager()
    processor = DicomProcessor(config, dicom_manager)
    progress_reporter = TqdmProgressReporter()

    # Scan directory
    num_files = dicom_manager.scan_directory(args.input_folder, config.skip_single_frame, config.hash_patient_id)
    logger.info(f"Found {num_files} DICOM files")

    # Save keys.csv
    if dicom_manager.dicom_df is not None:
        dicom_manager.dicom_df.drop(columns=['DICOMDataset'], inplace=False).to_csv(
            os.path.join(args.headers_folder, 'keys.csv'), index=False
        )

    # Initialize model
    processor.initialize_model()

    # Setup metrics CSV if ground truth directory provided
    metrics_csv_path = None
    metrics_file = None
    metrics_writer = None

    metrics_csv_path = os.path.join(args.headers_folder, "metrics.csv")
    os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
    metrics_file = open(metrics_csv_path, "w", newline="")
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=processor.get_metrics_fieldnames())
    metrics_writer.writeheader()

    # Process files
    progress_reporter.start(num_files, "Processing DICOM files")

    success_count = failed_count = skipped_count = 0

    try:
        for idx, row in dicom_manager.dicom_df.iterrows():
            dicom_manager.current_index = idx

            # Define callbacks
            def progress_callback(message: str):
                progress_reporter.update(idx, message)

            result = processor.process_single_dicom(
                row, args.output_folder, args.headers_folder,
                progress_callback, None
            )

            # Update counters
            if result.success and not result.skipped:
                success_count += 1
            elif result.skipped:
                skipped_count += 1
            else:
                failed_count += 1

            # Write metrics to CSV if available
            if metrics_writer and result.success and not result.skipped:
                csv_row = processor.format_metrics_for_csv(result)
                metrics_writer.writerow(csv_row)

    finally:
        progress_reporter.finish()
        if metrics_file:
            metrics_file.close()
            logger.info(f"Metrics saved to: {metrics_csv_path}")

    logger.info(f"Complete! Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}, Time: {time.time() - start_time:.2f}s")

    return {
        "status": f"Complete! Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}",
        "success": success_count,
        "failed": failed_count,
        "skipped": skipped_count,
    }

if __name__ == '__main__':
    main()

