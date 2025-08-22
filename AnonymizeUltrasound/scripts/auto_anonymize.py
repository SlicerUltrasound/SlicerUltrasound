#!/usr/bin/env python3
"""
This script anonymizes a directory of DICOM files using a pre-trained model for corner prediction.
It uses lib.dcm_inference for helper functions related to the AI-based corner prediction 
as well as lib.slicer_anonymizer for the anonymization logic and file management.
It copies the behavior of SlicerUltrasound/AnonymizeUltrasound

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
    overview_dir: The directory to save the overview images. default: None
    no_mask_generation: Whether to NOT generate a mask. This means that only the headers will be anonymized. default: False

"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, List
import time
import matplotlib.pyplot as plt
import shutil

from AnonymizeUltrasound.common.dcm_inference import (
    load_model,
    preprocess_image,
    compute_masks_and_configs
)
from AnonymizeUltrasound.AnonymizeUltrasound import UltrasoundAnonymizer
from AnonymizeUltrasound.common.create_frames import read_frames_from_dicom
from AnonymizeUltrasound.common.logging_utils import setup_logging

def process_dicom_file(dicom_info: dict, model, device: str, anonymizer: UltrasoundAnonymizer, output_folder: str, headers_folder: str, preserve_directory_structure: bool, resume_anonymization: bool, overview_dir: Optional[str], skip_single_frame: bool, no_mask_generation: bool, logger=logging.getLogger(__name__)) -> tuple[bool, bool]:
    """
    Process a single DICOM file: run inference, create mask, and save anonymized version.
    
    Args:
        dicom_info: Dictionary containing DICOM file information
        model: Loaded AI model for corner prediction
        device: Device to run inference on
        anonymizer: UltrasoundAnonymizer instance
        output_folder: Output directory for anonymized DICOM files
        headers_folder: Output directory for DICOM headers
        preserve_directory_structure: Whether to preserve directory structure
        resume_anonymization: Whether to skip processing if the output file already exists
        overview_dir: Optional. Directory to save overview grids of original vs anonymized frames.
        skip_single_frame: Whether to skip single frame DICOM files.
    Returns:
        tuple[bool, bool]: (success, skipped)
        success: True if processing was successful, False otherwise
        skipped: True if the file was skipped, False otherwise
    """
    try:
        start_time = time.time()
        
        input_path = dicom_info['InputPath']
        output_path = dicom_info['OutputPath']
        anon_filename = dicom_info['AnonFilename']
        
        # Generate output file path to check for existence
        final_output_path = anonymizer.generate_output_filepath(
            output_folder, output_path, preserve_directory_structure
        )
        
        if resume_anonymization and os.path.exists(final_output_path):
            logger.info(f"Output file already exists, skipping: {final_output_path}")
            return True, True
            
        logger.info(f"\nProcessing: {input_path}")
        
        # 1. Read DICOM frames
        read_frames_start = time.time()
        try:
            original_image = read_frames_from_dicom(input_path)
            original_dims = (original_image.shape[-2], original_image.shape[-1])  # (height, width)
            if skip_single_frame and len(original_image.shape) == 3 and original_image.shape[0] == 1:
                logger.info(f"Skipping single frame DICOM file: {input_path}")
                return True, True
        except Exception as e:
            logger.error(f"Failed to read DICOM frames from {input_path}: {e}")
            return False, False
        read_frames_end = time.time()
        logger.info(f"Time to read DICOM frames: {read_frames_end - read_frames_start:.4f} seconds")
        
        # ensure defined for later plotting even if inference is skipped/failed
        curvilinear_mask = None
        
        if not no_mask_generation:
            # 2. Run AI inference for corner prediction
            inference_start = time.time()
            try:
                # Preprocess image for model input
                input_tensor = preprocess_image(original_image)
                
                # Run inference
                if model is not None:
                    # For coordinate models, get normalized coordinates directly
                    with torch.no_grad():
                        coords_normalized = model(input_tensor.to(device)).cpu().numpy()
                    
                    # Denormalize coordinates
                    coords = coords_normalized.reshape(4, 2)
                    coords[:, 0] *= original_dims[1]  # width
                    coords[:, 1] *= original_dims[0]  # height
                    
                    predicted_corners = {
                        "upper_left": tuple(coords[0]),
                        "upper_right": tuple(coords[1]),
                        "lower_left": tuple(coords[2]),
                        "lower_right": tuple(coords[3]),
                    }
                    
                    # Store predicted corners and original dimensions for JSON output
                    anonymizer.predicted_corners = predicted_corners
                    anonymizer.original_dims = original_dims
                    
                    # Compute mask from predicted corners
                    # technically this `curvilinear_mask` could be any type of mask, not just curvilinear
                    curvilinear_mask = compute_masks_and_configs(
                        original_dims=original_dims, 
                        predicted_corners=predicted_corners
                    )        
            except Exception as e:
                logger.error(f"Failed to run inference on {input_path}: {e}")
                return False, False
            inference_end = time.time()
            logger.info(f"Time for inference and mask computation: {inference_end - inference_start:.4f} seconds")
            
            # 3. Apply mask to anonymize the ultrasound images
            apply_mask_start = time.time()
            try:
                if curvilinear_mask is not None:
                    # Convert original image from (frames, channels, height, width) to (frames, height, width, channels)
                    image_array = np.transpose(original_image, (0, 2, 3, 1))
                    
                    # Apply mask to all frames
                    masked_image_array = anonymizer.apply_mask_to_sequence(image_array, curvilinear_mask)
                    
                    # Store basic mask parameters for later use
                    anonymizer.mask_parameters = {
                        'MaskApplied': True,
                        'ProcessedBy': 'auto_anonymize.py'
                    }
                else:
                    logger.warning(f"No mask created for {input_path}, skipping anonymization")
                    return False, False
            except Exception as e:
                logger.error(f"Failed to apply mask to {input_path}: {e}")
                return False, False
            apply_mask_end = time.time()
            logger.info(f"Time to apply mask: {apply_mask_end - apply_mask_start:.4f} seconds")
        else:
            # just copy the original image to the masked image array
            masked_image_array = original_image

            logger.info("Mask generation is disabled, so the original image will be copied to the masked image array.")
        

        # Generate overview if requested
        if overview_dir and original_image.shape[0] > 0:
            plot_start = time.time()
            try:
                fig, axes = plt.subplots(1, 3, figsize=(18, 4))
                
                axes[0].set_title('Original')
                axes[1].set_title('Mask Outline')
                axes[2].set_title('Anonymized')

                # Get the first frame
                orig_frame = original_image[0]
                orig_frame = np.transpose(orig_frame, (1, 2, 0))
                
                masked_frame = masked_image_array[0]
                
                # 1) Original
                axes[0].imshow(orig_frame.squeeze(), cmap='gray')
                axes[0].axis('off')
                
                # 2) Original + mask outline (only if we have a mask)
                axes[1].imshow(orig_frame.squeeze(), cmap='gray')
                if curvilinear_mask is not None:
                    # Draw a contour at the 0.5 level to trace the mask boundary
                    axes[1].contour(curvilinear_mask, levels=[0.5], colors='lime', linewidths=1.0)
                else:
                    axes[1].text(0.5, 0.5, 'No mask', ha='center', va='center', transform=axes[1].transAxes, color='red')
                axes[1].axis('off')

                # 3) Masked
                axes[2].imshow(masked_frame.squeeze(), cmap='gray')
                axes[2].axis('off')
                
                overview_filename = f"{os.path.splitext(anon_filename)[0]}_overview.png"
                overview_filepath = os.path.join(overview_dir, overview_filename)
                plt.tight_layout()
                plt.savefig(overview_filepath)
                plt.close(fig)
                logger.info(f"Saved overview image to {overview_filepath}")
            except Exception as e:
                logger.error(f"Failed to generate overview for {input_path}: {e}")
            plot_end = time.time()
            logger.info(f"Time to generate overview: {plot_end - plot_start:.4f} seconds")

        # 4. Save anonymized DICOM file
        save_dicom_start = time.time()
        try:
            anonymizer.save_anonymized_dicom(
                image_array=masked_image_array,
                output_path=final_output_path,
                new_patient_name=anon_filename.split('.')[0],
                new_patient_id=anon_filename.split('_')[0],
            )
        except Exception as e:
            logger.error(f"Failed to save anonymized DICOM {final_output_path}: {e}")
            return False, False
        save_dicom_end = time.time()
        logger.info(f"Time to save anonymized DICOM: {save_dicom_end - save_dicom_start:.4f} seconds")
        
        # 5. Save DICOM header if requested
        save_header_start = time.time()
        if headers_folder:
            try:
                anonymizer.save_anonymized_dicom_header(output_filename=anon_filename, headers_directory=headers_folder)
            except Exception as e:
                logger.error(f"Failed to save DICOM header for {input_path}: {e}")
                # Don't return False here as the main anonymization succeeded
        save_header_end = time.time()
        logger.info(f"Time to save DICOM header: {save_header_end - save_header_start:.4f} seconds")
        
        # 6. Save sequence info and mask info
        save_info_start = time.time()
        if not no_mask_generation:
            try:
                sop_instance_uid = dicom_info['InstanceUID']
                anonymizer.save_json(final_output_path, sop_instance_uid)
            except Exception as e:
                logger.error(f"Failed to save json for {final_output_path}: {e}")
                # Don't return False here as the main anonymization succeeded
        else:
            logger.info("Mask generation is disabled, so the original json will be copied over.")
            input_json_path = dicom_info['InputPath'].replace('.dcm', '.json')
            if os.path.exists(input_json_path):
                shutil.copy(input_json_path, final_output_path.replace('.dcm', '.json'))
            else:
                logger.warning(f"No json found at {input_json_path}, skipping copy.")
            
        save_info_end = time.time()
        logger.info(f"Time to save sequence info: {save_info_end - save_info_start:.4f} seconds")
        
        end_time = time.time()
        logger.info(f"Total processing time for {input_path}: {end_time - start_time:.4f} seconds")
        
        logger.info(f"Successfully processed: {input_path} -> {final_output_path}")
        return True, False
        
    except Exception as e:
        logger.error(f"Unexpected error processing {dicom_info.get('InputPath', 'unknown')}: {e}")
        return False, False


def main():
    """Main function for the auto_anonymize script."""
    print("""
░█████╗░██╗░░░██╗████████╗░█████╗░  ░█████╗░███╗░░██╗░█████╗░███╗░░██╗██╗░░░██╗███╗░░░███╗██╗███████╗███████╗
██╔══██╗██║░░░██║╚══██╔══╝██╔══██╗  ██╔══██╗████╗░██║██╔══██╗████╗░██║╚██╗░██╔╝████╗░████║██║╚════██║██╔════╝
███████║██║░░░██║░░░██║░░░██║░░██║  ███████║██╔██╗██║██║░░██║██╔██╗██║░╚████╔╝░██╔████╔██║██║░░███╔═╝█████╗░░
██╔══██║██║░░░██║░░░██║░░░██║░░██║  ██╔══██║██║╚████║██║░░██║██║╚████║░░╚██╔╝░░██║╚██╔╝██║██║██╔══╝░░██╔══╝░░
██║░░██║╚██████╔╝░░░██║░░░╚█████╔╝  ██║░░██║██║░╚███║╚█████╔╝██║░╚███║░░░██║░░░██║░╚═╝░██║██║███████╗███████╗
╚═╝░░╚═╝░╚═════╝░░░░╚═╝░░░░╚════╝░  ╚═╝░░╚═╝╚═╝░░╚══╝░╚════╝░╚═╝░░╚══╝░░░╚═╝░░░╚═╝░░░░░╚═╝╚═╝╚══════╝╚══════╝
""")
    parser = argparse.ArgumentParser(
        description='Anonymize ultrasound DICOM files using AI-based corner prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # required arguments
    parser.add_argument('input_folder', 
                       help='Directory containing DICOM files to anonymize')
    parser.add_argument('output_folder',
                       help='Directory to save anonymized DICOM files')
    parser.add_argument('headers_folder', 
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
    parser.add_argument('--overview-dir',
                          help='Directory to save overview images of original vs anonymized frames.')
    parser.add_argument('--no-mask-generation', action='store_true', default=False,
                       help='Do not generate a mask. This means that only the headers will be anonymized.')
    args = parser.parse_args()
    
    # Setup logging
    logger, log_file = setup_logging(process_name='auto_anonymize')
    
    # Validate arguments
    if not os.path.exists(args.input_folder):
        logger.error(f"Input folder does not exist: {args.input_folder}")
        sys.exit(1)
    
    if args.model_path and not os.path.exists(args.model_path):
        logger.error(f"Model file does not exist: {args.model_path}")
        sys.exit(1)
    
    # Create output directories
    os.makedirs(args.output_folder, exist_ok=True)
    if args.headers_folder:
        os.makedirs(args.headers_folder, exist_ok=True)
    else:
        args.headers_folder = args.output_folder
    
    if args.overview_dir:
        os.makedirs(args.overview_dir, exist_ok=True)
        logger.info(f"Saving overview grids to: {args.overview_dir}")
    else:
        # raise a big warning
        logger.warning("""
        ############################################################
        # WARNING: Overview directory is not specified.             #
        # Overview images will not be saved.                         #
        # PHI is at risk if you do not review the anonymized images. #
        ############################################################
        """)

    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        logger.warning("MPS requested but not available, falling back to CPU")
        args.device = 'cpu'
    
    logger.info(f"Starting anonymization process")
    logger.info(f"Input folder: {args.input_folder}")
    logger.info(f"Output folder: {args.output_folder}")
    logger.info(f"Headers folder: {args.headers_folder}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Resume anonymization: {args.resume_anonymization}")
    logger.info(f"Skip single frame: {args.skip_single_frame}")
    logger.info(f"No hash patient ID: {args.no_hash_patient_id}")
    logger.info(f"Filename prefix: {args.filename_prefix}")
    logger.info(f"Preserve directory structure: {args.preserve_directory_structure}")
    logger.info(f"Overview directory: {args.overview_dir}")
    logger.info(f"No mask generation: {args.no_mask_generation}")
    
    # Initialize anonymizer
    anonymizer = UltrasoundAnonymizer()
    
    # Determine if patient ID should be hashed
    hash_patient_id = not args.no_hash_patient_id
    if not hash_patient_id:
        logger.warning("""
        ############################################################
        # WARNING: Patient ID hashing is DISABLED (--no-hash-patient-id specified).   #
        # The PatientID field will be preserved in the output DICOM files.            #
        # Ensure the PatientID does not contain PHI.                                    #
        ############################################################
        """)
    else:
        logger.info("Patient ID will be hashed")
    
    if args.no_mask_generation:
        # raise a big warning
        logger.warning("""
        ############################################################
        # WARNING: Mask generation is DISABLED (--no-mask-generation specified).   #
        # The mask will not be generated. Burned in text will remain in the dcm frames. #
        ############################################################
        """)

    # 1. Scan directory for DICOM files
    logger.info("Scanning directory for DICOM files...")
    num_files = anonymizer.scan_directory(args.input_folder, args.skip_single_frame, hash_patient_id=hash_patient_id)
    
    if num_files == 0:
        logger.error("No valid DICOM files found in input directory")
        sys.exit(1)
    
    logger.info(f"Found {num_files} DICOM files to process")

    # save the keys.csv file, which is just the anonymizer.dicom_df
    anonymizer.dicom_df.to_csv(os.path.join(args.headers_folder, 'keys.csv'), index=False)
    
    # 2. Load model if provided
    model = None
    if args.model_path and not args.no_mask_generation:
        logger.info(f"Loading model from {args.model_path}")
        try:
            model = load_model(args.model_path, args.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)
    elif args.no_mask_generation:
        logger.info("Mask generation is disabled, so no model is loaded.")
    else:
        logger.error("No model provided")
        sys.exit(1)

    if args.no_hash_patient_id:
        logger.info("Patient ID will not be hashed")
        hash_patient_id = False
    else:
        logger.info("Patient ID will be hashed")
        hash_patient_id = True
    
    # 3. Process each DICOM file
    logger.info("Starting DICOM file processing...")
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Create progress bar
    pbar = tqdm(anonymizer.dicom_df.iterrows(), total=len(anonymizer.dicom_df), 
                desc="Processing DICOM files")
    
    for idx, dicom_info in pbar:
        anonymizer.current_dicom_index = idx
        success, skipped = process_dicom_file(
            dicom_info.to_dict(),
            model,
            args.device,
            anonymizer,
            args.output_folder,
            args.headers_folder,
            args.preserve_directory_structure,
            args.resume_anonymization,
            args.overview_dir,
            args.skip_single_frame,
            args.no_mask_generation,
            logger
        )
        
        if success and not skipped:
            successful_count += 1
        elif skipped:
            skipped_count += 1
        else:
            failed_count += 1
        
        pbar.set_postfix({
            'Success': successful_count,
            'Failed': failed_count,
            'Skipped': skipped_count,
            'Success Rate': f'{successful_count/(successful_count+failed_count)*100:.1f}%' if (successful_count+failed_count) > 0 else '0%'
        })
    
    # 4. Summary
    logger.info(f"Anonymization complete!")
    logger.info(f"Successfully processed: {successful_count} files")
    logger.info(f"Skipped: {skipped_count} files")
    logger.info(f"Failed to process: {failed_count} files")
    logger.info(f"Total files considered: {successful_count + failed_count}")
    logger.info(f"Success rate: {successful_count/(successful_count+failed_count)*100:.1f}%" if (successful_count+failed_count) > 0 else "0%")
    
    if failed_count > 0:
        logger.warning(f"Some files failed to process. Check the logs above for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()

