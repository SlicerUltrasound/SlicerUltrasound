from typing import Optional, Dict, Any, Tuple, Callable, List
import math
import logging
import numpy as np
import os
import json
import time
from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

from .dicom_file_manager import DicomFileManager
from .inference import load_model, preprocess_image, download_model
from .masking import compute_masks_and_configs, mask_config_to_corner_points, corner_points_to_fan_mask_config
from .evaluation import compare_masks, load_mask_config

@dataclass
class ProcessingConfig:
    """Configuration for DICOM processing"""
    model_path: str
    device: str
    preserve_directory_structure: bool = True
    resume_anonymization: bool = False
    skip_single_frame: bool = False
    hash_patient_id: bool = True
    no_mask_generation: bool = False
    overview_dir: Optional[str] = None
    ground_truth_dir: Optional[str] = None
    top_ratio: float = 0.1
    phi_only_mode: bool = False  # If True, only apply top redaction, skip fan mask
    remove_phi_from_image: bool = True  # If True, apply PHI redaction to image and generate PDF; if False, only remove PHI from metadata
    overwrite_files: bool = False  # If True, overwrite existing output files; if False, skip them

@dataclass
class ProcessingResult:
    """Result of processing a single DICOM file"""
    success: bool
    skipped: bool
    input_path: str
    output_path: str
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metrics: Optional[Dict[str, Any]] = None

class GroundTruthManager:
    """Manages ground truth file indexing and lookup for metrics computation"""

    def __init__(self, ground_truth_dir: Optional[str] = None):
        self.ground_truth_dir = ground_truth_dir
        self.gt_index = {}
        self.logger = logging.getLogger(__name__)

        if ground_truth_dir:
            self._build_ground_truth_index()

    def _build_ground_truth_index(self):
        """Build index of ground truth JSON files for fast lookup"""
        if not self.ground_truth_dir or not os.path.exists(self.ground_truth_dir):
            self.logger.warning(f"Ground truth directory not found: {self.ground_truth_dir}")
            return

        try:
            for root, _, files in os.walk(self.ground_truth_dir):
                for f in files:
                    if f.lower().endswith(".json"):
                        full_path = os.path.join(root, f)
                        if f not in self.gt_index:
                            self.gt_index[f] = full_path
                        else:
                            self.logger.warning(
                                f"Duplicate ground truth filename '{f}' found. "
                                f"Keeping '{self.gt_index[f]}', ignoring '{full_path}'."
                            )
            self.logger.info(f"Indexed {len(self.gt_index)} ground truth files")
        except Exception as e:
            self.logger.warning(f"Failed to index ground truth directory '{self.ground_truth_dir}': {e}")

    def get_ground_truth_path(self, anon_filename: str) -> Optional[str]:
        """Get ground truth file path for given anonymized filename"""
        if not self.gt_index:
            return None

        gt_basename = f"{os.path.splitext(anon_filename)[0]}.json"
        return self.gt_index.get(gt_basename)

class DicomProcessor:
    """Shared core processor for DICOM anonymization with metrics support"""

    def __init__(self, config: ProcessingConfig, dicom_manager: DicomFileManager):
        self.config = config
        self.dicom_manager = dicom_manager
        self.model = None
        self.device = None
        self.logger = logging.getLogger(__name__)

        # Initialize ground truth manager if ground truth directory is provided
        self.gt_manager = GroundTruthManager(config.ground_truth_dir)

    def initialize_model(self):
        """Load and initialize the AI model"""
        if not self.config.no_mask_generation:
            # Check if model exists, download if not
            if not os.path.exists(self.config.model_path):
                self.logger.info(f"Model not found at {self.config.model_path}. Attempting to download...")
                success = download_model(output_path=self.config.model_path)
                if not success:
                    raise FileNotFoundError(f"Could not download model to {self.config.model_path}")

            self.model = load_model(self.config.model_path, self.config.device)
            self.device = self.config.device
            self.logger.info(f"Model loaded on {self.device}")


    def evaluate_single_dicom(
        self,
        row,
        progress_callback: Optional[Callable[[str], None]] = None,
        overview_callback: Optional[Callable[[str, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[Dict[str, Any]]], None]] = None
    ) -> ProcessingResult:
        start_time = time.time()
        input_path = row.InputPath
        try:
            if progress_callback:
                progress_callback(f"Evaluating: {input_path}")

            # Read frames [N,H,W,C]
            original_image = self.dicom_manager.read_frames_from_dicom(input_path)
            original_dims = (original_image.shape[1], original_image.shape[2])

            # Load ground truth mask configuration
            dicom_basename = os.path.basename(input_path)
            gt_config_path = self.gt_manager.get_ground_truth_path(dicom_basename)
            gt_mask_config = load_mask_config(gt_config_path)if gt_config_path else None
            gt_corners = mask_config_to_corner_points(gt_mask_config) if gt_mask_config else None

            # Run AI
            coords_norm = self._run_inference(original_image)
            predicted_corners = self._denormalize_coords(coords_norm, original_dims)

            # Predicted mask and config
            predicted_mask_2d, predicted_mask_config = compute_masks_and_configs(
                original_dims=original_dims, predicted_corners=predicted_corners
            )

            # For overview, produce a masked visualization
            masked_image = self._apply_mask(original_image, predicted_mask_2d)

            # Metrics: match GT by DICOM filename
            metrics = self._compute_metrics_with_ground_truth(dicom_basename, gt_mask_config, original_dims, gt_corners, predicted_mask_config, predicted_corners)

            # Overview callback
            if overview_callback and original_image.shape[0] > 0:
                overview_callback(dicom_basename, original_image, masked_image, predicted_mask_2d, metrics)

            return ProcessingResult(
                success=True, skipped=False, input_path=input_path,
                output_path="", processing_time=time.time() - start_time,
                metrics=metrics
            )
        except Exception as e:
            err = f"Failed to evaluate {input_path}: {e}"
            self.logger.error(err)
            return ProcessingResult(
                success=False, skipped=False, input_path=input_path,
                output_path="", error_message=err, processing_time=time.time() - start_time
            )

    def process_single_dicom(
        self,
        row,
        output_folder: str,
        headers_folder: str,
        progress_callback: Optional[Callable[[str], None]] = None,
        overview_callback: Optional[Callable[[str, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[Dict[str, Any]]], None]] = None
    ) -> ProcessingResult:
        """
        Process a single DICOM file with shared logic.

        Args:
            row: DataFrame row containing DICOM info
            output_folder: Output directory for anonymized DICOMs
            headers_folder: Directory for DICOM headers
            progress_callback: Optional callback for progress updates
            overview_callback: Optional callback for overview generation

        Returns:
            ProcessingResult with success/failure info and metrics
        """
        start_time = time.time()
        input_path = row.InputPath

        try:
            # Check if should skip (resume mode)
            final_output_path = self.dicom_manager.generate_output_filepath(
                output_folder, row.OutputPath, self.config.preserve_directory_structure
            )

            # Check if file exists and should be skipped
            if os.path.exists(final_output_path):
                if not self.config.overwrite_files:
                    self.logger.info(f"Skipping existing file: {final_output_path}")
                    return ProcessingResult(
                        success=True, skipped=True, input_path=input_path,
                        output_path=final_output_path, processing_time=time.time() - start_time
                    )
                else:
                    self.logger.info(f"Overwriting existing file: {final_output_path}")

            if progress_callback:
                progress_callback(f"Processing: {input_path}")

            # Read DICOM frames and returns numpy array (N, H, W, C)
            original_image = self.dicom_manager.read_frames_from_dicom(input_path)
            original_dims = (original_image.shape[1], original_image.shape[2]) # (height, width)

            # Skip single frame if requested
            if (self.config.skip_single_frame and len(original_image.shape) == 3
                and original_image.shape[0] == 1):
                self.logger.info(f"Skipping single frame: {input_path}")
                return ProcessingResult(
                    success=True, skipped=True, input_path=input_path,
                    output_path=final_output_path, processing_time=time.time() - start_time
                )

            # Generate mask (if enabled)
            curvilinear_mask = None
            mask_config = None
            predicted_corners = None

            if not self.config.no_mask_generation and self.model is not None:
                # Run AI inference to get predicted corners
                coords_normalized = self._run_inference(original_image)
                predicted_corners = self._denormalize_coords(coords_normalized, original_dims)

                # Generate mask only if not in PHI-only mode
                if not self.config.phi_only_mode:
                    curvilinear_mask, mask_config = compute_masks_and_configs(
                        original_dims=original_dims,
                        predicted_corners=predicted_corners
                    )

            # Apply mask to image (only if both PHI-only mode and remove_phi_from_image are enabled)
            if self.config.phi_only_mode and self.config.remove_phi_from_image:
                # In PHI-only mode, start with original image (no fan mask)
                masked_image_array = original_image.copy()
            elif not self.config.phi_only_mode:
                # Normal mode: apply fan mask (regardless of remove_phi_from_image setting)
                masked_image_array = self._apply_mask(original_image, curvilinear_mask)
            else:
                # PHI-only mode is enabled but remove_phi_from_image is disabled - use original image
                masked_image_array = original_image.copy()

            # Apply top redaction if enabled (only in PHI-only mode with remove_phi_from_image enabled)
            if (self.config.phi_only_mode and self.config.remove_phi_from_image and
                self.config.top_ratio > 0 and predicted_corners is not None):
                masked_image_array = self._apply_top_redaction(
                    masked_image_array, predicted_corners, self.config.top_ratio
                )

                # Store data for PDF generation (will be processed at the end)
                if headers_folder:
                    os.makedirs(headers_folder, exist_ok=True)
                    filename = os.path.splitext(row.AnonFilename)[0]

                    # Store PDF data for later batch processing
                    # Determine the correct subdirectory within headers tree to match the output structure
                    if self.config.preserve_directory_structure:
                        # Extract the relative path from the output path to determine headers subdirectory
                        relative_path = os.path.relpath(final_output_path, output_folder)
                        relative_dir = os.path.dirname(relative_path)
                        if relative_dir:
                            headers_subdir = os.path.join(headers_folder, relative_dir)
                        else:
                            headers_subdir = headers_folder
                    else:
                        # Flat structure - use headers folder directly
                        headers_subdir = headers_folder

                    if not hasattr(self, '_pdf_data'):
                        self._pdf_data = {}

                    if headers_subdir not in self._pdf_data:
                        self._pdf_data[headers_subdir] = {
                            'files': []
                        }

                    self._pdf_data[headers_subdir]['files'].append({
                        'original_image': original_image,
                        'redacted_image': masked_image_array,
                        'predicted_corners': predicted_corners,
                        'filename': filename,
                        'top_ratio': self.config.top_ratio
                    })

            # Save anonymized DICOM
            anon_filename = row.AnonFilename
            new_patient_name = os.path.splitext(anon_filename)[0]
            new_patient_id = anon_filename.split('_')[0]

            os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
            self.dicom_manager.save_anonymized_dicom(
                image_array=masked_image_array,
                output_path=final_output_path,
                new_patient_name=new_patient_name,
                new_patient_id=new_patient_id,
                labels=None
            )

            # Save header
            if headers_folder:
                self.dicom_manager.save_anonymized_dicom_header(
                    current_dicom_record=row,
                    output_filename=anon_filename,
                    headers_directory=headers_folder
                )

            # Save sequence info JSON
            self._save_sequence_info(final_output_path, row, mask_config)

            # Generate overview if callback provided
            if overview_callback and original_image.shape[0] > 0:
                overview_callback(row.AnonFilename, original_image, masked_image_array, curvilinear_mask, None)

            processing_time = time.time() - start_time
            self.logger.info(f"Successfully processed: {input_path} -> {final_output_path} ({processing_time:.2f}s)")

            return ProcessingResult(
                success=True, skipped=False, input_path=input_path,
                output_path=final_output_path, processing_time=processing_time,
                metrics=None
            )

        except Exception as e:
            error_msg = f"Failed to process {input_path}: {str(e)}"
            self.logger.error(error_msg)
            return ProcessingResult(
                success=False, skipped=False, input_path=input_path,
                output_path="", error_message=error_msg,
                processing_time=time.time() - start_time
            )

    def generate_all_pdfs(self):
        """Generate PDFs for all collected data (call this after processing all files)"""
        if not (self.config.phi_only_mode and self.config.remove_phi_from_image):
            self.logger.info("Skipping PDF generation - PHI-only mode or remove_phi_from_image is disabled")
            return

        if not hasattr(self, '_pdf_data') or not self._pdf_data:
            return

        for headers_subdir, data in self._pdf_data.items():
            files = data['files']

            if not files:
                continue

            # Create PDF in the headers subdirectory where *_DicomHeader.json files are saved
            os.makedirs(headers_subdir, exist_ok=True)
            pdf_path = os.path.join(headers_subdir, "redaction.pdf")

            # Generate PDF with all files for this headers directory
            with PdfPages(pdf_path) as pdf:
                for file_data in files:
                    self._add_page_to_pdf(
                        pdf,
                        file_data['original_image'],
                        file_data['redacted_image'],
                        file_data['predicted_corners'],
                        file_data['top_ratio'],
                        file_data['filename'],
                        frame_idx=0
                    )

            self.logger.info(f"Generated redaction PDF with {len(files)} pages: {pdf_path}")

        # Clear the PDF data after generation
        self._pdf_data = {}

    def _convert_numpy_float_to_python_float(self, coords: Dict[str, float]) -> Dict[str, float]:
        """
        Convert numpy float to python float and round to 2 decimal places
        :param coords: dictionary of predicted corners with numpy floats
        :return: dictionary of predicted corners with python float
        """
        coords_dict = {
            "upper_left_x": float(coords['upper_left'][0]),
            "upper_left_y": float(coords['upper_left'][1]),
            "upper_right_x": float(coords['upper_right'][0]),
            "upper_right_y": float(coords['upper_right'][1]),
            "lower_left_x": float(coords['lower_left'][0]),
            "lower_left_y": float(coords['lower_left'][1]),
            "lower_right_x": float(coords['lower_right'][0]),
            "lower_right_y": float(coords['lower_right'][1]),
        }

        return self._round_metrics_to_decimal_places(coords_dict, 2)

    def _run_inference(self, original_image: np.ndarray) -> np.ndarray:
        """
        Run AI inference to predict corners
        :param original_image: 4D image array (N, H, W, C)
        :return: 2D array of predicted corners (4, 2)
        """
        input_tensor = preprocess_image(original_image)
        with torch.no_grad():
            assert self.model is not None
            assert self.device is not None
            coords_normalized = self.model(input_tensor.to(self.device)).cpu().numpy()
        return coords_normalized

    def _denormalize_coords(self, coords_normalized: np.ndarray, original_dims: Tuple[int, int]) -> Dict[str, Tuple[np.float32, np.float32]]:
        """
        Convert normalized coordinates to pixel coordinates
        :param coords_normalized: 2D array of predicted corners (4, 2)
        :param original_dims: tuple of original image dimensions (height, width)
        :return: dictionary of predicted corners
        """
        coords = coords_normalized.reshape(4, 2)
        coords[:, 0] *= original_dims[1]  # width
        coords[:, 1] *= original_dims[0]  # height

        return {
            "upper_left": tuple(coords[0]),
            "upper_right": tuple(coords[1]),
            "lower_left": tuple(coords[2]),
            "lower_right": tuple(coords[3]),
        }

    def _apply_mask(self, original_image: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Apply mask to image sequence
        :param original_image: 4D image array (N, H, W, C)
        :param mask: 2D binary mask (H, W)
        :return: 4D image array (N, H, W, C)
        """
        if mask is not None:
            # Apply mask to all frames/channels
            masked = original_image.copy()
            mask_broadcast = mask[np.newaxis, :, :, np.newaxis]
            return masked * mask_broadcast
        else:
            return original_image

    def _apply_top_redaction(self, image_array: np.ndarray, predicted_corners: Dict[str, Tuple[int, int]], top_ratio: float) -> np.ndarray:
        """
        Apply top redaction to image array using predicted corners as hints.

        Args:
            image_array: Image array (N, H, W, C)
            predicted_corners: Dictionary with corner points
            top_ratio: Ratio of image height to redact from top (0.0-1.0)

        Returns:
            Image array with top redaction applied
        """
        if top_ratio <= 0:
            return image_array

        result = image_array.copy()
        height = image_array.shape[1]

        # Calculate base redaction height
        base_redaction_height = int(height * top_ratio)

        # Get top points from predicted corners
        top_left = predicted_corners.get('upper_left')
        top_right = predicted_corners.get('upper_right')

        # Find the highest Y coordinate among top points
        highest_y = self._get_highest_y_from_top_points(top_left, top_right)

        # Calculate final redaction height considering top points and image bounds
        final_redaction_height = self._calculate_final_redaction_height(base_redaction_height, highest_y, height)

        # Apply redaction to each frame
        for i in range(image_array.shape[0]):
            # Ensure we have valid bounds for slicing
            y_start = 0
            y_end = min(final_redaction_height, height)

            if y_start < y_end:
                if image_array.shape[-1] == 1:  # Grayscale
                    result[i, y_start:y_end, :, 0] = 0
                else:  # RGB
                    for c in range(image_array.shape[-1]):
                        result[i, y_start:y_end, :, c] = 0

        self.logger.info(f"Applied top redaction: {final_redaction_height}px (base: {base_redaction_height}px, top points: {highest_y}px)")
        return result


    def _normalize_image(self, img):
        """Normalize image to 0-1 range for display (from redact_dicom_image_phi.py)"""
        img = img.astype("float32", copy=False)
        if img.ndim == 2:
            lo, hi = np.percentile(img, [1, 99])
            if hi <= lo:
                lo, hi = float(img.min()), float(img.max())
            img = (img - lo) / max(1e-6, (hi - lo))
        else:
            lo = img.min(axis=(0,1), keepdims=True)
            hi = img.max(axis=(0,1), keepdims=True)
            img = (img - lo) / np.maximum(hi - lo, 1e-6)
        return np.clip(img, 0, 1)

    def _get_highest_y_from_top_points(self, top_left: Optional[tuple], top_right: Optional[tuple]) -> int:
        """Calculate the highest Y coordinate from top left and top right points"""
        highest_y = 0
        if top_left and len(top_left) >= 2:
            try:
                highest_y = max(highest_y, int(top_left[1]))
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid top_left coordinate: {top_left}")
        if top_right and len(top_right) >= 2:
            try:
                highest_y = max(highest_y, int(top_right[1]))
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid top_right coordinate: {top_right}")
        return highest_y

    def _calculate_final_redaction_height(self, base_redaction_height: int, highest_y: int, image_height: int) -> int:
        """Calculate the final redaction height considering top points and image bounds"""
        # Use top points as upper limit - don't redact past the highest top point
        if highest_y > 0:
            # If we have top points, use the minimum of base redaction and top points
            final_redaction_height = min(base_redaction_height, highest_y)
        else:
            # If no top points detected, use base redaction height
            final_redaction_height = base_redaction_height

        # Ensure we don't redact more than the image height
        final_redaction_height = min(final_redaction_height, image_height - 1)
        return int(final_redaction_height)

    def _add_page_to_pdf(self, pdf, original_image, redacted_image, predicted_corners, top_ratio, filename, frame_idx=0):
        """Add a single page to the PDF with original, redacted, and diff images"""
        # Use only the first frame since redaction is the same across all frames
        orig_frame = original_image[frame_idx]
        redacted_frame = redacted_image[frame_idx]

        # Normalize images for display (prevents matplotlib clipping warnings)
        orig_norm = self._normalize_image(orig_frame)
        redacted_norm = self._normalize_image(redacted_frame)

        # Calculate difference using normalized images (matching redact_dicom_image_phi.py approach)
        if orig_norm.shape == redacted_norm.shape:
            diff_frame = np.abs(orig_norm - redacted_norm)
        else:
            # Handle size mismatches
            h = min(orig_norm.shape[0], redacted_norm.shape[0])
            w = min(orig_norm.shape[1], redacted_norm.shape[1])
            orig_crop = orig_norm[:h, :w] if orig_norm.ndim == 2 else orig_norm[:h, :w, ...]
            redacted_crop = redacted_norm[:h, :w] if redacted_norm.ndim == 2 else redacted_norm[:h, :w, ...]
            diff_frame = np.abs(orig_crop - redacted_crop)

        # Prepare display versions
        if orig_norm.ndim == 2:
            orig_frame_2d = orig_norm
            redacted_frame_2d = redacted_norm
        else:
            orig_frame_2d = orig_norm
            redacted_frame_2d = redacted_norm

        # Calculate redaction parameters
        height = original_image.shape[1]
        base_redaction_height = int(height * top_ratio)

        top_left = predicted_corners.get('upper_left')
        top_right = predicted_corners.get('upper_right')
        highest_y = self._get_highest_y_from_top_points(top_left, top_right)

        # Calculate final redaction height considering top points and image bounds
        final_redaction_height = self._calculate_final_redaction_height(base_redaction_height, highest_y, height)

        # Create figure with 3 subplots: Original, Redacted, Diff
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{filename} - Frame {frame_idx + 1}', fontsize=14)

        # Original image (using normalized data)
        if orig_frame_2d.ndim == 2:
            axes[0].imshow(orig_frame_2d, cmap='gray')
        else:
            axes[0].imshow(orig_frame_2d)

        axes[0].set_title('Original')
        axes[0].axis('off')

        # Add top redaction visualization
        rect = patches.Rectangle((0, 0), orig_frame_2d.shape[1], final_redaction_height,
                               linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
        axes[0].add_patch(rect)

        # Add top points if available (using different colors)
        if top_left:
            axes[0].plot(top_left[0], top_left[1], 'ro', markersize=8, label='Top Left')  # Red circle
        if top_right:
            axes[0].plot(top_right[0], top_right[1], 'bo', markersize=8, label='Top Right')  # Blue circle

        if top_left or top_right:
            axes[0].legend()

        # Redacted image (using normalized data)
        if redacted_frame_2d.ndim == 2:
            axes[1].imshow(redacted_frame_2d, cmap='gray')
        else:
            axes[1].imshow(redacted_frame_2d)

        axes[1].set_title('Redacted')
        axes[1].axis('off')

        # Diff image
        if diff_frame.ndim == 2:
            axes[2].imshow(diff_frame, cmap='hot')
        else:
            axes[2].imshow(diff_frame)

        axes[2].set_title('Difference')
        axes[2].axis('off')

        # Add text with redaction info
        info_text = f'Top Ratio: {top_ratio:.1%}\nBase Height: {base_redaction_height}px\nFinal Height: {final_redaction_height}px'
        if top_left or top_right:
            info_text += f'\nTop Points: {highest_y}px'

        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


    def _save_sequence_info(self, final_output_path: str, row, mask_config: Optional[dict]):
        """Save sequence info JSON"""
        if not self.config.no_mask_generation and not self.config.phi_only_mode:
            sequence_info = {
                'SOPInstanceUID': getattr(row.DICOMDataset, 'SOPInstanceUID', 'None') or 'None',
                'GrayscaleConversion': False
            }
            if mask_config is not None:
                sequence_info['MaskConfig'] = mask_config

            json_path = final_output_path.replace(".dcm", ".json")
            with open(json_path, 'w') as f:
                json.dump(sequence_info, f, indent=2)

    def _load_ground_truth_mask_config(self, anon_filename: str, predicted_mask_config: Optional[dict],
                        original_dims: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """
        Compute segmentation metrics by comparing predicted mask with ground truth.

        This method leverages the existing evaluation functions to compute comprehensive
        segmentation metrics including Dice coefficient, IoU, precision, recall, etc.

        Args:
            anon_filename: Anonymized filename to find corresponding ground truth
            predicted_mask_config: Predicted mask configuration from AI model
            original_dims: Original image dimensions (height, width)

        Returns:
            Dictionary containing computed metrics or None if no ground truth available
        """
        # Early return if no ground truth directory or predicted mask config
        if not self.gt_manager.gt_index or not predicted_mask_config:
            return None

        try:
            # Find ground truth file for this anonymized filename
            gt_config_path = self.gt_manager.get_ground_truth_path(anon_filename)

            if not gt_config_path:
                self.logger.debug(f"No ground truth found for {anon_filename}")
                return None

            # Load ground truth mask configuration using existing function
            gt_mask_config = load_mask_config(gt_config_path)

            return gt_mask_config
        except Exception as e:
            self.logger.warning(f"Failed to load ground truth mask configuration for {anon_filename}: {e}")
            return None
    def _json_to_csv_columns(self, json_data: dict, prefix: str) -> dict:
        """
        Convert JSON object to CSV columns with prefix.

        Args:
            json_data: Dictionary containing JSON data to convert
            prefix: Prefix to add to column names (e.g., 'gt', 'predicted')

        Returns:
            Dictionary with flattened keys and processed values
        """
        csv_columns = {}

        # Handle nested MaskConfig if present (for backward compatibility)
        if "MaskConfig" in json_data:
            # Flatten MaskConfig into parent level
            mask_config = json_data.pop("MaskConfig")
            json_data.update(mask_config)

        for key, value in json_data.items():
            # Convert key to snake_case and add prefix
            csv_key = f"{prefix}_{self._camel_to_snake(key)}"

            # Process different value types
            if isinstance(value, (float, np.floating)):
                # Round floats to 2 decimal places
                csv_columns[csv_key] = round(float(value), 2)
            elif isinstance(value, (int, np.integer)):
                # Keep integers as-is
                csv_columns[csv_key] = int(value)
            elif isinstance(value, bool):
                # Convert boolean to lowercase string
                csv_columns[csv_key] = str(value).lower()
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                csv_columns[csv_key] = ", ".join(str(item) for item in value)
            else:
                # Keep strings and other types as-is
                csv_columns[csv_key] = str(value)

        return csv_columns

    def _camel_to_snake(self, camel_str: str) -> str:
        """
        Convert CamelCase to snake_case.

        Args:
            camel_str: String in CamelCase format

        Returns:
            String in snake_case format
        """
        # Insert underscore before uppercase letters and convert to lowercase
        import re
        snake_str = re.sub('([a-z0-9])([A-Z])', r'\1_\2', camel_str)
        return snake_str.lower()

    def _compute_metrics_with_ground_truth(self,
                        anon_filename: str, gt_mask_config: Optional[dict], original_dims: Tuple[int, int],
                        gt_corners: Optional[dict] = None, predicted_mask_config: Optional[dict] = None,
                        predicted_corners: Optional[dict] = None) -> Optional[Dict[str, Any]]:
        """
        Compute segmentation metrics by comparing predicted mask with ground truth.

        This method leverages the existing evaluation functions to compute comprehensive
        segmentation metrics including Dice coefficient, IoU, precision, recall, etc.

        Args:
            anon_filename: Anonymized filename to find corresponding ground truth
            predicted_mask_config: Predicted mask configuration from AI model
            original_dims: Original image dimensions (height, width)
            gt_corners: Ground truth corners
            predicted_corners: Predicted corners

        Returns:
            Dictionary containing computed metrics or None if no ground truth available
        """
        # Early return if missing any required parameters
        if not self.gt_manager.gt_index or not predicted_mask_config or not original_dims or not gt_corners or not predicted_corners:
            return None

        try:
            # Find ground truth file for this anonymized filename
            gt_config_path = self.gt_manager.get_ground_truth_path(anon_filename)

            if not gt_config_path:
                self.logger.debug(f"No ground truth found for {anon_filename}")
                return None

            # Load ground truth mask configuration using existing function
            gt_mask_config = load_mask_config(gt_config_path)

            # Compute metrics using existing comparison function
            metrics = compare_masks(gt_mask_config, predicted_mask_config, gt_corners, predicted_corners, original_dims)

            # Round floating point values to 2 decimal places
            metrics = self._round_metrics_to_decimal_places(metrics, decimal_places=2)

            metrics['filename'] = anon_filename
            metrics['ground_truth_filename'] = os.path.basename(gt_config_path)
            metrics['dicom_input_path'] = self.dicom_manager.dicom_df.loc[self.dicom_manager.current_index, 'InputPath']
            metrics['ground_truth_path'] = gt_config_path

            # Convert JSON objects to CSV columns with prefixes
            if gt_mask_config:
                gt_csv_columns = self._json_to_csv_columns(gt_mask_config.copy(), 'gt')
                metrics.update(gt_csv_columns)

            if predicted_mask_config:
                predicted_csv_columns = self._json_to_csv_columns(predicted_mask_config.copy(), 'pred')
                metrics.update(predicted_csv_columns)

            # Handle corners data
            if gt_corners:
                gt_corners_csv = self._json_to_csv_columns(
                    self._convert_numpy_float_to_python_float(gt_corners), 'gt_corners'
                )
                metrics.update(gt_corners_csv)

            if predicted_corners:
                predicted_corners_csv = self._json_to_csv_columns(
                    self._convert_numpy_float_to_python_float(predicted_corners), 'pred_corners'
                )
                metrics.update(predicted_corners_csv)

            if gt_corners and predicted_corners:
                metrics['angle_upper_left'] = self.calculate_angle_degrees(gt_corners['upper_left'][0], gt_corners['upper_left'][1], predicted_corners['upper_left'][0], predicted_corners['upper_left'][1])
                metrics['angle_upper_right'] = self.calculate_angle_degrees(gt_corners['upper_right'][0], gt_corners['upper_right'][1], predicted_corners['upper_right'][0], predicted_corners['upper_right'][1])
                metrics['angle_lower_left'] = self.calculate_angle_degrees(gt_corners['lower_left'][0], gt_corners['lower_left'][1], predicted_corners['lower_left'][0], predicted_corners['lower_left'][1])
                metrics['angle_lower_right'] = self.calculate_angle_degrees(gt_corners['lower_right'][0], gt_corners['lower_right'][1], predicted_corners['lower_right'][0], predicted_corners['lower_right'][1])

            return metrics

        except Exception as e:
            self.logger.warning(f"Failed to compute metrics for {anon_filename}: {e}")
            return None

    def calculate_angle_degrees(self, ground_truth_x: float, ground_truth_y: float, predicted_x: float, predicted_y: float, cv_convention: bool = False) -> float:
        """
        Calculate the angle between two points in degrees.
        :param ground_truth_x: x coordinate of the ground truth point
        :param ground_truth_y: y coordinate of the ground truth point
        :param predicted_x: x coordinate of the predicted point
        :param predicted_y: y coordinate of the predicted point
        :param cv_convention: whether to apply computer vision convention (Y-axis flipped)
        :return: angle in degrees
        """

        # Translate coordinates to make ground truth the origin
        dx = predicted_x - ground_truth_x
        dy = predicted_y - ground_truth_y

        if dx == 0 and dy == 0:
            return 0.0

        # Apply computer vision convention if needed (Y-axis flipped)
        if cv_convention:
            dy = -dy

        # Calculate angle in radians
        angle_rad = math.atan2(dy, dx)

        # Convert to degrees
        angle_deg = math.degrees(angle_rad)

        if angle_deg < 0:
            angle_deg += 360

        return round(angle_deg, 2)

    def _round_metrics_to_decimal_places(self, metrics: Dict[str, Any], decimal_places: int = 2) -> Dict[str, Any]:
        """
        Round floating point values in metrics dictionary to specified decimal places.

        Args:
            metrics: Dictionary containing metrics with potential floating point values
            decimal_places: Number of decimal places to round to (default: 2)

        Returns:
            Dictionary with floating point values rounded to specified decimal places
        """
        rounded_metrics = {}

        for key, value in metrics.items():
            if isinstance(value, (float, np.floating)):
                # Round floating point numbers to specified decimal places
                rounded_metrics[key] = round(float(value), decimal_places)
            elif isinstance(value, (int, np.integer)):
                # Keep integers as-is (like image_height, image_width)
                rounded_metrics[key] = int(value)
            else:
                # Keep non-numeric values as-is (strings, etc.)
                rounded_metrics[key] = value

        return rounded_metrics

    def get_evaluate_fieldnames(self) -> List[str]:
        """
        Get standard fieldnames for metrics CSV output including flattened JSON columns.
        """
        base_fields = [
            "filename",
            "ground_truth_filename",
            "dicom_input_path",
            "ground_truth_path",
        ]

        # Ground truth config fields
        gt_config_fields = [
            "gt_sopinstance_uid",
            "gt_grayscale_conversion",
            "gt_mask_type",
            "gt_angle1",
            "gt_angle2",
            "gt_center_rows_px",
            "gt_center_cols_px",
            "gt_radius1",
            "gt_radius2",
            "gt_image_size_rows",
            "gt_image_size_cols",
            "gt_annotation_labels",
        ]

        # Predicted config fields (same structure with different prefix)
        predicted_config_fields = [field.replace("gt_", "pred_") for field in gt_config_fields]

        # Corner fields (if needed)
        corner_fields = [
            "gt_corners_upper_left_x", "gt_corners_upper_left_y", "gt_corners_upper_right_x", "gt_corners_upper_right_y", "gt_corners_lower_left_x", "gt_corners_lower_left_y", "gt_corners_lower_right_x", "gt_corners_lower_right_y",
            "pred_corners_upper_left_x", "pred_corners_upper_left_y", "pred_corners_upper_right_x", "pred_corners_upper_right_y", "pred_corners_lower_left_x", "pred_corners_lower_left_y", "pred_corners_lower_right_x", "pred_corners_lower_right_y",
        ]

        corner_angle_fields = [
            "angle_upper_left", "angle_upper_right", "angle_lower_left", "angle_lower_right",
        ]

        # Metric fields
        metric_fields = [
            "dice_mean",
            "iou_mean",
            "mean_distance_error",
            "upper_left_error",
            "upper_right_error",
            "lower_left_error",
            "lower_right_error",
        ]

        return base_fields + gt_config_fields + predicted_config_fields + corner_fields + corner_angle_fields + metric_fields

    def generate_overview_pdf(self, overview_manifest: List[Dict[str, Any]], output_dir: str) -> str:
        """Generate overview PDF using OverviewGenerator"""
        from .overview_generator import OverviewGenerator

        if not overview_manifest:
            self.logger.warning("No overview images to include in PDF")
            return ""

        try:
            generator = OverviewGenerator(output_dir)
            pdf_path = generator.generate_overview_pdf(overview_manifest, output_dir)
            self.logger.info(f"Generated overview PDF: {pdf_path}")
            return pdf_path
        except Exception as e:
            self.logger.error(f"Failed to generate overview PDF: {e}")
            return ""
