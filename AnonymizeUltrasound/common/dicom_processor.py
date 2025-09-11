from typing import Optional, Dict, Any, Tuple, Callable, List
import logging
import numpy as np
import os
import json
import time
from dataclasses import dataclass
import torch

from .dicom_file_manager import DicomFileManager
from .inference import load_model, preprocess_image
from .masking import compute_masks_and_configs, mask_config_to_corner_points
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
            # 1. Check if should skip (resume mode)
            final_output_path = self.dicom_manager.generate_output_filepath(
                output_folder, row.OutputPath, self.config.preserve_directory_structure
            )

            if self.config.resume_anonymization and os.path.exists(final_output_path):
                self.logger.info(f"Skipping existing file: {final_output_path}")
                return ProcessingResult(
                    success=True, skipped=True, input_path=input_path,
                    output_path=final_output_path, processing_time=time.time() - start_time
                )

            if progress_callback:
                progress_callback(f"Processing: {input_path}")

            # 2. Read DICOM frames and returns numpy array (N, H, W, C)
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

            # 3. Generate mask (if enabled)
            curvilinear_mask = None
            mask_config = None
            predicted_corners = None

            if not self.config.no_mask_generation and self.model is not None:
                # Run AI inference
                coords_normalized = self._run_inference(original_image)
                predicted_corners = self._denormalize_coords(coords_normalized, original_dims)

                # Generate mask
                curvilinear_mask, mask_config = compute_masks_and_configs(
                    original_dims=original_dims,
                    predicted_corners=predicted_corners
                )

            # 4. Apply mask to image
            masked_image_array = self._apply_mask(original_image, curvilinear_mask)

            # 5. Save anonymized DICOM
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

            # 6. Save header
            if headers_folder:
                self.dicom_manager.save_anonymized_dicom_header(
                    current_dicom_record=row,
                    output_filename=anon_filename,
                    headers_directory=headers_folder
                )

            # 7. Save sequence info JSON
            self._save_sequence_info(final_output_path, row, mask_config)

            # 8. Load ground truth mask config if available
            metrics = {}
            gt_mask_config = self._load_ground_truth_mask_config(anon_filename, mask_config, original_dims)
            if gt_mask_config is not None:
                metrics['ground_truth_config_json'] = json.dumps(gt_mask_config)

            # 9. Add predicted corners to metrics for CSV export
            if isinstance(predicted_corners, dict):
                metrics['predicted_corners_json'] = json.dumps(self._convert_numpy_float_to_python_float(predicted_corners))

            # 10. Generate overview if callback provided
            if overview_callback and original_image.shape[0] > 0:
                overview_callback(row.AnonFilename, original_image, masked_image_array, curvilinear_mask, metrics)

            processing_time = time.time() - start_time
            self.logger.info(f"Successfully processed: {input_path} -> {final_output_path} ({processing_time:.2f}s)")

            return ProcessingResult(
                success=True, skipped=False, input_path=input_path,
                output_path=final_output_path, processing_time=processing_time,
                metrics=metrics
            )

        except Exception as e:
            error_msg = f"Failed to process {input_path}: {str(e)}"
            self.logger.error(error_msg)
            return ProcessingResult(
                success=False, skipped=False, input_path=input_path,
                output_path="", error_message=error_msg,
                processing_time=time.time() - start_time
            )

    def _convert_numpy_float_to_python_float(self, coords: Dict[str, Tuple[np.float32, np.float32]]) -> Dict[str, Tuple[float, float]]:
        """
        Convert numpy float to python float
        :param coords: dictionary of predicted corners with numpy floats
        :return: dictionary of predicted corners with python float
        """
        return {
            "upper_left": (float(coords['upper_left'][0]), float(coords['upper_left'][1])),
            "upper_right": (float(coords['upper_right'][0]), float(coords['upper_right'][1])),
            "lower_left": (float(coords['lower_left'][0]), float(coords['lower_left'][1])),
            "lower_right": (float(coords['lower_right'][0]), float(coords['lower_right'][1])),
        }

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

    def _save_sequence_info(self, final_output_path: str, row, mask_config: Optional[dict]):
        """Save sequence info JSON"""
        if not self.config.no_mask_generation:
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

            # Add ground truth path to metrics for tracking
            metrics['ground_truth_path'] = gt_config_path
            metrics['ground_truth_config_json'] = json.dumps(gt_mask_config)
            metrics['ground_truth_corners_json'] = json.dumps(self._convert_numpy_float_to_python_float(gt_corners)) if gt_corners else None
            metrics['predicted_config_json'] = json.dumps(predicted_mask_config)
            metrics['predicted_corners_json'] = json.dumps(self._convert_numpy_float_to_python_float(predicted_corners)) if predicted_corners else None

            return metrics

        except Exception as e:
            self.logger.warning(f"Failed to compute metrics for {anon_filename}: {e}")
            return None

    def get_evaluate_fieldnames(self) -> List[str]:
        """
        Get standard fieldnames for metrics CSV output.

        This provides a consistent set of field names that can be used
        across different implementations (CLI vs Slicer).

        Returns:
            List of field names for metrics CSV
        """
        return [
            "dicom_input_path",
            "ground_truth_config_path",
            "ground_truth_config_json",
            "ground_truth_corners_json",
            "predicted_config_json",
            "predicted_corners_json",
            "dice_mean",
            "iou_mean",
            "pixel_accuracy_mean"
        ]

    def get_metrics_fieldnames(self) -> List[str]:
        """
        Get standard fieldnames for metrics CSV output.

        This provides a consistent set of field names that can be used
        across different implementations (CLI vs Slicer).

        Returns:
            List of field names for metrics CSV
        """
        return [
            "dicom_output_path",
            "dicom_filename",
            "predicted_config_json",
            "predicted_corners_json",
        ]

    def format_metrics_for_csv(self, result: ProcessingResult) -> Dict[str, Any]:
        """
        Format ProcessingResult metrics for CSV output.

        Args:
            result: ProcessingResult containing metrics data

        Returns:
            Dictionary formatted for CSV writing
        """
        metrics = result.metrics or {}

        return {
            "dicom_output_path": result.output_path,
            "dicom_filename": os.path.basename(result.output_path) if result.output_path else "",
            "predicted_config_json": metrics.get("predicted_config_json", ""),
            "predicted_corners_json": metrics.get("predicted_corners_json", ""),
        }

    def format_evaluate_metrics_for_csv(self, result: ProcessingResult) -> Dict[str, Any]:
        """
        Format ProcessingResult metrics for CSV output.
        """
        metrics = result.metrics or {}
        return {
            "dicom_input_path": result.input_path,
            "ground_truth_config_path": metrics.get("ground_truth_path", ""),
            "ground_truth_config_json": metrics.get("ground_truth_config_json", ""),
            "ground_truth_corners_json": metrics.get("ground_truth_corners_json", ""),
            "predicted_config_json": metrics.get("predicted_config_json", ""),
            "predicted_corners_json": metrics.get("predicted_corners_json", ""),
            "dice_mean": metrics.get("dice_mean", ""),
            "iou_mean": metrics.get("iou_mean", ""),
            "pixel_accuracy_mean": metrics.get("pixel_accuracy_mean", ""),
        }

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
