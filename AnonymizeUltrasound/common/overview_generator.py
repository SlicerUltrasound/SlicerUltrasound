import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import logging

class OverviewGenerator:
    """Generates overview images for anonymization results"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def generate_overview(
        self,
        filename: str,
        original_image: np.ndarray,
        masked_image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        ground_truth_config: Optional[Dict[str, Any]] = None,
        predicted_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate overview image comparing original vs anonymized"""

        # Validate input has frames
        if original_image.shape[0] == 0:
            raise ValueError("No frames available in original_image")
        if masked_image.shape[0] == 0:
            raise ValueError("No frames available in masked_image")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
        fig.patch.set_facecolor('white')

        # Set individual axes backgrounds to white
        for ax in axes:
            ax.set_facecolor('white')

        axes[0].set_title('Original')
        axes[1].set_title('Mask Outline')

        # Create max-pooled snapshots (same as AI model preprocessing)
        # This ensures we're showing the same "background" image that the AI model analyzed
        orig_frame = self._create_snapshot(original_image)

        # Display original snapshot
        axes[0].imshow(orig_frame.squeeze(), cmap='gray')
        axes[0].axis('off')

        # Display original snapshot with mask outline
        axes[1].imshow(orig_frame.squeeze(), cmap='gray')
        if mask is not None:
            axes[1].contour(mask, levels=[0.5], colors='lime', linewidths=1.0)
        else:
            axes[1].text(0.5, 0.5, 'No mask', ha='center', va='center',
                        transform=axes[1].transAxes, color='red')
        axes[1].axis('off')


        # If ground truth and predicted configs are provided, show comparison, otherwise show anonymized
        if ground_truth_config is not None and predicted_config is not None:
            axes[2].set_title('GT (Blue) vs Predicted (Red)')
            comparison_image = self._create_mask_comparison_image(
                original_image, ground_truth_config, predicted_config
            )
            axes[2].imshow(comparison_image)
        else:
            axes[2].set_title('Anonymized')
            masked_frame = self._create_snapshot(masked_image)

            if orig_frame.shape != masked_frame.shape:
                raise ValueError("Original and masked frame shapes do not match")

            axes[2].imshow(masked_frame)

        axes[2].axis('off')

        # Save overview
        overview_filename = f"{os.path.splitext(filename)[0]}.png"
        overview_path = os.path.join(self.output_dir, overview_filename)
        plt.tight_layout()
        plt.savefig(overview_path, dpi=300, bbox_inches='tight', pad_inches=0.05, facecolor='white')
        plt.close(fig)

        return overview_path

    def _create_mask_comparison_image(self,
        original_image: np.ndarray,
        ground_truth_config: Optional[dict],
        predicted_config: Optional[dict]
        ) -> np.ndarray:
        """
        Create a comparison image showing ground truth mask in blue,
        predicted mask in red, and overlap in purple.

        Args:
            original_image: Original ultrasound image array (N, H, W, C)
            ground_truth_config: Ground truth mask configuration
            predicted_config: Predicted mask configuration

        Returns:
            RGB image with color-coded mask comparison
        """
        from .masking import create_mask

        # Create snapshot for consistent visualization
        orig_frame = self._create_snapshot(original_image)

        # Get image dimensions
        if orig_frame.ndim == 2:
            height, width = orig_frame.shape
        else:
            height, width = orig_frame.shape[:2]

        # Create ground truth and predicted masks
        gt_mask = create_mask(ground_truth_config, image_size=(height, width))
        pred_mask = create_mask(predicted_config, image_size=(height, width))

        # Convert to binary masks
        gt_binary = (gt_mask > 0).astype(np.uint8)
        pred_binary = (pred_mask > 0).astype(np.uint8)

        # Create RGB comparison image
        # Start with grayscale original image as base
        if orig_frame.ndim == 2:
            # Convert grayscale to RGB
            base_image = np.stack([orig_frame, orig_frame, orig_frame], axis=2)
        else:
            base_image = orig_frame.copy()

        # Ensure base image is in 0-255 range
        if base_image.max() <= 1.0:
            base_image = (base_image * 255).astype(np.uint8)
        else:
            base_image = base_image.astype(np.uint8)

        # Create color overlay
        overlay = base_image.copy().astype(np.float32)

        # Define colors (in RGB)
        blue_color = np.array([0, 0, 255], dtype=np.float32)    # Ground truth
        red_color = np.array([255, 0, 0], dtype=np.float32)     # Predicted
        purple_color = np.array([128, 0, 128], dtype=np.float32) # Overlap

        # Calculate overlap
        overlap_mask = (gt_binary & pred_binary).astype(bool)
        gt_only_mask = (gt_binary & ~pred_binary).astype(bool)
        pred_only_mask = (pred_binary & ~gt_binary).astype(bool)

        # Apply color overlays with transparency
        alpha = 0.4  # Transparency factor

        # Ground truth only (blue)
        if np.any(gt_only_mask):
            overlay[gt_only_mask] = (1 - alpha) * overlay[gt_only_mask] + alpha * blue_color

        # Predicted only (red)
        if np.any(pred_only_mask):
            overlay[pred_only_mask] = (1 - alpha) * overlay[pred_only_mask] + alpha * red_color

        # Overlap (purple)
        if np.any(overlap_mask):
            overlay[overlap_mask] = (1 - alpha) * overlay[overlap_mask] + alpha * purple_color

        # Clip values and convert back to uint8
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return overlay

    def _create_snapshot(self, image_array: np.ndarray) -> np.ndarray:
        """
        Create a max-pooled snapshot from multi-frame image array.
        This replicates the same preprocessing step used by the AI model.

        Args:
            image_array: Multi-frame image array with shape (N, H, W, C)

        Returns:
            Single frame snapshot with shape (H, W, C) or (H, W) for grayscale
        """
        # Validate input shape - should be (N, H, W, C)
        if len(image_array.shape) != 4:
            raise ValueError(f"Expected 4D array (N, H, W, C), got {len(image_array.shape)}D array")

        # Step 1: Max-pool frames to get single frame (same as AI model preprocessing)
        snapshot = image_array.max(axis=0)  # (H, W, C)

        # If single channel, we can optionally squeeze the channel dimension for display
        # but keep it consistent with how the AI model would see it
        if snapshot.shape[2] == 1:
            # For display purposes, we can squeeze to (H, W) for grayscale
            return snapshot.squeeze(axis=2)  # (H, W)
        else:
            # Keep RGB format
            return snapshot  # (H, W, 3)

    def _format_metric(self, value) -> str:
        """Format metric value for display"""
        try:
            return f"{float(value):.3f}"
        except Exception:
            return "N/A"


    def generate_overview_pdf(self, overview_manifest: List[Dict[str, Any]], output_dir: str) -> str:
        """Generate a comprehensive PDF report with metrics tables."""
        from matplotlib.backends.backend_pdf import PdfPages
        from datetime import datetime
        import matplotlib.gridspec as gridspec

        if not overview_manifest:
            raise ValueError("Overview manifest is empty")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_filename = f"overview_report_{timestamp}.pdf"
        overview_pdf_path = os.path.join(output_dir, pdf_filename)

        try:
            with PdfPages(overview_pdf_path) as pdf:
                for item in overview_manifest:
                    if "path" not in item or not os.path.exists(item["path"]):
                        self.logger.warning(f"Skipping item - image not found: {item.get('path', 'N/A')}")
                        continue

                    img = plt.imread(item["path"])

                    # Create figure with structured layout
                    fig = plt.figure(figsize=(11, 8.5), dpi=300)
                    fig.patch.set_facecolor('white')

                    # Create vertical layout: image on top (75% height), table on bottom (25% height)
                    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05,
                                         top=0.95, bottom=0.05, left=0.03, right=0.97)

                    ax_img = fig.add_subplot(gs[0, 0])
                    ax_img.imshow(img, interpolation='nearest')
                    ax_img.axis('off')

                    ax_table = fig.add_subplot(gs[1, 0])
                    ax_table.axis('off')

                    # Prepare metrics data
                    metrics_data = [
                        ["Filename", item.get("filename", "Unknown")],
                        ["Dice Score", f"{item.get('dice', 0):.3f}"],
                        ["IoU", f"{item.get('iou', 0):.3f}"],
                        ["MDE", f"{item.get('mean_distance_error', 0):.3f}"],
                        ["Upper Left Error", f"{item.get('upper_left_error', 0):.3f}"],
                        ["Upper Right Error", f"{item.get('upper_right_error', 0):.3f}"],
                        ["Lower Left Error", f"{item.get('lower_left_error', 0):.3f}"],
                        ["Lower Right Error", f"{item.get('lower_right_error', 0):.3f}"]
                    ]

                    # Create horizontal table layout
                    table = ax_table.table(
                        cellText=[list(row[1] for row in metrics_data)],  # Values only
                        colLabels=[row[0] for row in metrics_data],       # Metrics as column headers
                        cellLoc='center',
                        loc='center',
                        bbox=[0,0,1,1]
                    )

                    # Optimize table appearance for landscape layout
                    table.auto_set_font_size(False)
                    table.set_fontsize(6)
                    table.scale(1, 1.2)

                    pdf.savefig(fig, facecolor='white', bbox_inches='tight', pad_inches=0.05)
                    plt.close(fig)

            return overview_pdf_path

        except Exception as e:
            # Cleanup code remains the same
            if os.path.exists(overview_pdf_path):
                try:
                    os.remove(overview_pdf_path)
                except OSError as cleanup_error:
                    logging.debug(f"Could not remove partial PDF file {overview_pdf_path}: {cleanup_error}")
            raise Exception(f"Failed to create overview PDF: {e}") from e