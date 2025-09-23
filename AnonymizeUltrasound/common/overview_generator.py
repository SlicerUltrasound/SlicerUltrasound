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
        metrics: Optional[Dict[str, Any]] = None
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
        axes[2].set_title('Anonymized')

        # Create max-pooled snapshots (same as AI model preprocessing)
        # This ensures we're showing the same "background" image that the AI model analyzed
        orig_frame = self._create_snapshot(original_image)
        masked_frame = self._create_snapshot(masked_image)
        
        if orig_frame.shape != masked_frame.shape:
            raise ValueError("Original and masked frame shapes do not match")

        # Convert mask to boolean for processing
        mask2d = None
        if mask is not None:
            mask2d = (mask > 0)

        # Helper function to apply white background outside the fan
        def _with_white_bg(frame, mask2d):
            """Apply white background outside the mask area"""
            if mask2d is None:
                return frame, 'gray'

            if frame.ndim == 2:
                out = frame.copy()
                out[~mask2d] = 255
                return out, 'gray'
            elif frame.ndim == 3 and frame.shape[2] == 3:
                out = frame.copy()
                m3 = np.repeat((~mask2d)[..., None], 3, axis=2)
                out[m3] = 255
                return out, None
            else:
                # single-channel but kept as HxWx1
                out = frame[..., 0].copy()
                out[~mask2d] = 255
                return out, 'gray'

        # Apply white background to masked frame
        masked_disp, cmap2 = _with_white_bg(masked_frame, mask2d)

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

        # Display anonymized with white background
        axes[2].imshow(masked_disp, cmap=cmap2)
        axes[2].axis('off')

        # Save overview
        overview_filename = f"{os.path.splitext(filename)[0]}.png"
        overview_path = os.path.join(self.output_dir, overview_filename)
        plt.tight_layout()
        plt.savefig(overview_path, dpi=300, bbox_inches='tight', pad_inches=0.05, facecolor='white')
        plt.close(fig)

        return overview_path

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
                    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)

                    ax_img = fig.add_subplot(gs[0, 0])
                    ax_img.imshow(img, interpolation='nearest')
                    ax_img.axis('off')

                    ax_table = fig.add_subplot(gs[1, 0])
                    ax_table.axis('off')
                    
                    # # Use gridspec for precise layout control
                    # gs = gridspec.GridSpec(1, 2, width_ratios=[7, 3], 
                    #                     left=0.02, right=0.98, 
                    #                     top=0.95, bottom=0.05,
                    #                     wspace=0.05)
                    
                    # Prepare metrics data
                    filename = item.get("filename", "Unknown")
                    metrics_data = [
                        ["Filename", filename],
                        ["Dice Score", f"{item.get('dice', 0):.3f}"],
                        ["IoU", f"{item.get('iou', 0):.3f}"],
                        ["Mean Distance Error", f"{item.get('mean_distance_error', 0):.3f}"],
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
                        loc='center'
                    )
                    
                    self._style_metrics_table(table, len(metrics_data))
                    
                    pdf.savefig(fig, facecolor='white')
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

    def _style_metrics_table(self, table, num_rows: int):
        """Apply consistent styling to metrics table"""
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)  # Increase row height for better readability
        
        # Style header row
        for col in range(2):
            table[(0, col)].set_facecolor('#2E7D32')  # Dark green
            table[(0, col)].set_text_props(weight='bold', color='white')
            table[(0, col)].set_edgecolor('white')
        
        # Style data rows with alternating colors
        for row in range(1, num_rows + 1):
            color = '#E8F5E8' if row % 2 == 0 else 'white'  # Light green alternating
            for col in range(2):
                table[(row, col)].set_facecolor(color)
                table[(row, col)].set_edgecolor('#CCCCCC')
                
        # Make filename row stand out
        table[(1, 0)].set_text_props(weight='bold')  # Filename label
        table[(1, 1)].set_text_props(weight='bold')  # Filename value