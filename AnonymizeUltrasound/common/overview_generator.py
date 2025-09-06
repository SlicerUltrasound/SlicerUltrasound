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
        
        orig_frame = original_image[0]
        masked_frame = masked_image[0]
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
        
        # Display original
        axes[0].imshow(orig_frame.squeeze(), cmap='gray')
        axes[0].axis('off')
        
        # Display original with mask outline
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
        
        # Add metrics if available
        if metrics:
            dice_txt = self._format_metric(metrics.get("dice_mean"))
            iou_txt = self._format_metric(metrics.get("iou_mean"))
            metrics_str = f"Dice: {dice_txt}  IoU: {iou_txt}"
            axes[1].text(0.02, 0.98, metrics_str, transform=axes[1].transAxes,
                        fontsize=12, color='yellow', ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
        
        # Save overview
        overview_filename = f"{os.path.splitext(filename)[0]}.png"
        overview_path = os.path.join(self.output_dir, overview_filename)
        plt.tight_layout()
        plt.savefig(overview_path, dpi=300, bbox_inches='tight', pad_inches=0.05, facecolor='white')
        plt.close(fig)
        
        return overview_path

    def _format_metric(self, value) -> str:
        """Format metric value for display"""
        try:
            return f"{float(value):.3f}"
        except Exception:
            return "N/A"

    def generate_overview_pdf(self, overview_manifest: List[Dict[str, Any]], output_dir: str) -> str:
        """
        Generate a comprehensive PDF report from overview PNG images.
        
        Args:
            overview_manifest: List of dictionaries containing overview info
            output_dir: Directory to save the PDF
            
        Returns:
            Path to the generated PDF file
            
        Raises:
            Exception: If PDF generation fails
        """
        from matplotlib.backends.backend_pdf import PdfPages
        from datetime import datetime
        
        if not overview_manifest:
            raise ValueError("Overview manifest is empty")
            
        # Generate timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_filename = f"overview_report_{timestamp}.pdf"
        overview_pdf_path = os.path.join(output_dir, pdf_filename)
        
        try:
            with PdfPages(overview_pdf_path) as pdf:
                for item in overview_manifest:
                    # Validate required fields
                    if "path" not in item or not os.path.exists(item["path"]):
                        self.logger.warning(f"Skipping item - image not found: {item.get('path', 'N/A')}")
                        continue
                        
                    # Load the overview PNG image
                    img = plt.imread(item["path"])
                    
                    # Create full-page landscape figure with high DPI
                    fig = plt.figure(figsize=(11, 8.5), dpi=300)
                    fig.patch.set_facecolor('white')
                    
                    # Add image with minimal margins (1% margin on all sides)
                    ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
                    ax.imshow(img, interpolation='nearest')
                    ax.axis('off')
                    
                    # Add filename overlay at top-left corner
                    filename = item.get("filename", "Unknown")
                    fig.text(
                        0.012, 0.992,  # Position: 1.2% from left, 99.2% from bottom
                        f"{filename}",
                        ha='left', va='top', 
                        fontsize=10,
                        bbox=dict(
                            boxstyle='round,pad=0.2', 
                            facecolor='white', 
                            edgecolor='none', 
                            alpha=0.7
                        )
                    )
                    
                    # Optionally add metrics overlay at top-right corner
                    dice = item.get("dice")
                    iou = item.get("iou")
                    if dice is not None and iou is not None:
                        metrics_text = f"Dice: {dice:.3f}, IoU: {iou:.3f}"
                        fig.text(
                            0.988, 0.992,  # Position: 98.8% from left, 99.2% from bottom
                            metrics_text,
                            ha='right', va='top',
                            fontsize=9,
                            bbox=dict(
                                boxstyle='round,pad=0.2',
                                facecolor='lightblue',
                                edgecolor='none',
                                alpha=0.8
                            )
                        )
                    
                    # Save page to PDF (no bbox_inches to maintain full-page layout)
                    pdf.savefig(fig, facecolor='white')
                    plt.close(fig)
                    
            return overview_pdf_path
            
        except Exception as e:
            # Clean up partial file if it exists
            if os.path.exists(overview_pdf_path):
                try:
                    os.remove(overview_pdf_path)
                except OSError:
                    pass
            raise Exception(f"Failed to create overview PDF: {e}") from e
