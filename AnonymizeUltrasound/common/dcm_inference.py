
import torch
import numpy as np
from common.create_masks import create_mask, corner_points_to_fan_mask_config
from PIL import Image
import cv2
from typing import Optional

def load_model(model_path: str, device: str = 'cpu'):
    """
    Loads a PyTorch model, handling both traced and non-traced checkpoints.
    
    Args:
        model_path (str): Path to the model file
        device (str): Device to load the model on
        
    Returns:
        torch.nn.Module or torch.jit.ScriptModule: Loaded model
    """
    model = torch.jit.load(model_path, map_location=torch.device(device))
    model.eval()  # Set model to evaluation mode
    model.to(device)  # Move model to device (GPU if available)
    return model
        

def preprocess_image(
    image: np.ndarray, # (N, C, H, W)
    target_size: tuple[int, int] = (240, 320),  # (height, width) - matches training spatial_size
) -> torch.Tensor:
    """
    Preprocess an image to match the EXACT training preprocessing pipeline.
    
    Training pipeline (from configs/models/attention_unet_with_dsnt/train.yaml):
    1. Transposed: indices [2, 0, 1] 
    2. Resized: spatial_size [240, 320]
    3. ToTensord + EnsureTyped: float32
    
    This function replicates that exact sequence.
    """
    # Step 1: Max-pool frames to get single frame
    snapshot = image.max(axis=0)  # (C, H, W)
    
    # Step 2: Convert to grayscale using PIL method (matching training dataset)
    # First transpose to (H, W, C) for PIL processing
    snapshot = np.transpose(snapshot, (1, 2, 0))  # (H, W, C)
    
    # Handle single channel case - squeeze if needed
    if snapshot.shape[2] == 1:
        snapshot_for_pil = snapshot.squeeze(axis=2)  # (H, W)
    else:
        snapshot_for_pil = snapshot
    
    pil_image = Image.fromarray(snapshot_for_pil.astype(np.uint8))
    grayscale_image = pil_image.convert('L')
    snapshot = np.array(grayscale_image)  # (H, W)
    
    # Step 3: Add channel dimension to get (H, W, C) format
    snapshot = np.expand_dims(snapshot, axis=-1)  # (H, W, 1)
    
    # Step 4: Apply Transposed transform [2, 0, 1] - this goes from (H, W, C) to (C, H, W)
    snapshot = np.transpose(snapshot, (2, 0, 1))  # (1, H, W)
    
    # Step 5: Apply Resized transform to spatial_size [240, 320]
    # Since we have (1, H, W), we need to work with (H, W) for cv2.resize
    h, w = snapshot.shape[1], snapshot.shape[2]
    resized = cv2.resize(snapshot[0], (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)  # (240, 320)
    
    # Add channel dimension back: (H, W) -> (1, H, W)
    resized = np.expand_dims(resized, axis=0)  # (1, 240, 320)
    
    # Step 6: Convert to tensor and ensure float32 (EnsureTyped)
    tensor = torch.from_numpy(resized).float()  # (1, 240, 320)
    
    # Step 7: Add batch dimension to get (1, 1, 240, 320)
    tensor = tensor.unsqueeze(0)
    
    return tensor

def compute_masks_and_configs(original_dims, predicted_corners: Optional[dict] = None):
    """
    Compute curvilinear mask and fan mask configuration from predicted corners.
    
    Args:
        original_dims (tuple): Original dimensions of the image (height, width)
        predicted_corners (dict): Dictionary containing predicted corners
            - 'top_left' (tuple): Top-left corner coordinates (x, y)
            - 'top_right' (tuple): Top-right corner coordinates (x, y)
            - 'bottom_left' (tuple): Bottom-left corner coordinates (x, y)
            - 'bottom_right' (tuple): Bottom-right corner coordinates (x, y)
    
    Returns:
        np.ndarray: Curvilinear mask
        dict: Fan mask configuration
    """
    if predicted_corners is None:
        raise ValueError("predicted_corners must be provided for 'direct' algorithm")
    else:
        cfg = corner_points_to_fan_mask_config(predicted_corners, original_dims)
        curvilinear_mask = create_mask(cfg, image_size=original_dims, intensity=1)

    return curvilinear_mask, cfg
