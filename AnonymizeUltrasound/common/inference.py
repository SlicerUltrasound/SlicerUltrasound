import torch
import numpy as np
from PIL import Image
import cv2
import logging

def get_device(device: str = 'cpu'):
    """ Set the Device to run the model on """
    if device is not None and device != '':
        return device

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logging.info(f"The model will run on Device: {device}")

    return device

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

def validate_image_shape(image: np.ndarray) -> np.ndarray:
    """
    Validate the shape of the image to be (N, H, W, C)
    If the shape is not (N, H, W, C), it will be converted to (N, H, W, C)
    :param image: np.ndarray
    :return: np.ndarray
    """
    logging.info(f"Validating image shape: {image.shape}")

    # Validate input shape
    if image.ndim != 4:
        raise ValueError(f"Expected 4D array (N, H, W, C), got {image.ndim}D array with shape {image.shape}")

    # Check if it's (N, C, H, W) format and convert to (N, H, W, C)
    if image.shape[1] in (1, 3, 4) and image.shape[2] > image.shape[1] and image.shape[3] > image.shape[1]:
        logging.info(f"Converting from (N, C, H, W) to (N, H, W, C) format")
        image = image.transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

    N, H, W, C = image.shape

    # Validate dimensions make sense
    if N < 1:
        raise ValueError(f"Number of frames must be >= 1, got {N}")
    if H < 1 or W < 1:
        raise ValueError(f"Height and width must be >= 1, got H={H}, W={W}")
    if C not in (1, 3, 4):  # grayscale, RGB, or RGBA
        raise ValueError(f"Number of channels must be 1, 3, or 4, got {C}")

    # Additional sanity checks
    if H > 10000 or W > 10000:  # reasonable upper bounds
        logging.warning(f"Unusually large image dimensions: {H}×{W}")

    return image

def preprocess_image(
    image: np.ndarray, # (N, H, W, C)
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
    pil_image = validate_image_shape(image)

    # Step 1: Max-pool frames to get single frame
    snapshot = pil_image.max(axis=0)  # (H, W, C)

    # Step 2: Convert to grayscale using PIL method (matching training dataset)

    # Handle single channel case.
    # PIL expects 2D arrays for grayscale images, not 3D arrays with a single channel (H, W)
    if snapshot.shape[2] == 1:
        snapshot_for_pil = snapshot.squeeze(axis=2)
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
