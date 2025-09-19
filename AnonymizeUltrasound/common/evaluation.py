import torch
from monai.metrics.meandice import DiceMetric
from monai.metrics.meaniou import MeanIoU
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import json
from common.masking import create_mask

def compare_masks(ground_truth_config: dict, predicted_config: dict, ground_truth_corners: dict, predicted_corners: dict, original_dims: tuple[int, int]) -> dict:
    """
    Compares two mask configurations and returns segmentation metrics.
    :param ground_truth_config: path to the ground truth mask configuration
    :param predicted_config: path to the predicted mask configuration
    :param ground_truth_corners: ground truth corners
    :param predicted_corners: predicted corners
    :param original_dims: original dimensions
    :return: dictionary with segmentation metrics
    """
    gt_mask = create_mask(ground_truth_config, image_size=original_dims)
    pred_mask = create_mask(predicted_config, image_size=original_dims)

    return calculate_segmentation_metrics(gt_mask, pred_mask, ground_truth_corners, predicted_corners, original_dims)

def load_mask_config(config_path: str) -> dict:
    """
    Load mask config from config file, handling missing image size fields.
    Example config:
    {
        "SOPInstanceUID": "1.2.156",
        "GrayscaleConversion": false,
        "MaskConfig": {
            "mask_type": "fan",
            "angle1": 50.163514981742004,
            "angle2": 113.2971648675992,
            "center_rows_px": 798,
            "center_cols_px": 694,
            "radius1": 741,
            "radius2": 146,
            "image_size_rows": 880,
            "image_size_cols": 1290
        }
    }

    or

    {
        "SOPInstanceUID": "1.2.156",
        "GrayscaleConversion": false,
        "mask_type": "fan",
        "angle1": 120.740816880876,
        "angle2": 59.46198675207232,
        "center_rows_px": -113,
        "center_cols_px": 877,
        "radius1": 270,
        "radius2": 1024,
        "image_size_rows": 920,
        "image_size_cols": 1590,
        "AnnotationLabels": [
            "1- R1 T-H"
        ]
    }
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        mask_config = config.get("MaskConfig")
        if mask_config is None:
            mask_config = config

        return mask_config

    except Exception as e:
        raise ValueError(f"Error loading {config_path}: {e}")

def corners_to_array(corners_dict):
    """
    Convert corner dictionaries to numpy arrays
    :param corners_dict: dictionary of corners
    :return: numpy array of corners
    Example:
    {
        "upper_left": (100, 100),
        "upper_right": (200, 100),
        "lower_left": (100, 200),
        "lower_right": (200, 200),
    }

    Returns:
    [[100.0, 100.0],
     [200.0, 100.0],
     [100.0, 200.0],
     [200.0, 200.0]]
    """
    corner_order = ['upper_left', 'upper_right', 'lower_left', 'lower_right']
    return np.array([[float(corners_dict[corner][0]), float(corners_dict[corner][1])]
                    for corner in corner_order], dtype=np.float32)

def calculate_segmentation_metrics(
    ground_truth_mask: np.ndarray,
    predicted_mask: np.ndarray,
    ground_truth_corners: dict,
    predicted_corners: dict,
    original_dims: tuple[int, int],
    include_background_for_dice: bool = True,
    include_background_for_iou: bool = True,
) -> dict:
    """
    Calculates segmentation metrics for two mask pairs.
    :param ground_truth_mask: numpy array of the ground truth mask
    :param predicted_mask: numpy array of the predicted mask
    :param ground_truth_corners: ground truth corners
    :param predicted_corners: predicted corners
    :param include_background_for_dice: whether to include background in Dice calculation
    :param include_background_for_iou: whether to include background in IoU calculation
    :return: dictionary with segmentation metrics
    """
    if ground_truth_mask.shape != predicted_mask.shape:
        raise ValueError("Mask shapes must match.")

    # Binarize
    gt_bin = (ground_truth_mask > 0).astype(np.uint8)
    pred_bin = (predicted_mask > 0).astype(np.uint8)

    # Corner distances calculations
    gt_array = corners_to_array(ground_truth_corners)
    pred_array = corners_to_array(predicted_corners)
    diff = pred_array - gt_array
    pixel_distances = np.sqrt(np.sum(diff**2, axis=1))

    # Basic distance metrics
    mean_distance_error = float(np.mean(pixel_distances))
    per_corner_errors = pixel_distances.astype(float)

    # Dice & IoU
    gt_tensor = torch.from_numpy(gt_bin.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pred_tensor = torch.from_numpy(pred_bin.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    dice_metric = DiceMetric(include_background=include_background_for_dice, reduction="mean")
    iou_metric = MeanIoU(include_background=include_background_for_iou, reduction="mean")
    dice_val = dice_metric(y_pred=pred_tensor, y=gt_tensor)
    iou_val = iou_metric(y_pred=pred_tensor, y=gt_tensor)

    # Handle different return types from MONAI metrics
    dice_score = float(dice_val[0]) if hasattr(dice_val, '__getitem__') else float(dice_val)
    iou_score = float(iou_val[0]) if hasattr(iou_val, '__getitem__') else float(iou_val)

    # Pixel accuracy
    pixel_acc = (gt_bin == pred_bin).sum() / gt_bin.size

    return {
        # Image data
        'image_height': original_dims[0],
        'image_width': original_dims[1],

          # Basic metrics
        'mean_distance_error': mean_distance_error,
        'corner_0_error': per_corner_errors[0],
        'corner_1_error': per_corner_errors[1],
        'corner_2_error': per_corner_errors[2],
        'corner_3_error': per_corner_errors[3],
        'dice_mean': dice_score,
        'iou_mean': iou_score,
        'pixel_accuracy_mean': pixel_acc,
    }
