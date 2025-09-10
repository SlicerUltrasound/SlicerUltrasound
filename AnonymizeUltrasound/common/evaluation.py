import torch
from monai.metrics.meandice import DiceMetric
from monai.metrics.meaniou import MeanIoU
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import json
from common.masking import create_mask

def compare_masks(ground_truth_config: dict, predicted_config: dict, original_dims: tuple[int, int]) -> dict:
    """
    Compares two mask configurations and returns segmentation metrics.
    :param ground_truth_config: path to the ground truth mask configuration
    :param predicted_config: path to the predicted mask configuration
    :return: dictionary with segmentation metrics
    """
    gt_mask = create_mask(ground_truth_config, image_size=original_dims)
    pred_mask = create_mask(predicted_config, image_size=original_dims)

    return calculate_segmentation_metrics(gt_mask, pred_mask)

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

def calculate_segmentation_metrics(
    ground_truth_mask: np.ndarray,
    predicted_mask: np.ndarray,
    include_background_for_dice: bool = True,
    include_background_for_iou: bool = True,
) -> dict:
    """
    Calculates segmentation metrics for two mask pairs.
    :param ground_truth_mask: numpy array of the ground truth mask
    :param predicted_mask: numpy array of the predicted mask
    :param include_background_for_dice: whether to include background in Dice calculation
    :param include_background_for_iou: whether to include background in IoU calculation
    :return: dictionary with segmentation metrics
    """
    if ground_truth_mask.shape != predicted_mask.shape:
        raise ValueError("Mask shapes must match.")

    # Binarize
    gt_bin = (ground_truth_mask > 0).astype(np.uint8)
    pred_bin = (predicted_mask > 0).astype(np.uint8)

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

    # Precision, recall, F1
    precision = precision_score(gt_bin.flatten(), pred_bin.flatten(), zero_division='warn')
    recall = recall_score(gt_bin.flatten(), pred_bin.flatten(), zero_division='warn')
    f1 = f1_score(gt_bin.flatten(), pred_bin.flatten(), zero_division='warn')

    # Sensitivity, specificity
    tn, fp, fn, tp = confusion_matrix(gt_bin.flatten(), pred_bin.flatten(), labels=[0,1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'dice_mean': dice_score,
        'iou_mean': iou_score,
        'pixel_accuracy_mean': pixel_acc,
        'precision_mean': precision,
        'recall_mean': recall,
        'f1_mean': f1,
        'sensitivity_mean': sensitivity,
        'specificity_mean': specificity,
    }
