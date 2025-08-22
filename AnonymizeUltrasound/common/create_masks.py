import json
import os
import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def create_masks(masks_dir, mask_type="curvilinear", config_file='config.json'):
    """
    Create and save masks of the specified type in the given directory.

    Parameters:
    masks_dir (str): The directory where the masks will be saved.
    mask_type (str): The type of mask to create. Can be "curvilinear" or "linear". Default is "curvilinear".

    Returns:
    None
    """
    # Load the configuration from the JSON file
    config = json.load(open(config_file))
    data_dir = config['data_dir']
    # Ensure frames_dir exists
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)
    dcm_path = config['dcm_path']
    annotations_path = config['annotations_path']

    # Read the CSV files into DataFrames
    df_annotations = pd.read_csv(annotations_path)
    df_dcm = pd.read_csv(dcm_path)

    # Set 'id' as the index for df_dcm
    df_dcm.set_index('id', inplace=True)

    # Drop any rows with NaN
    df_annotations.dropna(subset=['Case', 'Clip', 'FileName', 'Frame'], inplace=True)

    # Group by 'Case' and 'Clip'
    grouped = df_annotations.groupby(['Case', 'Clip', 'FileName'])
    grouped_progress = tqdm(grouped, desc='Processing groups', unit='group')

    for (case, clip, dcm_id), group in grouped_progress:
        grouped_progress.set_description(f'Processing Case: {case}, Clip: {clip}, DICOM ID: {dcm_id}')
        grouped_progress.refresh()
        # Extract case number from the case pattern
        case_number = int(re.search(r'\d+', case).group())
        # print(f'Case: {case_number:03}')
        # print(f'  Clip: {int(clip):03}')

        # Read DICOM frames
        # print(f'  DICOM: {dcm_id}')
        dcm_file = df_dcm.at[dcm_id, 'path']
        json_file = dcm_file.replace(".dcm", ".json")
        json_filepath = os.path.join(data_dir, json_file)
        
        if not os.path.exists(json_filepath):
            print(f"File {json_filepath} does not exist")
            continue

        config = json.load(open(json_filepath))
        try:
            config = update_config(config)
        except KeyError as e:
            print(f"Skipping DICOM ID {dcm_id} due to missing key: {e}")
            continue

        # Create a mask for each frame based on the mask type in the config
        mask_type_from_config = config.get("mask_type", "fan")  # Default to fan for backward compatibility
        
        if mask_type_from_config == "rectangle":
            mask = create_rectangle_mask(config)
        elif mask_type == "curvilinear" and mask_type_from_config == "fan":
            mask = create_curvilinear_mask(config)
        elif mask_type == "trapezoid" and mask_type_from_config == "fan":
            mask = create_trapezoid_mask(config)
        else:
            print(f"Unsupported mask type: {mask_type_from_config}. Skipping DICOM ID {dcm_id}.")
            continue

        # Iterate over frames in the group
        for _, row in group.iterrows():
            frame_number = int(row['Frame'])
            # print(f'    Frame: {frame_number:03}')
            
            frame_filename = f'{case_number:03}_{int(clip):03}_{frame_number:03}.png'
            frame_path = os.path.join(masks_dir, frame_filename)
            cv2.imwrite(frame_path, mask)

def update_config(config):
    """
    Updates the config with the number of lines and number of samples along lines. Also calculates the center_coordinate_pixel, angle_min_degrees, angle_max_degrees, radius_min_px, and radius_max_px.
    :param config: dictionary containing the configuration parameters
    :return: updated config
    """
    mask_type = config.get("mask_type", "fan")  # Default to fan for backward compatibility
    
    if mask_type == "rectangle":
        # For rectangle masks, no additional processing needed
        return config
    else:
        # For fan masks, process the original parameters
        config["center_coordinate_pixel"] = [int(config["center_rows_px"]), int(config["center_cols_px"])]
        angle1 = float(config['angle1'])
        angle2 = float(config['angle2'])
        config['angle_min_degrees'] = min(angle1, angle2)
        config['angle_max_degrees'] = max(angle1, angle2)
        radius1 = float(config['radius1'])
        radius2 = float(config['radius2'])
        config['radius_min_px'] = min(radius1, radius2)
        config['radius_max_px'] = max(radius1, radius2)
        return config

def scanconversion_config_to_corner_points(scanconversion_config):
    """
    Create corner points from a scan conversion configuration dictionary.

    Args:
        scanconversion_config (dict): Dictionary with scan conversion parameters.
        Must contain the following keys for fan masks:
            "mask_type", "angle_min_degrees", "angle_max_degrees", "radius_min_px", "radius_max_px", "center_coordinate_pixel".
        Must contain the following keys for rectangle masks:
            "mask_type", "top_left_coordinate_pixel", "top_right_coordinate_pixel", "bottom_left_coordinate_pixel", "bottom_right_coordinate_pixel".

    Returns:
        corner_points (dict): Dictionary with the following keys: "upper_left", "upper_right", "lower_left", "lower_right".
    """
    if scanconversion_config["mask_type"] == "fan":
        angle1 = scanconversion_config["angle1"]
        angle2 = scanconversion_config["angle2"]
        center_rows_px = scanconversion_config["center_rows_px"]
        center_cols_px = scanconversion_config["center_cols_px"]
        radius1 = scanconversion_config["radius1"]
        radius2 = scanconversion_config["radius2"]
        
        upper_left = (center_cols_px + radius1 * np.cos(np.radians(angle1)), center_rows_px + radius1 * np.sin(np.radians(angle1)))
        upper_right = (center_cols_px + radius1 * np.cos(np.radians(angle2)), center_rows_px + radius1 * np.sin(np.radians(angle2)))
        lower_left = (center_cols_px + radius2 * np.cos(np.radians(angle1)), center_rows_px + radius2 * np.sin(np.radians(angle1)))
        lower_right = (center_cols_px + radius2 * np.cos(np.radians(angle2)), center_rows_px + radius2 * np.sin(np.radians(angle2)))
        
    elif scanconversion_config["mask_type"] == "rectangle":
        upper_left = (scanconversion_config["rectangle_left"], scanconversion_config["rectangle_top"])
        upper_right = (scanconversion_config["rectangle_right"], scanconversion_config["rectangle_top"])
        lower_left = (scanconversion_config["rectangle_left"], scanconversion_config["rectangle_bottom"])
        lower_right = (scanconversion_config["rectangle_right"], scanconversion_config["rectangle_bottom"])
    
    corner_points = {
        "upper_left": upper_left,
        "upper_right": upper_right,
        "lower_left": lower_left,
        "lower_right": lower_right
    }
    
    return corner_points

import cv2
import numpy as np

def create_trapezoid_mask(config):
    """
    Create a trapezoid mask with the given configuration.

    Parameters:
    config (dict): A dictionary containing the configuration for the trapezoid mask.

    Returns:
    np.ndarray: A mask with the trapezoid filled in white and the background in black.
    """

    corner_points = scanconversion_config_to_corner_points(config)
    frame_shape = (config['image_size_rows'], config['image_size_cols'])

    # Extract corner points
    upper_left = corner_points['upper_left']
    upper_right = corner_points['upper_right']
    lower_left = corner_points['lower_left']
    lower_right = corner_points['lower_right']
    
    # Create a black mask
    mask = np.zeros(frame_shape, dtype=np.uint8)
    
    # Define the points of the trapezoid
    points = np.array([upper_left, upper_right, lower_right, lower_left], dtype=np.int32)
    
    # Draw the trapezoid on the mask
    cv2.fillPoly(mask, [points], 255)
    
    return mask


def create_mask(config, edge_erosion=0.0, image_size=None, intensity=255):
    """
    Generate a binary mask based on the mask type (curvilinear fan or rectangle).

    Args:
        config (dict): Dictionary with scan conversion parameters.
        edge_erosion (float): Fraction of the image size (number of rows) to be eroded from the edges of the mask.
        image_size (tuple): Image size as (height, width). Used if not specified in config.

    Returns:
        mask_array (np.ndarray): Binary mask with ones inside the scan area and zeros outside.
    """
    mask_type = config.get("mask_type", "fan")  # Default to fan for backward compatibility
    
    if mask_type == "rectangle":
        return create_rectangle_mask(config, edge_erosion, image_size)
    elif mask_type == "fan":
        return create_curvilinear_mask(config, edge_erosion, image_size, intensity=intensity)
    else:
        raise ValueError(f"Unsupported mask type: {mask_type}. Supported types are 'rectangle' and 'fan'.")


def create_rectangle_mask(config, edge_erosion=0.0, image_size=None):
    """
    Generate a binary mask for rectangle ultrasound regions.

    Args:
        config (dict): Dictionary with rectangle parameters containing:
                      rectangle_left, rectangle_right, rectangle_top, rectangle_bottom
        edge_erosion (float): Fraction of the image size to be eroded from the edges of the mask.
        image_size (tuple): Image size as (height, width). Used if not specified in config.

    Returns:
        mask_array (np.ndarray): Binary mask for the rectangle region.
    """
    # Get image dimensions
    try:
        image_rows = int(config['image_size_rows'])
        image_cols = int(config['image_size_cols'])
    except KeyError:
        if image_size is not None:
            image_rows, image_cols = image_size
        else:
            raise ValueError("Image size must be specified in the configuration or as an argument.")

    # Get rectangle boundaries
    left = int(config['rectangle_left'])
    right = int(config['rectangle_right'])
    top = int(config['rectangle_top'])
    bottom = int(config['rectangle_bottom'])

    # Validate rectangle boundaries
    if left >= right or top >= bottom:
        raise ValueError(f"Invalid rectangle boundaries: left={left}, right={right}, top={top}, bottom={bottom}")
    
    # Create a black mask
    mask = np.zeros((image_rows, image_cols), dtype=np.uint8)
    
    # Fill the rectangle region with white
    mask[top:bottom+1, left:right+1] = 255
    
    # Apply edge erosion if specified
    if edge_erosion > 0:
        # Repaint the borders of the mask to zero to allow erosion from all sides
        mask[0, :] = 0
        mask[:, 0] = 0
        mask[-1, :] = 0
        mask[:, -1] = 0
        # Erode the mask
        erosion_size = int(edge_erosion * image_rows)
        mask = cv2.erode(mask, np.ones((erosion_size, erosion_size), np.uint8), iterations=1)
    
    return mask


def create_curvilinear_mask(config, edge_erosion=0.0, image_size=None, intensity=255):
    """
    Generate a binary mask for the curvilinear image with ones inside the scan lines area and zeros outside.

    Args:
        config (dict): Dictionary with scan conversion parameters.
        edge_erosion (float): Fraction of the image size (number of rows) to be eroded from the edges of the mask.

    Returns:
        mask_array (np.ndarray): Binary mask for the curvilinear image with ones inside the scan lines area and zeros outside.
    """
    angle1 = float(config.get("angle_min_degrees", config.get("angle1")))
    angle2 = float(config.get("angle_max_degrees", config.get("angle2")))
    # center_rows_px = int(config.get('center_rows_px', config.get("center_coordinate_pixel")[0]))
    # center_cols_px = int(config.get("center_cols_px", config.get("center_coordinate_pixel")[1]))
    try:
        center_rows_px = int(config['center_rows_px'])
        center_cols_px = int(config['center_cols_px'])
    except KeyError:
        if 'center_coordinate_pixel' in config:
            center_rows_px = int(config['center_coordinate_pixel'][0])
            center_cols_px = int(config['center_coordinate_pixel'][1])
        else:
            raise ValueError("Center coordinates must be specified in the configuration.")
    radius1 = int(config.get("radius_min_px", config.get("radius1")))
    radius2 = int(config.get("radius_max_px", config.get("radius2")))
    # image_rows = config.get("image_size_rows", image_size[0] if image_size else None)
    # image_cols = config.get("image_size_cols", image_size[1] if image_size else None)
    try:
        image_rows = int(config['image_size_rows'])
        image_cols = int(config['image_size_cols'])
    except KeyError:
        if image_size is not None:
            image_rows, image_cols = image_size
        else:
            raise ValueError("Image size must be specified in the configuration or as an argument.")

    # Validate input parameters
    if image_rows is None or image_cols is None:
        raise ValueError("Image size must be specified in the configuration or as an argument.")
    if (angle1 is None) or (angle2 is None) or (center_rows_px is None) or (center_cols_px is None) or (radius1 is None) or (radius2 is None):
        raise ValueError("Missing required parameters in the configuration for curvilinear mask.")

    

    mask = np.zeros((image_rows, image_cols), dtype=np.int8)
    mask = cv2.ellipse(mask, (center_cols_px, center_rows_px), (radius2, radius2), 0.0, angle1, angle2, 1, -1)
    mask = cv2.circle(mask, (center_cols_px, center_rows_px), radius1, 0, -1)
    mask = mask.astype(np.uint8)  # Convert mask_array to uint8
    
    # Erode mask by 10 percent of the image size to remove artifacts on the edges
    if edge_erosion > 0:
        # Repaint the borders of the mask to zero to allow erosion from all sides
        mask[0, :] = 0
        mask[:, 0] = 0
        mask[-1, :] = 0
        mask[:, -1] = 0
        # Erode the mask
        erosion_size = int(edge_erosion * image_rows)
        mask = cv2.erode(mask, np.ones((erosion_size, erosion_size), np.uint8), iterations=1)
    
    mask = mask * intensity
    return mask


def line_coefficients(p1, p2):
    """
    Given two points p1=(x1,y1), p2=(x2,y2), return (A, B, C) for the line equation A*x + B*y + C = 0.
    """
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C

def corner_points_to_fan_mask_config(corner_points, image_size=None):
    """
    Invert scanconversion_config_to_corner_points: recover the fan or rectangle mask
    parameters from the four corner points. 
    Logic is taken from createFanMask from SlicerUltrasound/AnonymizeUltrasound/AnonymizeUltrasound.py
    """
    # unpack
    topLeft = corner_points["upper_left"]
    topRight = corner_points["upper_right"]
    bottomLeft =  corner_points["lower_left"]
    bottomRight = corner_points["lower_right"]
    image_size_rows = image_size[0] if image_size else None
    image_size_cols = image_size[1] if image_size else None

    # Detect if the mask is a fan or a rectangle for 4-point mode (this logic comes from SlicerUltrasound/AnonymizeUltrasound/AnonymizeUltrasound.py)
    maskHeight = abs(topLeft[1] - bottomLeft[1])
    tolerancePixels = round(0.1 * maskHeight) 
    if abs(topLeft[0] - bottomLeft[0]) < tolerancePixels and abs(topRight[0] - bottomRight[0]) < tolerancePixels:
        # This is a rectangle mask
        rectangle_config = {
            "mask_type": "rectangle",
            "rectangle_left": int(round(min(topLeft[0], bottomLeft[0]))),
            "rectangle_right": int(round(max(topRight[0], bottomRight[0]))),
            "rectangle_top": int(round(min(topLeft[1], topRight[1]))),
            "rectangle_bottom": int(round(max(bottomLeft[1], bottomRight[1]))),
        }
        if image_size is not None:
            rectangle_config["image_size_rows"] = image_size_rows
            rectangle_config["image_size_cols"] = image_size_cols
        return rectangle_config

    if topRight is not None:
        # Compute the angle of the fan mask in degrees

        if abs(topLeft[0] - bottomLeft[0]) < 0.001:
            angle1 = 90.0
        else:
            angle1 = np.arctan2((bottomLeft[1] - topLeft[1]), (bottomLeft[0] - topLeft[0])) * 180 / np.pi 
        if angle1 > 180.0:
            angle1 -= 180.0
        if angle1 < 0.0:
            angle1 += 180.0
        
        if abs(topRight[0] - bottomRight[0]) < 0.001:
            angle2 = 90.0
        else:
            angle2 = np.arctan((topRight[1] - bottomRight[1]) / (topRight[0] - bottomRight[0])) * 180 / np.pi
        if angle2 > 180.0:
            angle2 -= 180.0
        if angle2 < 0.0:
            angle2 += 180.0
        # Fit lines to the top and bottom points
        leftLineA, leftLineB, leftLineC = line_coefficients(topLeft, bottomLeft)
        rightLineA, rightLineB, rightLineC = line_coefficients(topRight, bottomRight)

        # Handle the case when the lines are parallel
        if leftLineB != 0 and rightLineB != 0 and leftLineA / leftLineB == rightLineA / rightLineB:
            raise ValueError("The left and right lines are parallel; cannot determine unique angles.")
        # Compute intersection point of the two lines
        det = leftLineA * rightLineB - leftLineB * rightLineA
        if det == 0:
            raise ValueError(f"The lines do not intersect; they are parallel or coincident. topLeft: {topLeft}, topRight: {topRight}, bottomLeft: {bottomLeft}, bottomRight: {bottomRight}")

        intersectionX = (leftLineB * rightLineC - rightLineB * leftLineC) / det
        intersectionY = (rightLineA * leftLineC - leftLineA * rightLineC) / det

        # Compute average distance of top points to the intersection point

        topDistance = np.sqrt((topLeft[0] - intersectionX) ** 2 + (topLeft[1] - intersectionY) ** 2) + \
                        np.sqrt((topRight[0] - intersectionX) ** 2 + (topRight[1] - intersectionY) ** 2)
        topDistance /= 2

        # Compute average distance of bottom points to the intersection point

        bottomDistance = np.sqrt((bottomLeft[0] - intersectionX) ** 2 + (bottomLeft[1] - intersectionY) ** 2) + \
                            np.sqrt((bottomRight[0] - intersectionX) ** 2 + (bottomRight[1] - intersectionY) ** 2)
        bottomDistance /= 2

        # Mask parameters

        center_rows_px = round(intersectionY)
        center_cols_px = round(intersectionX)
        radius1 = round(topDistance)
        radius2 = round(bottomDistance)

        fan_config_dict = {
            "mask_type": "fan",
            "angle1": float(angle1),
            "angle2": float(angle2),
            "center_rows_px": center_rows_px,
            "center_cols_px": center_cols_px,
            "radius1": radius1,
            "radius2": radius2,
            "image_size_rows": image_size_rows,
            "image_size_cols": image_size_cols,
        }
    else:
        # 3-point fan: apex at topLeft, bottomLeft/bottomRight define span
        if image_size_rows is None or image_size_cols is None:
            raise ValueError("image_size must be provided for 3-point fan mask")
        # compute radii from apex to bottom points
        r1 = np.hypot(bottomLeft[0] - topLeft[0], bottomLeft[1] - topLeft[1])
        r2 = np.hypot(bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1])
        radius = int(round((r1 + r2) / 2))
        # compute angles in degrees
        a1 = np.degrees(np.arctan2(bottomLeft[1] - topLeft[1], bottomLeft[0] - topLeft[0]))
        a2 = np.degrees(np.arctan2(bottomRight[1] - topLeft[1], bottomRight[0] - topLeft[0]))
        angle1, angle2 = (a1, a2) if a1 <= a2 else (a2, a1)
        # apex coordinates as center
        cx = int(round(topLeft[0]))
        cy = int(round(topLeft[1]))
        fan_config_dict = {
            "mask_type": "fan",
            "angle1": float(angle1),
            "angle2": float(angle2),
            "center_rows_px": cy,
            "center_cols_px": cx,
            "radius1": 0,
            "radius2": radius,
            "image_size_rows": image_size_rows,
            "image_size_cols": image_size_cols,
        }
    return fan_config_dict

# Example usage
if __name__ == '__main__':
    config = json.load(open('config.json'))
    masks_dir = config['masks_dir']
    create_masks(masks_dir)
