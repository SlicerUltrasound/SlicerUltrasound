import cv2
import numpy as np

def create_rectangle_mask(config, edge_erosion=0.0, image_size=None) -> np.ndarray:
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

def create_curvilinear_mask(config, edge_erosion=0.0, image_size=None, intensity=255) -> np.ndarray:
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

def create_mask(config, edge_erosion=0.0, image_size=None, intensity=255) -> np.ndarray:
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

def corner_points_to_fan_mask_config(corner_points, image_size=None) -> dict:
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

def mask_config_to_corner_points(config: dict) -> dict:
    """
    Inverse of corner_points_to_fan_mask_config.
    Given a mask config, return corner points as:
      {
        "upper_left": (x, y),
        "upper_right": (x, y) or None (for 3-point fan),
        "lower_left": (x, y),
        "lower_right": (x, y),
      }

    Supports:
    - mask_type == "rectangle": uses rectangle_left/right/top/bottom
    - mask_type == "fan": uses angles, center, and radii. Accepts legacy keys:
      angle_min_degrees/angle_max_degrees, radius_min_px/radius_max_px, center_coordinate_pixel.
    """
    mask_type = config.get("mask_type", "fan")

    if mask_type == "rectangle":
        left = int(round(config["rectangle_left"]))
        right = int(round(config["rectangle_right"]))
        top = int(round(config["rectangle_top"]))
        bottom = int(round(config["rectangle_bottom"]))
        return {
            "upper_left": (left, top),
            "upper_right": (right, top),
            "lower_left": (left, bottom),
            "lower_right": (right, bottom),
        }

    if mask_type == "fan":
        # Angles (degrees)
        angle1 = config.get("angle1", config.get("angle_min_degrees"))
        angle2 = config.get("angle2", config.get("angle_max_degrees"))
        if angle1 is None or angle2 is None:
            raise ValueError("Fan config must provide angle1/angle2 or angle_min_degrees/angle_max_degrees.")
        angle1 = float(angle1)
        angle2 = float(angle2)

        # Center (row, col) or (y, x)
        if "center_rows_px" in config and "center_cols_px" in config:
            cy = int(config["center_rows_px"])
            cx = int(config["center_cols_px"])
        elif "center_coordinate_pixel" in config and config["center_coordinate_pixel"] is not None:
            cy = int(config["center_coordinate_pixel"][0])
            cx = int(config["center_coordinate_pixel"][1])
        else:
            raise ValueError("Fan config must provide center_rows_px/center_cols_px or center_coordinate_pixel.")

        # Radii
        r1 = config.get("radius1", config.get("radius_min_px", 0))
        r2 = config.get("radius2", config.get("radius_max_px"))
        if r2 is None:
            raise ValueError("Fan config must provide radius2 or radius_max_px.")
        r1 = int(round(float(r1)))
        r2 = int(round(float(r2)))

        t1 = np.deg2rad(angle1)
        t2 = np.deg2rad(angle2)

        def pt(theta, r):
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            return (int(round(x)), int(round(y)))

        top1 = pt(t1, r1)
        top2 = pt(t2, r1)
        bot1 = pt(t1, r2)
        bot2 = pt(t2, r2)

        # 3-point fan: apex at center, r1 == 0
        if r1 == 0:
            apex = (cx, cy)
            if bot1[0] <= bot2[0]:
                lower_left, lower_right = bot1, bot2
            else:
                lower_left, lower_right = bot2, bot1
            # Return None for upper_right so corner_points_to_fan_mask_config uses its 3-point branch
            result = {
                "upper_left": apex,
                "upper_right": None,
                "lower_left": lower_left,
                "lower_right": lower_right,
            }
        else:
            # Keep angle pairing consistent when deciding left/right using x of the top points
            if top1[0] <= top2[0]:
                upper_left, lower_left = top1, bot1
                upper_right, lower_right = top2, bot2
            else:
                upper_left, lower_left = top2, bot2
                upper_right, lower_right = top1, bot1
            result = {
                "upper_left": upper_left,
                "upper_right": upper_right,
                "lower_left": lower_left,
                "lower_right": lower_right,
            }

        # Optional: clip to image bounds if available
        rows = config.get("image_size_rows")
        cols = config.get("image_size_cols")
        if rows is not None and cols is not None:
            rows = int(rows)
            cols = int(cols)

            def clip(p):
                if p is None:
                    return None
                x, y = p
                x = max(0, min(cols - 1, x))
                y = max(0, min(rows - 1, y))
                return (x, y)

            for k in ("upper_left", "upper_right", "lower_left", "lower_right"):
                result[k] = clip(result[k])

        return result

    raise ValueError(f"Unsupported mask type: {mask_type}. Supported types are 'rectangle' and 'fan'.")

def compute_masks_and_configs(original_dims: tuple[int, int], predicted_corners: dict) -> tuple[np.ndarray, dict]:
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
    cfg = corner_points_to_fan_mask_config(predicted_corners, original_dims)
    curvilinear_mask = create_mask(cfg, image_size=original_dims, intensity=1)

    return curvilinear_mask, cfg