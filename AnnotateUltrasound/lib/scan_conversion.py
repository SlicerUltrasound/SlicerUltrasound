import numpy as np
from scipy.spatial import Delaunay
from scipy.ndimage import map_coordinates


# ---------- 1. JSON → full config ------------------------------------
def update_config_dict(config_dict, num_lines, num_samples_along_lines, image_width, image_height):
    """
    Updates the config_dict with the number of lines and number of samples along lines. Also calculates the center_coordinate_pixel, angle_min_degrees, angle_max_degrees, radius_min_px, and radius_max_px.
    :param config_dict: dictionary containing the configuration parameters
    :param num_lines: number of lines
    :param num_samples_along_lines: number of samples along lines
    :return: updated config_dict
    """
    config_dict["center_coordinate_pixel"] = [int(config_dict["center_rows_px"]), int(config_dict["center_cols_px"])]
    angle1 = float(config_dict['angle1'])
    angle2 = float(config_dict['angle2'])
    config_dict['angle_min_degrees'] = min(angle1, angle2)
    config_dict['angle_max_degrees'] = max(angle1, angle2)
    config_dict['num_lines'] = num_lines
    config_dict['num_samples_along_lines'] = num_samples_along_lines
    radius1 = float(config_dict['radius1'])
    radius2 = float(config_dict['radius2'])
    config_dict['radius_min_px'] = min(radius1, radius2)
    config_dict['radius_max_px'] = max(radius1, radius2)
    config_dict['num_cartesian_image_cols'] = image_width
    config_dict['num_cartesian_image_rows'] = image_height

    return config_dict


# ---------- 2. Polar grid (θ,r) → Cartesian (x,y) ---------------------
def cartesian_coordinates(scanconversion_config):
    """
    Compute cartesian coordianates for conversion from cartesian to polar representation.
    The returned cartesian coordinates can be used to map the curvilinear image to a rectangular image using scipy.ndimage.map_coordinates.

    Args:
        scanconversion_config (dict): Dictionary with scan conversion parameters. Must contain the following keys:
        "angle_min_degrees", "angle_max_degrees", "radius_min_px", "radius_max_px", "num_samples_along_lines", "num_lines", "center_coordinate_pixel".

    Rerturns:
        x_cart (np.ndarray): x coordinates of the cartesian grid.
        y_cart (np.ndarray): y coordinates of the cartesian grid.

    Example:
        >>> x_cart, y_cart = scan_conversion_inverse(scanconversion_config)
        >>> scan_converted_image = map_coordinates(ultrasound_data[0, :, :, 0], [x_cart, y_cart], order=3, mode="nearest")
        >>> scan_converted_segmentation = map_coordinates(segmentation_data[0, :, :, 0], [x_cart, y_cart], order=0, mode="nearest")
    """

    # Create sampling points in polar coordinates (theta, r)

    angle_min_deg = np.deg2rad(scanconversion_config["angle_min_degrees"])
    angle_max_deg = np.deg2rad(scanconversion_config["angle_max_degrees"])
    radius_min_px = scanconversion_config["radius_min_px"]
    radius_max_px = scanconversion_config["radius_max_px"]

    theta, r = np.meshgrid(np.linspace(angle_min_deg, angle_max_deg, scanconversion_config["num_lines"]),
                           np.linspace(radius_min_px, radius_max_px, scanconversion_config["num_samples_along_lines"]))

    # Convert the polar coordinates to cartesian coordinates using the center coordinate as origin

    x_cart = r * np.cos(theta) + scanconversion_config["center_coordinate_pixel"][1]
    y_cart = r * np.sin(theta) + scanconversion_config["center_coordinate_pixel"][0]

    return x_cart, y_cart


# ---------- 3. Interpolation weights for fast inverse mapping ---------
def scan_interpolation_weights(scanconversion_config):
    """
    Compute the interpolation weights for scan conversion.
    
    Args:
        scanconversion_config (dict): Dictionary with scan conversion parameters. Must contain the following keys:
        "curvilinear_image_size", "angle_min_degrees", "angle_max_degrees", "radius_min_px", "radius_max_px", "num_samples_along_lines", "num_lines", "center_coordinate_pixel".
    
    Returns:
        vertices (np.ndarray): Vertices of the triangulation.
        weights (np.ndarray): Interpolation weights.
    """
    num_rows = scanconversion_config["num_cartesian_image_rows"]
    num_cols = scanconversion_config["num_cartesian_image_cols"]

    x_cart, y_cart = cartesian_coordinates(scanconversion_config)
    triangulation = Delaunay(np.vstack((x_cart.flatten(), y_cart.flatten())).T)

    grid_y, grid_x = np.mgrid[0:num_rows, 0:num_cols]
    simplices = triangulation.find_simplex(np.vstack((grid_x.flatten(), grid_y.flatten())).T)
    vertices = triangulation.simplices[simplices]

    X = triangulation.transform[simplices, :2]
    Y = np.vstack((grid_x.flatten(), grid_y.flatten())).T - triangulation.transform[simplices, 2]
    b = np.einsum('ijk,ik->ij', X, Y)
    weights = np.c_[b, 1 - b.sum(axis=1)]

    return vertices, weights


# ---------- 4. Curvilinear → Scan‑lines -------------------------------
def curvilinear_to_scanlines(image, scanconversion_config, x_cart, y_cart, interpolation_order=1):
    """
    Scan convert a curvilinear image to a linear image (scanlines).
    
    Args:
        image (np.ndarray): Curvilinear image to be scan converted. Must be in the format of [Height, Width] or [Channels, Height, Width].
        scanconversion_config (dict): Dictionary with scan conversion parameters.
        x_cart (np.ndarray): x coordinates of the cartesian grid.
        y_cart (np.ndarray): y coordinates of the cartesian grid.
        interpolation_order (int): Order of the interpolation. 0 = nearest neighbor, 1 = linear, 2 = cubic.
        
    Returns:
        converted_image (np.ndarray): Scan converted image. Dimensions are [Sampes, Lines] or [Channels, Samples, Lines], corresponding to the input image dimensions.
    """
    num_samples = scanconversion_config["num_samples_along_lines"]
    num_lines = scanconversion_config["num_lines"]
    if len(image.shape) == 2:
        converted_image = np.zeros((num_samples, num_lines))
        converted_image[:, :] = map_coordinates(image, [y_cart, x_cart], order=interpolation_order, mode='constant', cval=0.0)
    else:
        num_channels = image.shape[0]
        converted_image = np.zeros((num_channels, num_samples, num_lines), dtype=image.dtype)
        for channel in range(num_channels):
            converted_image[channel, :, :] = map_coordinates(image[channel, :, :], [y_cart, x_cart], order=interpolation_order, mode='constant', cval=0.0)
    
    return converted_image

# ---------- 5. Scan‑lines → Curvilinear -------------------------------
def scanlines_to_curvilinear(linear_data, scanconversion_config, vertices, weights):
    """
    Scan convert a linear image (scanlines) to a curvilinear image.

    Args:
        linear_data (np.ndarray): Linear image to be scan converted. Must be in the format of [Lines, Samples], or [Channels, Lines, Samples].
        scanconversion_config (dict): Dictionary with scan conversion parameters.

    Returns:
        scan_converted_image (np.ndarray): Scan converted image.
    """
        
    num_image_rows = scanconversion_config["num_cartesian_image_rows"]
    num_image_cols = scanconversion_config["num_cartesian_image_cols"]
    
    if len(linear_data.shape) == 2:
        z = linear_data.flatten()
        zi = np.einsum('ij,ij->i', np.take(z, vertices), weights)
        return zi.reshape(num_image_rows, num_image_cols)
    elif len(linear_data.shape) == 3:
        num_channels = linear_data.shape[0]
        scan_converted_image = np.zeros((num_channels, num_image_rows, num_image_cols), dtype=linear_data.dtype)
        for i in range(num_channels):
            z = linear_data[i, :, :].flatten()
            zi = np.einsum('ij,ij->i', np.take(z, vertices), weights)
            scan_converted_image[i, :, :] = zi.reshape(num_image_rows, num_image_cols)
        return scan_converted_image