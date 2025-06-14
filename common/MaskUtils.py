import numpy as np
import cv2
from typing import Tuple, Optional, List
import vtk
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLVectorVolumeNode

class MaskUtils:
    """Utility class for masking operations"""

    @staticmethod
    def create_rectangle_mask(image_array: np.ndarray, top_left: np.ndarray, top_right: np.ndarray,
                            bottom_left: np.ndarray, bottom_right: np.ndarray) -> np.ndarray:
        """Create rectangular mask"""
        image_size_rows = image_array.shape[1]
        image_size_cols = image_array.shape[2]

        rectangle_left = round((top_left[0] + bottom_left[0]) / 2)
        rectangle_right = round((top_right[0] + bottom_right[0]) / 2)
        rectangle_top = round((top_left[1] + top_right[1]) / 2)
        rectangle_bottom = round((bottom_left[1] + bottom_right[1]) / 2)

        mask_array = np.zeros((image_size_rows, image_size_cols), dtype=np.uint8)
        mask_array[rectangle_top:rectangle_bottom, rectangle_left:rectangle_right] = 1

        return mask_array

    @staticmethod
    def create_fan_mask(image_array: np.ndarray, top_left: np.ndarray, top_right: np.ndarray,
                       bottom_left: np.ndarray, bottom_right: np.ndarray, value: int = 255) -> np.ndarray:
        """Create fan-shaped mask"""
        image_size_rows = image_array.shape[1]
        image_size_cols = image_array.shape[2]
        mask_array = np.zeros((image_size_rows, image_size_cols), dtype=np.uint8)

        # Calculate center point
        center = np.array([image_size_cols / 2, image_size_rows / 2])
        
        # Calculate angles for each corner
        angles = []
        for point in [top_left, top_right, bottom_right, bottom_left]:
            angle = np.arctan2(point[1] - center[1], point[0] - center[0])
            angles.append(np.degrees(angle))
        
        # Sort angles
        angles.sort()
        
        # Calculate radii
        r1 = np.min([np.linalg.norm(top_left[:2] - center),
                    np.linalg.norm(top_right[:2] - center)])
        r2 = np.max([np.linalg.norm(bottom_left[:2] - center),
                    np.linalg.norm(bottom_right[:2] - center)])

        # Draw fan mask
        mask_array = MaskUtils.draw_circle_segment(mask_array, center, r2, angles[0], angles[2], value)
        mask_array = MaskUtils.draw_circle_segment(mask_array, center, r1, angles[0], angles[2], 0)

        return mask_array

    @staticmethod
    def draw_circle_segment(image: np.ndarray, center: np.ndarray, radius: float,
                          start_angle: float, end_angle: float, color: int) -> np.ndarray:
        """Draw circle segment for fan mask"""
        mask = np.zeros_like(image)

        start_angle_rad = np.deg2rad(start_angle)
        end_angle_rad = np.deg2rad(end_angle)

        thetas = np.linspace(start_angle_rad, end_angle_rad, 360)
        xs = center[0] + radius * np.cos(thetas)
        ys = center[1] + radius * np.sin(thetas)

        pts = np.array([np.round(xs), np.round(ys)]).T.astype(int)
        cv2.polylines(mask, [pts], False, color, 1)
        cv2.line(mask, tuple(center.astype(int)), tuple(pts[0]), color, 1)
        cv2.line(mask, tuple(center.astype(int)), tuple(pts[-1]), color, 1)
        cv2.fillPoly(mask, [np.vstack([center, pts])], color)

        return cv2.bitwise_or(image, mask)

    @staticmethod
    def update_overlay_volume(overlay_volume: vtkMRMLVectorVolumeNode,
                            current_volume: vtkMRMLScalarVolumeNode,
                            mask_array: Optional[np.ndarray] = None) -> None:
        """Update overlay volume with mask"""
        if not overlay_volume or not current_volume:
            return

        # Get current volume array
        current_volume_array = slicer.util.arrayFromVolume(current_volume)
        
        # Create RGB overlay
        overlay_shape = (1, current_volume_array.shape[1], current_volume_array.shape[2], 3)
        rgb_mask = np.zeros(overlay_shape, dtype=np.uint8)
        
        if mask_array is not None:
            # Convert mask to RGB
            rgb_mask[0, :, :, 1] = (mask_array > 0) * 255  # Green channel
        
        # Update overlay volume
        overlay_volume.SetSpacing(current_volume.GetSpacing())
        overlay_volume.SetOrigin(current_volume.GetOrigin())
        ijk_to_ras = vtk.vtkMatrix4x4()
        current_volume.GetIJKToRASMatrix(ijk_to_ras)
        overlay_volume.SetIJKToRASMatrix(ijk_to_ras)

        # Create image data
        overlay_image_data = vtk.vtkImageData()
        overlay_image_data.SetDimensions(current_volume_array.shape[2], current_volume_array.shape[1], 1)
        overlay_image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
        overlay_volume.SetAndObserveImageData(overlay_image_data)
        
        # Update array
        slicer.util.updateVolumeFromArray(overlay_volume, rgb_mask)

    @staticmethod
    def find_four_corners(mask: np.ndarray) -> Optional[np.ndarray]:
        """Find the four corners of the foreground in the mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx_corners = cv2.approxPolyDP(contour, epsilon, True)
        approx_corners = approx_corners.reshape(-1, 2)

        if len(approx_corners) < 3:
            return None

        return MaskUtils.find_extreme_corners(approx_corners)

    @staticmethod
    def find_extreme_corners(points: np.ndarray) -> Optional[np.ndarray]:
        """Find extreme corners from a set of points"""
        points = np.array(points)
        top_left = list(points[np.argmin(points[:, 0] + points[:, 1])])
        top_right = list(points[np.argmax(points[:, 0] - points[:, 1])])
        bottom_left = list(points[np.argmin(points[:, 0] - points[:, 1])])
        bottom_right = list(points[np.argmax(points[:, 0] + points[:, 1])])

        corners = [tuple(top_left), tuple(top_right), tuple(bottom_left), tuple(bottom_right)]
        unique_corners = set(corners)
        num_unique_corners = len(unique_corners)

        epsilon = 2
        if num_unique_corners == 3:
            top_point = top_left if top_left[1] < top_right[1] else top_right
            top_left = list(top_point)
            top_right = list(top_point)
            top_left[0] -= epsilon
            top_right[0] += epsilon

        if num_unique_corners < 3:
            return None

        return np.array([top_left, top_right, bottom_left, bottom_right]) 