import os
import logging
import vtk
import numpy as np
import cv2
from typing import Optional, List, Dict, Any, Tuple
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic
from slicer.util import VTKObservationMixin
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLVectorVolumeNode, vtkMRMLSequenceBrowserNode

class BaseUltrasoundLogic(ScriptedLoadableModuleLogic, VTKObservationMixin):
    """Base class for ultrasound module logic components"""

    def __init__(self) -> None:
        """Initialize base logic class"""
        ScriptedLoadableModuleLogic.__init__(self)
        VTKObservationMixin.__init__(self)
        
        # Common variables
        self.dicom_manager = None
        self.sequenceBrowserNode = None
        self.parameterNode = None

    def getCurrentProxyNode(self) -> Optional[vtkMRMLScalarVolumeNode]:
        """Get the proxy node of the master sequence node of the currently selected sequence browser node"""
        if not self.parameterNode or not self.parameterNode.ultrasoundSequenceBrowser:
            logging.error("Current sequence browser node not found")
            return None

        masterSequenceNode = self.parameterNode.ultrasoundSequenceBrowser.GetMasterSequenceNode()
        if not masterSequenceNode:
            logging.error("Master sequence node missing")
            return None

        proxyNode = self.parameterNode.ultrasoundSequenceBrowser.GetProxyNode(masterSequenceNode)
        if not proxyNode:
            logging.error("Proxy node not found")
            return None

        return proxyNode

    def createFanMask(self, imageArray: np.ndarray, topLeft: np.ndarray, topRight: np.ndarray, 
                     bottomLeft: np.ndarray, bottomRight: np.ndarray, value: int = 255) -> np.ndarray:
        """Create fan-shaped mask"""
        image_size_rows = imageArray.shape[1]
        image_size_cols = imageArray.shape[2]
        mask_array = np.zeros((image_size_rows, image_size_cols), dtype=np.uint8)

        # Calculate angles and intersection point
        center = np.array([image_size_cols / 2, image_size_rows / 2])
        
        # Calculate angles for each corner
        angles = []
        for point in [topLeft, topRight, bottomRight, bottomLeft]:
            angle = np.arctan2(point[1] - center[1], point[0] - center[0])
            angles.append(np.degrees(angle))
        
        # Sort angles
        angles.sort()
        
        # Calculate radii
        r1 = np.min([np.linalg.norm(topLeft[:2] - center),
                    np.linalg.norm(topRight[:2] - center)])
        r2 = np.max([np.linalg.norm(bottomLeft[:2] - center),
                    np.linalg.norm(bottomRight[:2] - center)])

        # Draw fan mask
        mask_array = self.draw_circle_segment(mask_array, center, r2, angles[0], angles[2], value)
        mask_array = self.draw_circle_segment(mask_array, center, r1, angles[0], angles[2], 0)

        return mask_array

    def draw_circle_segment(self, image: np.ndarray, center: np.ndarray, radius: float, 
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

    def line_coefficients(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float, float]:
        """Returns the coefficients of the line equation Ax + By + C = 0"""
        if p1[0] == p2[0]:  # Vertical line
            A = 1
            B = 0
            C = -p1[0]
        else:
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
            A = -m
            B = 1
            C = m * p1[0] - p1[1]
        return A, B, C

    def updateOverlayVolume(self, overlayVolume: vtkMRMLVectorVolumeNode, 
                          currentVolumeNode: vtkMRMLScalarVolumeNode) -> None:
        """Update overlay volume with current settings"""
        if not overlayVolume or not currentVolumeNode:
            return

        # Get current volume array
        currentVolumeArray = slicer.util.arrayFromVolume(currentVolumeNode)
        
        # Create RGB overlay
        overlay_shape = (1, currentVolumeArray.shape[1], currentVolumeArray.shape[2], 3)
        rgb_mask = np.zeros(overlay_shape, dtype=np.uint8)
        
        # Update overlay volume
        overlayVolume.SetSpacing(currentVolumeNode.GetSpacing())
        overlayVolume.SetOrigin(currentVolumeNode.GetOrigin())
        ijkToRas = vtk.vtkMatrix4x4()
        currentVolumeNode.GetIJKToRASMatrix(ijkToRas)
        overlayVolume.SetIJKToRASMatrix(ijkToRas)

        # Create image data
        overlayImageData = vtk.vtkImageData()
        overlayImageData.SetDimensions(currentVolumeArray.shape[2], currentVolumeArray.shape[1], 1)
        overlayImageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
        overlayVolume.SetAndObserveImageData(overlayImageData)
        
        # Update array
        slicer.util.updateVolumeFromArray(overlayVolume, rgb_mask)

    def clearScene(self) -> None:
        """Clear the scene of all nodes"""
        if self.sequenceBrowserNode:
            self.removeObserver(self.sequenceBrowserNode, vtk.vtkCommand.ModifiedEvent, 
                              self.onSequenceBrowserModified)
            self.sequenceBrowserNode = None
        slicer.mrmlScene.Clear()

    def onSequenceBrowserModified(self, caller, event) -> None:
        """Handle sequence browser modification"""
        pass  # To be implemented by derived classes
