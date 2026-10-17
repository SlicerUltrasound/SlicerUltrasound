import csv
import json
import logging
import math
import numpy as np
import os
import glob
import pydicom
import qt
import shutil
import slicer
import vtk
import colorsys
import copy
import re
import zlib
import datetime
import colorsys

# Import AnnotateUltrasound base module
import AnnotateUltrasound as annotate

try:
    import pandas as pd
except ImportError:
    slicer.util.pip_install('pandas')
    import pandas as pd

try:
    import cv2
except ImportError:
    slicer.util.pip_install('opencv-python')
    import cv2

from collections import defaultdict
from DICOMLib import DICOMUtils
from typing import Annotated, Optional

from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLVectorVolumeNode
from slicer import vtkMRMLNode
FAN_SATURATION=0.8 #(between 0-1.0) Smaller the number, the more the saturation is reduced of the overlay fan
#
# AdjudicateUltrasound
#

class AdjudicateUltrasound(annotate.AnnotateUltrasound):
    """AdjudicateUltrasound module, subclassing AnnotateUltrasound.AnnotateUltrasound."""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Adjudicate Ultrasound"
        icon_path = os.path.join(os.path.dirname(__file__), 'Resources', 'Icons', 'AdjudicateUltrasound.png')
        self.parent.icon = qt.QIcon(icon_path)
        self.parent.categories = ["Ultrasound"]
        self.parent.dependencies = []
        self.parent.contributors = ["Tamas Ungi (Queen's University)"]
        self.parent.helpText = f"""
This module facilitates the process of adjudicating segmentations of B-lines and the pleura in series of B-mode lung ultrasound videos.<br><br>

See more information in <a href="https://github.com/SlicerUltrasound/SlicerUltrasound/blob/main/README.md">README</a> <a href="https://github.com/SlicerUltrasound/SlicerUltrasound/tree/main/AdjudicateUltrasound">Source Code</a>.
"""
        self.parent.acknowledgementText = """
This file was originally developed by Tamas Ungi (Queen's University), with support from MLSC Bits to Bytes grant for Point of Care Ultrasound, and NIH grants R21EB034075 and R01EB035679.
"""
        slicer.app.connect("startupCompleted()", postModuleDiscoveryTasks)

#
# Register sample data sets in Sample Data module
#

def postModuleDiscoveryTasks():
    """
    Performs initialization tasks after Slicer has been fully loaded.
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.
    pass


#
# AdjudicateUltrasoundParameterNode
#

@parameterNodeWrapper
class AdjudicateUltrasoundParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """
    inputVolume: vtkMRMLScalarVolumeNode
    depthGuideVolume: vtkMRMLScalarVolumeNode
    overlayVolume: vtkMRMLVectorVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    invertedVolume: vtkMRMLScalarVolumeNode
    lineBeingPlaced: vtkMRMLNode = None
    dfLoaded: bool = False
    pleuraPercentage: float = -1.0
    unsavedChanges: bool = False
    depthGuideVisible: bool = True
    rater = ''
    showInvalidAndDuplicate: bool = True

#
# AdjudicateUltrasoundWidget
#
#
# global singleton instance of the widget

class CustomObserverMixin:
    def addObserver(self, obj, event, method, group="none", priority=0.0):
        if not hasattr(self, '_VTKObservationMixin__observations'):
            self._VTKObservationMixin__observations = {}

        events = self._VTKObservationMixin__observations.setdefault(obj, {})
        methods = events.setdefault(event, {})

        if method in methods:
            return  # Observer already added

        tag = obj.AddObserver(event, method, priority)
        methods[method] = (group, tag, priority)

    def removeObserver(self, obj, event, method):
        if not hasattr(self, '_VTKObservationMixin__observations'):
            return

        try:
            events = self._VTKObservationMixin__observations[obj]
            methods = events[event]
            group, tag, priority = methods.pop(method)
            obj.RemoveObserver(tag)

            if not methods:
                del events[event]
            if not events:
                del self._VTKObservationMixin__observations[obj]

        except KeyError:
            raise KeyError(f"No observer found for: {obj}, {event}, {method}")
adjudicateUltrasoundWidgetInstance = None
def getAdjudicateUltrasoundWidget():
    """
    Get the singleton instance of the AdjudicateUltrasoundWidget.
    """
    global adjudicateUltrasoundWidgetInstance
    if adjudicateUltrasoundWidgetInstance is None:
        raise RuntimeError("AdjudicateUltrasoundWidget instance is not initialized")
    return adjudicateUltrasoundWidgetInstance

class AdjudicateUltrasoundWidget(annotate.AnnotateUltrasoundWidget,CustomObserverMixin):
    """AdjudicateUltrasoundWidget, subclassing AnnotateUltrasound.AnnotateUltrasoundWidget."""
    def __init__(self, parent=None) -> None:
        # Set up logic to point to AdjudicateUltrasoundLogic
        self.logic = AdjudicateUltrasoundLogic()

        super().__init__(parent)

        self.updatingGUI = False
        self._parameterNode = None

        self._parameterNodeGuiTag = None
        self.notEnteredYet = True
        self._lastFrameIndex = -1

        # Flag to track if the user manually expanded the rater table
        self._userManuallySetRaterTableState = False
        self._lastUserManualCollapsedState = None  # Track the last state the user manually set

    def _updateCornerAnnotation(self):
        """
        Update the corner annotation with per-rater % pleura.
        Called from updateGuiFromAnnotations (on frame change) and from _updateGUIFromParamaterNode (on parameter change)
        """
        # Update corner annotation with _parameterNode.pleuraPercentage
        selectedRaters = self.logic.getSelectedRaters()
        view = slicer.app.layoutManager().sliceWidget("Red").sliceView()
        if selectedRaters is not None:
            if self.ui.showPleuraPercentageCheckBox.checked and self._parameterNode.pleuraPercentage >= 0:
                rater_pcts = getattr(self.logic, "raterPercentages", {})
                if rater_pcts:
                    lines = [f"{r}: {p:.1f}%" for r, p in sorted(rater_pcts.items())]
                    text = "\n".join(lines)
                else:
                    text = f"B-line/Pleura = {self._parameterNode.pleuraPercentage:.1f} %"
                view.cornerAnnotation().SetText(vtk.vtkCornerAnnotation.UpperLeft, text)
                view.cornerAnnotation().GetTextProperty().SetColor(1, 1, 0)
                view.forceRender()
            else:
                view.cornerAnnotation().SetText(vtk.vtkCornerAnnotation.UpperLeft, "")
                view.forceRender()
        else:
            view.cornerAnnotation().SetText(vtk.vtkCornerAnnotation.UpperLeft, "")
            view.forceRender()

    def updateGuiFromAnnotations(self):
        super().updateGuiFromAnnotations()
        self._updateCornerAnnotation() # Update corner annotation with % pleura for all raters when annotations are updated
    
    # def onNextButton(self):
    #     super().onNextButton()
    #     self._updateCornerAnnotation() # Update corner annotation with % pleura for all raters when going to next scan


    def resourcePath(self, filename):
        """Return the absolute path of the module ``Resources`` directory."""
        # since we inherit from AnnotateUltrasound and use its AnnotateUltrasound.ui, we use its resource path
        annotatePath = os.path.dirname(slicer.util.modulePath("AnnotateUltrasound"))
        return os.path.join(annotatePath, "Resources", filename)

    def initializeShortcuts(self):
        super().initializeShortcuts()

    def populateRaterColorTable(self):
        if not hasattr(self.ui, 'raterColorTable'):
            return
        self.ui.raterColorTable.blockSignals(True)
        self.ui.raterColorTable.clearContents()
        colors = list(self.logic.getAllRaterColors())

        #!!!!!!!!!Added: We also filter out current_rater from the display table because we are adjudicating and not annotating
        # Filter out __selected_node__ and __adjudicated_node__ before setting row count, we don't want to show it in the UI.
        # Note: populateRaterColorTable is shared with AdjudicatedUltrasound, it is leaking __adjudicated_node__ knowledge
        # but better than copying it to AdjudicatedUltrasound for this and we do not want to have the __selected_node__ "red"
        # and __adjudicated_node__ "blue"/"magenta" colors used by raters in any module.
        current_rater=self.logic.getCurrentRater()  #Added
        visible_colors = [(r, (pleura_color, bline_color)) for r, (pleura_color, bline_color) in colors
                  if r != "__selected_node__" and r != "__adjudicated_node__"
                  and r != current_rater] #Added the filter of current_rater

        self.ui.raterColorTable.setRowCount(len(visible_colors))
        self.ui.raterColorTable.setColumnCount(3)
        self.ui.raterColorTable.setHorizontalHeaderLabels(["Rater", "Pleura", "B-line"])
        header = self.ui.raterColorTable.horizontalHeader()
        header.setSectionResizeMode(0, qt.QHeaderView.Stretch)
        # Columns 1 & 2: Color indicators — just enough to show the color
        header.setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, qt.QHeaderView.ResizeToContents)

        self.ui.raterColorTable.setColumnWidth(1, 30)
        self.ui.raterColorTable.setColumnWidth(2, 30)
        for row, (r, (pleura_color, bline_color)) in enumerate(visible_colors):
            rater_item = qt.QTableWidgetItem(r)
            rater_item.setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
            if not hasattr(self, "selectedRaters") or r in self.selectedRaters:
                rater_item.setCheckState(qt.Qt.Checked)
            else:
                rater_item.setCheckState(qt.Qt.Unchecked)

            pleura_item = qt.QTableWidgetItem()
            pleura_item.setFlags(qt.Qt.ItemIsEnabled)
            pleura_item.setBackground(qt.QColor(*(int(c * 255) for c in pleura_color)))

            bline_item = qt.QTableWidgetItem()
            bline_item.setFlags(qt.Qt.ItemIsEnabled)
            bline_item.setBackground(qt.QColor(*(int(c * 255) for c in bline_color)))

            self.ui.raterColorTable.setItem(row, 0, rater_item)
            self.ui.raterColorTable.setItem(row, 1, pleura_item)
            self.ui.raterColorTable.setItem(row, 2, bline_item)
        self.ui.raterColorTable.blockSignals(False)
    

    def _createAdjudicationShortcuts(self):
        # Remove old shortcuts if needed
        for name in dir(self):
            if name.startswith("adjudicationShortcut"):
                try:
                    getattr(self, name).activated.disconnect()
                    getattr(self, name).setParent(None)
                    delattr(self, name)
                except Exception:
                    pass

        def make_shortcut(attr_name, key, slot):
            shortcut = qt.QShortcut(qt.QKeySequence(key), slicer.util.mainWindow())
            shortcut.setContext(qt.Qt.ApplicationShortcut)
            shortcut.activated.connect(slot)
            setattr(self, attr_name, shortcut)

        make_shortcut("adjudicationShortcutV", "V", self.onToggleValidation)
        make_shortcut("adjudicationShortcutTab", "Tab", self.selectNextVisibleLine)
        make_shortcut("adjudicationShortcutShiftTab", "Shift+Tab", self.selectPreviousVisibleLine)
        make_shortcut("adjudicationShortcutR", "R", self.onResetAllAdjudication)

    def connectKeyboardShortcuts(self):
        super().connectKeyboardShortcuts()
        # disconnect the shortcut for add line, remove line, clear all lines etc as we don't want
        # to allow modifying the annotations
        self.disconnectDrawingShortcuts()
        self._createAdjudicationShortcuts()

    def disconnectKeyboardShortcuts(self):
        super().disconnectKeyboardShortcuts()
        # Disconnect AdjudicateUltrasound-specific shortcuts
        if hasattr(self, "_adjudicationShortcuts"):
            for shortcut in self._adjudicationShortcuts:
                shortcut.activated.disconnect()
                shortcut.setParent(None)
            self._adjudicationShortcuts.clear()

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        super().setup()
        self.connectKeyboardShortcuts()
        # Set up state tracking for rater table collapse handling
        self._userManuallySetRaterTableState = False
        self._lastUserManualCollapsedState = True
        # Ensure programmatic collapse is not interpreted as manual
        self._ignoreCollapsedChangedSignal = False
        if hasattr(self.ui, 'raterColorsCollapsibleButton'):
            self.ui.raterColorsCollapsibleButton.blockSignals(True)
            self.ui.raterColorsCollapsibleButton.collapsed = True
            self.ui.raterColorsCollapsibleButton.blockSignals(False)

        # Update directory button directory from settings
        self.ui.inputDirectoryButton.directory = slicer.app.settings().value("AdjudicateUltrasound/InputDirectory", "")

        # Reconnect readInputButton, inputDirectoryButton, saveButton, and saveAndLoadNextButton
        # to the overridden methods
        for name, method in {
            "readInputButton": self.onReadInputButton,
            "inputDirectoryButton": self.onInputDirectorySelected,
            "saveButton": self.onSaveButton,
            "saveAndLoadNextButton": self.onSaveAndLoadNextButton,
            "nextButton": self.onNextButton
        }.items():
            btn = getattr(self.ui, name, None)
            if not btn:
                continue
            try:
                if name == "inputDirectoryButton":
                    btn.directoryChanged.disconnect()
                    btn.directoryChanged.connect(method)
                else:
                    btn.clicked.disconnect()
                    btn.clicked.connect(method)
            except TypeError:
                # Not connected yet
                if name == "inputDirectoryButton":
                    btn.directoryChanged.connect(method)
                else:
                    btn.clicked.connect(method)

        # Hide drawing buttons for Add, Remove and Clear All Lines
        self.ui.addPleuraButton.setVisible(False)
        self.ui.addBlineButton.setVisible(False)
        self.ui.removePleuraButton.setVisible(False)
        self.ui.removeBlineButton.setVisible(False)
        self.ui.clearAllLinesButton.setVisible(False)
        self._adjustOverlayButtonLayout()
        # Also hide the Add and Remove Current Frame buttons
        self.ui.addCurrentFrameButton.setVisible(False)
        self.ui.removeCurrentFrameButton.setVisible(False)

        # Move adjudication-related widgets out of workflowCollapsibleButton and into their own layout
        # Find the parent layout containing workflowCollapsibleButton
        raterNameLabel = self.ui.raterNameLabel
        raterNameLabel.setText("Adjudicator:")
        parentLayout = self.ui.workflowCollapsibleButton.parent().layout()
        # Only create if not already present
        if not hasattr(self, "_adjudicationToolsWidget") or self._adjudicationToolsWidget is None:
            self._adjudicationShortcuts = []
            # Create a widget and layout for adjudication controls
            self._adjudicationToolsWidget = qt.QWidget()
            self._adjudicationToolsWidget.setObjectName("adjudicationToolsWidget")
            adjudicationToolsLayout = qt.QVBoxLayout()
            adjudicationToolsLayout.setSpacing(4)
            adjudicationToolsLayout.setContentsMargins(4, 4, 4, 4)

            def make_button(label, tooltip, shortcutKey, slot):
                btn = qt.QPushButton(label)
                btn.setToolTip(tooltip)
                btn.clicked.connect(slot)
                return btn

            # Row 1: Validate/Invalidate/Unadjudicated
            row1 = qt.QHBoxLayout()
            row1.addWidget(make_button("Toggle valid/invalid/unmarked for Line [V]", "Toggle selected line between valid/invalid/unmarked (Shortcut V key)", "V", self.onToggleValidation))
            adjudicationToolsLayout.addLayout(row1)

            # Row 2: Next/Prev visible
            row2 = qt.QHBoxLayout()
            row2.addWidget(make_button("Next Visible [Tab]", "Navigate to next visible annotation line (Shortcut Tab)", "Tab", self.selectNextVisibleLine))
            row2.addWidget(make_button("Prev Visible [Shift+Tab]", "Navigate to previous visible annotation line (Shortcut Shift+Tab)", "Shift+Tab", self.selectPreviousVisibleLine))
            adjudicationToolsLayout.addLayout(row2)

            # Row 3: Validate/Invalidate rest
            row3 = qt.QHBoxLayout()
            row3.addWidget(make_button("Validate Rest", "Mark all visible unadjudicated lines as validated", "", self.onValidateAllUnadjudicated))
            row3.addWidget(make_button("Invalidate Rest", "Mark all visible unadjudicated lines as invalidated", "", self.onInvalidateAllUnadjudicated))
            row3.addWidget(make_button("Validate Clip", "Mark all unadjudicated lines as validated for each frame in current clip", "", self.onValidateClip))
            adjudicationToolsLayout.addLayout(row3)

            # Reset all row
            resetRow = qt.QHBoxLayout()
            resetRow.addWidget(make_button("Reset All [R]", "Reset all lines to unadjudicated (Shortcut R key)", "R", self.onResetAllAdjudication))
            adjudicationToolsLayout.addLayout(resetRow)

            # Checkbox for show invalidated
            self.showInvalidatedCheckBox = qt.QCheckBox("Show Invalidated Lines")
            self.showInvalidatedCheckBox.setToolTip("Show Invalidated Lines")
            self.showInvalidatedCheckBox.stateChanged.connect(self.onShowInvalidToggled)
            self.showInvalidatedCheckBox.setChecked(True)
            adjudicationToolsLayout.addWidget(self.showInvalidatedCheckBox)

            self._adjudicationToolsWidget.setLayout(adjudicationToolsLayout)
            # Insert below workflowCollapsibleButton
            index = parentLayout.indexOf(self.ui.workflowCollapsibleButton)
            parentLayout.insertWidget(index + 1, self._adjudicationToolsWidget)

        # Ensure parameter node is initialized after setup is complete
        self.initializeParameterNode()
        # Load fixed labels file
        self.onLabelsFileSelected()

    def cleanup(self):
        super().cleanup()
        self.disconnectKeyboardShortcuts()

    def onDepthGuideToggled(self, toggled):
        # Save new state in application settings and update depth guide volume to show/hide the depth guide
        settings = slicer.app.settings()
        settings.setValue('AdjudicateUltrasound/DepthGuide', toggled)
        if toggled:
            self.logic.parameterNode.depthGuideVisible = True
        else:
            self.logic.parameterNode.depthGuideVisible = False
        self.logic.updateDepthGuideVolume()

    def onRaterNameChanged(self):
        if self._parameterNode:
            self._parameterNode.rater = self.ui.raterName.text.strip().lower()
            statusText = f"Adjudicator name changed to {self._parameterNode.rater}"
            slicer.util.mainWindow().statusBar().showMessage(statusText, 3000)

    def findNextUnlabeledScan(self):
        """
        Find the index of the next unadjudicated scan in the DICOM dataframe.
        :return: Index of the next unadjudicated scan or None if no such scan is found.
        """
        if self.logic.dicomDf is None:
            return None

        for idx in range(self.logic.nextDicomDfIndex, len(self.logic.dicomDf)):
            inputDirectory = self.logic.dicomDf.iloc[idx]['InputDirectory']
            dcm_filepath = self.logic.dicomDf.iloc[idx]['Filepath']
            base_name = os.path.splitext(os.path.basename(dcm_filepath))[0]
            adjudication_file = os.path.join(inputDirectory, f"{base_name}.adjudication.json")

            # Check if the annotation file exists
            if not os.path.exists(adjudication_file):
                # File doesn't exist, so this scan is unlabeled
                return idx

            try:
                with open(adjudication_file, 'r') as f:
                    annotations = json.load(f)
                    # Check if frame annotations exist and are empty
                    if 'frame_annotations' not in annotations or not annotations['frame_annotations']:
                        return idx
            except Exception as e:
                logging.error(f"Error reading annotations file {adjudication_file}: {e}")
                # If there's an error reading the file, treat it as unlabeled
                return idx

        return None

    def saveUserSettings(self):
        settings = qt.QSettings()
        settings.setValue('AdjudicateUltrasound/ShowPleuraPercentage', self.ui.showPleuraPercentageCheckBox.checked)
        settings.setValue('AdjudicateUltrasound/DepthGuide', self.ui.depthGuideCheckBox.checked)
        settings.setValue('AdjudicateUltrasound/Rater', self.ui.raterName.text.strip())
        ratio=self.logic.updateOverlayVolume()

        if ratio is not None:
            self._parameterNode.pleuraPercentage = ratio * 100

        self._updateGUIFromParameterNode()

    def onClearAllLines(self):
        logging.debug('onClearAllLines')
        self.logic.clearAllLines()
        ratio = self.logic.updateOverlayVolume()
        if ratio is not None:
            self._parameterNode.unsavedChanges = True
        self.updateGuiFromAnnotations()

    def onInputDirectorySelected(self):
        logging.info('!!!!!!!!!!onInputDirectorySelected Pressed!!!!!!!!!')

        inputDirectory = self.ui.inputDirectoryButton.directory
        if not inputDirectory:
            statusText = '⚠️ Please select an input directory'
            slicer.util.mainWindow().statusBar().showMessage(statusText, 5000)
            self.ui.statusLabel.setText(statusText)
            return

        # Update local settings
        slicer.app.settings().setValue("AdjudicateUltrasound/InputDirectory", inputDirectory)

    def onToggleValidation(self):
        # Get the selected markup node
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectedNodeID = selectionNode.GetActivePlaceNodeID()
        if not selectedNodeID:
            slicer.util.showStatusMessage(f"No line selected to adjudicate.", 3000)
            return
        markupNode = slicer.mrmlScene.GetNodeByID(selectedNodeID)
        if not markupNode or not markupNode.GetClassName() == "vtkMRMLMarkupsLineNode":
            slicer.util.showStatusMessage("Selected node is not a line.", 3000)
            return
        # Only allow validation if node is visible
        displayNode = markupNode.GetDisplayNode()
        if not displayNode or not displayNode.GetVisibility():
            slicer.util.showStatusMessage("Cannot adjudicate: selected line is not visible.", 3000)
            return
        validation_json = markupNode.GetAttribute("validation")
        current_status = None
        if validation_json:
            try:
                validation = json.loads(validation_json)
                current_status = validation.get("status", None)
            except Exception:
                current_status = None

        # Set next status based on current status
        if current_status == "validated":
            self._setLineValidationStatus("invalidated", "invalidate", "invalidated")
            displayNode.SetOpacity(1.0)
        elif current_status == "invalidated":
            self._setLineValidationStatus("unadjudicated", "unadjudicate", "unadjudicated")
            displayNode.SetOpacity(1.0)
        else:
            self._setLineValidationStatus("validated", "validate", "validated")

    def _setLineValidationStatus(self, status, action_verb, status_description):
        """
        Set the validation status of the selected line.
        Args:
            status: The validation status to set ("validated", "invalidated")
            action_verb: The verb for the error message (e.g., "validate", "invalidate")
            status_description: The description for the success message (e.g., "validated", "invalidated")
        """
        # Get the selected markup node
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectedNodeID = selectionNode.GetActivePlaceNodeID()
        if not selectedNodeID:
            slicer.util.showStatusMessage(f"No line selected to {action_verb}.", 3000)
            return
        markupNode = slicer.mrmlScene.GetNodeByID(selectedNodeID)
        if not markupNode or not markupNode.GetClassName() == "vtkMRMLMarkupsLineNode":
            slicer.util.showStatusMessage("Selected node is not a line.", 3000)
            return
        # Only allow validation if node is visible
        displayNode = markupNode.GetDisplayNode()
        if not displayNode or not displayNode.GetVisibility():
            slicer.util.showStatusMessage("Cannot adjudicate: selected line is not visible.", 3000)
            return
        # Get current rater (adjudicator)
        adjudicator = self.getCurrentRater()
        validation = {
            "status": status,
            "adjudicator": adjudicator,
            "timestamp": datetime.datetime.now().isoformat()
        }
        markupNode.SetAttribute("validation", json.dumps(validation))
        # Set opacity based on status
        if status == "validated":
            displayNode.SetOpacity(1.0)
        else:
            displayNode.SetOpacity(0.3)
        # Also update the corresponding annotation line
        self._updateAnnotationLineValidation(markupNode, validation)
        # sync the annotations back to markup nodes to change any display properties for the line
        self.logic.syncAnnotationsToMarkups()
        self.logic.refreshDisplay(updateOverlay=True, updateGui=True)
        slicer.util.showStatusMessage(f"Line {status_description}.", 3000)

    def onValidateAllUnadjudicated(self):
        self._setAllUnadjudicatedLinesStatus("validated", "Validated", 1.0)

    def onInvalidateAllUnadjudicated(self):
        self._setAllUnadjudicatedLinesStatus("invalidated", "Invalidated", 0.3)

    def onValidateClip(self):
        """Validate all unadjudicated lines for each frame in the current clip."""
        if self.logic.sequenceBrowserNode is None:
            slicer.util.errorDisplay("No sequence loaded. Please load a DICOM sequence first.")
            return

        if self.logic.annotations is None or 'frame_annotations' not in self.logic.annotations:
            slicer.util.errorDisplay("No annotations found for this sequence.")
            return

        total_validated = 0
        frames_processed = 0

        # Get the total number of frames in the sequence
        total_frames = self.logic.sequenceBrowserNode.GetNumberOfItems()

        # Show progress dialog
        progress_dialog = qt.QProgressDialog("Validating clip...", "Cancel", 0, total_frames)
        progress_dialog.setWindowModality(qt.Qt.WindowModal)
        progress_dialog.show()

        # Set programmatic update flag to prevent unsavedChanges from being set
        self.logic._isProgrammaticUpdate = True

        try:
            # Loop through each frame in the clip
            for frame_index in range(total_frames):
                if progress_dialog.wasCanceled:
                    break

                # Set the current frame
                self.logic.sequenceBrowserNode.SetSelectedItemNumber(frame_index)
                slicer.app.processEvents()  # Allow UI to update

                # Validate all unadjudicated lines in the current frame
                frame_validated = self._validateUnadjudicatedLinesInCurrentFrame("validated")

                total_validated += frame_validated
                if frame_validated > 0:
                    frames_processed += 1

                # Update progress
                progress_dialog.setValue(frame_index + 1)
                progress_dialog.setLabelText(f"Validating frame {frame_index + 1}/{total_frames}...")
                slicer.app.processEvents()

        finally:
            # Reset programmatic update flag
            self.logic._isProgrammaticUpdate = False
            progress_dialog.close()

        # Return to the first frame
        self.logic.sequenceBrowserNode.SetSelectedItemNumber(0)
        slicer.app.processEvents()

        # Sync annotations to markups and refresh display
        self.logic.syncAnnotationsToMarkups()
        self.logic.refreshDisplay(updateOverlay=True, updateGui=True)

        # Show completion message
        if total_validated > 0:
            slicer.util.showStatusMessage(f"Validated {total_validated} lines across {frames_processed} frames in the clip.", 5000)
        else:
            slicer.util.showStatusMessage("No unadjudicated lines found in the clip.", 3000)

    def _validateUnadjudicatedLinesInCurrentFrame(self, status):
        """
        Validate all unadjudicated lines in the current frame.
        Args:
            status: The validation status to set ("validated", "invalidated")
        Returns:
            int: Number of lines validated
        """
        # Get current rater (adjudicator)
        adjudicator = self.getCurrentRater()
        count = 0

        for nodeIndex in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLMarkupsLineNode")):
            markupNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, "vtkMRMLMarkupsLineNode")
            displayNode = markupNode.GetDisplayNode()
            if not displayNode or not displayNode.GetVisibility():
                continue # Skip if not visible
            validation_json = markupNode.GetAttribute("validation")
            current_status = None
            if validation_json:
                try:
                    validation = json.loads(validation_json)
                    current_status = validation.get("status", None)
                except Exception:
                    current_status = None
            if not current_status or current_status == "unadjudicated":
                validation = {
                    "status": status,
                    "adjudicator": adjudicator,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                markupNode.SetAttribute("validation", json.dumps(validation))

                # Update the annotation data for this line
                self._updateAnnotationLineValidation(markupNode, validation)

                count += 1

        return count

    def _setAllUnadjudicatedLinesStatus(self, status, status_past_tense, opacity):
        """
        Set the validation status of all unadjudicated lines.
        Args:
            status: The validation status to set ("validated", "invalidated")
            status_past_tense: The past tense form for the success message (e.g., "Validated", "Invalidated")
            opacity: The opacity to set for the lines (1.0 for validated, 0.3 for others)
        """
        # Set programmatic update flag to prevent unsavedChanges from being set
        self.logic._isProgrammaticUpdate = True

        try:
            count = self._validateUnadjudicatedLinesInCurrentFrame(status)

            # Update the visual appearance of all lines since annotations are already updated above
            self.logic.syncAnnotationsToMarkups()

            # Refresh display
            self.logic.refreshDisplay(updateOverlay=True, updateGui=True)

        finally:
            # Reset programmatic update flag
            self.logic._isProgrammaticUpdate = False

        # Set unsavedChanges since this is a user action that modifies validation status
        self._parameterNode.unsavedChanges = True

        slicer.util.showStatusMessage(f"{status_past_tense} {count} unadjudicated lines.", 3000)

    def onShowInvalidToggled(self, checked):
        if self._parameterNode:
            self._parameterNode.showInvalidAndDuplicate = (checked != 0)
        for nodeIndex in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLMarkupsLineNode")):
            markupNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, "vtkMRMLMarkupsLineNode")
            validation_json = markupNode.GetAttribute("validation")
            status = None
            if validation_json:
                try:
                    validation = json.loads(validation_json)
                    status = validation.get("status", None)
                except Exception:
                    status = None
            if status == "invalidated":
                displayNode = markupNode.GetDisplayNode()
                if displayNode:
                    if checked:
                        displayNode.SetVisibility(True)
                        displayNode.SetOpacity(0.3)
                    else:
                        displayNode.SetVisibility(False)

        slicer.util.showStatusMessage(f"{'Showing' if checked else 'Hiding'} invalidated lines.", 3000)

    def getCurrentRater(self):
        # Return the current rater name from parameter node or UI
        if hasattr(self, '_parameterNode') and hasattr(self._parameterNode, 'rater'):
            return self._parameterNode.rater
        if hasattr(self.ui, 'raterName'):
            return self.ui.raterName.text().strip()
        return ""

    def _updateAnnotationLineValidation(self, markupNode, validation):
        """
        Update the validation field in self.logic.annotations for the line matching this markupNode in the current frame.
        """
        if self.logic.annotations is None or 'frame_annotations' not in self.logic.annotations:
            return
        if self.logic.sequenceBrowserNode is None:
            return
        currentFrameIndex = max(0, self.logic.sequenceBrowserNode.GetSelectedItemNumber())
        frame = next((item for item in self.logic.annotations['frame_annotations'] if int(item.get("frame_number", -1)) == currentFrameIndex), None)
        if not frame:
            return
        # Get rater and points from markupNode
        rater = markupNode.GetAttribute("rater")
        if not rater:
            logging.warning("Attempted to adjudicate a markup node with no rater set. Node name: %s, points: %s", markupNode.GetName(), [markupNode.GetNthControlPointPosition(i, [0,0,0]) for i in range(markupNode.GetNumberOfControlPoints())])
            return
        points = []
        for i in range(markupNode.GetNumberOfControlPoints()):
            coord = [0, 0, 0]
            markupNode.GetNthControlPointPosition(i, coord)
            points.append(coord)
        # Try to match in pleura_lines and b_lines
        for key in ["pleura_lines", "b_lines"]:
            for line in frame.get(key, []):
                if line.get("rater") == rater:
                    line_points = line.get("line", {}).get("points", [])
                    if len(line_points) == len(points) and all([all(abs(a-b)<1e-6 for a,b in zip(pt1,pt2)) for pt1,pt2 in zip(line_points, points)]):
                        line["validation"] = validation
                        return

    def _selectLineByFilter(self, filter_fn, direction, empty_message, status_message_prefix):
        # Gather all lines matching the filter in the current frame
        nodes = []
        for node in self.logic.pleuraLines + self.logic.bLines:
            if node.GetDisplayNode() and node.GetDisplayNode().GetVisibility() and filter_fn(node):
                nodes.append(node)
        if not nodes:
            slicer.util.showStatusMessage(empty_message, 3000)
            return
        # Find currently selected node
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectedNodeID = selectionNode.GetActivePlaceNodeID()
        try:
            idx = [n.GetID() for n in nodes].index(selectedNodeID)
            if direction == "next":
                new_idx = (idx + 1) % len(nodes)
            else:
                new_idx = (idx - 1) % len(nodes)
        except ValueError:
            new_idx = 0 if direction == "next" else len(nodes) - 1
        newNode = nodes[new_idx]
        selectionNode.SetActivePlaceNodeID(newNode.GetID())
        slicer.util.showStatusMessage(f"{status_message_prefix} {new_idx+1} of {len(nodes)}.", 2000)
        self.logic.syncAnnotationsToMarkups()
        self.logic.refreshDisplay(updateOverlay=True, updateGui=True)
        rater = newNode.GetAttribute("rater") or "unknown"
        line_type = "pleura" if newNode in self.logic.pleuraLines else "b-line"
        logging.info(f"{direction[0].upper()}: Selected {line_type} line from rater '{rater}' (line {new_idx+1} of {len(nodes)})")

    def _isUnadjudicated(self, node):
        """Check if a node is unadjudicated (no validation status or status is 'unadjudicated')."""
        validation_json = node.GetAttribute("validation")
        status = None
        if validation_json:
            try:
                validation = json.loads(validation_json)
                status = validation.get("status", None)
            except Exception:
                status = None
        return not status or status == "unadjudicated"

    def selectNextVisibleLine(self):
        self._selectLineByFilter(lambda node: True, "next", "No visible lines in this frame.", "Selected visible line")

    def selectPreviousVisibleLine(self):
        self._selectLineByFilter(lambda node: True, "previous", "No visible lines in this frame.", "Selected visible line")

    def onResetAllAdjudication(self):
        """Reset all lines in the current frame to unadjudicated status."""

        currentFrameIndex = max(0, self.logic.sequenceBrowserNode.GetSelectedItemNumber())
        frame = next((item for item in self.logic.annotations['frame_annotations'] if str(item.get("frame_number")) == str(currentFrameIndex)), None)
        for line in frame.get("pleura_lines", []):
            line["validation"] = {"status": "unadjudicated"}
        for line in frame.get("b_lines", []):
            line["validation"] = {"status": "unadjudicated"}
        self.logic.syncAnnotationsToMarkups() # sync the annotations back to markup nodes to change display properties for all the lines
        self.logic.refreshDisplay(updateOverlay=True, updateGui=True)
        slicer.util.showStatusMessage("All lines reset to unadjudicated.", 3000)

    def extractSeenAndSelectedRaters(self):
        """
        Extracts the set of raters that have contributed lines in the current annotations,
        ensuring the current rater is included even if not present in any frame annotations.
        Sets self.seenRaters to a sorted list of rater names.
        """
        # Use the Logic's centralized method to extract and set up raters
        super().extractSeenAndSelectedRaters()


    def saveAnnotations(self):
        """
        Saves current annotations to rater-specific json file.
        Returns True if save was successful, False otherwise.
        """
        try:
            # Add annotation line control points to the annotations dictionary and save it to file
            if self.logic.annotations is None:
                logging.error("saveAnnotations (adjudicate): No annotations loaded")
                return False

            # Check if rater name is set and not empty; if not, prompt user to enter one
            rater = self._parameterNode.rater
            if not rater:
                qt.QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Missing Rater Name",
                    "Rater name is not set. Please enter your rater name before saving."
                )
                self.ui.statusLabel.setText("⚠️ Please enter a rater name before saving.")
                return False

            waitDialog = self.createWaitDialog("Saving annotations", "Saving annotations...")

            # Check if any labels are checked
            annotationLabels = []
            for i in reversed(range(self.ui.labelsScrollAreaWidgetContents.layout().count())):
                widget = self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget()
                if not isinstance(widget, qt.QGroupBox):
                    continue
                # Find all checkboxes in groupBox
                for j in reversed(range(widget.layout().count())):
                    checkBox = widget.layout().itemAt(j).widget()
                    if isinstance(checkBox, qt.QCheckBox) and checkBox.isChecked():
                        # Use original category/label for saving
                        origCategory = checkBox.property('originalCategory')
                        origLabel = checkBox.property('originalLabel')
                        annotationLabels.append(f"{origCategory}/{origLabel}")
            self.logic.annotations['labels'] = annotationLabels

            # We save all lines for adjudication
            inputDirectory = self.logic.dicomDf.iloc[self.logic.nextDicomDfIndex - 1]['InputDirectory']
            file_path = self.logic.dicomDf.iloc[self.logic.nextDicomDfIndex - 1]['Filepath']
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            adjudication_file = os.path.join(inputDirectory, f"{base_name}.adjudication.json")
            # Save all lines (combined, no filtering) to adjudication file
            save_data = self.logic.convert_ras_to_lps(self.logic.annotations.get("frame_annotations", []))
            with open(adjudication_file, 'w') as f:
                json.dump(save_data, f)

            waitDialog.close()

            self._parameterNode.unsavedChanges = False

            statusText = f"✅ Adjudications saved successfully to {os.path.basename(adjudication_file)}"
            slicer.util.mainWindow().statusBar().showMessage(statusText, 3000)
            self.ui.statusLabel.setText(statusText)
            logging.info(f"Adjudications saved to {adjudication_file}")

            return True

        except Exception as e:
            statusText = f"❌ Failed to save adjudications: {str(e)}"
            slicer.util.mainWindow().statusBar().showMessage(statusText, 5000)
            self.ui.statusLabel.setText(statusText)
            logging.error(f"Error saving adjudications: {e}")
            return False

    def onSaveButton(self):
        """
        Saves current annotations to json file only
        """
        logging.info('onSaveButton (save)')
        success = self.saveAnnotations()
        if not success:
            # Error message already shown by saveAnnotations
            return

    def onSaveAndLoadNextButton(self):
        """
        Saves current annotations to json file and loads next sequence.
        """
        logging.info('onSaveAndLoadNextButton (save and load next scan)')

        success = self.saveAnnotations()
        if success:
            self.onNextButton()
            # Focus restoration is already handled by onNextButton(), no need to call it again
        else:
            # Error message already shown by saveAnnotations, don't proceed to next
            return

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        self.disconnectKeyboardShortcuts()  # Disconnect shortcuts when leaving the module

        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None

    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        widget = slicer.modules.adjudicateultrasound.widgetRepresentation()
        if widget and widget.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        # Guard: ensure logic is initialized
        if self.logic is None:
            logging.error("Logic not initialized before initializeParameterNode")
            return

        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

        settings = slicer.app.settings()
        showDepthGuide = settings.value('AdjudicateUltrasound/DepthGuide', False)
        # be consistent and just read bool, convert if was string
        if isinstance(showDepthGuide, str):
            showDepthGuide = showDepthGuide.lower() == 'true'
        self._parameterNode.rater = settings.value('AdjudicateUltrasound/Rater', '')
        self.ui.raterName.setText(self._parameterNode.rater)
        if self._parameterNode.rater != '':
            self.logic.setRater(self._parameterNode.rater)
            self.logic.getColorsForRater(self._parameterNode.rater)
        self.ui.depthGuideCheckBox.setChecked(showDepthGuide)

    def setParameterNode(self, inputParameterNode: AdjudicateUltrasoundParameterNode) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self.logic and self._parameterNode:
            self.logic.parameterNode = self._parameterNode

        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateGUIFromParameterNode)
            self._updateGUIFromParameterNode()

    def _updateGUIFromParameterNode(self, caller=None, event=None) -> None:
        if self.updatingGUI:
            return
        self.updatingGUI = True
        try:
            if self._parameterNode is None:
                logging.debug("No parameter node")
                return

            # Update line buttons
            if self._parameterNode.lineBeingPlaced is None:
                self.ui.addPleuraButton.setChecked(False)
                self.ui.addBlineButton.setChecked(False)
            else:
                if self._parameterNode.lineBeingPlaced.GetName() == "Pleura":
                    self.ui.addPleuraButton.setChecked(True)
                    self.ui.addBlineButton.setChecked(False)
                elif self._parameterNode.lineBeingPlaced.GetName() == "B-line":
                    self.ui.addPleuraButton.setChecked(False)
                    self.ui.addBlineButton.setChecked(True)
                else:
                    logging.error(f"Unknown line type {self._parameterNode.lineBeingPlaced.GetName()}")
                    return

            # If the frame index changed, then we want to make sure no row is selected in the frames table
            if self.logic.sequenceBrowserNode is not None:
                currentFrameIndex = self.logic.sequenceBrowserNode.GetSelectedItemNumber()
                if currentFrameIndex != self._lastFrameIndex:
                    self._lastFrameIndex = currentFrameIndex
                    self.ui.framesTableWidget.clearSelection()

            self._updateCornerAnnotation() #Adds percent pleura in corner for all raters

            # Update collapse/expand buttons
            if not self._parameterNode.dfLoaded:
                self.ui.inputsCollapsibleButton.collapsed = False
                self.ui.workflowCollapsibleButton.collapsed = True
                self.ui.sectorAnnotationsCollapsibleButton.collapsed = True
                self.ui.labelAnnotationsCollapsibleButton.collapsed = True
                # Collapse rater color table when no DICOM is loaded (no raters to display)
                if hasattr(self.ui, 'raterColorsCollapsibleButton'):
                    self._setRaterColorTableCollapsedState(True)
            else:
                self.ui.inputsCollapsibleButton.collapsed = True
                self.ui.workflowCollapsibleButton.collapsed = False
                self.ui.sectorAnnotationsCollapsibleButton.collapsed = False
                self.ui.labelAnnotationsCollapsibleButton.collapsed = False

                # Handle rater table collapse/expand logic
                if hasattr(self.ui, 'raterColorsCollapsibleButton'):
                    # During navigation, allow content updates but prevent collapse/expand state changes
                    if self._isNavigating:
                        # Skip collapse/expand logic during navigation, but still populate content
                        pass
                    else:
                        if self._userManuallySetRaterTableState and self._lastUserManualCollapsedState is not None:
                            self._setRaterColorTableCollapsedState(self._lastUserManualCollapsedState)

            # Save rater name to settings
            settings = qt.QSettings()
            settings.setValue('AdjudicateUltrasound/Rater', self.ui.raterName.text.strip())

            # Only update raterColorTable if present and DICOM is loaded
            if hasattr(self.ui, 'raterColorTable') and self._parameterNode.dfLoaded:
                self.populateRaterColorTable()
        finally:
            self.updatingGUI = False

#
# AdjudicateUltrasoundLogic
#

class AdjudicateUltrasoundLogic(annotate.AnnotateUltrasoundLogic):
    """
    AdjudicateUltrasoundLogic, subclassing annotate.AnnotateUltrasoundLogic.
    """

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        super().__init__()

        # These variables keep their values when the scene is cleared
        self.dicomDf = None
        self.nextDicomDfIndex = 0

        # These variables need to be reset when the scene is cleared
        self.annotations = None
        self.pleuraLines = []
        self.bLines = []
        self.sequenceBrowserNode = None
        self.depthGuideMode = 1
        self.seen_basenames = set()
        self.dcm_by_base = {}  # base_name → dcm filepath
        self.useFreeList = False

        # Flag to track when we're doing programmatic updates (to avoid setting unsavedChanges)
        self._isProgrammaticUpdate = False
        self._suppressSync = False

    # Static variable to track seen raters and their order
    seenRaters = []
    realRaters = []
    selectedRaters = []

    def _getOrCreateParameterNode(self):
        if not hasattr(self, "parameterNode"):
            baseWrapper = super()._getOrCreateParameterNode()
            rawNode = baseWrapper.parameterNode  # unwrap
            self.parameterNode = AdjudicateUltrasoundParameterNode(rawNode)
        return self.parameterNode

    def getParameterNode(self):
        return self._getOrCreateParameterNode()

    def getCurrentRater(self):
        return self.getParameterNode().rater.strip().lower()

    def updateInputDf(self, rater, input_folder):
        """
        Update the dicomDf dataframe with the DICOM files in the input folder.

        :param input_folder: The input folder to search for DICOM files.
        :return: The number of rows in the dataframe and the number of annotations files created.
        """
        logging.info('adjudicate: updateInputDf')

        dicom_data = []

        # Recursively walk through the input folder
        for dcm_path in glob.glob(f"{input_folder}/**/*.dcm", recursive=True):
            base = os.path.splitext(os.path.basename(dcm_path))[0]
            if base not in self.seen_basenames:
                self.seen_basenames.add(base)
                self.dcm_by_base[base] = dcm_path
            else:
                continue

        # Get the total number of files
        total_files = len(self.dcm_by_base)

        # Create a QProgressDialog
        progress_dialog = qt.QProgressDialog("Parsing DICOM files...", "Cancel", 0, total_files)
        progress_dialog.setWindowModality(qt.Qt.WindowModal)
        progress_dialog.show()

        sorted_dcm_by_base = dict(sorted(self.dcm_by_base.items()))
        self.dcm_by_base = sorted_dcm_by_base

        file_count = 0
        for key, file_path in self.dcm_by_base.items():
            progress_dialog.setValue(file_count)
            slicer.app.processEvents()

            file_count += 1
            try:
                # Try to read the file as a DICOM file
                dicom_file = pydicom.dcmread(file_path)

                # Skip non-ultrasound modalities
                if dicom_file.get("Modality", "") != "US":
                    continue

                # Extract required information
                patient_uid = dicom_file.PatientID if 'PatientID' in dicom_file else None
                study_uid = dicom_file.StudyInstanceUID if 'StudyInstanceUID' in dicom_file else None
                series_uid = dicom_file.SeriesInstanceUID if 'SeriesInstanceUID' in dicom_file else None
                instance_uid = dicom_file.SOPInstanceUID if 'SOPInstanceUID' in dicom_file else None

                if patient_uid and study_uid and series_uid and instance_uid:
                    dicom_data.append([file_path, input_folder, patient_uid, study_uid, series_uid, instance_uid])
            except Exception as e:
                # If the file is not a valid DICOM file, continue to the next file
                continue

            # Update dicomDf
            self.dicomDf = pd.DataFrame(dicom_data, columns=['Filepath', 'InputDirectory', 'PatientUID', 'StudyUID', 'SeriesUID', 'InstanceUID'])
            self.nextDicomDfIndex = 0

        # Close the progress dialog
        progress_dialog.setValue(total_files)
        progress_dialog.close()

        # Return the number of rows in the dataframe
        return len(self.dicomDf), 0

    def loadNextSequence(self):
        """
        Load the next sequence in the dataframe.
        Returns the index of the loaded sequence in the dataframe or None if no more sequences are available.
        """
        # Save current depth guide mode
        currentDepthGuideMode = self.depthGuideMode
        parameterNode = self.getParameterNode()
        if not parameterNode:
            logging.error("No parameter node found, cannot load next sequence.")
            return None

        # Clear the scene
        self.clearScene()

        if self.dicomDf is None:
            parameterNode.dfLoaded = False
            return None
        else:
            parameterNode.dfLoaded = True

        if self.nextDicomDfIndex >= len(self.dicomDf):
            return None

        nextDicomFilepath = self.dicomDf.iloc[self.nextDicomDfIndex]['Filepath']
        dir_path = self.dicomDf.iloc[self.nextDicomDfIndex]['InputDirectory']
        # increment the index to the next DICOM file
        self.nextDicomDfIndex += 1
        base_name = os.path.splitext(os.path.basename(nextDicomFilepath))[0]

        # Make sure a temporary folder for the DICOM files exists
        tempDicomDir = slicer.app.temporaryPath + '/AdjudicateUltrasound'
        if not os.path.exists(tempDicomDir):
            os.makedirs(tempDicomDir)

        # Delete all files in the temporary folder
        for file in os.listdir(tempDicomDir):
            os.remove(os.path.join(tempDicomDir, file))

        # Copy DICOM file to temporary folder
        shutil.copy(nextDicomFilepath, tempDicomDir)

        loadedNodeIDs = []
        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(tempDicomDir, db)
            patientUIDs = db.patients()
            for patientUID in patientUIDs:
                loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))

        logging.info(f"Loaded {len(loadedNodeIDs)} nodes")

        # Check loadedNodeIDs and collect sequence browser nodes to display them later
        currentSequenceBrowser = None
        for nodeID in loadedNodeIDs:
            currentSequenceBrowser = slicer.mrmlScene.GetNodeByID(nodeID)
            if currentSequenceBrowser is not None and currentSequenceBrowser.IsA("vtkMRMLSequenceBrowserNode"):
                self.currentDicomHeader = self.dicomHeaderDictForBrowserNode(currentSequenceBrowser)
                if self.currentDicomHeader is None:
                    logging.error(f"Could not find DICOM header for sequence browser node {currentSequenceBrowser.GetID()}")
                break

        # Get the current proxy node of the master sequence node of the selected sequence browser node
        masterSequenceNode = currentSequenceBrowser.GetMasterSequenceNode()
        inputUltrasoundNode = currentSequenceBrowser.GetProxyNode(masterSequenceNode)

        # Make sure the proxy node is a volume node and save it for later
        if inputUltrasoundNode is not None:
            if not (inputUltrasoundNode.IsA("vtkMRMLScalarVolumeNode") or inputUltrasoundNode.IsA("vtkMRMLVectorVolumeNode")):
                logging.error(f"Proxy node is not a volume node")
                return None

        previousNodeState = parameterNode.StartModify()

        self.sequenceBrowserNode = currentSequenceBrowser
        self.sequenceBrowserNode.SetPlaybackLooped(False)
        parameterNode.inputVolume = inputUltrasoundNode
        parameterNode.unsavedChanges = False  # Reset unsaved changes when loading new sequence

        # Restore depth guide mode
        self.depthGuideMode = currentDepthGuideMode
        logging.debug(f"Restored depthGuideMode to {self.depthGuideMode} after loading sequence")

        ultrasoundArray = slicer.util.arrayFromVolume(inputUltrasoundNode)
        # Mask array should be the same size as the ultrasound array, but with 3 channels
        maskArray = np.zeros([1, ultrasoundArray.shape[1], ultrasoundArray.shape[2], 3], dtype=np.uint8)

        # Initialize the mask volume to be the same size as the ultrasound volume but with all voxels set to 0
        if parameterNode.overlayVolume is None:
            overlayVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVectorVolumeNode", "Overlay")
            overlayVolume.SetSpacing(inputUltrasoundNode.GetSpacing())
            overlayVolume.SetOrigin(inputUltrasoundNode.GetOrigin())
            ijkToRas = vtk.vtkMatrix4x4()
            inputUltrasoundNode.GetIJKToRASMatrix(ijkToRas)
            overlayVolume.SetIJKToRASMatrix(ijkToRas)
            overlayImageData = vtk.vtkImageData()
            overlayImageData.SetDimensions(ultrasoundArray.shape[1], ultrasoundArray.shape[2], 1)
            overlayImageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
            overlayVolume.SetAndObserveImageData(overlayImageData)
            # overlayVolume.CreateDefaultDisplayNodes()
            slicer.util.updateVolumeFromArray(overlayVolume, maskArray)

            # Force the slice viewer to re-render with the corrected display settings
            slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceNode().Modified()
            parameterNode.overlayVolume = overlayVolume

        # Load all annotations with the same base prefix and deeply merge frame_annotations by frame_number
        self.seenRaters = []
        merged_data = {}
        merged_data["frame_annotations"] = []
        adjudication_path = os.path.join(dir_path, f"{base_name}.adjudication.json")
        if os.path.exists(adjudication_path):
            try:
                with open(adjudication_path, 'r') as f:
                    adjudication = json.load(f)
                    self.convert_lps_to_ras(adjudication.get("frame_annotations", []))
                    # just use the data we read as it has all the lines and other data we need
                    merged_data = adjudication
            except Exception as e:
                logging.warning(f"Failed to load adjudication file {adjudication_path}: {e}")
        else:
            filepaths = glob.glob(f"{dir_path}/**/{base_name}.json", recursive=True) + glob.glob(f"{dir_path}/**/{base_name}.*.json", recursive=True)
            for filepath in filepaths:
                try:
                    with open(filepath, 'r') as f:
                        ann = json.load(f)
                        self.convert_lps_to_ras(ann.get("frame_annotations", []))
                        # Merge non-frame_annotations keys with conflict check
                        for k, v in ann.items():
                            if k == "frame_annotations":
                                continue
                            if k not in merged_data:
                                merged_data[k] = v
                            elif merged_data[k] in [None, "", [], {}]:
                                merged_data[k] = v
                            elif isinstance(merged_data[k], list) and isinstance(v, list):
                                merged_data[k].extend(v)
                                merged_data[k] = list(dict.fromkeys(merged_data[k]))
                            elif merged_data[k] != v:
                                logging.warning(f"Conflicting values for key '{k}': {merged_data[k]} vs {v}, keeping first value")

                        # Only merge pleura_lines and b_lines for frames with matching frame_number
                        for frame in ann.get("frame_annotations", []):
                            frame_number = frame["frame_number"]
                            matched = next((f for f in merged_data["frame_annotations"] if f["frame_number"] == frame_number), None)
                            if matched:
                                matched["pleura_lines"].extend(frame.get("pleura_lines", []))
                                matched["b_lines"].extend(frame.get("b_lines", []))
                            else:
                                merged_data["frame_annotations"].append({
                                    "frame_number": frame["frame_number"],
                                    "coordinate_space": frame.get("coordinate_space", "RAS"),
                                    "pleura_lines": frame.get("pleura_lines", []),
                                    "b_lines": frame.get("b_lines", []),
                                })
                except Exception as e:
                    logging.warning(f"Failed to load annotation file {filepath}: {e}")

        self.annotations = merged_data

        # Extract and set up raters using the centralized method
        self.extractAndSetupRaters()

        # Initialize markup nodes based on loaded annotations
        if self.useFreeList:
            self.initializeMarkupNodesFromAnnotations()

        current_rater = self.getParameterNode().rater.strip().lower()
        if current_rater in self.seenRaters:
            self.seenRaters.remove(current_rater)
        if "__selected_node__" in self.seenRaters:
            self.seenRaters.remove("__selected_node__")
        if "__adjudicated_node__" in self.seenRaters:
            self.seenRaters.remove("__adjudicated_node__")
        # put current rater at the top
        self.seenRaters = [current_rater, "__selected_node__", "__adjudicated_node__"] + sorted(self.seenRaters)
        self.realRaters = [r for r in self.seenRaters if r != "__selected_node__" and r != "__adjudicated_node__"]
        self.setSelectedRaters(self.realRaters)

        # Set programmatic update flag to prevent unsavedChanges from being set
        self.refreshDisplay(updateOverlay=True, updateGui=True)
        parameterNode.EndModify(previousNodeState)

        # Set overlay volume as foreground in slice viewers
        redSliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceCompositeNode()
        redSliceCompositeNode.SetForegroundVolumeID(overlayVolume.GetID())
        redSliceCompositeNode.SetForegroundOpacity(0.12)
        redSliceCompositeNode.SetCompositing(2)

        displayNode = overlayVolume.GetDisplayNode()
        displayNode.SetWindow(255)
        displayNode.SetLevel(127)

        # Observe the ultrasound image for changes
        self.addObserver(self.sequenceBrowserNode, vtk.vtkCommand.ModifiedEvent, self.onSequenceBrowserModified)

        # Return the index of the loaded sequence in the dataframe
        return self.nextDicomDfIndex

    def createMarkupLine(self, name, rater, coordinates, color=[1, 1, 0], validation=None):
        self._suppressSync = True
        markupNode = super().createMarkupLine(name, rater, coordinates, color)
        if markupNode is not None and validation is not None:
            markupNode.SetAttribute("validation", json.dumps(validation))
        self._suppressSync = False
        return markupNode

    def _updateMarkupNode(self, node, entry, selectedNodeID):
        """
        Override: Update a markup node with the given entry.
        This is used to update the markup node for the selected rater.
        """

        # Check if node is still valid
        if not node or not slicer.mrmlScene.IsNodePresent(node):
            return

        coordinates = entry.get("line", {}).get("points", [])
        rater = entry.get("rater", "")
        validation = entry.get("validation", None)
        if not validation or not validation.get("status"):
            validation = {"status": "unadjudicated"}
        node.SetAttribute("rater", rater)
        node.SetAttribute("validation", json.dumps(validation))
        node.SetLocked(True)

        status = validation.get("status", "unadjudicated")

        # Ensure display node exists
        displayNode = node.GetDisplayNode()
        if displayNode is None:
            # Check if scene is valid before creating display nodes
            if not slicer.mrmlScene:
                return
            try:
                node.CreateDefaultDisplayNodes()
                displayNode = node.GetDisplayNode()
                if displayNode is None:
                    logging.error(f"Failed to create display node for markup node {node.GetName()}")
                    return
            except Exception as e:
                logging.error(f"Exception creating display node for {node.GetName()}: {e}")
                return

        # Check if this node is selected - apply selected highlighting regardless of validation status
        isSelected = (node.GetID() == selectedNodeID)

        if isSelected:
            # Apply selected highlighting using the __selected_node__ color
            _, selected_node_color = self.getColorsForRater("__selected_node__")
            displayNode.SetSelectedColor(selected_node_color)
        else:
            # Apply colors based on mode and validation status
            if status == "unadjudicated":
                # In adjudicator mode, unadjudicated lines use current rater's colors
                current_rater = self.getCurrentRater()
                if node in self.pleuraLines:
                    color_pleura, _ = self.getColorsForRater(rater)
                    displayNode.SetSelectedColor(color_pleura)
                else:
                    _, color_bline = self.getColorsForRater(rater)
                    displayNode.SetSelectedColor(color_bline)
            else:
                # In adjudicator mode, validated/invalidated lines use __adjudicated_node__ color
                if node in self.pleuraLines:
                    color_pleura, _ = self.getColorsForRater("__adjudicated_node__")
                    displayNode.SetSelectedColor(color_pleura)
                else:
                    _, color_bline = self.getColorsForRater("__adjudicated_node__")
                    displayNode.SetSelectedColor(color_bline)
        # Set visibility, opacity, and line style based on validation status and adjudicator mode
        status = validation.get("status", "unadjudicated")
        displayNode.SetGlyphTypeFromString("Circle2D")
        displayNode.SetGlyphScale(2.0)
        displayNode.SetLineThickness(0.25)
        if status == "invalidated":
            showInvalidAndDuplicate = self.parameterNode.showInvalidAndDuplicate if self.parameterNode else True
            displayNode.SetVisibility(showInvalidAndDuplicate and self.showHideLines)
            displayNode.SetGlyphTypeFromString("Cross2D")
            displayNode.SetOpacity(0.40)
        elif status == "unadjudicated":
            displayNode.SetVisibility(self.showHideLines)
            if not isSelected:
                displayNode.SetOpacity(0.65)
        else:  # validated
            displayNode.SetVisibility(self.showHideLines)
            if isSelected:
                # selected and validated lines are diamonds to differentiate from selected unadjudicated and invalidated lines
                displayNode.SetGlyphTypeFromString("Diamond2D")
            else:
                displayNode.SetOpacity(0.85)

        # Selected node highlighting is independent of validation status
        if isSelected:
            displayNode.SetGlyphScale(3.0)
            displayNode.SetOpacity(1.0)

        # Update control points
        # hasPointModifiedObserver = self.hasObserver(node, node.PointModifiedEvent, self.onPointModified)
        # hasPointPositionDefinedObserver = self.hasObserver(node, node.PointPositionDefinedEvent, self.onPointPositionDefined)
        # if hasPointModifiedObserver:
        #     self.removeObserver(node, node.PointModifiedEvent, self.onPointModified)
        # if hasPointPositionDefinedObserver:
        #     self.removeObserver(node, node.PointPositionDefinedEvent, self.onPointPositionDefined)

        # node.RemoveAllControlPoints()
        # for pt in coordinates:
        #     node.AddControlPointWorld(*pt)
        #     node.Modified()

        # if not hasPointModifiedObserver:
        #     self.addObserver(node, node.PointModifiedEvent, self.onPointModified)
        # if not hasPointPositionDefinedObserver:
        #     self.addObserver(node, node.PointPositionDefinedEvent, self.onPointPositionDefined)
        try:
            self.removeObserver(node, node.PointModifiedEvent, self.onPointModified)
        except:
            pass
        try:
            self.removeObserver(node, node.PointPositionDefinedEvent, self.onPointPositionDefined)
        except:
            pass
        try:
            self.removeObserver(node, node.PointRemovedEvent, self.onPointRemoved)
        except:
            pass
        node.RemoveAllControlPoints()
        for pt in coordinates:
            node.AddControlPointWorld(*pt)
            node.Modified()

        try:
            self.addObserver(node, node.PointModifiedEvent, self.onPointModified)
        except:
            pass
        try:
            self.addObserver(node, node.PointPositionDefinedEvent, self.onPointPositionDefined)
        except:
            pass
        try:
            self.addObserver(node, node.PointRemovedEvent, self.onPointRemoved)
        except:
            pass

    def _updateMarkupNodesForFrame(self, frame):
        """
        Override: Update markup nodes for pleura and b-lines for the given frame.
        Only updates the markup nodes for the selected raters. Passed selected node
        to _updateMarkupNode to highlight the selected node.
        """

        # Check if scene is valid before proceeding
        if not slicer.mrmlScene:
            return

        pleura_entries = [entry for entry in frame.get("pleura_lines", []) if entry.get("rater") in self.selectedRaters]
        bline_entries = [entry for entry in frame.get("b_lines", []) if entry.get("rater") in self.selectedRaters]

        # Get currently selected node ID for highlighting
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectedNodeID = selectionNode.GetActivePlaceNodeID() if selectionNode else None

        # Update pleura markups
        # Update pleura markups
        for i, entry in enumerate(pleura_entries):
            if i >= len(self.pleuraLines):
                node = self.createMarkupLine("Pleura", entry.get("rater", ""), entry.get("coordinates", []), [1,1,0])
                self.pleuraLines.append(node)
            else:
                node = self.pleuraLines[i]
            self._updateMarkupNode(node, entry, selectedNodeID)

        # free unused pleura markups
        unused_pleura_lines = len(self.pleuraLines) - len(pleura_entries)
        for i in range(unused_pleura_lines):
            node = self.pleuraLines.pop()
            self._freeMarkupNode(node)

        # Update b-line markups
        for i, entry in enumerate(bline_entries):
            if i >= len(self.bLines):
                node = self.createMarkupLine("B-line", entry.get("rater", ""), entry.get("coordinates", []), [0,1,1])
                self.bLines.append(node)
            else:
                node = self.bLines[i]
            self._updateMarkupNode(node, entry, selectedNodeID)

        # free unused b-line markups
        unused_b_lines = len(self.bLines) - len(bline_entries)
        for i in range(unused_b_lines):
            node = self.bLines.pop()
            self._freeMarkupNode(node)

    # def updateOverlayVolume(self):
    #     """
    #     Override: Update the overlay volume based on the validated annotations.
    #     This was not working before, now fixed on June 11, 2026
    #     Also, we make the fan overlay the same colour as the line between annotation points but a shade lighter

    #     :return: The ratio of green pixels to blue pixels in the overlay volume. None if inputs not defined yet.
    #     """
    #     parameterNode = self.getParameterNode()

    #     if parameterNode is None or parameterNode.overlayVolume is None:
    #         logging.debug("updateOverlayVolume: No overlay volume found! Cannot update overlay volume.")
    #         return None

    #     if self.annotations is None:
    #         logging.warning("updateOverlayVolume (adjudicate): No annotations loaded")
    #         # Make sure all voxels are set to 0
    #         parameterNode.overlayVolume.GetImageData().GetPointData().GetScalars().Fill(0)
    #         return None

    #     if parameterNode.inputVolume is None:
    #         logging.debug("No input volume found, not updating overlay volume.")
    #         # Make sure all voxels are set to 0
    #         parameterNode.overlayVolume.GetImageData().GetPointData().GetScalars().Fill(0)
    #         return None

    #     # If no raters are selected, do not draw any mask
    #     if hasattr(self, "selectedRaters") and not self.selectedRaters:
    #         overlayArray = slicer.util.arrayFromVolume(parameterNode.overlayVolume)
    #         overlayArray[:] = 0
    #         overlayArray = self._applyDepthGuideToMask(overlayArray, parameterNode)
    #         slicer.util.updateVolumeFromArray(parameterNode.overlayVolume, overlayArray)
    #         displayNode = parameterNode.overlayVolume.GetDisplayNode()
    #         if displayNode:
    #             displayNode.SetAutoWindowLevel(0)   # disable auto window/level permanently
    #             displayNode.SetWindow(255)
    #             displayNode.SetLevel(127)

    #         # Force the slice viewer to re-render with the corrected display settings
    #         slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceNode().Modified()
    #         slicer.util.showStatusMessage("Overlay hidden: no raters selected", 3000)
    #         return None

    #     if parameterNode.inputVolume is None:
    #         logging.debug("No input volume found, not updating overlay volume.")
    #         return None

    #     ultrasoundArray = slicer.util.arrayFromVolume(parameterNode.inputVolume)

    #     # Mask array should be the same size as the ultrasound array
    #     # Make the mask array RGB color regardless of the number of channels in the ultrasound array
    #     maskArray = np.zeros([1, ultrasoundArray.shape[1], ultrasoundArray.shape[2], 3], dtype=np.uint8)
    #     ijkToRas = vtk.vtkMatrix4x4()
    #     parameterNode.inputVolume.GetIJKToRASMatrix(ijkToRas)
    #     rasToIjk = vtk.vtkMatrix4x4()
    #     vtk.vtkMatrix4x4.Invert(ijkToRas, rasToIjk)

    #     # Build binary pleura and b-line masks and render light-shaded fans
    #     rows = ultrasoundArray.shape[1]
    #     cols = ultrasoundArray.shape[2]
    #     pleura_mask = np.zeros((rows, cols), dtype=np.uint8)
    #     bline_mask = np.zeros((rows, cols), dtype=np.uint8)

    #     # First: pleura fans (collect mask and render light color)
    #     for markupNode in self.pleuraLines:
    #         nodeRater = markupNode.GetAttribute("rater") if markupNode else None
    #         validation_json = markupNode.GetAttribute("validation")
    #         status = None
    #         if validation_json:
    #             try:
    #                 validation = json.loads(validation_json)
    #                 status = validation.get("status", None)
    #             except Exception:
    #                 status = None
    #         #Before it only showed fans for validated lines, now we show for unadjudicated lines as well
    #         # if status != "validated":
    #         #     continue

    #         if status == "invalidated":
    #             continue

    #         if hasattr(self, "selectedRaters") and self.selectedRaters and nodeRater not in self.selectedRaters:
    #             continue
    #         if not markupNode.GetDisplayNode().GetVisibility():
    #             continue

    #         # Get the line color and create a light version
    #         # displayNode = markupNode.GetDisplayNode()
    #         # r,g,b=displayNode.GetSelectedColor() #Gets the selected colour for this rater's pleura line, between 0-1
    #         nodeRater=markupNode.GetAttribute("rater")
    #         r,g,b=self.getColorsForRater(nodeRater)[0] #Gets the pleura line color for this rater, between 0-1
    #         h,s,v=colorsys.rgb_to_hsv(r,g,b) #Converts to HSV space
    #         lr,lg,lb=colorsys.hsv_to_rgb(h,s*FAN_SATURATION, 1.0) #Creates a lighter colour by reducing saturation and increasing value, keeping hue the same
    #         # Create lighter version for the fan colour
    #         lightColor = [int(lr*255),int(lg*255),int(lb*255)]
    #         # lineColor = [int(c * 255) for c in displayNode.GetSelectedColor()]
    #         # lightColor = lineColor  # remove the blend-to-white step

    #         for i in range(markupNode.GetNumberOfControlPoints() - 1):
    #             coord1 = [0, 0, 0]
    #             coord2 = [0, 0, 0]
    #             markupNode.GetNthControlPointPosition(i, coord1)
    #             markupNode.GetNthControlPointPosition(i + 1, coord2)
    #             # Skip if the two control points are the same, this sometimes happens when we start placing a line
    #             if coord1 == coord2:
    #                 continue
    #             coord1 = rasToIjk.MultiplyPoint(coord1 + [1])
    #             coord2 = rasToIjk.MultiplyPoint(coord2 + [1])
    #             coord1 = [int(round(coord1[0])), int(round(coord1[1])), int(round(coord1[2]))]
    #             coord2 = [int(round(coord2[0])), int(round(coord2[1])), int(round(coord2[2]))]
    #             # Draw mask fan between coord1 and coord2
    #             sectorArray = self.createSectorMaskBetweenPoints(ultrasoundArray, coord1, coord2, value=255)

    #             # Ensure sectorArray is 2D (rows, cols). Some codepaths return (1,rows,cols).
    #             if hasattr(sectorArray, 'ndim') and sectorArray.ndim == 3 and sectorArray.shape[0] == 1:
    #                 sector2d = sectorArray[0]
    #             else:
    #                 sector2d = sectorArray

    #             # Find where the sector array is >0 and that is where the pleural fan is, then we add in the light color
    #             sector_bool = sector2d > 0
    #             pleura_mask = np.logical_or(pleura_mask, sector_bool).astype(np.uint8)

    #             # Blend sector with the light-shaded pleura line color
    #             for c in range(3):
    #                 maskArray[0, :, :, c] = np.where(sector_bool,
    #                                                    np.maximum(maskArray[0, :, :, c], lightColor[c]),
    #                                                    maskArray[0, :, :, c])

    #     # Add B-lines to mask array with their individual colors as light-shaded fans
    #     for markupNode in self.bLines:
    #         nodeRater = markupNode.GetAttribute("rater") if markupNode else None
    #         validation_json = markupNode.GetAttribute("validation")
    #         status = None
    #         if validation_json:
    #             try:
    #                 validation = json.loads(validation_json)
    #                 status = validation.get("status", None)
    #             except Exception:
    #                 status = None
    #         #Before it only showed fans for validated lines, now we show for unadjudicated lines as well
    #         # if status != "validated":
    #         #      continue
    #         if status == "invalidated":
    #             continue

    #         if hasattr(self, "selectedRaters") and self.selectedRaters and nodeRater not in self.selectedRaters:
    #             continue
    #         if not markupNode.GetDisplayNode().GetVisibility():
    #             continue

    #         # Get the line color and create a light version
    #         #displayNode = markupNode.GetDisplayNode()
    #         #r,g,b=displayNode.GetSelectedColor() #Gets the selected colour for this rater's pleura line, between 0-1
    #         nodeRater=markupNode.GetAttribute("rater")
    #         r,g,b=self.getColorsForRater(nodeRater)[1] #Gets the pleura line color for this rater, between 0-1
    #         h,s,v=colorsys.rgb_to_hsv(r,g,b) #Converts to HSV space
    #         lr,lg,lb=colorsys.hsv_to_rgb(h,s*FAN_SATURATION, 1.0) #Creates a lighter colour by reducing saturation and increasing value, keeping hue the same
    #         # Create lighter version for the fan colour
    #         lightColor = [int(lr*255),int(lg*255),int(lb*255)]
    #         # lineColor = [int(c * 255) for c in displayNode.GetSelectedColor()]
    #         # lightColor = lineColor  # remove the blend-to-white step

    #         for i in range(markupNode.GetNumberOfControlPoints() - 1):
    #             coord1 = [0, 0, 0]
    #             coord2 = [0, 0, 0]
    #             markupNode.GetNthControlPointPosition(i, coord1)
    #             markupNode.GetNthControlPointPosition(i + 1, coord2)
    #             # Skip if the two control points are the same, this sometimes happens when we start placing a line
    #             if coord1 == coord2:
    #                 continue
    #             coord1 = rasToIjk.MultiplyPoint(coord1 + [1])
    #             coord2 = rasToIjk.MultiplyPoint(coord2 + [1])
    #             coord1 = [int(round(coord1[0])), int(round(coord1[1])), int(round(coord1[2]))]
    #             coord2 = [int(round(coord2[0])), int(round(coord2[1])), int(round(coord2[2]))]
    #             # Draw mask fan between coord1 and coord2
    #             sectorArray = self.createSectorMaskBetweenPoints(ultrasoundArray, coord1, coord2,value=255)
    #             if hasattr(sectorArray, 'ndim') and sectorArray.ndim == 3 and sectorArray.shape[0] == 1:
    #                 sector2d = sectorArray[0]
    #             else:
    #                 sector2d = sectorArray
    #             # Blend sector with the light-shaded B-line color
    #             sector_bool = sector2d > 0
    #             # restrict to pleura area
    #             #sector_bool = np.logical_and(sector_bool, pleura_mask.astype(bool))
    #             bline_mask = np.logical_or(bline_mask, sector_bool).astype(np.uint8)
    #             for c in range(3):
    #                 maskArray[0, :, :, c] = np.where(sector_bool,
    #                                                 np.maximum(maskArray[0, :, :, c], lightColor[c]),
    #                                                 maskArray[0, :, :, c])

    #     # (This erases b-line fan pixels where no pleura fan exists at all)
    #     bline_drawn_mask = bline_mask.astype(bool) & ~pleura_mask.astype(bool)
    #     # Only keep b-line pixels that are either within the pleura fan OR adjacent to it
    #     # Simplest version — erase only where absolutely no pleura coverage exists:
    #     bline_outside_pleura=bline_mask.astype(bool) & (pleura_mask == 0)
    #     for c in range(3):
    #         maskArray[0, :, :, c] = np.where(bline_outside_pleura,0,maskArray[0, :, :, c])

    #     # Calculate pleura and b-line coverage (B-lines within pleura)
    #     pleuraPixels = np.count_nonzero(pleura_mask)
    #     blineWithinPleura = np.count_nonzero(np.logical_and(bline_mask, pleura_mask))

    #     # apply depthGuide if enabled
    #     maskArray = self._applyDepthGuideToMask(maskArray, parameterNode)

    #     # Update the overlay volume
    #     slicer.util.updateVolumeFromArray(parameterNode.overlayVolume, maskArray)
    #     displayNode = parameterNode.overlayVolume.GetDisplayNode()
    #     if displayNode:
    #         displayNode.SetAutoWindowLevel(0)   # disable auto window/level permanently
    #         displayNode.SetWindow(255)
    #         displayNode.SetLevel(127)

    #     # Force the slice viewer to re-render with the corrected display settings
    #     slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceNode().Modified()

    #     # Initialize the depth guide volume to be the same size as the ultrasound volume
    #     # Create depth guide as scalar volume (same as input volume)
    #     if parameterNode.depthGuideVolume is None:
    #         depthGuideVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "DepthGuide")
    #         depthGuideImageData = vtk.vtkImageData()
    #         depthGuideImageData.SetDimensions(ultrasoundArray.shape[1], ultrasoundArray.shape[2], 1)
    #         depthGuideImageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    #         depthGuideVolume.SetSpacing(parameterNode.inputVolume.GetSpacing())
    #         depthGuideVolume.SetOrigin(parameterNode.inputVolume.GetOrigin())
    #         depthGuideVolume.SetIJKToRASMatrix(ijkToRas)
    #         depthGuideVolume.SetAndObserveImageData(depthGuideImageData)
    #         depthGuideVolume.CreateDefaultDisplayNodes()
    #         parameterNode.depthGuideVolume = depthGuideVolume

    #     # Update depth guide visibility
    #     self.updateDepthGuideVolume()

    #     # Return the ratio of B-line pixels within pleura to pleura pixels
    #     if pleuraPixels == 0:
    #         parameterNode.pleuraPercentage = 0.0
    #         return 0.0
    #     else:
    #         ratio = float(blineWithinPleura) / float(pleuraPixels)
    #         parameterNode.pleuraPercentage = ratio * 100.0
    #         return ratio

    def updateOverlayVolume(self):
        """
        Override: Update the overlay volume based on the validated annotations.
        This was not working before, now fixed on June 11, 2026
        Also, we make the fan overlay the same colour as the line between annotation points but a shade lighter

        :return: The ratio of green pixels to blue pixels in the overlay volume. None if inputs not defined yet.
        """
        parameterNode = self.getParameterNode()

        if parameterNode is None or parameterNode.overlayVolume is None:
            logging.debug("updateOverlayVolume: No overlay volume found! Cannot update overlay volume.")
            return None

        if self.annotations is None:
            logging.warning("updateOverlayVolume (adjudicate): No annotations loaded")
            # Make sure all voxels are set to 0
            parameterNode.overlayVolume.GetImageData().GetPointData().GetScalars().Fill(0)
            return None

        if parameterNode.inputVolume is None:
            logging.debug("No input volume found, not updating overlay volume.")
            # Make sure all voxels are set to 0
            parameterNode.overlayVolume.GetImageData().GetPointData().GetScalars().Fill(0)
            return None

        # If no raters are selected, do not draw any mask
        if hasattr(self, "selectedRaters") and not self.selectedRaters:
            overlayArray = slicer.util.arrayFromVolume(parameterNode.overlayVolume)
            overlayArray[:] = 0
            overlayArray = self._applyDepthGuideToMask(overlayArray, parameterNode)
            slicer.util.updateVolumeFromArray(parameterNode.overlayVolume, overlayArray)
            displayNode = parameterNode.overlayVolume.GetDisplayNode()
            if displayNode:
                displayNode.SetAutoWindowLevel(0)   # disable auto window/level permanently
                displayNode.SetWindow(255)
                displayNode.SetLevel(127)

            # Force the slice viewer to re-render with the corrected display settings
            slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceNode().Modified()
            slicer.util.showStatusMessage("Overlay hidden: no raters selected", 3000)
            return None

        if parameterNode.inputVolume is None:
            logging.debug("No input volume found, not updating overlay volume.")
            return None

        ultrasoundArray = slicer.util.arrayFromVolume(parameterNode.inputVolume)

        # Mask array should be the same size as the ultrasound array
        # Make the mask array RGB color regardless of the number of channels in the ultrasound array
        maskArray = np.zeros([1, ultrasoundArray.shape[1], ultrasoundArray.shape[2], 3], dtype=np.uint8)
        ijkToRas = vtk.vtkMatrix4x4()
        parameterNode.inputVolume.GetIJKToRASMatrix(ijkToRas)
        rasToIjk = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(ijkToRas, rasToIjk)

        # Build binary pleura and b-line masks and render light-shaded fans
        rows = ultrasoundArray.shape[1]
        cols = ultrasoundArray.shape[2]
        pleura_mask = np.zeros((rows, cols), dtype=np.uint8)
        bline_mask = np.zeros((rows, cols), dtype=np.uint8)

        ###############Step 1: Collect per-rater pleura sectors#########################

        pleura_fans_by_rater = {}

        for markupNode in self.pleuraLines:
            nodeRater = markupNode.GetAttribute("rater") if markupNode else None
            validation_json = markupNode.GetAttribute("validation")
            status = None
            if validation_json:
                try:
                    status = json.loads(validation_json).get("status", None)
                except Exception:
                    pass
            if status == "invalidated":
                continue
            if hasattr(self, "selectedRaters") and self.selectedRaters and nodeRater not in self.selectedRaters:
                continue
            if not markupNode.GetDisplayNode().GetVisibility():
                continue

            r, g, b = self.getColorsForRater(nodeRater)[0]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            lr, lg, lb = colorsys.hsv_to_rgb(h, s * FAN_SATURATION, 1.0)
            lightColor = [int(lr * 255), int(lg * 255), int(lb * 255)]

            if nodeRater not in pleura_fans_by_rater:
                pleura_fans_by_rater[nodeRater] = [np.zeros((rows, cols), dtype=bool), lightColor]

            node_sector = pleura_fans_by_rater[nodeRater][0]
            for i in range(markupNode.GetNumberOfControlPoints() - 1):
                coord1, coord2 = [0, 0, 0], [0, 0, 0]
                markupNode.GetNthControlPointPosition(i, coord1)
                markupNode.GetNthControlPointPosition(i + 1, coord2)
                if coord1 == coord2:
                    continue
                coord1 = rasToIjk.MultiplyPoint(coord1 + [1])
                coord2 = rasToIjk.MultiplyPoint(coord2 + [1])
                coord1 = [int(round(coord1[0])), int(round(coord1[1])), int(round(coord1[2]))]
                coord2 = [int(round(coord2[0])), int(round(coord2[1])), int(round(coord2[2]))]
                sectorArray = self.createSectorMaskBetweenPoints(ultrasoundArray, coord1, coord2, value=255)
                if hasattr(sectorArray, 'ndim') and sectorArray.ndim == 3 and sectorArray.shape[0] == 1:
                    sectorArray = sectorArray[0]
                node_sector = np.logical_or(node_sector, sectorArray > 0)
            pleura_fans_by_rater[nodeRater][0] = node_sector

        
        ################Step 2: Blend Pleura Fans Using Colour Averaging##################
        pleura_color_sum = np.zeros((rows, cols, 3), dtype=np.float32)
        pleura_count     = np.zeros((rows, cols),    dtype=np.float32)
        rater_pleura_masks = {}   # {rater: sector_bool}  kept for per-rater metrics

        for rater, (sector_bool, lightColor) in pleura_fans_by_rater.items():
            rater_pleura_masks[rater] = sector_bool
            pleura_mask = np.logical_or(pleura_mask, sector_bool).astype(np.uint8)
            for c in range(3):
                pleura_color_sum[:, :, c] += sector_bool.astype(np.float32) * lightColor[c]
            pleura_count += sector_bool.astype(np.float32)

        any_pleura = pleura_count > 0
        for c in range(3):
            maskArray[0, :, :, c] = np.where(
                any_pleura,
                (pleura_color_sum[:, :, c] / np.maximum(pleura_count, 1)).astype(np.uint8),
                maskArray[0, :, :, c]
            )

        
        ##############Step 3: collect per-rater b-line sectors###################

        bline_fans_by_rater = {}

        for markupNode in self.bLines:
            nodeRater = markupNode.GetAttribute("rater") if markupNode else None
            validation_json = markupNode.GetAttribute("validation")
            status = None
            if validation_json:
                try:
                    status = json.loads(validation_json).get("status", None)
                except Exception:
                    pass
            if status == "invalidated":
                continue
            if hasattr(self, "selectedRaters") and self.selectedRaters and nodeRater not in self.selectedRaters:
                continue
            if not markupNode.GetDisplayNode().GetVisibility():
                continue

            r, g, b = self.getColorsForRater(nodeRater)[1]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            lr, lg, lb = colorsys.hsv_to_rgb(h, s * FAN_SATURATION, 1.0)
            lightColor = [int(lr * 255), int(lg * 255), int(lb * 255)]

            if nodeRater not in bline_fans_by_rater:
                bline_fans_by_rater[nodeRater] = [np.zeros((rows, cols), dtype=bool), lightColor]

            node_sector = bline_fans_by_rater[nodeRater][0]
            for i in range(markupNode.GetNumberOfControlPoints() - 1):
                coord1, coord2 = [0, 0, 0], [0, 0, 0]
                markupNode.GetNthControlPointPosition(i, coord1)
                markupNode.GetNthControlPointPosition(i + 1, coord2)
                if coord1 == coord2:
                    continue
                coord1 = rasToIjk.MultiplyPoint(coord1 + [1])
                coord2 = rasToIjk.MultiplyPoint(coord2 + [1])
                coord1 = [int(round(coord1[0])), int(round(coord1[1])), int(round(coord1[2]))]
                coord2 = [int(round(coord2[0])), int(round(coord2[1])), int(round(coord2[2]))]
                sectorArray = self.createSectorMaskBetweenPoints(ultrasoundArray, coord1, coord2, value=255)
                if hasattr(sectorArray, 'ndim') and sectorArray.ndim == 3 and sectorArray.shape[0] == 1:
                    sectorArray = sectorArray[0]
                node_sector = np.logical_or(node_sector, sectorArray > 0)
            bline_fans_by_rater[nodeRater][0] = node_sector

        

        ###############Step 4: Blend B-line fans (restricted to pleura area)###########
        bline_color_sum = np.zeros((rows, cols, 3), dtype=np.float32)
        bline_count     = np.zeros((rows, cols),    dtype=np.float32)
        rater_bline_masks = {}   # {rater: sector_bool restricted to that rater's own pleura}

        for rater, (sector_bool, lightColor) in bline_fans_by_rater.items():
            # For the visual: restrict to the UNION of all pleura fans
            sector_visible = np.logical_and(sector_bool, pleura_mask.astype(bool))
            # For the per-rater METRIC: restrict to this rater's own pleura
            rater_own_pleura = rater_pleura_masks.get(rater, np.zeros((rows, cols), dtype=bool))
            rater_bline_masks[rater] = np.logical_and(sector_bool, rater_own_pleura)
            bline_mask = np.logical_or(bline_mask, sector_visible).astype(np.uint8)
            for c in range(3):
                bline_color_sum[:, :, c] += sector_visible.astype(np.float32) * lightColor[c]
            bline_count += sector_visible.astype(np.float32)

        any_bline = bline_count > 0
        for c in range(3):
            maskArray[0, :, :, c] = np.where(
                any_bline,
                (bline_color_sum[:, :, c] / np.maximum(bline_count, 1)).astype(np.uint8),
                maskArray[0, :, :, c]
            )


        # apply depthGuide if enabled
        maskArray = self._applyDepthGuideToMask(maskArray, parameterNode)

        # Update the overlay volume
        slicer.util.updateVolumeFromArray(parameterNode.overlayVolume, maskArray)
        displayNode = parameterNode.overlayVolume.GetDisplayNode()
        if displayNode:
            displayNode.SetAutoWindowLevel(0)   # disable auto window/level permanently
            displayNode.SetWindow(255)
            displayNode.SetLevel(127)

        # Force the slice viewer to re-render with the corrected display settings
        slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceNode().Modified()

        # Initialize the depth guide volume to be the same size as the ultrasound volume
        # Create depth guide as scalar volume (same as input volume)
        if parameterNode.depthGuideVolume is None:
            depthGuideVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "DepthGuide")
            depthGuideImageData = vtk.vtkImageData()
            depthGuideImageData.SetDimensions(ultrasoundArray.shape[1], ultrasoundArray.shape[2], 1)
            depthGuideImageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

            depthGuideVolume.SetSpacing(parameterNode.inputVolume.GetSpacing())
            depthGuideVolume.SetOrigin(parameterNode.inputVolume.GetOrigin())
            depthGuideVolume.SetIJKToRASMatrix(ijkToRas)
            depthGuideVolume.SetAndObserveImageData(depthGuideImageData)
            depthGuideVolume.CreateDefaultDisplayNodes()
            parameterNode.depthGuideVolume = depthGuideVolume

        


        ##############Stage 5: Per-rater metrics#################

        rater_percentages = {}   # {rater: percentage float}
        all_raters = set(list(rater_pleura_masks.keys()) + list(rater_bline_masks.keys()))
        for rater in all_raters:
            r_pleura = rater_pleura_masks.get(rater, np.zeros((rows, cols), dtype=bool))
            r_bline  = rater_bline_masks.get(rater,  np.zeros((rows, cols), dtype=bool))
            r_pleura_px = np.count_nonzero(r_pleura)
            r_bline_px  = np.count_nonzero(r_bline)
            rater_percentages[rater] = (float(r_bline_px) / float(r_pleura_px) * 100.0
                                        if r_pleura_px > 0 else 0.0)

        # Store on logic for the widget to read
        self.raterPercentages = rater_percentages   # dict {rater: float}
        # Also store overall summary for backward compatibility
        overall_pleura_px = np.count_nonzero(pleura_mask)
        overall_bline_px  = np.count_nonzero(np.logical_and(bline_mask, pleura_mask))

        if overall_pleura_px == 0:
            parameterNode.pleuraPercentage = 0.0
            return 0.0
        else:
            parameterNode.pleuraPercentage = float(float(overall_bline_px) / float(overall_pleura_px)) * 100.0
            return float(float(overall_bline_px) / float(overall_pleura_px))

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')

    def syncMarkupsToAnnotations(self):
        """
        One-way sync: Save current markup nodes to annotations for the current frame.
        This is the single source of truth for persisting markup changes.
        Overridden to handle validation data.
        """
        if self.sequenceBrowserNode is None:
            logging.warning("No sequence browser node found, cannot sync markups to annotations.")
            return

        currentFrameIndex = max(0, self.sequenceBrowserNode.GetSelectedItemNumber())
        logging.debug(f"Syncing markups to annotations for frame {currentFrameIndex} (adjudicate)")

        # Check if annotations already has a list of frame annotations
        if 'frame_annotations' not in self.annotations:
            self.annotations['frame_annotations'] = []

        # Find existing frame annotation for currentFrameIndex
        existing = next((f for f in self.annotations['frame_annotations']
                         if int(f.get("frame_number", -1)) == currentFrameIndex), None)
        if not existing:
            # create an empty frame and append it to annotations
            existing = {
                "frame_number": currentFrameIndex,
                "coordinate_space": "RAS",
                "pleura_lines": [],
                "b_lines": []
            }
            self.annotations['frame_annotations'].append(existing)

        # Remove all lines, we will add only the visible lines back in and any updates will take here
        existing['pleura_lines'] = []
        existing['b_lines'] = []

        # Add visible pleura lines to annotations
        for markupNode in self.pleuraLines:
            displayNode = markupNode.GetDisplayNode() if markupNode else None
            is_visible = displayNode.GetVisibility() if displayNode else False
            num_points = markupNode.GetNumberOfControlPoints() if markupNode else 0

            # Only save visible nodes with valid coordinates
            if not is_visible:
                continue  # Skip hidden nodes

            coordinates = []
            for j in range(num_points):
                coord = [0, 0, 0]
                markupNode.GetNthControlPointPosition(j, coord)
                coordinates.append(coord)

            validation_json = markupNode.GetAttribute("validation")
            try:
                validation = json.loads(validation_json) if validation_json else {"status": "unadjudicated"}
            except json.JSONDecodeError:
                validation = {"status": "unadjudicated"}

            if coordinates and len(coordinates) >= 2:  # Only save lines with at least 2 points
                line_data = {
                    "rater": markupNode.GetAttribute("rater"),
                    "line": {"points": coordinates},
                    "validation": validation
                }
                existing['pleura_lines'].append(line_data)

        # Add visible B-lines to annotations
        for markupNode in self.bLines:
            displayNode = markupNode.GetDisplayNode() if markupNode else None
            is_visible = displayNode.GetVisibility() if displayNode else False
            num_points = markupNode.GetNumberOfControlPoints() if markupNode else 0

            # Only save visible nodes with valid coordinates
            if not is_visible:
                continue  # Skip hidden nodes

            coordinates = []
            for j in range(num_points):
                coord = [0, 0, 0]
                markupNode.GetNthControlPointPosition(j, coord)
                coordinates.append(coord)

            validation_json = markupNode.GetAttribute("validation")
            try:
                validation = json.loads(validation_json) if validation_json else {"status": "unadjudicated"}
            except json.JSONDecodeError:
                validation = {"status": "unadjudicated"}

            if coordinates and len(coordinates) >= 2:  # Only save lines with at least 2 points
                line_data = {
                    "rater": markupNode.GetAttribute("rater"),
                    "line": {"points": coordinates},
                    "validation": validation
                }
                existing['b_lines'].append(line_data)

    def syncAnnotationsToMarkups(self):
        """
        One-way sync: Update markup nodes from annotations for the current frame.
        This is the single source of truth for displaying annotations.
        Overridden to handle validation data and adjudication-specific display logic.
        """
        if not slicer.mrmlScene:
            return

        if self.annotations is None  or 'frame_annotations' not in self.annotations:
            logging.debug("syncAnnotationsToMarkups (adjudicate): No annotations loaded")
            # Hide all markups
            for node in self.pleuraLines:
                displayNode = node.GetDisplayNode()
                if displayNode:
                    displayNode.SetVisibility(False)
            for node in self.bLines:
                displayNode = node.GetDisplayNode()
                if displayNode:
                    displayNode.SetVisibility(False)
            return

        if self.sequenceBrowserNode is None:
            logging.warning("No sequence browser node found")
            return

        currentFrameIndex = max(0, self.sequenceBrowserNode.GetSelectedItemNumber())

        frame = next((item for item in self.annotations['frame_annotations']
                     if str(item.get("frame_number")) == str(currentFrameIndex)), None)

        # Set programmatic update flag to prevent unsavedChanges from being set
        self._isProgrammaticUpdate = True

        # Batch scene updates using StartState/EndState
        slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
        try:
            # Hide all markups if no frame data
            for node in self.pleuraLines:
                displayNode = node.GetDisplayNode()
                if displayNode:
                    displayNode.SetVisibility(False)
            for node in self.bLines:
                displayNode = node.GetDisplayNode()
                if displayNode:
                    displayNode.SetVisibility(False)

            if frame is not None:
                self._updateMarkupNodesForFrame(frame)
        finally:
            slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)
            # Reset programmatic update flag
            self._isProgrammaticUpdate = False

    def refreshDisplay(self, updateOverlay=True, updateGui=True):
        """
        Override: Central method to refresh the display after any changes.
        This ensures consistent updates across all UI elements.
        """
        parameterNode = self.getParameterNode()

        # Update overlay volume if requested
        if updateOverlay:
            ratio=self.updateOverlayVolume()

        # Update GUI if requested and we have a widget
        if updateGui:
            try:
                widget = getAdjudicateUltrasoundWidget()
                if widget:
                    widget.updateGuiFromAnnotations()
            except RuntimeError:
                # Widget not initialized yet, skip GUI update
                pass

    def fanCornersFromSectorLine(self, p1, p2, center, r1, r2):
        op1 = np.array(p1) - np.array(center)
        op2 = np.array(p2) - np.array(center)

        unit_op1 = op1 / np.linalg.norm(op1)
        unit_op2 = op2 / np.linalg.norm(op2)

        A = center + unit_op1 * r1
        C = center + unit_op1 * r2
        B = center + unit_op2 * r1
        D = center + unit_op2 * r2

        return A, B, C, D
    
    def line_coefficients(self, p1, p2):
        """
        Returns the coefficients of the line equation Ax + By + C = 0
        """
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

    def createFanMask(self, imageArray, topLeft, topRight, bottomLeft, bottomRight, value=255):
        # Caching: store last-used parameters and mask
        if not hasattr(self, '_lastFanMaskParams'):
            self._lastFanMaskParams = None
            self._lastFanMaskArray = None
        # Create a tuple of all relevant parameters
        params = (
            imageArray.shape,
            tuple(topLeft), tuple(topRight), tuple(bottomLeft), tuple(bottomRight),
            value
        )
        if self._lastFanMaskParams == params:
            return self._lastFanMaskArray.copy()
        image_size_rows = imageArray.shape[1]
        image_size_cols = imageArray.shape[2]
        mask_array = np.zeros((image_size_rows, image_size_cols), dtype=np.uint8)

        # Compute the angle of the fan mask in degrees

        if abs(topLeft[0] - bottomLeft[0]) < 0.001:
            angle1 = 90.0
        else:
            angle1 = np.arctan((topLeft[1] - bottomLeft[1]) / (topLeft[0] - bottomLeft[0])) * 180 / np.pi + 180.0
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
        leftLineA, leftLineB, leftLineC = self.line_coefficients(topLeft, bottomLeft)
        rightLineA, rightLineB, rightLineC = self.line_coefficients(topRight, bottomRight)

        # Handle the case when the lines are parallel
        if leftLineB != 0 and rightLineB != 0 and leftLineA / leftLineB == rightLineA / rightLineB:
            logging.warning(f"Left and right lines are parallel: topLeft: {topLeft}, topRight: {topRight}, bottomLeft: {bottomLeft}, bottomRight: {bottomRight}, leftLineA: {leftLineA}, leftLineB: {leftLineB}, rightLineA: {rightLineA}, rightLineB: {rightLineB}")
            return mask_array

        # Compute intersection point of the two lines
        det = leftLineA * rightLineB - leftLineB * rightLineA
        if det == 0:
            logging.warning("No intersection point found")
            return mask_array

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

        # Create a mask image

        # mask_array = cv2.ellipse(mask_array, (center_cols_px, center_rows_px), (radius2, radius2), 0.0, angle2, angle1, value, -1)
        mask_array = self.draw_circle_segment(mask_array, (center_cols_px, center_rows_px), radius2, angle2, angle1, value)
        mask_array = cv2.circle(mask_array, (center_cols_px, center_rows_px), radius1, 0, -1)

        # Cache the result before returning
        self._lastFanMaskParams = params
        self._lastFanMaskArray = mask_array.copy()
        return mask_array

    def draw_circle_segment(self, image, center, radius, start_angle, end_angle, color):
        """
        Draws a segment of a circle with floating point start and end angles on a numpy array image.

        :param image: Image as a numpy array.
        :param center: Center of the circle (x, y).
        :param radius: Radius of the circle.
        :param start_angle: Start angle in degrees (floating point).
        :param end_angle: End angle in degrees (floating point).
        :param color: Color of the segment (B, G, R).
        :return: Image with the drawn circle segment.
        """
        mask = np.zeros_like(image)

        # Convert angles to radians
        start_angle_rad = np.deg2rad(start_angle)
        end_angle_rad = np.deg2rad(end_angle)

        # Generate points for the circle segment
        thetas = np.linspace(start_angle_rad, end_angle_rad, 360)
        xs = center[0] + radius * np.cos(thetas)
        ys = center[1] + radius * np.sin(thetas)

        # Draw the outer arc
        pts = np.array([np.round(xs), np.round(ys)]).T.astype(int)
        cv2.polylines(mask, [pts], False, color, 1)

        # Draw two lines from the center to the start and end points
        cv2.line(mask, center, tuple(pts[0]), color, 1)
        cv2.line(mask, center, tuple(pts[-1]), color, 1)

        # Fill the segment
        cv2.fillPoly(mask, [np.vstack([center, pts])], color)

        # Combine the mask with the original image
        return cv2.bitwise_or(image, mask)

    def createSectorMaskBetweenPoints(self, imageArray, point1, point2, value=255):
        if "mask_type" not in self.annotations:
            logging.error("No mask type found in annotations. Assuming rectangular mask.")
        # Caching: store last-used parameters and mask
        if not hasattr(self, '_lastSectorMaskParams'):
            self._lastSectorMaskParams = None
            self._lastSectorMaskArray = None
        params = (
            imageArray.shape,
            tuple(point1), tuple(point2),
            value,
            self.annotations.get('mask_type', None),
            self.annotations.get('radius1', None),
            self.annotations.get('radius2', None),
            self.annotations.get('center_rows_px', None),
            self.annotations.get('center_cols_px', None),
            self.annotations.get('angle1', None),
            self.annotations.get('angle2', None)
        )
        if self._lastSectorMaskParams == params:
            return self._lastSectorMaskArray.copy()

        if "mask_type" not in self.annotations or self.annotations["mask_type"] != "fan":
            # Create a rectangular mask
            maskArray = np.zeros(imageArray.shape, dtype=np.uint8)
            maskArray[:, point1[1]:point2[1], point1[0]:point2[0]] = value
            # Cache and return
            self._lastSectorMaskParams = params
            self._lastSectorMaskArray = maskArray.copy()
        else:
            radius1 = self.annotations["radius1"]
            radius2 = self.annotations["radius2"]
            center_rows_px = self.annotations["center_rows_px"]
            center_cols_px = self.annotations["center_cols_px"]
            a, b, c, d = self.fanCornersFromSectorLine(point1[:2], point2[:2],
                                                       (center_cols_px, center_rows_px),
                                                       radius1, radius2)
            maskArray = self.createFanMask(imageArray, a, b, c, d, value)
            # Cache and return
            self._lastSectorMaskParams = params
            self._lastSectorMaskArray = maskArray.copy()

        return maskArray

    def clearSceneLines(self, sync=False):
        """
        Remove all pleura lines and B-lines from the scene and from the list of lines.
        """
        # Remove all pleura lines
        while self.removeLastPleuraLine(sync=sync):
            pass
        # Remove all B-lines
        while self.removeLastBline(sync=sync):
            pass

    def clearAllLines(self):
        """
        Remove all pleura lines and B-lines from the scene and from the list of lines.
        Only updates the annotation if the current frame is already in the annotations.
        """
        self.clearSceneLines(sync=False)
        # Only update annotation if current frame is already present
        if self.sequenceBrowserNode is not None and self.annotations is not None and 'frame_annotations' in self.annotations:
            currentFrameIndex = max(0, self.sequenceBrowserNode.GetSelectedItemNumber())
            if any(int(f.get("frame_number", -1)) == currentFrameIndex for f in self.annotations["frame_annotations"]):
                self.syncMarkupsToAnnotations()
                self.refreshDisplay(updateOverlay=True, updateGui=True)

    def removeLastPleuraLine(self, sync=True):
        """
        Remove the last pleura line from the scene and from the list of pleura lines.
        """
        if len(self.pleuraLines) > 0:
            # find the last pleura line for the rater
            parameterNode = self.getParameterNode()
            current_rater = parameterNode.rater.strip().lower()
            currentLine = None
            for line in reversed(self.pleuraLines):
                if line.GetAttribute("rater") == current_rater:
                    currentLine = line
                    break
            if currentLine is None:
                statusText = f"No pleura line found for rater {current_rater}"
                slicer.util.mainWindow().statusBar().showMessage(statusText, 3000)
                return False
            self.pleuraLines.remove(currentLine)
            self._freeMarkupNode(currentLine)
            if sync:
                self.syncMarkupsToAnnotations()
                self.refreshDisplay(updateOverlay=True, updateGui=True)

            return True
        return False

    def removeLastBline(self, sync=True):
        """
        Remove the last B-line from the scene and from the list of B-lines.
        """
        if len(self.bLines) > 0:
            # find the last B-line for the rater
            currentLine = None
            parameterNode = self.getParameterNode()
            current_rater = parameterNode.rater.strip().lower()
            for line in reversed(self.bLines):
                if line.GetAttribute("rater") == current_rater:
                    currentLine = line
                    break
            if currentLine is None:
                statusText = f"No B-line found for rater {current_rater}"
                slicer.util.mainWindow().statusBar().showMessage(statusText, 3000)
                return False
            self.bLines.remove(currentLine)
            self._freeMarkupNode(currentLine)
            if sync:
                self.syncMarkupsToAnnotations()
                self.refreshDisplay(updateOverlay=True, updateGui=True)
            return True
        return False
    
    def _freeMarkupNode(self, markupNode):
        # Handle None nodes gracefully
        if markupNode is None or markupNode.GetID() is None:
            return

        try:
            self.removeObserver(slicer.mrmlScene, slicer.mrmlScene.NodeRemovedEvent, self.onMarkupNodeRemoved)
        except:
            pass
        try:
            self.removeObserver(markupNode, markupNode.PointModifiedEvent, self.onPointModified)
        except:
            pass
        try:
            self.removeObserver(markupNode, markupNode.PointPositionDefinedEvent, self.onPointPositionDefined)
        except:
            pass
        try:
            self.removeObserver(markupNode, markupNode.PointRemovedEvent, self.onPointRemoved)
        except:
            pass

        markupNode.RemoveAllControlPoints()

        if self.useFreeList:
            markupNode.SetName("freeMarkupNode")
            markupNode.SetAttribute("rater", "")
            markupNode.Modified()
            self.freeMarkupNodes.append(markupNode)
        else:
            slicer.mrmlScene.RemoveNode(markupNode)

        try:
            self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeRemovedEvent, self.onMarkupNodeRemoved)
        except:
            pass

    
    


#
# Register the module
#

if __name__ == "__main__":
    import sys
    import os
    import slicer

    # Add the module path to sys.path
    modulePath = os.path.dirname(os.path.abspath(__file__))
    if modulePath not in sys.path:
        sys.path.insert(0, modulePath)

    # Register the module
    import AdjudicateUltrasound
    slicer.modules.adjudicateultrasound = AdjudicateUltrasound.AdjudicateUltrasound(slicer.qSlicerApplication().moduleManager())
