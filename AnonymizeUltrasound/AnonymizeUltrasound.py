from collections import defaultdict
import csv
import datetime
from enum import Enum
import hashlib
import io
import json
import logging
import random
import numpy as np
import math
import os
import pathlib
from PIL import Image
import pydicom
import shutil
import requests
from typing import Annotated, Optional

import qt
import vtk

from DICOMLib import DICOMUtils
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLVectorVolumeNode, vtkMRMLVolumeNode
from slicer import vtkMRMLSequenceBrowserNode, vtkMRMLSequenceNode
from slicer import vtkMRMLMarkupsFiducialNode


#
# AnonymizeUltrasound
#

class AnonymizeUltrasound(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Anonymize Ultrasound")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Ultrasound")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Tamas Ungi (Queen's University)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
            This is a module for anonymizing ultrasound images and sequences stored in DICOM folders.
            The mask (green contour) signals what part of the image will stay after applying the mask.
            The image area under the green contour will be kept along with the pixels inside the contour.
            See more information in <a href="https://github.com/SlicerUltrasound/SlicerUltrasound?tab=readme-ov-file#anonymize-ultrasound">module documentation</a>.
            """)
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
            This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
            and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
            """)

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", onSlicerStartupCompleted)

#
# Register sample data sets in Sample Data module
#

def onSlicerStartupCompleted():
    """
    Perform some initialization tasks that require the application to be fully started up.
    """
    # Install required packages

    global pd
    try:
        import pandas as pd
    except ImportError:
        logging.info("AnonymizeUltrasound: Pandas not found, installing...")
        slicer.util.pip_install('pandas')
        import pandas as pd

    global cv2
    try:
        import cv2
    except ImportError:
        slicer.util.pip_install('opencv-python')
        import cv2

    global torch
    try:
        import torch
    except ImportError:
        logging.info("AnonymizeUltrasound: torch not found, installing...")
        slicer.util.pip_install('torch')
        import torch

    global yaml
    try:
        import yaml
    except ImportError:
        logging.info("AnonymizeUltrasound: yaml not found, installing...")
        slicer.util.pip_install('PyYAML')
        import yaml

#
# AnonymizeUltrasoundParameterNode
#

class AnonymizerStatus(Enum):
    INITIAL = 0               # No data loaded yet
    INPUT_READY = 1           # Valid input folder parsed
    PATIENT_LOADED = 2        # Cannot mask without landmarks
    LANDMARK_PLACEMENT = 3    # Landmark placement mode for mouse click
    LANDMARKS_PLACED = 4      # Ready to mask
    STATUS_MASKING = 5        # Masking in progress

@parameterNodeWrapper
class AnonymizeUltrasoundParameterNode:
    """
    The parameters needed by module.
    """
    ultrasoundSequenceBrowser: vtkMRMLSequenceBrowserNode  # Sequence browser whose proxy node is ultrasoundVolume
    maskMarkups: vtkMRMLMarkupsFiducialNode                # Landmarks for masking
    overlayVolume: vtkMRMLVectorVolumeNode                 # Overlay volume to represent masking
    maskVolume: vtkMRMLScalarVolumeNode                    # Volume node to store the mask
    status: AnonymizerStatus = AnonymizerStatus.INITIAL    # Current status of the anonymizer to decide what actions are allowed
    patientId: str = ""                                    # Currently loaded patient
    studyInstanceUid: str = ""                             # Currently loaded study
    seriesInstanceUid: str = ""                            # Currently loaded series

#
# AnonymizeUltrasoundWidget
#


class AnonymizeUltrasoundWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    INPUT_FOLDER_SETTING = "AnonymizeUltrasound/InputFolder"
    OUTPUT_FOLDER_SETTING = "AnonymizeUltrasound/OutputFolder"
    HEADERS_FOLDER_SETTING = "AnonymizeUltrasound/HeadersFolder"
    AUTO_MASK_SETTING = "AnonymizeUltrasound/AutoMask"
    SKIP_SINGLE_FRAME_SETTING = "AnonymizeUltrasound/SkipSingleFrame"
    CONTINUE_PROGRESS_SETTING = "AnonymizeUltrasound/ContinueProgress"
    HASH_PATIENT_ID_SETTING = "AnonymizeUltrasound/HashPatientId"
    FILENAME_PREFIX_SETTING = "AnonymizeUltrasound/FilenamePrefix"
    LABELS_PATH_SETTING = "AnonymizeUltrasound/LabelsPath"
    THREE_POINT_FAN_SETTING = "AnonymizeUltrasound/ThreePointFan"
    ENABLE_MASK_CACHE_SETTING = "AnonymizeUltrasound/enableMaskCache"

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.compositingModeExit = None

        # --- Keyboard shortcuts ---
        # M: toggle Define Mask, N: next scan, Space: toggle auto overlay, E: export scan, A: export and load next scan
        self.shortcutM = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutM.setKey(qt.QKeySequence('M'))
        self.shortcutN = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutN.setKey(qt.QKeySequence('N'))
        self.shortcutC = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutC.setKey(qt.QKeySequence('C'))
        self.shortcutE = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutE.setKey(qt.QKeySequence('E'))
        self.shortcutA = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutA.setKey(qt.QKeySequence('A'))

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/AnonymizeUltrasound.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = AnonymizeUltrasoundLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons

        settings = slicer.app.settings()

        inputFolder = settings.value(self.INPUT_FOLDER_SETTING)
        if inputFolder:
            if os.path.exists(inputFolder):
                self.ui.inputDirectoryButton.directory = inputFolder
            else:
                logging.info(f"Settings input folder {inputFolder} does not exist")
        self.ui.inputDirectoryButton.connect("directoryChanged(QString)",
                                             lambda newValue: self.onSettingChanged(self.INPUT_FOLDER_SETTING, newValue))

        outputFolder = settings.value(self.OUTPUT_FOLDER_SETTING)
        if outputFolder:
            if os.path.exists(outputFolder):
                self.ui.outputDirectoryButton.directory = outputFolder
            else:
                logging.info(f"Settings output folder {outputFolder} does not exist")
        self.ui.outputDirectoryButton.connect("directoryChanged(QString)",
                                              lambda newValue: self.onSettingChanged(self.OUTPUT_FOLDER_SETTING, newValue))

        headersFolder = settings.value(self.HEADERS_FOLDER_SETTING)
        if headersFolder:
            if os.path.exists(headersFolder):
                self.ui.headersDirectoryButton.directory = headersFolder
            else:
                logging.info(f"Settings headers folder {headersFolder} does not exist")
        self.ui.headersDirectoryButton.connect("directoryChanged(QString)",
                                               lambda newValue: self.onSettingChanged(self.HEADERS_FOLDER_SETTING, newValue))

        self.ui.importDicomButton.connect("clicked(bool)", self.onImportDicomButton)

        # Workflow control buttons

        self.ui.nextButton.clicked.connect(self.onNextButton)
        self.ui.defineMaskButton.toggled.connect(self.onMaskLandmarksButton)
        self.ui.exportButton.clicked.connect(self.onExportScanButton)
        if hasattr(self.ui, 'exportAndNextButton'):
            self.ui.exportAndNextButton.clicked.connect(self.onExportAndNextShortcut)

        # Settings widgets

        enableMaskCache = settings.value(self.ENABLE_MASK_CACHE_SETTING)
        if enableMaskCache and enableMaskCache.lower() == "true":
            self.ui.enableMaskCacheCheckBox.checked = True
        else:
            self.ui.enableMaskCacheCheckBox.checked = False
        self.ui.enableMaskCacheCheckBox.connect('toggled(bool)',
                                                lambda newValue: self.onSettingChanged(self.ENABLE_MASK_CACHE_SETTING, str(newValue)))

        autoMaskStr = settings.value(self.AUTO_MASK_SETTING)
        if autoMaskStr and autoMaskStr.lower() == "true":
            self.ui.autoMaskCheckBox.checked = True
        else:
            self.ui.autoMaskCheckBox.checked = False
        self.ui.autoMaskCheckBox.connect('toggled(bool)', lambda newValue: self.onSettingChanged(self.AUTO_MASK_SETTING, str(newValue)))

        self.ui.autoOverlayCheckBox.checked = False
        self.ui.autoOverlayCheckBox.connect('toggled(bool)', self.onAutoOverlayCheckBoxToggled)

        # Developer gating for Auto‑Overlay check box
        self._updateAutoOverlayCheckBoxVisibility()

        continueProgressStr = settings.value(self.CONTINUE_PROGRESS_SETTING)
        if continueProgressStr and continueProgressStr.lower() == "true":
            self.ui.continueProgressCheckBox.checked = True
        else:
            self.ui.continueProgressCheckBox.checked = False
        self.ui.continueProgressCheckBox.connect('toggled(bool)', lambda newValue: self.onSettingChanged(self.CONTINUE_PROGRESS_SETTING, str(newValue)))

        skipSingleFrameStr = settings.value(self.SKIP_SINGLE_FRAME_SETTING)
        if skipSingleFrameStr and skipSingleFrameStr.lower() == "true":
            self.ui.skipSingleframeCheckBox.checked = True
        else:
            self.ui.skipSingleframeCheckBox.checked = False
        self.ui.skipSingleframeCheckBox.connect('toggled(bool)', lambda newValue: self.onSettingChanged(self.SKIP_SINGLE_FRAME_SETTING, str(newValue)))

        hashPatientIdStr = settings.value(self.HASH_PATIENT_ID_SETTING)
        if hashPatientIdStr and hashPatientIdStr.lower() == "true":
            self.ui.hashPatientIdCheckBox.checked = True
        else:
            self.ui.hashPatientIdCheckBox.checked = False
        self.ui.hashPatientIdCheckBox.connect('toggled(bool)', lambda newValue: self.onSettingChanged(self.HASH_PATIENT_ID_SETTING, str(newValue)))

        filenamePrefix = settings.value(self.FILENAME_PREFIX_SETTING)  # This has been moved to Processing tab on the UI
        if filenamePrefix:
            self.ui.namePrefixLineEdit.text = filenamePrefix
        self.ui.namePrefixLineEdit.connect('textChanged(QString)', lambda newValue: self.onSettingChanged(self.FILENAME_PREFIX_SETTING, newValue))

        # Three-point fan mask setting
        threePointStr = settings.value(self.THREE_POINT_FAN_SETTING)
        if threePointStr and threePointStr.lower() == "true":
            self.ui.threePointFanCheckBox.checked = True
        else:
            self.ui.threePointFanCheckBox.checked = False
        self.ui.threePointFanCheckBox.connect('toggled(bool)', lambda newValue: self.onSettingChanged(self.THREE_POINT_FAN_SETTING, str(newValue)))

        self.ui.settingsCollapsibleButton.collapsed = True

        # Annotation labels

        self.ui.labelsFileSelector.connect('currentPathChanged(QString)', self.onLabelsPathChanged)
        labelsPath = settings.value(self.LABELS_PATH_SETTING)
        if not labelsPath or labelsPath == '':
            labelsPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Resources/default_labels.csv')
        self.ui.labelsFileSelector.currentPath = labelsPath
        self.ui.labelsCollapsibleButton.collapsed = True

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # Start on red-only view. Allow other layouts later.
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)

        self.connectKeyboardShortcuts()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()
        self.disconnectKeyboardShortcuts()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

        sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode()
        self.compositingModeExit = sliceCompositeNode.GetCompositing()  # Save compositing mode to restore it when exiting the module
        sliceCompositeNode.SetCompositing(2)

        # Collapse DataProbe widget
        mw = slicer.util.mainWindow()
        if mw:
            w = slicer.util.findChild(mw, "DataProbeCollapsibleWidget")
            if w:
                w.collapsed = True


    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)

        # Restore compositing mode to the value it was before entering the module
        sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode()
        sliceCompositeNode.SetCompositing(self.compositingModeExit)

        # Remove keyboard shortcuts when leaving the module
        self.disconnectKeyboardShortcuts()

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: Optional[AnonymizeUltrasoundParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)
            self._onParameterNodeModified()

    def onLabelsPathChanged(self, filePath):
        settings = qt.QSettings()
        settings.setValue(self.LABELS_PATH_SETTING, filePath)

        categories = defaultdict(list)

        try:
            with open(filePath, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    category, label = map(str.strip, row)
                    categories[category].append(label)
        except (FileNotFoundError, PermissionError) as e:
            logging.error(f"Cannot read labels file: {filePath}, error: {e}")

        # Remove all existing labels from the scroll area
        for i in reversed(range(self.ui.labelsScrollAreaWidgetContents.layout().count())):
            self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget().deleteLater()

        # Populate labels scroll area
        for category, labels in categories.items():
            categoryGroupBox = qt.QGroupBox(category, self.ui.labelsScrollAreaWidgetContents)
            categoryLayout = qt.QVBoxLayout(categoryGroupBox)
            for label in labels:
                checkBox = qt.QCheckBox(label, categoryGroupBox)
                categoryLayout.addWidget(checkBox)
            categoryGroupBox.setLayout(categoryLayout)
            self.ui.labelsScrollAreaWidgetContents.layout().addWidget(categoryGroupBox)

        self.ui.labelsScrollAreaWidgetContents.layout().addStretch(1)

    def onSettingChanged(self, settingName: str, newValue: str) -> None:
        """
        Update setting value and GUI based on user selection.
        @param settingName: setting name
        @param newValue: new value, if "" then setting is removed
        """
        settings = slicer.app.settings()
        if newValue and newValue != "":
            settings.setValue(settingName, newValue)
        else:
            settings.remove(settingName)
        if settingName == self.THREE_POINT_FAN_SETTING and self._parameterNode:
            # clear any existing points and reset overlay
            markupsNode = self._parameterNode.maskMarkups
            threePointFanModeEnabled = self.ui.threePointFanCheckBox.checked
            if markupsNode is not None:
                markupsNode.RemoveAllControlPoints()
                # redraw mask (will be empty)
                self.logic.updateMaskVolume(three_point=threePointFanModeEnabled)
                self.logic.showMaskContour()

        if settingName == self.ENABLE_MASK_CACHE_SETTING and newValue.lower() == "false":
            logging.info("Mask cache disabled, clearing existing cache")
            self.logic.clearMaskCache()

    def _onParameterNodeModified(self, caller=None, event=None) -> None:
        """
        Update GUI based on parameter node values.
        """
        if not self._parameterNode:
            logging.error("Parameter node not set")
            return

        numInstances = self.logic.getNumberOfInstances()
        if numInstances > 0:
            self.ui.progressBar.maximum = numInstances
            self.ui.inputsCollapsibleButton.collapsed = True
            self.ui.dataProcessingCollapsibleButton.collapsed = False
            self.ui.dataProcessingCollapsibleButton.enabled = True
            self.ui.labelsCollapsibleButton.enabled = True
        else:
            self.ui.inputsCollapsibleButton.collapsed = False
            self.ui.dataProcessingCollapsibleButton.collapsed = True
            self.ui.dataProcessingCollapsibleButton.enabled = False
            self.ui.labelsCollapsibleButton.enabled = False
            self.ui.statusLabel.text = "Select input folder and press Read DICOM folder button to load DICOM files"



    def onImportDicomButton(self) -> None:
        logging.info("Import DICOM button clicked")

        # Check input and output folders

        inputDirectory = self.ui.inputDirectoryButton.directory
        if not inputDirectory:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Anonymize Ultrasound", "Please select an input directory")
            return
        if not os.path.exists(inputDirectory):
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Anonymize Ultrasound", "Input directory does not exist")
            return

        outputDirectory = self.ui.outputDirectoryButton.directory
        if not outputDirectory:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Anonymize Ultrasound", "Please select an output directory")
            return
        if not os.path.exists(outputDirectory):
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Anonymize Ultrasound", "Output directory does not exist")
            return

        outputHeadersDirectory = self.ui.headersDirectoryButton.directory
        if not outputHeadersDirectory:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Anonymize Ultrasound", "Please select a headers directory")
            return
        if not os.path.exists(outputHeadersDirectory):
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Anonymize Ultrasound", "Headers directory does not exist")
            return

        numFiles = self.logic.updateDicomDf(inputDirectory, self.ui.skipSingleframeCheckBox.checked)
        logging.info(f"Found {numFiles} DICOM files in input folder")

        if numFiles > 0:
            self._parameterNode.status = AnonymizerStatus.INPUT_READY
        else:
            self._parameterNode.status = AnonymizerStatus.INITIAL

        # Export self.logic.dicomDf as a CSV file in the headers directory
        outputFilePath = os.path.join(outputHeadersDirectory, "keys.csv")
        self.logic.dicomDf.to_csv(outputFilePath, index=False)

        statusText = str(numFiles)
        if self.ui.skipSingleframeCheckBox.checked:
            statusText += " multi-frame dicom files found in input folder."
        else:
            statusText += " dicom files found in input folder."

        if self.ui.continueProgressCheckBox.checked:
            numDone = self.logic.updateProgressDicomDf(inputDirectory, outputDirectory)
            if numDone is None:
                statusText += '\nAll files have been processed. Cannot load more files from input folder.'
            elif numDone < 1:
                statusText += '\nNo files already processed. Starting from first in alphabetical order.'
            else:
                statusText += '\n' + str(numDone) + ' files already processed in output folder. Continue at next.'
        self.ui.statusLabel.text = statusText

    def onNextButton(self) -> None:

        threePointFanModeEnabled = self.ui.threePointFanCheckBox.checked
        continueProgress = self.ui.continueProgressCheckBox.checked

        # If continue progress is checked and nextDicomDfIndex is None, there is nothing more to load
        if self.logic.nextDicomDfIndex is None and continueProgress:
            self.ui.statusLabel.text = "All files from input folder have been processed to output folder. No more files to load."
            return

        # Remove observers for the mask markups node, because loading a new series will reset the scene and createa a new markups node

        maskMarkupsNode = self._parameterNode.maskMarkups
        if maskMarkupsNode:
            self.removeObserver(maskMarkupsNode, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onPointModified)

        # Load the next series

        dialog = self.createWaitDialog("Loading series", "Please wait until the DICOM file is loaded...")
        currentDicomDfIndex = None
        try:
            outputDirectory = self.ui.outputDirectoryButton.directory
            currentDicomDfIndex = self.logic.loadNextSequence(outputDirectory, continueProgress)
            if currentDicomDfIndex is None:
                statusText = "No more series to load"
                self.ui.statusLabel.text = statusText
                dialog.close()
            else:
                self.ui.progressBar.value = currentDicomDfIndex
                dialog.close()
        except Exception as e:
            dialog.close()
            logging.warning("Error loading series: " + str(e))  # Known error is raised on Windows if loading from outside C: drive

        # Add observers for the mask markups node

        maskMarkupsNode = self._parameterNode.maskMarkups
        if maskMarkupsNode:
            self.addObserver(maskMarkupsNode, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onPointModified)

        # Uncheck all label checkboxes

        for i in range(self.ui.labelsScrollAreaWidgetContents.layout().count()):
            groupBox = self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget()
            if groupBox is None:
                continue
            # Find all checkboxes in groupBox
            for j in range(groupBox.layout().count()):
                checkBox = groupBox.layout().itemAt(j).widget()
                if isinstance(checkBox, qt.QCheckBox):
                    checkBox.setChecked(False)

        # Update GUI

        patientID = self.logic.currentDicomDataset.PatientID
        if patientID:
            self.ui.patientIdLabel.text = patientID
        else:
            logging.error("Patient ID is missing")
            self.ui.patientIdLabel.text = 'None'

        instanceUID = self.logic.currentDicomDataset.SOPInstanceUID
        if instanceUID is None:
            logging.error("Instance UID is missing")
            self.ui.sopInstanceUidLabel.text = 'None'
        else:
            self.ui.sopInstanceUidLabel.text = instanceUID

        statusText = f"Instance {instanceUID} loaded from file:\n"

        # Get the file path from the dataframe

        if currentDicomDfIndex is not None:
            filepath = self.logic.dicomDf.iloc[currentDicomDfIndex].Filepath
            statusText += filepath
            self.ui.statusLabel.text = statusText
        self.logic.updateMaskVolume(three_point=threePointFanModeEnabled)
        self.logic.showMaskContour()

        # Set red slice compositing mode to 2
        sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode()
        sliceCompositeNode.SetCompositing(2)

        # Reactivate the main window to ensure keyboard shortcuts work
        slicer.util.mainWindow().activateWindow()
        slicer.util.mainWindow().raise_()
        slicer.util.mainWindow().setFocus()

    def onAutoOverlayCheckBoxToggled(self, checked):
        self.logic.showAutoOverlay = checked  # Pass to logic
        self.logic._composeAndPushOverlay()

    def _updateAutoOverlayCheckBoxVisibility(self):
        self.ui.autoOverlayCheckBox.setVisible(self.developerMode)

    #
    # Placement of mask markups
    #

    def onMaskLandmarksButton(self, toggled):
        logging.info('Mask landmarks button pressed')

        maskMarkupsNode = self._parameterNode.maskMarkups
        if maskMarkupsNode is None:
            logging.error(f"Landmark node not found: {self.logic.MASK_FAN_LANDMARKS}")
            return

        # determine if three-point fan mode is active
        threePointFanModeEnabled = self.ui.threePointFanCheckBox.checked

        # Automatic mask via AI only when NOT in three-point fan mode
        # TODO: Support for three-point fan mode auto mask
        autoMaskSuccessful = False
        if self.ui.autoMaskCheckBox.checked:
            if threePointFanModeEnabled:
                logging.info("Auto mask not applied because in three-point fan mode")
            else:
                # Get the mask control points
                maskMarkupsNode.RemoveAllControlPoints()
                coords_IJK = self.logic.getAutoMask()
                if coords_IJK is None:
                    logging.error("Auto mask not found")
                else:
                    autoMaskSuccessful = True

                # Try to apply the automatic mask markups
                currentVolumeNode = self.logic.getCurrentProxyNode()
                if autoMaskSuccessful == True and currentVolumeNode is not None:
                    ijkToRas = vtk.vtkMatrix4x4()
                    currentVolumeNode.GetIJKToRASMatrix(ijkToRas)

                    num_points = coords_IJK.shape[0]
                    coords_RAS = np.zeros((num_points, 4))
                    for i in range(num_points):
                        point_IJK = np.array([coords_IJK[i, 0], coords_IJK[i, 1], 0, 1])
                        # convert to IJK
                        coords_RAS[i, :] = ijkToRas.MultiplyPoint(point_IJK)

                    for i in range(num_points):
                        coord = coords_RAS[i, :]
                        maskMarkupsNode.AddControlPoint(coord[0], coord[1], coord[2])

                    # Update the status
                    self._parameterNode.status = AnonymizerStatus.LANDMARKS_PLACED
                    self.ui.defineMaskButton.checked = False
                else:
                    logging.error("Ultraosund volume node not found")
                    autoMaskSuccessful = False

        # If markups are not automatically defined, start the manual process using mouse interactions

        if toggled and autoMaskSuccessful == False:
            maskMarkupsNode = self._parameterNode.maskMarkups
            maskMarkupsNode.RemoveAllControlPoints()
            self.logic.updateMaskVolume(three_point=threePointFanModeEnabled)
            
            self._parameterNode.status = AnonymizerStatus.LANDMARK_PLACEMENT
            self.addObserver(maskMarkupsNode, slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onPointAdded)
            self.addObserver(maskMarkupsNode, slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.onPointDefined)

            maskMarkupsNode.SetDisplayVisibility(True)

            selectionNode = slicer.app.applicationLogic().GetSelectionNode()
            selectionNode.SetReferenceActivePlaceNodeClassName(maskMarkupsNode.GetClassName())
            selectionNode.SetReferenceActivePlaceNodeID(maskMarkupsNode.GetID())

            # Switch mouse mode to place mode
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SwitchToPersistentPlaceMode()
            interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        else:
            # Make sure mouse mode is default
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

    def onPointModified(self, caller=None, event=None):
        markupsNode = self._parameterNode.maskMarkups
        if not markupsNode:
            logging.error("Markups node not found")
            return

        threePointFanModeEnabled = self.ui.threePointFanCheckBox.checked
        required = 3 if threePointFanModeEnabled else 4
        count = markupsNode.GetNumberOfControlPoints()
        if count == required:
            self.logic.updateMaskVolume(three_point=threePointFanModeEnabled)
            maskContourVolumeNode = self._parameterNode.overlayVolume
            if maskContourVolumeNode:
                sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode()
                if sliceCompositeNode.GetForegroundVolumeID() != maskContourVolumeNode.GetID():
                    sliceCompositeNode.SetForegroundVolumeID(maskContourVolumeNode.GetID())
                    sliceCompositeNode.SetForegroundOpacity(0.5)
                    displayNode = maskContourVolumeNode.GetDisplayNode()
                    displayNode.SetWindow(255)
                    displayNode.SetLevel(127)

    def onPointAdded(self, caller=None, event=None):
        logging.info('Point added')
        markupsNode = self._parameterNode.maskMarkups
        # determine required points based on 3-point fan mode
        threePointFanModeEnabled = self.ui.threePointFanCheckBox.checked
        count = markupsNode.GetNumberOfControlPoints()
        required = 3 if threePointFanModeEnabled else 4
        if count == required:
            # finalize mask placement
            self.logic.updateMaskVolume(three_point=threePointFanModeEnabled)
            self.logic.showMaskContour()
        else:
            slicer.util.setSliceViewerLayers(foreground=None)

    def onPointDefined(self, caller=None, event=None):
        logging.info('Point defined')
        markupsNode = self._parameterNode.maskMarkups
        if not markupsNode:
            logging.error("Markups node not found")
            return
        threePointFanModeEnabled = self.ui.threePointFanCheckBox.checked
        count = markupsNode.GetNumberOfControlPoints()
        required = 3 if threePointFanModeEnabled else 4
        if count == required:
            self.logic.updateMaskVolume(three_point=threePointFanModeEnabled)
            self.logic.showMaskContour()
            self.ui.defineMaskButton.checked = False
            self._parameterNode.status = AnonymizerStatus.LANDMARKS_PLACED
            self.removeObserver(markupsNode, slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onPointAdded)
            self.removeObserver(markupsNode, slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.onPointDefined)

            # Switch mouse mode to default
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

            # Pop the markup button
            self.ui.defineMaskButton.checked = False

    #
    # Export scan
    #

    def onExportScanButton(self):
        """
        Callback function for the export scan button.
        """
        logging.info('Export scan button pressed')

        threePointFanModeEnabled = self.ui.threePointFanCheckBox.checked
        currentSequenceBrowser = self._parameterNode.ultrasoundSequenceBrowser
        if currentSequenceBrowser is None:
            self.ui.statusLabel.text = "Load a DICOM sequence before trying to export"
            logging.info("No sequence browser found, nothing exported.")
            return

        selectedItemNumber = currentSequenceBrowser.GetSelectedItemNumber()  # Save current frame index for sequence so we can restore it after exporting the scan

        # Check if any labels are checked. If yes, we need to save them as annotations

        annotationLabels = []
        for i in reversed(range(self.ui.labelsScrollAreaWidgetContents.layout().count())):
            groupBox = self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget()
            if groupBox is None:
                continue
            # Find all checkboxes in groupBox
            for j in reversed(range(groupBox.layout().count())):
                checkBox = groupBox.layout().itemAt(j).widget()
                if isinstance(checkBox, qt.QCheckBox) and checkBox.isChecked():
                    annotationLabels.append(checkBox.text)

        # If there are not mask markups, confirm with the user that they really want to proceed.
        required = 4 if not threePointFanModeEnabled else 3
        count = self._parameterNode.maskMarkups.GetNumberOfControlPoints()
        if count < required:
            if not slicer.util.confirmOkCancelDisplay("No mask defined. Do you want to proceed without masking?"):
                return

        # Mask images to erase the unwanted parts
        self.logic.maskSequence(three_point=threePointFanModeEnabled)
        
        # Set up output directory and filename

        hashPatientId = self.ui.hashPatientIdCheckBox.checked

        # If hashPatientId is not checked, confirm with the user that they really want to proceed.

        if not hashPatientId:
            if not slicer.util.confirmOkCancelDisplay("Patient name will not be masked. Do you want to proceed?"):
                return

        outputDirectory = self.ui.outputDirectoryButton.directory
        headersDirectory = self.ui.headersDirectoryButton.directory

        filename, patient_uid, file_uid = self.logic.generateNameFromDicomData(self.logic.currentDicomDataset, hashPatientId)

        dialog = self.createWaitDialog("Exporting scan", "Please wait until the scan is exported...")

        if hashPatientId:
            new_patient_name = f"{self.ui.namePrefixLineEdit.text}_{patient_uid}"
            new_patient_id = patient_uid
        else:
            new_patient_name = None
            new_patient_id = None

        # Save current mask to cache before exporting
        if self.ui.enableMaskCacheCheckBox.checked:
            self.logic.saveCurrentMaskToCache()

        # Export the scan
        dicomFilePath, jsonFilePath, dicomHeaderFilePath = self.logic.exportDicom(
            outputDirectory=outputDirectory,
            outputFilename=filename,
            headersDirectory=headersDirectory,
            labels = annotationLabels,
            new_patient_name = new_patient_name,
            new_patient_id = new_patient_id)

        # Restore selected item number in sequence browser
        currentSequenceBrowser.SetSelectedItemNumber(selectedItemNumber)

        # Display file paths in the status label

        statusText = "DICOM saved to: " + dicomFilePath + "\nAnnotations saved to: " + jsonFilePath\
            + "\nDICOM header saved to: " + dicomHeaderFilePath

        self.ui.statusLabel.text = statusText

        # Close the modal dialog

        dialog.close()

    #
    # Dialog helpers
    #

    def createWaitDialog(self, title, message):
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle(title)
        dialogLayout = qt.QVBoxLayout(dialog)
        dialogLayout.setContentsMargins(20, 14, 20, 14)
        dialogLayout.setSpacing(4)
        dialogLayout.addStretch(1)
        dialogLabel = qt.QLabel(message)
        dialogLabel.setAlignment(qt.Qt.AlignCenter)
        dialogLayout.addWidget(dialogLabel)
        dialogLayout.addStretch(1)
        dialog.show()
        slicer.app.processEvents()

        return dialog

    def createWaitDialog(self, title, message):
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle(title)
        dialogLayout = qt.QVBoxLayout(dialog)
        dialogLayout.setContentsMargins(20, 14, 20, 14)
        dialogLayout.setSpacing(4)
        dialogLayout.addStretch(1)
        dialogLabel = qt.QLabel(message)
        dialogLabel.setAlignment(qt.Qt.AlignCenter)
        dialogLayout.addWidget(dialogLabel)
        dialogLayout.addStretch(1)
        dialog.show()
        slicer.app.processEvents()

        return dialog

    def connectKeyboardShortcuts(self):
        """Connect shortcut keys to their corresponding actions."""
        self.shortcutM.connect('activated()', lambda: self.ui.defineMaskButton.toggle())
        self.shortcutN.connect('activated()', self.onNextButton)
        self.shortcutC.connect('activated()', lambda: self.ui.threePointFanCheckBox.toggle())
        self.shortcutE.connect('activated()', self.onExportScanButton)
        self.shortcutA.connect('activated()', self.onExportAndNextShortcut)

    def disconnectKeyboardShortcuts(self):
        """Disconnect shortcut keys when leaving the module to avoid unwanted interactions."""
        try:
            self.shortcutM.activated.disconnect()
            self.shortcutN.activated.disconnect()
            self.shortcutC.activated.disconnect()
            self.shortcutE.activated.disconnect()
            self.shortcutA.activated.disconnect()
        except Exception:
            # If shortcuts were not connected yet, ignore
            pass

    def onExportAndNextShortcut(self):
        """Helper slot to export the current scan and immediately load the next one (shortcut 'A')."""
        self.onExportScanButton()
        # Load next only if export did not show blocking dialogs (user may have canceled)
        # We simply attempt; internal checks will guard.
        self.onNextButton()


#
# AnonymizeUltrasoundLogic
#


class AnonymizeUltrasoundLogic(ScriptedLoadableModuleLogic, VTKObservationMixin):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

        self.dicomDf = None
        self.nextDicomDfIndex = 0
        self.showAutoOverlay = False
        self._autoMaskRGB = None     # 1×H×W×3  uint8, red
        self._manualMaskRGB = None   # 1×H×W×3  uint8, green
        self._parameterNode = self._getOrCreateParameterNode()
        self.transducerMaskCache = {}   # TransducerModel -> mask volume node
        self.currentTransducerModel = 'unknown'

    def _getOrCreateParameterNode(self):
        if not hasattr(self, "_parameterNode"):
            self._parameterNode = AnonymizeUltrasoundParameterNode(super().getParameterNode())
        return self._parameterNode

    def getParameterNode(self):
        return self._parameterNode

    def getTransducerModel(self, transducerType: str) -> str:
        """
        Parse the transducer type string and return the transducer model or 'unknown'.
        For example, if transducerType is 'SC6-1s,02597', it returns 'sc6-1s'.
        """
        if not transducerType or transducerType == '':
            return 'unknown'

        return transducerType.split(",")[0].lower()

    def updateDicomDf(self, inputDirectory: str, skipSingleFrame: bool) -> int:
        """
        Update dicomDf with a list of all DICOM files in the input directory.
        """
        logging.info(f"Reading DICOM files from {inputDirectory}")
        dicom_data = []

        # Get the total number of files
        total_files = sum([len(files) for root, dirs, files in os.walk(inputDirectory)])

        # Create a QProgressDialog
        progress_dialog = qt.QProgressDialog("Parsing DICOM files...", "Cancel", 0, total_files, slicer.util.mainWindow())
        progress_dialog.setWindowModality(qt.Qt.WindowModal)
        progress_dialog.show()
        slicer.app.processEvents()

        # Recursively walk through the input folder
        file_count = 0
        for root, dirs, files in os.walk(inputDirectory):
            dirs.sort()
            files.sort()
            for file in files:
                progress_dialog.setValue(file_count)
                file_count += 1
                slicer.app.processEvents()

                # Construct the full file path
                file_path = os.path.join(root, file)

                try:
                    # Try to read the file as a DICOM file
                    dicom_ds = pydicom.dcmread(file_path, stop_before_pixels=True)

                    # Try to get image spacing.
                    physical_delta_x = None
                    physical_delta_y = None
                    to_patch = True
                    if hasattr(dicom_ds, "SequenceOfUltrasoundRegions"):
                        if len(dicom_ds.SequenceOfUltrasoundRegions) > 0:
                            region = dicom_ds.SequenceOfUltrasoundRegions[0]
                            if (hasattr(region, "PhysicalDeltaX") and hasattr(region, "PhysicalDeltaY")):
                                physical_delta_x = region.PhysicalDeltaX
                                physical_delta_y = region.PhysicalDeltaY
                                to_patch = False

                    # Extract required information
                    patient_id = dicom_ds.PatientID if 'PatientID' in dicom_ds else None
                    study_uid = dicom_ds.StudyInstanceUID if 'StudyInstanceUID' in dicom_ds else None
                    series_uid = dicom_ds.SeriesInstanceUID if 'SeriesInstanceUID' in dicom_ds else None
                    instance_uid = dicom_ds.SOPInstanceUID if 'SOPInstanceUID' in dicom_ds else None

                    content_date = dicom_ds.ContentDate if 'ContentDate' in dicom_ds else '19000101'
                    content_time = dicom_ds.ContentTime if 'ContentTime' in dicom_ds else '000000'

                    transducer_model = self.getTransducerModel(dicom_ds.TransducerType) if 'TransducerType' in dicom_ds else 'unknown'

                    if transducer_model == 'unknown':
                        logging.warning(f"Unknown transducer type in file {file_path}")

                    if patient_id is None:
                        logging.warning(f"Patient ID missing in file {file_path}")

                    exp_filename, _, _ = self.generateNameFromDicomData(dicom_ds)

                    if skipSingleFrame and ('NumberOfFrames' not in dicom_ds or dicom_ds.NumberOfFrames < 2):
                        continue

                    # Append the information to the list, if PatientID, StudyInstanceUID, and SeriesInstanceUID are present
                    if patient_id and study_uid and series_uid and instance_uid:
                        dicom_data.append([
                            file_path,
                            exp_filename,
                            patient_id,
                            study_uid,
                            series_uid,
                            instance_uid,
                            physical_delta_x,
                            physical_delta_y,
                            content_date,
                            content_time,
                            to_patch,
                            transducer_model
                        ])
                except Exception as e:
                    # If the file is not a valid DICOM file, continue to the next file
                    continue

        # Update dicomDf
        self.dicomDf = pd.DataFrame(dicom_data, columns=[
            'Filepath', 'AnonFilename', 'PatientUID', 'StudyUID',
            'SeriesUID', 'InstanceUID', 'PhysicalDeltaX', 'PhysicalDeltaY', 'ContentDate', 'ContentTime', 'Patch', 'TransducerModel'
        ])
        self.dicomDf = self.dicomDf.sort_values(by=['Filepath', 'ContentDate', 'ContentTime'])  # This makes a difference on Mac, not on Windows.

        # Add a new column to dicomDf named 'SeriesNumber' that is the index of the row in the group of rows with the same PatientUID and StudyUID.
        self.dicomDf['SeriesNumber'] = self.dicomDf.groupby(['PatientUID', 'StudyUID']).cumcount() + 1

        # This is a workaround for the issue that some DICOM files do not have spacing information. The information may be used when loading each file,
        # but patching the DICOM files before importing them would be a better option.
        self.dicomDf['PhysicalDeltaX'] = self.dicomDf.groupby('StudyUID')['PhysicalDeltaX'].transform(lambda x: x.ffill().bfill())
        self.dicomDf['PhysicalDeltaY'] = self.dicomDf.groupby('StudyUID')['PhysicalDeltaY'].transform(lambda x: x.ffill().bfill())

        # If PhysicalDeltaX is still missing from at least one row, log a warning.
        if self.dicomDf['PhysicalDeltaX'].isnull().sum() > 0:
            logging.warning("Some ultrasound scans have missing spacing information.")

        self.nextDicomDfIndex = 0

        # Close the progress dialog
        progress_dialog.setValue(total_files)
        progress_dialog.close()

        # Return the number of rows in the dataframe
        return len(self.dicomDf)

    def loadNextSequence(self, outputDirectory, continueProgress=True):
        """
        Load next sequence in the list of DICOM files.
        Returns the index of the loaded sequence in the dataframe of DICOM files, or None if no more sequences are available.
        """
        self.resetScene()

        parameterNode = self.getParameterNode()

        # Get next filepath from dicomDf. If nextDicomDfIndex is larger than the number of rows in dicomDf, then
        # return None.
        if self.nextDicomDfIndex is None or self.nextDicomDfIndex >= len(self.dicomDf):
            return None
        nextDicomDfRow = self.dicomDf.iloc[self.nextDicomDfIndex]

        # Make sure a temporary folder for the DICOM files exists
        tempDicomDir = slicer.app.temporaryPath + '/AnonymizeUltrasound'
        logging.info("Temporary DICOM directory: " + tempDicomDir)
        if not os.path.exists(tempDicomDir):
            os.makedirs(tempDicomDir)

        # Delete all files in the temporary folder
        for file in os.listdir(tempDicomDir):
            os.remove(os.path.join(tempDicomDir, file))

        # Copy DICOM file to temporary folder
        shutil.copy(nextDicomDfRow['Filepath'], tempDicomDir)
        logging.info(f"Copied DICOM file {nextDicomDfRow['Filepath']} to {tempDicomDir}")

        # TODO: Make this an option in the settings becuase some already patched dcm files are not loading with this option
        # # Patch the DICOM file to add spacing information if it is missing, but available from other rows
        # temporaryDicomFilepath = os.path.join(tempDicomDir, os.path.basename(nextDicomDfRow['Filepath']))
        # to_patch = nextDicomDfRow['Patch']
        # if to_patch:
        #     physical_delta_x = nextDicomDfRow['PhysicalDeltaX']
        #     physical_delta_y = nextDicomDfRow['PhysicalDeltaY']
        #     if physical_delta_x is not None and physical_delta_y is not None:
        #         ds = pydicom.dcmread(temporaryDicomFilepath)
        #         ds.PixelSpacing = [str(physical_delta_x*10.0), str(physical_delta_y*10.0)]  # Convert from cm/pixel to mm/pixel
        #         ds.save_as(temporaryDicomFilepath)
        #         logging.info(f"Patched DICOM file {temporaryDicomFilepath} with physical delta X and Y")

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
                parameterNode.ultrasoundSequenceBrowser = currentSequenceBrowser
                self.currentDicomHeader = self.dicomHeaderDictForBrowserNode(currentSequenceBrowser)
                # Add a "DicomFile" attribute to the currentSequenceBrowser, so we can check the source file later for information
                currentSequenceBrowser.SetAttribute("DicomFile", nextDicomDfRow['Filepath'])
                if self.currentDicomHeader is None:
                    logging.error(f"Could not find DICOM header for sequence browser node {currentSequenceBrowser.GetID()}")
                break

        # If accidentally pixel data gets in the dicom header, remove it
        if "Pixel Data" in self.currentDicomHeader:
            del self.currentDicomHeader["Pixel Data"]

        # After loading the DICOM, try to find a cached mask for the transducer model
        # If found, apply it. If not, the user will need to define it manually.
        if hasattr(self, 'currentDicomDataset') and self.currentDicomDataset:
            transducerType = self.currentDicomDataset.get("TransducerType", "unknown")
            self.currentTransducerModel = self.getTransducerModel(transducerType)
            cached_mask = self.getCachedMaskForTransducer(self.currentTransducerModel)

            if cached_mask:
                logging.info(f"Found cached mask for transducer {self.currentTransducerModel}")
                if self.applyCachedMask(cached_mask):
                    logging.info("Successfully applied cached mask")
                else:
                    logging.warning("Failed to apply cached mask, will need manual definition")

        # Increment nextDicomDfIndex
        nextIndex = self.incrementDicomDfIndex(None, outputDirectory, skip_existing=continueProgress)
        if nextIndex is None:
            slicer.util.mainWindow().statusBar().showMessage("No more DICOM files to process", 3000)
            return None

        # Delete files from temporary folder
        for file in os.listdir(tempDicomDir):
            os.remove(os.path.join(tempDicomDir, file))

        # Make this sequence browser node the current one in the toolbar
        slicer.modules.sequences.setToolBarActiveBrowserNode(currentSequenceBrowser)

        # Get the proxy node of the master sequence node of the selected sequence browser node
        masterSequenceNode = currentSequenceBrowser.GetMasterSequenceNode()
        if masterSequenceNode is None:
            logging.error("Master sequence node of sequence browser node with ID " + currentSequenceBrowser.GetID() + " not found")
            return None
        proxyNode = currentSequenceBrowser.GetProxyNode(masterSequenceNode)
        if proxyNode is None:
            logging.error("Proxy node of master sequence node with ID " + masterSequenceNode.GetID() + " not found")
            return None

        # If proxyNode is a vtkMRMLScalarVolumeNode or vtkMRMLVectorVolumeNode, then set it as the background volume
        if proxyNode.IsA("vtkMRMLScalarVolumeNode") or proxyNode.IsA("vtkMRMLVectorVolumeNode"):
            backgroundVolumeNode = proxyNode
            layoutManager = slicer.app.layoutManager()
            sliceLogic = layoutManager.sliceWidget('Red').sliceLogic()
            compositeNode = sliceLogic.GetSliceCompositeNode()
            compositeNode.SetBackgroundVolumeID(backgroundVolumeNode.GetID())

        return self.nextDicomDfIndex - 1

    def dicomHeaderDictForBrowserNode(self, browserNode):
        """
        Return DICOM header for the given browser node.
        Also sets up self.currentDicomDataset variable as a pydicom dataset with same contents as returned dictionary.

        :param browserNode: Sequence browser node
        :return: DICOM header dictionary
        """
        if browserNode is None:
            return None

        # Get the proxy node of the master sequence node of the selected sequence browser node
        masterSequenceNode = browserNode.GetMasterSequenceNode()
        proxyNode = browserNode.GetProxyNode(masterSequenceNode)

        # Get DICOM.instanceUID attribute from proxy node
        instanceUID = proxyNode.GetAttribute("DICOM.instanceUIDs")
        if instanceUID is None:
            logging.error("DICOM.instanceUIDs attribute not found in proxy node")
            return None

        # If instanceUID is a list, keep only the first item
        if isinstance(instanceUID, list):
            instanceUID = instanceUID[0]

        # Find row in self.dicomDf where instanceUID matches first instanceUID in instanceUIDs
        filepath = self.getFileForBrowserNode(browserNode)
        if filepath is None:
            logging.error(f"Could not find DICOM file for instanceUID {instanceUID}")
            return None

        ds = pydicom.dcmread(filepath)
        dsInstanceUID = ds["0008", "0018"].value
        if dsInstanceUID == instanceUID:
            self.currentDicomDataset = pydicom.dcmread(filepath)
            dicomHeaderDict = self.dicomHeaderToDict(ds)
            return dicomHeaderDict

        return None

    def dicomHeaderToDict(self, ds, parent=None):
        """
        Convert a DICOM dataset to a Python dictionary.
        """
        if parent is None:
            parent = {}
        for elem in ds:
            if elem.VR == "SQ":
                parent[elem.name] = []
                for item in elem:
                    child = {}
                    self.dicomHeaderToDict(item, child)
                    parent[elem.name].append(child)
            else:
                parent[elem.name] = elem.value
        return parent

    def resetScene(self):
        """
        Reset the scene by clearing it and setting it up again.
        """
        parameterNode = self.getParameterNode()

        # Clear the scene
        slicer.mrmlScene.Clear(0)
        self.currentDicomDataset = None
        self.currentDicomHeader = None
        self.setupScene()

    def setupScene(self):
        parameterNode = self.getParameterNode()
        markupsNode = parameterNode.maskMarkups
        if not markupsNode:
            markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "MaskFiducials")
            markupsNode.GetDisplayNode().SetTextScale(0.0)
            parameterNode.maskMarkups = markupsNode

        # Add observer for node added to mrmlScene
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAddedEvent, self.onNodeAdded)

    def updateProgressDicomDf(self, input_folder, output_folder, keep_folders=False):
        """
        Check the output folder to see what input files are already processed.

        :param input_folder: full path to the input folder where input DCM files are.
        :param output_folder: full path to the output folder where already processed files can be found.
        :param keep_folders: If True, output files are expected by the same name in the same subfolders as input files.
        :return:  index for dicomDf that points to the next row that needs to be processed.
        """
        self.nextDicomDfIndex = None
        self.incrementDicomDfIndex(input_folder, output_folder, skip_existing=True)
        return self.nextDicomDfIndex

    def incrementDicomDfIndex(self, input_folder=None, output_directory=None, skip_existing=False):
        """
        Increment the index of the DICOM dataframe. If skipExistingOutput is True, then skip the rows that have already been processed.

        :param skip_existing: If True, skip the rows that have already been processed.
        :param keep_folders: If True, keep the folder structure of the input DICOM files in the output directory.
        :return: None
        """
        listOfIndices = self.dicomDf.index.tolist()
        listOfIndices.sort()

        if self.nextDicomDfIndex is None:
            nextIndexIndex = 0
        else:
            nextIndexIndex = listOfIndices.index(self.nextDicomDfIndex)
            nextIndexIndex += 1

        if skip_existing:
            while nextIndexIndex < len(listOfIndices):
                nextDicomDfRow = self.dicomDf.iloc[listOfIndices[nextIndexIndex]]

                output_path = output_directory
                output_filename = nextDicomDfRow['AnonFilename']
                output_fullpath = os.path.join(output_path, output_filename)

                # Make sure output_fullpath has a .dcm extension
                if not output_fullpath.endswith('.dcm'):
                    output_fullpath += '.dcm'

                if not os.path.exists(output_fullpath):
                    break

                nextIndexIndex += 1

        if nextIndexIndex < len(listOfIndices):
            self.nextDicomDfIndex = listOfIndices[nextIndexIndex]
            logging.info(f"Next DICOM dataframe index: {self.nextDicomDfIndex}")
        else:
            self.nextDicomDfIndex = None
            slicer.util.mainWindow().statusBar().showMessage("No more DICOM files to process", 3000)

        return self.nextDicomDfIndex

    def getCurrentProxyNode(self):
        """
        Get the proxy node of the master sequence node of the currently selected sequence browser node
        """
        parameterNode = self.getParameterNode()
        currentBrowserNode = parameterNode.ultrasoundSequenceBrowser
        if currentBrowserNode is None:
            logging.error("Current sequence browser node not found")
            return None

        # Get the proxy node of the master sequence node of the selected sequence browser node

        masterSequenceNode = currentBrowserNode.GetMasterSequenceNode()
        if masterSequenceNode is None:
            logging.error("Master sequence node missing for browser node with ID " + currentBrowserNode.GetID())
            return None

        proxyNode = currentBrowserNode.GetProxyNode(masterSequenceNode)
        if proxyNode is None:
            logging.error("Proxy node of master sequence node with ID " + masterSequenceNode.GetID() + " not found")
            return None

        return proxyNode

    def getFileForBrowserNode(self, browserNode):
        if browserNode is None:
            return None

        # Get the proxy node of the master sequence node of the selected sequence browser node
        masterSequenceNode = browserNode.GetMasterSequenceNode()
        proxyNode = browserNode.GetProxyNode(masterSequenceNode)

        # Get DICOM.instanceUID attribute from proxy node
        instanceUID = proxyNode.GetAttribute("DICOM.instanceUIDs")
        if instanceUID is None:
            logging.error("DICOM.instanceUIDs attribute not found in proxy node")
            return None

        # If instanceUID is a list, keep only the first item
        if isinstance(instanceUID, list):
            instanceUID = instanceUID[0]

        # Find row in self.dicomDf where instanceUID matches first instanceUID in instanceUIDs
        filepath = None
        for index, row in self.dicomDf.iterrows():
            rowInstanceUID = row['InstanceUID']
            currentInstanceUID = instanceUID
            if rowInstanceUID == currentInstanceUID:
                filepath = row['Filepath']
                break
        if filepath is None:
            logging.error(f"Could not find DICOM file for instanceUID {instanceUID}")
            return None

        return filepath

    def getNumberOfInstances(self):
        """
        Return the number of instances in the current DICOM dataframe.
        """
        if self.dicomDf is None:
            return 0
        else:
            return len(self.dicomDf)

    def getAutoMask(self):
        if not hasattr(self, 'currentDicomDataset'):
            logging.error("No current DICOM dataset loaded")
            return None
        model, input_shape, device = self.downloadAndPrepareModel()
        if model is None:
            return None
        return self.findMaskAutomatic(model, input_shape, device)

    def downloadAndPrepareModel(self):
        """ Download the AI model and prepare it for inference """
        # Set the Device to run the model on
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"The model will run on Device: {device}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(script_dir, 'Resources/checkpoints/')
        os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)

        model_path = os.path.join(checkpoint_dir, 'model_traced.pt')
        model_config_path = os.path.join(checkpoint_dir, 'model_config.yaml')

        model_url = "https://www.dropbox.com/scl/fi/abgn6ln13thh0v9mq5kqj/model_traced.pt?rlkey=8a9eugxbqeuzwrglz55sh7hkd&st=mwclwtgv&dl=1"
        config_url = "https://www.dropbox.com/scl/fi/klnwakbysn95nae85lmjz/model_config.yaml?rlkey=p1jada30bvbsihtfiw80dq7h2&st=7a8y0ewy&dl=1"

        if not os.path.exists(model_path):
            logging.info(f"The AI model does not exist. Starting download...")
            dialog = AnonymizeUltrasoundWidget.createWaitDialog(self, "Downloading AI Model", "The AI model does not exist. Downloading...")
            success = self.download_model(model_url, model_path)
            dialog.close()
            if not success:
                return None, None, None

        if not os.path.exists(model_config_path):
            logging.info(f"The model config file does not exist. Starting download...")
            success = self.download_model(config_url, model_config_path)
            if not success:
                return None, None, None

        # Check if the model loaded successfully
        try:
            model = torch.jit.load(model_path).to(device).eval()
        except Exception as e:
            logging.error(f"Failed to load the model: {e}")
            logging.error("Automatic mode is disabled. Please define the mask manually.")
            # TODO: Disable the button of Auto mask generation?
            return None, None, None

        # Check if the model config loaded successfully
        try:
            with open(model_config_path, 'r') as file:
                model_config = yaml.safe_load(file)
            input_shape_str = model_config['input_shape']
            input_shape = tuple(map(int, input_shape_str.strip('()').split(',')))
        except Exception as e:
            logging.error(f"Failed to load the model config: {e}")
            logging.error("Automatic mode is disabled. Please define the mask manually.")
            # TODO: Disable the button of Auto mask generation?
            return None, None, None

        return model, input_shape, device

    def findMaskAutomatic(self, model, input_shape, device):
        """ Generate a mask automatically using the AI model """
        slicer.app.pauseRender()
        parameterNode = self.getParameterNode()
        currentSequenceBrowser = parameterNode.ultrasoundSequenceBrowser
        masterSequenceNode = currentSequenceBrowser.GetMasterSequenceNode()
        currentVolumeNode = masterSequenceNode.GetNthDataNode(0)
        currentVolumeArray = slicer.util.arrayFromVolume(currentVolumeNode)
        maxVolumeArray = np.copy(currentVolumeArray)

        for i in range(1, masterSequenceNode.GetNumberOfDataNodes()):
            currentVolumeNode = masterSequenceNode.GetNthDataNode(i)
            currentVolumeArray = slicer.util.arrayFromVolume(currentVolumeNode)
            maxVolumeArray = np.maximum(maxVolumeArray, currentVolumeArray)
        frame_item = maxVolumeArray[0, :, :]
        slicer.app.resumeRender()

        if len(frame_item.shape) == 3 and frame_item.shape[2] == 3:
            frame_item = cv2.cvtColor(frame_item, cv2.COLOR_RGB2GRAY)
        original_frame_size = frame_item.shape[::-1]
        logging.debug(f"Original frame size: {str(frame_item.shape)}")
        frame_item = cv2.resize(frame_item, input_shape)
        logging.debug(f"Resized frame size: {str(input_shape)}")

        with torch.no_grad():
            input_tensor = torch.tensor(np.expand_dims(np.expand_dims(np.array(frame_item), axis=0), axis=0)).float()
            input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        output = (torch.softmax(output, dim=1) > 0.5).cpu().numpy()
        mask_output = np.uint8(output[0, 1, :, :])
        mask_output = cv2.resize(np.uint8(output[0, 1, :, :]), original_frame_size)
        logging.info(f"({str(mask_output.shape)}) Mask generated successfully")

        # Paint the red overlay where the model says the fan is
        Hc, Wc = mask_output.shape
        rgb = np.zeros((1, Hc, Wc, 3), dtype=np.uint8)
        rgb[0, mask_output == 1, 0] = 255      # red channel only
        self._autoMaskRGB = rgb
        self._composeAndPushOverlay()

        approx_corners = self.find_four_corners(mask_output)

        if approx_corners is None:
            logging.error("Could not find the four corners of the foreground in the mask")
        else:
            top_left, top_right, bottom_right, bottom_left = approx_corners
            logging.debug(f"Approximate corners - Top-left: {top_left}, Top-right: {top_right}, Bottom-right: {bottom_right}, Bottom-left: {bottom_left}")
        return approx_corners

    def download_model(self, url, output_path):
        """ Download a file from a URL """
        try:
            # Send a GET request to the URL
            response = requests.get(url, stream=True)
            # Raise an exception if the request was unsuccessful
            response.raise_for_status()
            # Write the content to the file
            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                logging.info(f"Downloaded file saved to {output_path}")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download the file: {e}")
            # TODO: Disable the button of Auto mask generation?
            return False

    def find_extreme_corners(self, points):
        # Convert points to a numpy array for easier manipulation
        points = np.array(points)
        # Find the top-left corner (minimum x + y)
        top_left = list(points[np.argmin(points[:, 0] + points[:, 1])])
        # Find the top-right corner (maximum x - y)
        top_right = list(points[np.argmax(points[:, 0] - points[:, 1])])
        # Find the bottom-left corner (minimum x - y)
        bottom_left = list(points[np.argmin(points[:, 0] - points[:, 1])])
        # Find the bottom-right corner (maximum x + y)
        bottom_right = list(points[np.argmax(points[:, 0] + points[:, 1])])

        corners = [tuple(top_left), tuple(top_right), tuple(bottom_left), tuple(bottom_right)]
        unique_corners = set(corners)
        num_unique_corners = len(unique_corners)

        # If there are 3 unique corners, then the mask is a triangle
        epsilon = 2
        if num_unique_corners == 3:
            # Define the top point (which one is higher from top-left and top-right, higher means less y)
            top_point = top_left if top_left[1] < top_right[1] else top_right
            # Set top-left and top-right equal to top_point
            top_left = list(top_point)
            top_right = list(top_point)
            # Adjust x coordinates
            top_left[0] -= epsilon
            top_right[0] += epsilon

        # TODO: Is it possible?!
        if num_unique_corners < 3:
            return None

        return np.array([top_left, top_right, bottom_left, bottom_right])

    def find_four_corners(self, mask):
        """ Find the four corners of the foreground in the mask. """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        # Assuming the largest contour is the foreground
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx_corners = cv2.approxPolyDP(contour, epsilon, True)

        # Reshape the approx_corners array to a 2D array
        approx_corners = approx_corners.reshape(-1, 2)

        # If the contour has more than 4 corners, then find the extreme corners
        if len(approx_corners) < 3:
            return None

        approx_corners = self.find_extreme_corners(approx_corners)
        return approx_corners

    def showMaskContour(self, show=True):
        parameterNode = self.getParameterNode()
        maskContourVolumeNode = parameterNode.overlayVolume

        # Set mask contour as foreground volume in slice viewers, and make sure it has 50% opacity
        if maskContourVolumeNode and show:
            slicer.util.setSliceViewerLayers(foreground=maskContourVolumeNode)
            sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode()
            sliceCompositeNode.SetForegroundOpacity(0.5)
            # Make sure background and foreground are not alpha blended but added
            sliceCompositeNode.SetCompositing(2) # Add foreground and background
            # Set window and level so that mask is visible
            displayNode = maskContourVolumeNode.GetDisplayNode()
            displayNode.SetWindow(255)
            displayNode.SetLevel(127)
        else:
            slicer.util.setSliceViewerLayers(foreground=None)

    def onNodeAdded(self, caller, event):
        """
        Called when a node is added to the scene.
        If a sequence browser node is added, make that the current cultrasound sequence browser node.
        """
        node = caller
        if node.IsA("vtkMRMLSequenceBrowserNode"):
            logging.info(f"Sequence browser node added: {node.GetID()}")
            # Set newly added sequence browser node as current sequence browser node
            parameterNode = self.getParameterNode()
            parameterNode.ultrasoundSequenceBrowser = node

    def updateMaskVolume(self, three_point=False):
        """
        Update the mask volume based on the current mask landmarks. Returns a string that can displayed as status message.
        """
        parameterNode = self.getParameterNode()

        fanMaskMarkupsNode = parameterNode.maskMarkups
        # require 3 or 4 points depending on three-point setting
        count = fanMaskMarkupsNode.GetNumberOfControlPoints()
        required = 3 if three_point else 4
        if count < required:
            # Clear the overlay volume
            maskContourVolumeNode = parameterNode.overlayVolume
            if maskContourVolumeNode is not None:
                maskContourArray = slicer.util.arrayFromVolume(maskContourVolumeNode)
                maskContourArray.fill(0)
                slicer.util.updateVolumeFromArray(maskContourVolumeNode, maskContourArray)
            msg = f"At least {required} control points are needed to define a mask"
            logging.info(msg)
            return msg
        elif count > required: # I've never seen this happen, but just in case
            # Clear all the points
            fanMaskMarkupsNode.RemoveAllControlPoints()
            # Clear the overlay volume
            maskContourVolumeNode = parameterNode.overlayVolume                
            msg = f"Only {required} control points are needed to define a mask"
            logging.info(msg)
            return msg

        # Compute the center of the mask points

        currentVolumeNode = self.getCurrentProxyNode()
        if currentVolumeNode is None:
            logging.error("Current volume node not found")
            return("No ultrasound image is loaded")

        rasToIjk = vtk.vtkMatrix4x4()
        currentVolumeNode.GetRASToIJKMatrix(rasToIjk)

        # Allocate for max 4 points, but only fill what we have
        controlPoints_ijk = np.zeros((4, 4))
        actual_points = required  # 3 or 4
        
        for i in range(actual_points):
            markupPoint = [0, 0, 0]
            fanMaskMarkupsNode.GetNthControlPointPosition(i, markupPoint)
            ijkPoint = rasToIjk.MultiplyPoint([markupPoint[0], markupPoint[1], markupPoint[2], 1.0])
            controlPoints_ijk[i] = [ijkPoint[0], ijkPoint[1], ijkPoint[2], 1.0]

        if three_point:
            # For 3-point mode: assign points based on Y position
            # Point 0 (top) -> topLeft (apex)
            # Points 1,2 (bottom) -> bottomLeft, bottomRight
            points_by_y = sorted(range(actual_points), key=lambda i: controlPoints_ijk[i][1])
            
            topLeft = controlPoints_ijk[points_by_y[0]][:3]  # highest point (smallest Y)
            bottomLeft = controlPoints_ijk[points_by_y[1]][:3]
            bottomRight = controlPoints_ijk[points_by_y[2]][:3]
            
            # No topRight in 3-point mode
            topRight = None
        else:
            centerOfGravity = np.mean(controlPoints_ijk[:4], axis=0)

            topLeft = np.zeros(3)
            topRight = np.zeros(3)
            bottomLeft = np.zeros(3)
            bottomRight = np.zeros(3)

            for i in range(4):
                if controlPoints_ijk[i][0] < centerOfGravity[0] and controlPoints_ijk[i][1] < centerOfGravity[1]:
                    topLeft = controlPoints_ijk[i][:3]
                elif controlPoints_ijk[i][0] >= centerOfGravity[0] and controlPoints_ijk[i][1] < centerOfGravity[1]:
                    topRight = controlPoints_ijk[i][:3]
                elif controlPoints_ijk[i][0] < centerOfGravity[0] and controlPoints_ijk[i][1] >= centerOfGravity[1]:
                    bottomLeft = controlPoints_ijk[i][:3]
                elif controlPoints_ijk[i][0] >= centerOfGravity[0] and controlPoints_ijk[i][1] >= centerOfGravity[1]:
                    bottomRight = controlPoints_ijk[i][:3]

            if np.array_equal(topLeft, np.zeros(3)) or np.array_equal(topRight, np.zeros(3)) or \
                    np.array_equal(bottomLeft, np.zeros(3)) or np.array_equal(bottomRight, np.zeros(3)):
                logging.debug("Could not determine mask corners")
                return("Mask points should be in a fan or rectangular shape with two points in the top and two points in the bottom."
                    "\nMove points to try again.")

        imageArray = slicer.util.arrayFromVolume(currentVolumeNode)  # (z, y, x, channels)

        # Create mask based on mode
        if three_point:
            # Always create fan mask for 3-point mode
            assert topRight is None, "topRight should be None in 3-point mode"
            mask_array = self.createFanMask(imageArray, topLeft, None, bottomLeft, bottomRight, value=1, three_point=True)
        else:
            # Detect if the mask is a fan or a rectangle for 4-point mode
            maskHeight = abs(topLeft[1] - bottomLeft[1])
            tolerancePixels = round(0.1 * maskHeight)  #todo: Make this tolerance value a setting
            if abs(topLeft[0] - bottomLeft[0]) < tolerancePixels and abs(topRight[0] - bottomRight[0]) < tolerancePixels:
                # Mask is a rectangle
                mask_array = self.createRectangleMask(imageArray, topLeft, topRight, bottomLeft, bottomRight)
            else:
                # 4-point fan
                mask_array = self.createFanMask(imageArray, topLeft, topRight, bottomLeft, bottomRight, value=1, three_point=False)

        mask_contour_array = np.copy(mask_array)
        masking_kernel = np.ones((3, 3), np.uint8)
        mask_contour_array_eroded = cv2.erode(mask_contour_array, masking_kernel, iterations=3)
        mask_contour_array = mask_contour_array - mask_contour_array_eroded

        # Create RGB overlay mask
        maskContourVolumeNode = parameterNode.overlayVolume
        overlay_shape = (1, imageArray.shape[1], imageArray.shape[2], 3)
        rgb_mask = np.zeros(overlay_shape, dtype=np.uint8)
        # Set green channel for mask contour
        rgb_mask[0, :, :, 1] = (mask_contour_array > 0) * 255
        self._manualMaskRGB = rgb_mask

        if maskContourVolumeNode is None:
            maskContourVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVectorVolumeNode", "AnonymizeUltrasound Overlay")
            parameterNode.overlayVolume = maskContourVolumeNode
            maskContourVolumeNode.CreateDefaultDisplayNodes()
            maskContourDisplayNode = maskContourVolumeNode.GetDisplayNode()
            maskContourDisplayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeGreen")

        maskContourVolumeNode.SetSpacing(currentVolumeNode.GetSpacing())
        maskContourVolumeNode.SetOrigin(currentVolumeNode.GetOrigin())
        ijkToRas = vtk.vtkMatrix4x4()
        currentVolumeNode.GetIJKToRASMatrix(ijkToRas)
        maskContourVolumeNode.SetIJKToRASMatrix(ijkToRas)

        # Allocate and set image data for vector volume
        overlayImageData = vtk.vtkImageData()
        overlayImageData.SetDimensions(imageArray.shape[2], imageArray.shape[1], 1)
        overlayImageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
        maskContourVolumeNode.SetAndObserveImageData(overlayImageData)
        self._composeAndPushOverlay()

        # Add a dimension to the mask array

        mask_array = np.expand_dims(mask_array, axis=0)

        # Create a new volume node for the mask

        maskVolumeNode = parameterNode.maskVolume
        if maskVolumeNode is None:
            maskVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "AnonymizeUltrasound Mask")
            parameterNode.maskVolume = maskVolumeNode

        maskVolumeNode.SetSpacing(currentVolumeNode.GetSpacing())
        maskVolumeNode.SetOrigin(currentVolumeNode.GetOrigin())
        ijkToRas = vtk.vtkMatrix4x4()
        currentVolumeNode.GetIJKToRASMatrix(ijkToRas)
        maskVolumeNode.SetIJKToRASMatrix(ijkToRas)

        slicer.util.updateVolumeFromArray(maskVolumeNode, mask_array)

        return("Mask created successfully")

    def createRectangleMask(self, imageArray, topLeft, topRight, bottomLeft, bottomRight):
        image_size_rows = imageArray.shape[1]
        image_size_cols = imageArray.shape[2]

        rectangleLeft = round((topLeft[0] + bottomLeft[0]) / 2)
        rectangleRight = round((topRight[0] + bottomRight[0]) / 2)
        rectangleTop = round((topLeft[1] + topRight[1]) / 2)
        rectangleBottom = round((bottomLeft[1] + bottomRight[1]) / 2)

        self.maskParameters = {}
        self.maskParameters["mask_type"] = "rectangle"
        self.maskParameters["rectangle_left"] = rectangleLeft
        self.maskParameters["rectangle_right"] = rectangleRight
        self.maskParameters["rectangle_top"] = rectangleTop
        self.maskParameters["rectangle_bottom"] = rectangleBottom

        mask_array = np.zeros((image_size_rows, image_size_cols), dtype=np.uint8)
        mask_array[rectangleTop:rectangleBottom, rectangleLeft:rectangleRight] = 1

        return mask_array

    def createFanMask(self, imageArray, topLeft, topRight, bottomLeft, bottomRight, value=255, three_point=False):
        if three_point:
            image_size_rows, image_size_cols = imageArray.shape[1], imageArray.shape[2]
            mask_array = np.zeros((image_size_rows, image_size_cols), dtype=np.uint8)
            # apex is topLeft, bottom points are bottomLeft/bottomRight
            cx, cy = int(round(topLeft[0])), int(round(topLeft[1]))
            # compute radius as avg of the two bottom points radii
            r1 = math.hypot(bottomLeft[0]-topLeft[0], bottomLeft[1]-topLeft[1])
            r2 = math.hypot(bottomRight[0]-topLeft[0], bottomRight[1]-topLeft[1])
            radius = int(round((r1 + r2) / 2))
            # compute angles
            angle1 = math.degrees(math.atan2(bottomLeft[1]-topLeft[1], bottomLeft[0]-topLeft[0]))
            angle2 = math.degrees(math.atan2(bottomRight[1]-topLeft[1], bottomRight[0]-topLeft[0]))
            if angle2 < angle1:
                angle1, angle2 = angle2, angle1
            mask_array = self.draw_circle_segment(mask_array, (cx, cy), radius, angle1, angle2, value)
            self.maskParameters = {}
            self.maskParameters["mask_type"] = "fan"
            self.maskParameters["angle1"] = angle1
            self.maskParameters["angle2"] = angle2
            self.maskParameters["center_rows_px"] = cy
            self.maskParameters["center_cols_px"] = cx
            self.maskParameters["radius1"] = 0 # no radius for apex
            self.maskParameters["radius2"] = radius
            self.maskParameters["image_size_rows"] = image_size_rows
            self.maskParameters["image_size_cols"] = image_size_cols
            return mask_array
        else:
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
                logging.warning("Left and right lines are parallel")
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
            
            self.maskParameters = {}
            self.maskParameters["mask_type"] = "fan"
            self.maskParameters["angle1"] = angle1
            self.maskParameters["angle2"] = angle2
            self.maskParameters["center_rows_px"] = center_rows_px
            self.maskParameters["center_cols_px"] = center_cols_px
            self.maskParameters["radius1"] = radius1
            self.maskParameters["radius2"] = radius2
            self.maskParameters["image_size_rows"] = image_size_rows
            self.maskParameters["image_size_cols"] = image_size_cols

            # logging.debug(f"Radius1: {radius1}, Radius2: {radius2}, Angle1: {angle1}, Angle2: {angle2}, Center: ({center_cols_px}, {center_rows_px})") 
            return mask_array

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

    def maskSequence(self, three_point=False):
        self.updateMaskVolume(three_point=three_point)

        parameterNode = self.getParameterNode()
        currentSequenceBrowser = parameterNode.ultrasoundSequenceBrowser

        if currentSequenceBrowser is None:
            logging.error("No sequence browser node loaded!")
            return

        # Get mask volume

        maskVolumeNode = parameterNode.maskVolume
        if maskVolumeNode is None:
            logging.error("No mask volume loaded!")
            return

        maskArray = slicer.util.arrayFromVolume(maskVolumeNode)

        masterSequenceNode = currentSequenceBrowser.GetMasterSequenceNode()
        currentSequenceBrowser.SetRecording(masterSequenceNode, True)

        for index in range(masterSequenceNode.GetNumberOfDataNodes()):
            currentSequenceBrowser.SetSelectedItemNumber(index)
            currentVolumeNode = masterSequenceNode.GetNthDataNode(index)
            currentVolumeArray = slicer.util.arrayFromVolume(currentVolumeNode)

            # Mask every channel of the current volume array using maskArray

            for channel in range(currentVolumeArray.shape[3]):
                currentVolumeArray[:, :, :, channel] = np.multiply(currentVolumeArray[:, :, :, channel], maskArray)

        # Re-render views

        currentSequenceBrowser.SetSelectedItemNumber(0)
        proxyNode = self.getCurrentProxyNode()
        proxyNode.GetImageData().Modified()

    def convertToJsonCompatible(self, obj):
        if isinstance(obj, pydicom.multival.MultiValue):
            return list(obj)
        if isinstance(obj, pydicom.valuerep.PersonName):
            return str(obj)
        if isinstance(obj, bytes):
            return obj.decode('latin-1')
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    def compressFrameToJpeg(self, frame, quality=95):
        """
        Compress a single frame using JPEG compression.
        :param frame: a numpy array representing the image frame (W, H, C)
        :param quality: an integer from 0 to 100 setting the quality of the compression
        :return: compressed frame bytes
        """
        # If frame is 2-dimensional, expand to 3-dimensional
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, axis=-1)

        if frame.shape[2] == 1:
            image = Image.fromarray(frame[:, :, 0]).convert("L")
        else:
            image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
        with io.BytesIO() as output:
            image.save(output, format="JPEG", quality=quality)
            return output.getvalue()

    def generateNameFromDicomData(self, ds, hashPatientId = True):
        """
        Generate a filename from a DICOM header dictionary.
        Optionally, the name will be a hash of the PatientID and the SOP Instance UID.
        The name will consist of two parts:
        X_Y.dcm
        X is generated by hashing the original patient UID to a 10-digit number.
        Y is generated from the DICOM instance UID, but limited to 8 digits

        :param headerDict: DICOM header data
        :returns: tuple (filename, patientId, instanceId)
        """
        patientUID = ds.PatientID
        instanceUID = ds.SOPInstanceUID

        if patientUID is None or patientUID == "":
            logging.error("PatientID not found in DICOM header dict")
            return ""

        if instanceUID is None or instanceUID == "":
            logging.error("SOPInstanceUID not found in DICOM header dict")
            return ""

        if hashPatientId:
            hash_object = hashlib.sha256()
            hash_object.update(str(patientUID).encode())
            patientId = int(hash_object.hexdigest(), 16) % 10**10
        else:
            patientId = patientUID

        hash_object_instance = hashlib.sha256()
        hash_object_instance.update(str(instanceUID).encode())
        instanceId = int(hash_object_instance.hexdigest(), 16) % 10**8

        # Add trailing zeros
        patientId = str(patientId).zfill(10)
        instanceId = str(instanceId).zfill(8)

        return f"{patientId}_{instanceId}.dcm", patientId, instanceId

    def findKeyInDict(self, d: dict, target_key: str):
        """
        Recursively search for a key in a nested dictionary.
        Returns the value if the key is found, otherwise 'N/A'.
        """
        if target_key in d:
            return d[target_key]

        for key, value in d.items():
            if isinstance(value, dict):
                result = self.findKeyInDict(value, target_key)
                if result != 'N/A':
                    return result
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        result = self.findKeyInDict(item, target_key)
                        if result != 'N/A':
                            return result

        return 'N/A'

    def saveDicomFile(self, dicomFilePath, new_patient_name = None, new_patient_id = None):
        parameterNode = self.getParameterNode()

        # Collect all image frames in a numpy array

        currentSequenceBrowser = parameterNode.ultrasoundSequenceBrowser
        masterSequenceNode = currentSequenceBrowser.GetMasterSequenceNode()
        proxyNode = self.getCurrentProxyNode()

        proxyNodeArray = slicer.util.arrayFromVolume(proxyNode)

        imageArray = np.zeros((masterSequenceNode.GetNumberOfDataNodes(),
                               proxyNodeArray.shape[1], proxyNodeArray.shape[2], proxyNodeArray.shape[3]), dtype=np.int8)

        for index in range(masterSequenceNode.GetNumberOfDataNodes()):
            currentSequenceBrowser.SetSelectedItemNumber(index)
            currentVolumeNode = masterSequenceNode.GetNthDataNode(index)
            currentVolumeArray = slicer.util.arrayFromVolume(currentVolumeNode)
            imageArray[index, :, :, :] = currentVolumeArray

        # Create a new DICOM dataset
        anonymized_ds = pydicom.Dataset()
        dicom_header_data = self.currentDicomHeader
        original_ds = self.currentDicomDataset

        # Copy SequenceOfUltrasoundRegions if available
        if hasattr(original_ds, "SequenceOfUltrasoundRegions") and len(original_ds.SequenceOfUltrasoundRegions) > 0:
            anonymized_ds.SequenceOfUltrasoundRegions = original_ds.SequenceOfUltrasoundRegions

        # Copy spacing to conventional PixelSpacing tag for DICOM readers that don't support ultrasound regions
        deltaX = self.findKeyInDict(dicom_header_data, 'Physical Delta X')
        deltaY = self.findKeyInDict(dicom_header_data, 'Physical Delta Y')
        if deltaX != 'N/A' and deltaY != 'N/A':
            deltaXmm = float(deltaX) * 10
            deltaYmm = float(deltaY) * 10
            # Conver to string with maximum 14 digits
            deltaXmmStr = "{:.14f}".format(deltaXmm)
            deltaYmmStr = "{:.14f}".format(deltaYmm)
            anonymized_ds.PixelSpacing = [deltaXmmStr, deltaYmmStr]

        # Verify array shape and set DICOM tags accordingly
        if len(imageArray.shape) == 4:  # Multi-frame format
            frames, height, width, channels = imageArray.shape
            anonymized_ds.Rows = height
            anonymized_ds.Columns = width
            anonymized_ds.NumberOfFrames = frames
            anonymized_ds.SamplesPerPixel = channels
        elif len(imageArray.shape) == 3:  # Muti-frame, grayscale
            frames, height, width = imageArray.shape
            anonymized_ds.Rows = height
            anonymized_ds.Columns = width
            anonymized_ds.NumberOfFrames = frames
            anonymized_ds.SamplesPerPixel = 1
        elif len(imageArray.shape) == 2:  # Single frame, Grayscale
            height, width = imageArray.shape
            anonymized_ds.Rows = height
            anonymized_ds.Columns = width
            anonymized_ds.SamplesPerPixel = 1

        # Set PhotometricInterpretation based on number of channels
        if anonymized_ds.SamplesPerPixel == 1:
            anonymized_ds.PhotometricInterpretation = "MONOCHROME2"
        elif anonymized_ds.SamplesPerPixel == 3:
            anonymized_ds.PhotometricInterpretation = "YBR_FULL_422"  # For JPEG compressed images

        # Ensure the data type of numpy array matches the expected pixel data type
        anonymized_ds.Modality = 'US'

        # Compress each frame and set PixelData
        compressed_frames = []
        for frame in imageArray:
            compressed_frame = self.compressFrameToJpeg(frame)
            compressed_frames.append(compressed_frame)
        anonymized_ds.PixelData = pydicom.encaps.encapsulate(compressed_frames)
        anonymized_ds['PixelData'].VR = 'OB'
        anonymized_ds['PixelData'].is_undefined_length = True
        anonymized_ds.LossyImageCompression = '01'
        anonymized_ds.LossyImageCompressionMethod = 'ISO_10918_1'

        # Copy Manufacturer if available
        if hasattr(original_ds, "Manufacturer") and original_ds.Manufacturer:
            anonymized_ds.Manufacturer = original_ds.Manufacturer

        # Map additional DICOM tags from header data
        dicom_tag_mapping = {
            "BitsAllocated": "Bits Allocated",
            "BitsStored": "Bits Stored",
            "HighBit": "High Bit",
            "ManufacturerModelName": "Manufacturer's Model Name",
            "PatientAge": "Patient's Age",
            "PatientSex": "Patient's Sex",
            "PixelRepresentation": "Pixel Representation",
            "SeriesNumber": "Series Number",
            "StationName": "Station Name",
            "StudyDate": "Study Date",
            "StudyDescription": "Study Description",
            "StudyID": "Study ID",
            "StudyTime": "Study Time",
            "TransducerType": "Transducer Data"
        }
        for dicom_tag, header_key in dicom_tag_mapping.items():
            if header_key in dicom_header_data:
                setattr(anonymized_ds, dicom_tag, dicom_header_data[header_key])
            else:
                if dicom_tag in ["BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation"]:  # Mandatory for OHIF
                    logging.error(f"{dicom_tag} not found for DICOM header file: {dicomFilePath}")

        # Set or generate required UIDs
        if not hasattr(original_ds, 'SOPClassUID') or not original_ds.SOPClassUID:
            logging.error(f"SOPClassUID not found. Generating new one for {dicomFilePath}.")
            anonymized_ds.SOPClassUID = pydicom.uid.generate_uid()
        else:
            anonymized_ds.SOPClassUID = original_ds.SOPClassUID

        if original_ds.SOPInstanceUID is None or len(original_ds.SOPInstanceUID) < 1:
            logging.error(f"SOPInstanceUID not found. Generating new one for {dicomFilePath}. Exported data may be untraceable.")
            anonymized_ds.SOPInstanceUID = pydicom.uid.generate_uid()
        else:
            anonymized_ds.SOPInstanceUID = original_ds.SOPInstanceUID

        # Generate a unique SeriesInstanceUID. This is because ultrasound machines often reuse the same SeriesInstanceUID, which can cause issues in the viewer.
        anonymized_ds.SeriesInstanceUID = pydicom.uid.generate_uid()

        if original_ds.StudyInstanceUID is None or len(original_ds.StudyInstanceUID) < 1:
            logging.error(f"StudyInstanceUID not found. Generating new one for {dicomFilePath}. Exported data may be untraceable.")
            anonymized_ds.StudyInstanceUID = pydicom.uid.generate_uid()
        else:
            anonymized_ds.StudyInstanceUID = original_ds.StudyInstanceUID

        if new_patient_name is not None:
            anonymized_ds.PatientName = new_patient_name
        else:
            anonymized_ds.PatientName = original_ds.PatientName

        if new_patient_id is not None:
            anonymized_ds.PatientID = new_patient_id
        else:
            anonymized_ds.PatientID = original_ds.PatientID

        # Make the series desciption the filename, so we can easily identify the file later in the viewer
        new_series_description = os.path.basename(dicomFilePath)
        anonymized_ds.SeriesDescription = new_series_description

        # Add missing required attributes to satisfy DICOM conformance.
        # Type 2 elements must be present (they can be empty).
        if not hasattr(anonymized_ds, 'PatientBirthDate'):
            anonymized_ds.PatientBirthDate = ''
        if not hasattr(anonymized_ds, 'ReferringPhysicianName'):
            anonymized_ds.ReferringPhysicianName = ''
        if not hasattr(anonymized_ds, 'AccessionNumber'):
            anonymized_ds.AccessionNumber = ''

        patientId = original_ds.PatientID
        random.seed(patientId)
        random_number = random.randint(0, 30)

        # Get the Series Date and Content Data from the header, and add the random_number as an offset to the day, shifting the month if necessary
        study_date = original_ds.StudyDate if hasattr(original_ds, 'StudyDate') else '19000101'
        series_date = original_ds.SeriesDate if hasattr(original_ds, 'SeriesDate') else '19000101'
        content_date = original_ds.ContentDate if hasattr(original_ds, 'ContentDate') else '19000101'

        study_date = datetime.datetime.strptime(study_date, "%Y%m%d") + datetime.timedelta(days=random_number)
        series_date = datetime.datetime.strptime(series_date, "%Y%m%d") + datetime.timedelta(days=random_number)
        content_date = datetime.datetime.strptime(content_date, "%Y%m%d") + datetime.timedelta(days=random_number)
        anonymized_ds.StudyDate = study_date.strftime("%Y%m%d")
        anonymized_ds.SeriesDate = series_date.strftime("%Y%m%d")
        anonymized_ds.ContentDate = content_date.strftime("%Y%m%d")

        anonymized_ds.StudyTime = original_ds.StudyTime if hasattr(original_ds, 'StudyTime') else ''
        anonymized_ds.SeriesTime = original_ds.SeriesTime if hasattr(original_ds, 'SeriesTime') else ''
        anonymized_ds.ContentTime = original_ds.ContentTime if hasattr(original_ds, 'ContentTime') else ''

        # Get the SeriesNumber from the self.dicomDf table corresponding to the current DICOM file

        series_number = self.dicomDf.loc[self.dicomDf['InstanceUID'] == original_ds.SOPInstanceUID, 'SeriesNumber'].values
        anonymized_ds.SeriesNumber = series_number[0] if len(series_number) > 0 else '1'

        # Conditional elements: provide empty defaults if unknown.
        if not hasattr(anonymized_ds, 'Laterality'):
            anonymized_ds.Laterality = ''
        if not hasattr(anonymized_ds, 'InstanceNumber'):
            anonymized_ds.InstanceNumber = 1
        if not hasattr(anonymized_ds, 'PatientOrientation'):
            anonymized_ds.PatientOrientation = ''

        # For multi-frame images, add FrameIncrementPointer and FrameTime (Type 1C)
        if hasattr(anonymized_ds, 'NumberOfFrames') and int(anonymized_ds.NumberOfFrames) > 1:
            if hasattr(original_ds, 'FrameTime'):
                anonymized_ds.FrameTime = original_ds.FrameTime
            else:
                anonymized_ds.FrameTime = 0.1  # Default to 0.1 seconds
            if hasattr(original_ds, 'FrameIncrementPointer'):
                anonymized_ds.FrameIncrementPointer = original_ds.FrameIncrementPointer
            else:
                anonymized_ds.FrameIncrementPointer = pydicom.tag.Tag(0x0018, 0x1063)

        # For color images, set PlanarConfiguration (Type 1C)
        if anonymized_ds.SamplesPerPixel > 1 and not hasattr(anonymized_ds, 'PlanarConfiguration'):
            anonymized_ds.PlanarConfiguration = 0

        if not hasattr(anonymized_ds, "ImageType"):
            anonymized_ds.ImageType = r"ORIGINAL\PRIMARY\IMAGE"

        # Set meta information for the DICOM file
        meta = pydicom.Dataset()
        meta.FileMetaInformationGroupLength = 0
        meta.FileMetaInformationVersion = b'\x00\x01'
        meta.MediaStorageSOPClassUID = anonymized_ds.SOPClassUID  # This should match the SOPClassUID of the dataset
        meta.MediaStorageSOPInstanceUID = anonymized_ds.SOPInstanceUID  # This should match the SOPInstanceUID of the dataset
        meta.ImplementationClassUID = pydicom.uid.generate_uid(None)  # Generate a new UID for our implementation
        meta.TransferSyntaxUID = pydicom.uid.JPEGBaseline

        # Copy over the dataset values from ds to file_ds
        file_ds = pydicom.dataset.FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        for elem in anonymized_ds:
            file_ds.add(elem)

        # Set the is_implicit_VR and is_little_endian attributes for the encoding
        file_ds.is_implicit_VR = False
        file_ds.is_little_endian = True

        # Save the DICOM file
        file_ds.save_as(dicomFilePath)
        logging.info(f"DICOM generated successfully: {dicomFilePath}")

    def exportDicom(self,
                    outputDirectory,
                    outputFilename = None,
                    headersDirectory = None,
                    labels = None,
                    new_patient_name = None,
                    new_patient_id = None):
        """
        Export image array to DICOM files.

        :param outputDirectory: Output directory where the DICOM files will be saved.
        :param outputFilename: Output file name without extension. If None, a file name will be generated based
            on patient ID and instance UID.
        :param convertToGrayscale: If True, convert RGB images to grayscale.
        :param labels: List of annotation labels to be saved in accompanying CSV file.
        :param compression: If True, use JPEG compression (minimal) in output DICOM.
        """
        # Record sequence information to a dictionary. This will be saved in the annotations JSON file.
        SOPInstanceUID = self.currentDicomDataset.SOPInstanceUID
        if SOPInstanceUID is None:
            SOPInstanceUID = "None"

        sequenceInfo = {
            'SOPInstanceUID': SOPInstanceUID,
            'GrayscaleConversion': False
        }

        # Create output directory if it doesn't exist
        if not os.path.exists(outputDirectory):
            os.makedirs(outputDirectory)

        # Save DICOM image file
        if outputFilename is None:
            outputFilename, _, _ = self.generateNameFromDicomData(self.currentDicomDataset)
        if outputFilename == "":
            return None, None, None
        dicomFilePath = os.path.join(outputDirectory, outputFilename)
        self.saveDicomFile(dicomFilePath, new_patient_name, new_patient_id)

        # Save original DICOM header to a json file. This may not be completely anonymized.
        if headersDirectory is not None:
            if not os.path.exists(headersDirectory):
                os.makedirs(headersDirectory)
            dicomHeaderFileName = outputFilename.replace(".dcm", "_DICOMHeader.json")
            dicomHeaderFilePath = os.path.join(headersDirectory, dicomHeaderFileName)
            with open(dicomHeaderFilePath, 'w') as outfile:
                anonymizedDicomHeader = self.currentDicomHeader.copy()
                # Make the PatientName equal to the outputFilename without extension
                if "Patient's Name" in anonymizedDicomHeader:
                    anonymizedDicomHeader["Patient's Name"] = outputFilename.split(".")[0]
                # Make the month and day of the patient birth date 01 to anonymize the patient
                if "Patient's Birth Date" in anonymizedDicomHeader:
                    anonymizedDicomHeader["Patient's Birth Date"] = anonymizedDicomHeader["Patient's Birth Date"][:4] + "0101"
                json.dump(anonymizedDicomHeader, outfile, default=self.convertToJsonCompatible)

        # Add mask parameters to sequenceInfo
        for key, value in self.maskParameters.items():
            sequenceInfo[key] = value

        # Add annotation labels to sequenceInfo
        if labels is not None:
            sequenceInfo["AnnotationLabels"] = labels

        # Save sequenceInfo to a file
        annotationsFilename = outputFilename.replace(".dcm", ".json")
        sequenceInfoFilePath = os.path.join(outputDirectory, annotationsFilename)
        with open(sequenceInfoFilePath, 'w') as outfile:
            json.dump(sequenceInfo, outfile)

        return dicomFilePath, sequenceInfoFilePath, dicomHeaderFilePath

    def _composeAndPushOverlay(self):
        """Merge masks according to parameter-node switches and
        write into overlayVolume (always foreground)."""
        pnode = self.getParameterNode()
        # For now, always show both overlays if present
        rgb = None
        # Determine shape from available masks
        if self._manualMaskRGB is not None:
            _, h, w, _ = self._manualMaskRGB.shape
        elif self._autoMaskRGB is not None:
            _, h, w, _ = self._autoMaskRGB.shape
        else:
            return
        rgb = np.zeros((1, h, w, 3), dtype=np.uint8)
        # Optionally, you can add switches to pnode for visibility
        manualVisible = True
        autoVisible = self.showAutoOverlay
        if manualVisible and self._manualMaskRGB is not None:
            rgb[0] = np.maximum(rgb[0], self._manualMaskRGB[0])
        if autoVisible and self._autoMaskRGB is not None:
            rgb[0] = np.maximum(rgb[0], self._autoMaskRGB[0])
        if pnode.overlayVolume is not None:
            slicer.util.updateVolumeFromArray(pnode.overlayVolume, rgb)

    def cacheMaskForTransducer(self, transducer_model, control_points, mask_parameters):
        """
        Cache the current mask configuration for a transducer.

        :param transducer_model: unique identifier for the transducer
        :param control_points: list of control point coordinates
        :param mask_parameters: mask parameters dictionary
        """
        cached_info = CachedMaskInfo(transducer_model, control_points, mask_parameters)
        self.transducerMaskCache[transducer_model] = cached_info

        logging.info(f"Cached mask for transducer {transducer_model}: {cached_info.to_dict()}")

    def getCachedMaskForTransducer(self, transducer_model):
        """
        Retrieve cached mask information for a transducer.

        :param transducer_model: unique identifier for the transducer
        :return: CachedMaskInfo object or None if not found
        """
        if transducer_model in self.transducerMaskCache:
            cached_info = self.transducerMaskCache[transducer_model]
            return cached_info
        return None

    def applyCachedMask(self, cached_info):
        """
        Apply a cached mask to the current image.

        :param cached_info: CachedMaskInfo object
        :return: True if successfully applied, False otherwise
        """
        try:
            parameterNode = self.getParameterNode()
            maskMarkupsNode = parameterNode.maskMarkups

            if not maskMarkupsNode:
                logging.error("Mask markups node not found")
                return False

            maskMarkupsNode.RemoveAllControlPoints()

            # Add cached control points
            for point in cached_info.control_points:
                maskMarkupsNode.AddControlPoint(point[0], point[1], point[2])

            # Update mask volume and display
            self.updateMaskVolume()
            self.showMaskContour()

            # Update parameter node status
            if len(cached_info.control_points) >= 4:
                parameterNode.status = AnonymizerStatus.LANDMARKS_PLACED

            logging.info(f"Applied cached mask with {len(cached_info.control_points)} control points")
            return True

        except Exception as e:
            logging.error(f"Failed to apply cached mask: {str(e)}")
            return False

    def saveCurrentMaskToCache(self):
        """Save the current mask configuration to the mask cache."""
        try:
            parameterNode = self.getParameterNode()
            maskMarkupsNode = parameterNode.maskMarkups
            
            if not maskMarkupsNode or maskMarkupsNode.GetNumberOfControlPoints() < 3:
                logging.info("Insufficient control points to cache mask")
                return
            
            # Extract control points
            control_points = []
            for i in range(maskMarkupsNode.GetNumberOfControlPoints()):
                point = [0, 0, 0]
                maskMarkupsNode.GetNthControlPointPosition(i, point)
                control_points.append(point)
            
            # Cache the mask
            self.cacheMaskForTransducer(
                self.currentTransducerModel,
                control_points,
                self.maskParameters
            )
            logging.info(f"Saved current mask to cache for transducer {self.currentTransducerModel}")
            
        except Exception as e:
            logging.error(f"Failed to save current mask to cache: {str(e)}")

    def clearMaskCache(self):
        """Clear all cached masks."""
        self.transducerMaskCache.clear()
        logging.info("Cleared all cached masks")

class CachedMaskInfo:
    """Data structure to store cached mask information for a transducer."""

    def __init__(self, transducer_model, control_points, mask_parameters):
        self.transducer_model= transducer_model
        self.control_points = control_points
        self.mask_parameters = mask_parameters

    def to_dict(self):
        """Convert the cached mask information to a dictionary."""
        return {
            'transducer_model': self.transducer_model,
            'control_points': self.control_points,
            'mask_parameters': self.mask_parameters,
        }

#
# AnonymizeUltrasoundTest
#


class AnonymizeUltrasoundTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_AnonymizeUltrasound1()

    def test_AnonymizeUltrasound1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        onSlicerStartupCompleted()
        inputVolume = SampleData.downloadSample("AnonymizeUltrasound1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = AnonymizeUltrasoundLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
