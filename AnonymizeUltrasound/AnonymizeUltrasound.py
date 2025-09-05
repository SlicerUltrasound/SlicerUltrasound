from collections import defaultdict
import csv
from enum import Enum
from functools import cache
import re
import json
import logging
import numpy as np
import math
import os
import requests
from typing import Optional, Dict, List, Any
import time
import json
import shutil
from datetime import datetime

import qt
import vtk
import traceback

# Force matplotlib to not use a non-osx backend to avoid slicer crash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLVectorVolumeNode
from slicer import vtkMRMLSequenceBrowserNode
from slicer import vtkMRMLMarkupsFiducialNode
from DICOMLib import DICOMUtils

from common.dicom_file_manager import DicomFileManager
from common.masking import compute_masks_and_configs
from common.inference import load_model, preprocess_image, get_device
from common.evaluation import compare_masks, load_mask_config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'Resources/checkpoints/model_traced_unet_dsnt.pt')
MODEL_URL = "https://www.dropbox.com/scl/fi/mnu2k4n8fju6gy1glhieb/model_traced.pt?rlkey=eb0xmwzwsoesq3mp11s8xt9xd&dl=1"

class AnonymizeUltrasound(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Anonymize Ultrasound")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Ultrasound")]
        self.parent.dependencies = []
        self.parent.contributors = ["Tamas Ungi (Queen's University)"]
        self.parent.helpText = _("""
            This is a module for anonymizing ultrasound images and sequences stored in DICOM folders.
            The mask (green contour) signals what part of the image will stay after applying the mask.
            The image area under the green contour will be kept along with the pixels inside the contour.
            See more information in <a href="https://github.com/SlicerUltrasound/SlicerUltrasound?tab=readme-ov-file#anonymize-ultrasound">module documentation</a>.
            """)
        self.parent.acknowledgementText = _("""
            This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
            and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
            """)

        slicer.app.connect("startupCompleted()", onSlicerStartupCompleted)

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

    global monai
    try:
        import monai
    except ImportError:
        logging.info("AnonymizeUltrasound: monai not found, installing...")
        slicer.util.pip_install('monai')
        import monai

    global sklearn
    try:
        import sklearn
    except ImportError:
        logging.info("AnonymizeUltrasound: scikit-learn not found, installing...")
        slicer.util.pip_install('scikit-learn')
        import sklearn

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
    PRESERVE_DIRECTORY_STRUCTURE_SETTING = "AnonymizeUltrasound/preserveDirectoryStructure"
    AUTO_ANON_ENABLE_SETTING = "AnonymizeUltrasound/EnableAutoAnonymize"
    AUTO_ANON_INPUT_FOLDER_SETTING = "AnonymizeUltrasound/AutoAnonymizeInputFolder"
    AUTO_ANON_OUTPUT_FOLDER_SETTING = "AnonymizeUltrasound/AutoAnonymizeOutputFolder"
    AUTO_ANON_HEADERS_FOLDER_SETTING = "AnonymizeUltrasound/AutoAnonymizeHeadersFolder"
    AUTO_ANON_MODEL_PATH_SETTING = "AnonymizeUltrasound/AutoAnonymizeModelPath"
    AUTO_ANON_DEVICE_SETTING = "AnonymizeUltrasound/AutoAnonymizeDevice"
    AUTO_ANON_OVERVIEW_DIR_SETTING = "AnonymizeUltrasound/AutoAnonymizeOverviewDir"
    AUTO_ANON_GT_DIR_SETTING = "AnonymizeUltrasound/AutoAnonymizeGroundTruthDir"

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.compositingModeExit = None
        # If True, the user is processing DICOM files. If False, the user is not processing DICOM files.
        self.processing_mode = False

        # --- Keyboard shortcuts ---
        # M: toggle Define Mask, N: next scan, Space: toggle auto overlay, E: export scan, A: export and load next scan
        self.shortcutM = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutM.setKey(qt.QKeySequence('M'))
        self.shortcutP = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutP.setKey(qt.QKeySequence('P'))
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
                logging.error(f"Settings input folder {inputFolder} does not exist")
                self.ui.inputDirectoryButton.directory = self.get_default_user_data_directory()
        else:
            self.ui.inputDirectoryButton.directory = self.get_default_user_data_directory()

        self.ui.inputDirectoryButton.connect("directoryChanged(QString)",
                                             lambda newValue: self.onSettingChanged(self.INPUT_FOLDER_SETTING, newValue))

        outputFolder = settings.value(self.OUTPUT_FOLDER_SETTING)
        if outputFolder:
            if os.path.exists(outputFolder):
                self.ui.outputDirectoryButton.directory = outputFolder
            else:
                logging.error(f"Settings output folder {outputFolder} does not exist")
                self.ui.outputDirectoryButton.directory = self.get_default_user_data_directory()
        else:
            self.ui.outputDirectoryButton.directory = self.get_default_user_data_directory()

        self.ui.outputDirectoryButton.connect("directoryChanged(QString)",
                                              lambda newValue: self.onSettingChanged(self.OUTPUT_FOLDER_SETTING, newValue))

        headersFolder = settings.value(self.HEADERS_FOLDER_SETTING)
        if headersFolder:
            if os.path.exists(headersFolder):
                self.ui.headersDirectoryButton.directory = headersFolder
            else:
                logging.error(f"Settings headers folder {headersFolder} does not exist")
                self.ui.headersDirectoryButton.directory = self.get_default_user_data_directory()
        else:
            self.ui.headersDirectoryButton.directory = self.get_default_user_data_directory()

        self.ui.headersDirectoryButton.connect("directoryChanged(QString)",
                                               lambda newValue: self.onSettingChanged(self.HEADERS_FOLDER_SETTING, newValue))

        self.ui.importDicomButton.connect("clicked(bool)", self.onImportDicomButton)

        # Workflow control buttons

        self.ui.nextButton.clicked.connect(self.onNextButton)
        self.ui.prevButton.clicked.connect(self.onPreviousButton)
        self.ui.defineMaskButton.toggled.connect(self.onMaskLandmarksButton)
        self.ui.exportButton.clicked.connect(self.onExportScanButton)
        self.ui.exportAndNextButton.clicked.connect(self.onExportAndNextButton)

        # Settings widgets

        preserveDirectoryStructure = settings.value(self.PRESERVE_DIRECTORY_STRUCTURE_SETTING)
        if preserveDirectoryStructure and preserveDirectoryStructure.lower() == "true":
            self.ui.preserveDirectoryStructureCheckBox.checked = True
        else:
            self.ui.preserveDirectoryStructureCheckBox.checked = False
        self.ui.preserveDirectoryStructureCheckBox.connect('toggled(bool)',
                                                          lambda newValue: self.on_critical_setting_changed(self.PRESERVE_DIRECTORY_STRUCTURE_SETTING, str(newValue)))

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
            labelsPath = self.resourcePath('default_labels.csv')
        self.ui.labelsFileSelector.currentPath = labelsPath
        self.ui.labelsCollapsibleButton.collapsed = True

        # Auto anonymize settings
        enable_val = settings.value(self.AUTO_ANON_ENABLE_SETTING)
        self.ui.enableAutoAnonymizeCheckBox.checked = bool(enable_val and str(enable_val).lower() == "true")
        self.ui.enableAutoAnonymizeCheckBox.toggled.connect(self._on_auto_anon_enable_toggled)
        self._apply_auto_anon_visibility(self.ui.enableAutoAnonymizeCheckBox.checked)

        auto_anon_input_folder = settings.value(self.AUTO_ANON_INPUT_FOLDER_SETTING)
        if auto_anon_input_folder:
            if os.path.exists(auto_anon_input_folder):
                self.ui.inputDirectoryButtonBatch.directory = auto_anon_input_folder
            else:
                logging.error(f"Settings input folder {auto_anon_input_folder} does not exist")
                self.ui.inputDirectoryButtonBatch.directory = self.get_default_user_data_directory()
        else:
            # No setting saved, default to best user data directory
            self.ui.inputDirectoryButtonBatch.directory = self.get_default_user_data_directory()

        self.ui.inputDirectoryButtonBatch.connect("directoryChanged(QString)",
                                             lambda newValue: self.onSettingChanged(self.AUTO_ANON_INPUT_FOLDER_SETTING, newValue))

        auto_anon_output_folder = settings.value(self.AUTO_ANON_OUTPUT_FOLDER_SETTING)
        if auto_anon_output_folder:
            if os.path.exists(auto_anon_output_folder):
                self.ui.outputDirectoryButtonBatch.directory = auto_anon_output_folder
            else:
                logging.error(f"Settings output folder {auto_anon_output_folder} does not exist")
                self.ui.outputDirectoryButtonBatch.directory = self.get_default_user_data_directory()
        else:
            self.ui.outputDirectoryButtonBatch.directory = self.get_default_user_data_directory()

        self.ui.outputDirectoryButtonBatch.connect("directoryChanged(QString)",
                                              lambda newValue: self.onSettingChanged(self.AUTO_ANON_OUTPUT_FOLDER_SETTING, newValue))

        auto_anon_headers_folder = settings.value(self.AUTO_ANON_HEADERS_FOLDER_SETTING)
        if auto_anon_headers_folder:
            if os.path.exists(auto_anon_headers_folder):
                self.ui.headersDirectoryButtonBatch.directory = auto_anon_headers_folder
            else:
                logging.error(f"Settings headers folder {auto_anon_headers_folder} does not exist")
                self.ui.headersDirectoryButtonBatch.directory = self.get_default_user_data_directory()
        else:
            self.ui.headersDirectoryButtonBatch.directory = self.get_default_user_data_directory()

        self.ui.headersDirectoryButtonBatch.connect("directoryChanged(QString)",
                                               lambda newValue: self.onSettingChanged(self.AUTO_ANON_HEADERS_FOLDER_SETTING, newValue))

        model_path = settings.value(self.AUTO_ANON_MODEL_PATH_SETTING)
        if model_path:
            if os.path.exists(model_path):
                self.ui.autoAnonModelPathButton.currentPath = model_path
            else:
                logging.error(f"Settings model path {model_path} does not exist")
                self.ui.autoAnonModelPathButton.currentPath = MODEL_PATH
        else:
            self.ui.autoAnonModelPathButton.currentPath = MODEL_PATH

        target_device = settings.value(self.AUTO_ANON_DEVICE_SETTING)
        if target_device:
            device_index = self.ui.autoAnonDeviceComboBox.findText(target_device)
            if device_index >= 0:
                self.ui.autoAnonDeviceComboBox.setCurrentIndex(device_index)
            else:
                self.ui.autoAnonDeviceComboBox.setCurrentIndex(0)  # Default to CPU
            
        overview_dir = settings.value(self.AUTO_ANON_OVERVIEW_DIR_SETTING)
        if overview_dir:
            if os.path.exists(overview_dir):
                self.ui.autoAnonOverviewDirButton.directory = overview_dir
            else:
                logging.error(f"Settings overview directory {overview_dir} does not exist")

        gt_dir = settings.value(self.AUTO_ANON_GT_DIR_SETTING)
        if gt_dir:
            if os.path.exists(gt_dir):
                self.ui.autoAnonGroundTruthDirButton.directory = gt_dir
            else:
                logging.error(f"Settings ground truth directory {gt_dir} does not exist")
                self.ui.autoAnonGroundTruthDirButton.directory = self.get_default_user_data_directory()
        else:
            self.ui.autoAnonGroundTruthDirButton.directory = self.get_default_user_data_directory()

        self.ui.autoAnonModelPathButton.connect("currentPathChanged(QString)",
                                                lambda newValue: self.onSettingChanged(self.AUTO_ANON_MODEL_PATH_SETTING, newValue))
        
        self.ui.autoAnonDeviceComboBox.connect("currentTextChanged(QString)",
                                                lambda newValue: self.onSettingChanged(self.AUTO_ANON_DEVICE_SETTING, newValue))

        self.ui.autoAnonOverviewDirButton.connect("directoryChanged(QString)",
                                                  lambda newValue: self.onSettingChanged(self.AUTO_ANON_OVERVIEW_DIR_SETTING, newValue))
        self.ui.autoAnonGroundTruthDirButton.connect("directoryChanged(QString)",
                                                     lambda newValue: self.onSettingChanged(self.AUTO_ANON_GT_DIR_SETTING, newValue))

        # Run button
        self.ui.runAutoAnonymizeButton.clicked.connect(self.on_run_auto_anon_clicked)

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

    def onLabelsPathChanged(self, labelsFilepath=None):
        # Use provided path or default to resource path
        if labelsFilepath is None:
            # Try to load from settings first
            settings = qt.QSettings()
            savedPath = settings.value(self.LABELS_PATH_SETTING)
            if savedPath and os.path.exists(savedPath):
                labelsFilepath = savedPath
            else:
                labelsFilepath = self.resourcePath('default_labels.csv')

        # Save the path to settings for next time
        settings = qt.QSettings()
        settings.setValue(self.LABELS_PATH_SETTING, labelsFilepath)

        logging.info(f"Loading labels file from: {labelsFilepath}")
        categories = defaultdict(list)
        try:
            with open(labelsFilepath, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    category, label = map(str.strip, row)
                    # Store original values in categories dict but display humanized versions
                    categories[category].append(label)
        except (FileNotFoundError, PermissionError) as e:
            logging.error(f"Cannot read labels file: {labelsFilepath}, error: {e}")
            return

        # Clear existing widgets
        for i in reversed(range(self.ui.labelsScrollAreaWidgetContents.layout().count())):
            self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget().deleteLater()

        # humanize category and label, splitting them on CamelCase
        def humanize(text):
            # Split on CamelCase, handling consecutive capitals
            # This will split "LiverSpleenDiaphragm" into ["Liver", "Spleen", "Diaphragm"]
            words = re.findall('[A-Z][a-z\d]+|[A-Z\d]+(?=[A-Z][a-z\d]|[^a-zA-Z\d]|$)|[a-z\d]+', text)
            return ' '.join(words)
        # Create widgets with humanized display text
        for category, labels in categories.items():
            categoryGroupBox = qt.QGroupBox(humanize(category), self.ui.labelsScrollAreaWidgetContents)
            categoryGroupBox.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Preferred)
            categoryLayout = qt.QVBoxLayout(categoryGroupBox)
            categoryLayout.setSpacing(0)  # Remove spacing between items
            categoryLayout.setContentsMargins(2, 2, 2, 2)  # Reduce margins to minimum
            for label in labels:
                checkBox = qt.QCheckBox(humanize(label), categoryGroupBox)
                checkBox.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Preferred)
                checkBox.toggled.connect(lambda checked, cb=checkBox: self.onLabelCheckBoxToggled(cb, checked))
                # Store original category and label for saving
                checkBox.setProperty('originalCategory', category)
                checkBox.setProperty('originalLabel', label)
                categoryLayout.addWidget(checkBox)
            categoryGroupBox.setLayout(categoryLayout)
            self.ui.labelsScrollAreaWidgetContents.layout().addWidget(categoryGroupBox)

    def get_default_user_data_directory(self):
        """Get the most appropriate default directory for user data across platforms."""
        home_dir = os.path.expanduser("~")
        
        # Try Documents folder first (most user-friendly)
        documents_dir = os.path.join(home_dir, "Documents")
        if os.path.exists(documents_dir):
            return documents_dir
        
        # Fall back to home directory if Documents doesn't exist
        return home_dir

    def confirm_setting_change(self, message: str = "Are you sure you want to change this setting?") -> bool:
        """
        Show a confirmation dialog for setting changes.
        Returns True if user confirms, False if they cancel.
        """
        alert = qt.QMessageBox()
        alert.setIcon(qt.QMessageBox.Warning)
        alert.setWindowTitle("Confirm Setting Change")
        alert.setText(message)
        alert.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
        alert.setDefaultButton(qt.QMessageBox.No)

        result = alert.exec_()
        return result == qt.QMessageBox.Yes

    def message_box(self, title: str, message: str, icon: qt.QMessageBox.Icon = qt.QMessageBox.Information) -> None:
        """
        Show a message box with a given title and message.
        """
        alert = qt.QMessageBox()
        alert.setIcon(icon)
        alert.setWindowTitle(title)
        alert.setText(message)
        alert.setStandardButtons(qt.QMessageBox.Ok)
        alert.exec_()

    def format_setting_name(self, setting_name: str) -> str:
        """
        Convert a setting name to a human-readable format.
        Example: "AnonymizeUltrasound/preserveDirectoryStructure" -> "Preserve Directory Structure"
        """
        # Extract the part after the last '/'
        if '/' in setting_name:
            name_part = setting_name.split('/')[-1]
        else:
            name_part = setting_name

        # Convert camelCase to Title Case with spaces
        import re
        spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', name_part)
        return spaced.title()

    def revert_setting(self, settingName: str, previousValue: str) -> None:
        """
        Revert a UI setting to its previous value when user cancels a change.
        """
        if settingName == self.PRESERVE_DIRECTORY_STRUCTURE_SETTING:
            try:
                # Temporarily disconnect to prevent recursion
                self.ui.preserveDirectoryStructureCheckBox.disconnect('toggled(bool)')
                # Convert string back to boolean for checkbox
                should_be_checked = previousValue and previousValue.lower() == "true"
                self.ui.preserveDirectoryStructureCheckBox.checked = should_be_checked
            finally:
                # Reconnect the signal after setting the value
                self.ui.preserveDirectoryStructureCheckBox.connect('toggled(bool)', lambda newValue: self.on_critical_setting_changed(self.PRESERVE_DIRECTORY_STRUCTURE_SETTING, str(newValue)))
        else:
            logging.error(f"Reverting UI setting for {settingName} not implemented")

    def on_critical_setting_changed(self, settingName: str, newValue: str) -> None:
        """
        Handle changes to critical settings that may require user confirmation.

        Args:
            settingName (str): The full setting key/name identifier (e.g.,
                            "AnonymizeUltrasound/preserveDirectoryStructure")
            newValue (str): The new value for the setting as a string representation.
                        For boolean settings, this will be "True" or "False"

        Returns:
            None

        Raises:
            ValueError: If the setting name is not supported for critical setting changes

        Behavior:
            1. Retrieves the current/previous value from persistent settings
            2. Checks if the application is in processing mode for the specific setting
            3. If in processing mode and changing a critical setting:
                - Shows a confirmation dialog explaining potential consequences
                - If user cancels: reverts the UI to the previous state without persisting
                - If user confirms: proceeds with the change
            4. For unsupported settings, raises a ValueError
        """
        settings = slicer.app.settings()
        previousValue = settings.value(settingName)

        if self.processing_mode and settingName == self.PRESERVE_DIRECTORY_STRUCTURE_SETTING:
            message = f"Changing the '{self.format_setting_name(self.PRESERVE_DIRECTORY_STRUCTURE_SETTING)}'setting while processing DICOM files may result in duplicated files in the output directory. If you want to change the setting, reload the DICOM folder."
            if not self.confirm_setting_change(message):
                self.revert_setting(settingName, previousValue)
                return
        else:
            raise ValueError(f"Reverting UI setting for {settingName} not implemented")

    def onSettingChanged(self, settingName: str, newValue: str) -> None:
        """
        Update setting value and GUI based on user selection.
        @param settingName: setting name
        @param newValue: new value, if "" then setting is removed
        """
        settings = slicer.app.settings()

        # If the setting is not set, remove it from the settings
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
        self.set_processing_mode(True)

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

        numFiles = self.logic.dicom_manager.scan_directory(inputDirectory, self.ui.skipSingleframeCheckBox.checked, self.ui.hashPatientIdCheckBox.checked)
        logging.info(f"Found {numFiles} DICOM files in input folder")

        if numFiles > 0:
            self._parameterNode.status = AnonymizerStatus.INPUT_READY
        else:
            self._parameterNode.status = AnonymizerStatus.INITIAL

        # Export self.logic.dicom_manager.dicom_df as a CSV file in the headers directory
        if self.logic.dicom_manager.dicom_df is not None:
            outputFilePath = os.path.join(outputHeadersDirectory, "keys.csv")
            self.logic.dicom_manager.dicom_df.drop(columns=['DICOMDataset'], inplace=False).to_csv(outputFilePath, index=False)

        statusText = str(numFiles)
        if self.ui.skipSingleframeCheckBox.checked:
            statusText += " multi-frame dicom files found in input folder."
        else:
            statusText += " dicom files found in input folder."

        if self.ui.continueProgressCheckBox.checked:
            # Find the number of files already processed in the output directory
            numDone = self.logic.dicom_manager.update_progress_from_output(outputDirectory, self.ui.preserveDirectoryStructureCheckBox.checked)
            if numDone is None:
                statusText += '\nAll files have been processed. Cannot load more files from input folder.'
            elif numDone < 1:
                statusText += '\nNo files already processed. Starting from first in alphabetical order.'
            else:
                statusText += '\n' + str(numDone) + ' files already processed in output folder. Continue at next.'
        self.ui.statusLabel.text = statusText

        # Set the patient name prefix to the input directory name
        self.ui.namePrefixLineEdit.text = inputDirectory.split('/')[-1]
        if numFiles > 0:
            self.onNextButton()

    def set_processing_mode(self, mode: bool):
        self.processing_mode = mode

    def onPreviousButton(self) -> None:
        logging.info("Prev button clicked")
        threePointFanModeEnabled = self.ui.threePointFanCheckBox.checked
        continueProgress = self.ui.continueProgressCheckBox.checked

        # If continue progress is checked and nextDicomDfIndex is None, there is nothing more to load
        if self.logic.dicom_manager.next_index is None and continueProgress:
            self.ui.statusLabel.text = "All files from input folder have been processed to output folder. No more files to load."
            return

        # Remove observers for the mask markups node, because loading a new series will reset the scene and create a new markups node

        maskMarkupsNode = self._parameterNode.maskMarkups
        if maskMarkupsNode:
            self.removeObserver(maskMarkupsNode, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onPointModified)

        # Load the next series

        dialog = self.createWaitDialog("Loading series", "Please wait until the DICOM file is loaded...")
        currentDicomDfIndex = None
        try:
            outputDirectory = self.ui.outputDirectoryButton.directory
            currentDicomDfIndex = self.logic.loadPreviousSequence(outputDirectory, continueProgress)
            logging.info(f"Loaded series {currentDicomDfIndex}")
            if currentDicomDfIndex is None:
                statusText = "No more series to load"
                self.ui.statusLabel.text = statusText
                dialog.close()
            else:
                self.ui.progressBar.value = currentDicomDfIndex + 1
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

        if currentDicomDfIndex is not None and self.logic.dicom_manager.dicom_df is not None:
            current_dicom_record = self.logic.dicom_manager.dicom_df.iloc[currentDicomDfIndex]

            patientID = current_dicom_record.DICOMDataset.PatientID if current_dicom_record is not None else "N/A"
            if patientID:
                self.ui.patientIdLabel.text = patientID
            else:
                logging.error("Patient ID is missing")
                self.ui.patientIdLabel.text = 'None'

            instanceUID = current_dicom_record.DICOMDataset.SOPInstanceUID if current_dicom_record is not None else "N/A"
            if instanceUID is None:
                logging.error("Instance UID is missing")
                self.ui.sopInstanceUidLabel.text = 'None'
            else:
                self.ui.sopInstanceUidLabel.text = instanceUID

            statusText = f"Instance {instanceUID} loaded from file:\n"

            # Get the file path from the dataframe

            filepath = current_dicom_record['InputPath']
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
        else:
            self.ui.statusLabel.text = "No DICOM file loaded"

    def on_run_auto_anon_clicked(self):
        in_dir = self.ui.inputDirectoryButtonBatch.directory
        out_dir = self.ui.outputDirectoryButtonBatch.directory
        hdr_dir = self.ui.headersDirectoryButtonBatch.directory
        if not in_dir or not os.path.exists(in_dir):
            slicer.util.errorDisplay("Please select a valid input directory (Import DICOM folder).")
            return
        if not out_dir or not os.path.exists(out_dir):
            slicer.util.errorDisplay("Please select a valid output directory (Import DICOM folder).")
            return
        if not hdr_dir or not os.path.exists(hdr_dir):
            slicer.util.errorDisplay("Please select a valid headers directory (Import DICOM folder).")
            return

        model_path = (self.ui.autoAnonModelPathButton.currentPath or MODEL_PATH).strip()
        
        # Get selected value from combo box instead of text input
        device = self.ui.autoAnonDeviceComboBox.currentText or "cpu"
        
        overview_dir = (self.ui.autoAnonOverviewDirButton.directory or "").strip()
        ground_truth_dir = (self.ui.autoAnonGroundTruthDirButton.directory or "").strip()

        skip_single = self.ui.skipSingleframeCheckBox.checked
        hash_pid = self.ui.hashPatientIdCheckBox.checked
        preserve_dirs = self.ui.preserveDirectoryStructureCheckBox.checked
        resume = self.ui.continueProgressCheckBox.checked

        self.ui.statusLabel.text = "Running auto‑anonymize..."
        slicer.app.processEvents()

        try:
            result = self.logic.batch_auto_anonymize(
                input_folder=in_dir,
                output_folder=out_dir,
                headers_folder=hdr_dir,
                model_path=model_path,
                device=device,
                preserve_directory_structure=preserve_dirs,
                resume_anonymization=resume,
                skip_single_frame=skip_single,
                hash_patient_id=hash_pid,
                overview_dir=overview_dir,
                ground_truth_dir=ground_truth_dir,
                metrics_csv_path=os.path.join(overview_dir, f"{os.path.basename(in_dir)}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
            )
            self.ui.statusLabel.text = result['status']
        except Exception as e:
            logging.error(f"Auto‑anonymize failed: {e} {traceback.format_exc()}")
            slicer.util.errorDisplay(str(e))

    def onNextButton(self) -> None:
        logging.info("Next button clicked")
        threePointFanModeEnabled = self.ui.threePointFanCheckBox.checked
        continueProgress = self.ui.continueProgressCheckBox.checked

        # If continue progress is checked and nextDicomDfIndex is None, there is nothing more to load
        if self.logic.dicom_manager.next_index is None and continueProgress:
            self.ui.statusLabel.text = "All files from input folder have been processed to output folder. No more files to load."
            return

        # Remove observers for the mask markups node, because loading a new series will reset the scene and create a new markups node

        maskMarkupsNode = self._parameterNode.maskMarkups
        if maskMarkupsNode:
            self.removeObserver(maskMarkupsNode, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onPointModified)

        # Load the next series

        dialog = self.createWaitDialog("Loading series", "Please wait until the DICOM file is loaded...")
        currentDicomDfIndex = None
        try:
            outputDirectory = self.ui.outputDirectoryButton.directory
            currentDicomDfIndex = self.logic.loadNextSequence(outputDirectory, continueProgress)
            logging.info(f"Loaded series {currentDicomDfIndex}")
            if currentDicomDfIndex is None:
                statusText = "No more series to load"
                self.ui.statusLabel.text = statusText
                dialog.close()
            else:
                self.ui.progressBar.value = currentDicomDfIndex + 1
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

        if currentDicomDfIndex is not None and self.logic.dicom_manager.dicom_df is not None:
            current_dicom_record = self.logic.dicom_manager.dicom_df.iloc[currentDicomDfIndex]

            patientID = current_dicom_record.DICOMDataset.PatientID if current_dicom_record is not None else "N/A"
            if patientID:
                self.ui.patientIdLabel.text = patientID
            else:
                logging.error("Patient ID is missing")
                self.ui.patientIdLabel.text = 'None'

            instanceUID = current_dicom_record.DICOMDataset.SOPInstanceUID if current_dicom_record is not None else "N/A"
            if instanceUID is None:
                logging.error("Instance UID is missing")
                self.ui.sopInstanceUidLabel.text = 'None'
            else:
                self.ui.sopInstanceUidLabel.text = instanceUID

            statusText = f"Instance {instanceUID} loaded from file:\n"

            # Get the file path from the dataframe

            filepath = current_dicom_record['InputPath']
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
        else:
            self.ui.statusLabel.text = "No DICOM file loaded"

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
        autoMaskSuccessful = False
        if self.ui.autoMaskCheckBox.checked:
            # Get the mask control points
            maskMarkupsNode.RemoveAllControlPoints()
            coords_IJK = self.logic.getAutoMask()
            if coords_IJK is None:
                logging.error("Auto mask not found, resetting three-point fan mode from settings")
                # Three-point fan mask setting
                settings = slicer.app.settings()
                threePointStr = settings.value(self.THREE_POINT_FAN_SETTING)
                if threePointStr and threePointStr.lower() == "true":
                    self.ui.threePointFanCheckBox.checked = True
                else:
                    self.ui.threePointFanCheckBox.checked = False
            else:
                autoMaskSuccessful = True
                num_points = coords_IJK.shape[0]
                if num_points == 3:
                    self.ui.threePointFanCheckBox.checked = True
                else:
                    self.ui.threePointFanCheckBox.checked = False

            # Try to apply the automatic mask markups
            currentVolumeNode = self.logic.getCurrentProxyNode()
            if autoMaskSuccessful == True and currentVolumeNode is not None:
                ijkToRas = vtk.vtkMatrix4x4()
                currentVolumeNode.GetIJKToRASMatrix(ijkToRas)

                coords_RAS = np.zeros((num_points, 4))
                for i in range(num_points):
                    point_IJK = np.array([coords_IJK[i, 0], coords_IJK[i, 1], 0, 1])
                    # convert to IJK
                    coords_RAS[i, :] = ijkToRas.MultiplyPoint(point_IJK)

                for i in range(num_points):
                    coord = coords_RAS[i, :]
                    logging.debug(f"Adding control point {coord}")
                    maskMarkupsNode.AddControlPoint(coord[0], coord[1], coord[2])

                # Update the status
                self._parameterNode.status = AnonymizerStatus.LANDMARKS_PLACED
                self.ui.defineMaskButton.checked = False
            else:
                logging.error("Ultrasound volume node not found")
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

    def onExportScanButton(self) -> bool:
        """
        Callback function for the export scan button.
        """
        logging.info('Export scan button pressed')
        preserve_directory_structure = self.ui.preserveDirectoryStructureCheckBox.checked
        threePointFanModeEnabled = self.ui.threePointFanCheckBox.checked
        currentSequenceBrowser = self._parameterNode.ultrasoundSequenceBrowser
        if currentSequenceBrowser is None:
            self.ui.statusLabel.text = "Load a DICOM sequence before trying to export"
            logging.info("No sequence browser found, nothing exported.")
            return False

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
                return False

        # Mask images to erase the unwanted parts
        self.logic.maskSequence(three_point=threePointFanModeEnabled)

        # Set up output directory and filename

        hashPatientId = self.ui.hashPatientIdCheckBox.checked

        # If hashPatientId is not checked, confirm with the user that they really want to proceed.

        if not hashPatientId:
            if not slicer.util.confirmOkCancelDisplay("Patient name will not be masked. Do you want to proceed?"):
                return False

        outputDirectory = self.ui.outputDirectoryButton.directory
        headersDirectory = self.ui.headersDirectoryButton.directory

        current_dicom_record = self.logic.dicom_manager.dicom_df.iloc[self.logic.dicom_manager.current_index]
        filename, patient_uid, _ = self.logic.dicom_manager.generate_filename_from_dicom_dataset(current_dicom_record.DICOMDataset, hashPatientId)

        dialog = self.createWaitDialog("Exporting scan", "Please wait until the scan is exported...")

        if hashPatientId:
            patient_name_prefix = self.ui.namePrefixLineEdit.text

            # if patient_name_prefix is empty, alert the user
            if not patient_name_prefix:
                if not slicer.util.confirmOkCancelDisplay("A `Patient Name Prefix` is required when Patient ID hashing is enabled. Do you want to proceed without a prefix?"):
                    dialog.close()
                    return False

            new_patient_name = f"{patient_name_prefix}_{patient_uid}"
            new_patient_id = patient_uid
        else:
            new_patient_name = None
            new_patient_id = None

        # Save current mask to cache before exporting
        if self.ui.enableMaskCacheCheckBox.checked:
            self.logic.saveCurrentMaskToCache()

        # Export the scan
        dicomFilePath, jsonFilePath, dicomHeaderFilePath = self.logic.exportDicom(
            output_directory=outputDirectory,
            output_filename=filename,
            headers_directory=headersDirectory,
            labels = annotationLabels,
            new_patient_name = new_patient_name,
            new_patient_id = new_patient_id,
            preserve_directory_structure = preserve_directory_structure)

        # Restore selected item number in sequence browser
        currentSequenceBrowser.SetSelectedItemNumber(selectedItemNumber)

        # Display file paths in the status label
        statusText = f"Export successful!\nDICOM: {dicomFilePath}\nLabels: {jsonFilePath}"
        if dicomHeaderFilePath:
            statusText += f"\nHeader: {dicomHeaderFilePath}"

        self.ui.statusLabel.text = statusText

        # Close the modal dialog
        dialog.close()

        return True

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

    def connectKeyboardShortcuts(self):
        """Connect shortcut keys to their corresponding actions."""
        self.shortcutM.connect('activated()', lambda: self.ui.defineMaskButton.toggle())
        self.shortcutP.connect('activated()', self.onPreviousButton)
        self.shortcutN.connect('activated()', self.onNextButton)
        self.shortcutC.connect('activated()', lambda: self.ui.threePointFanCheckBox.toggle())
        self.shortcutE.connect('activated()', self.onExportScanButton)
        self.shortcutA.connect('activated()', self.onExportAndNextButton)

    def disconnectKeyboardShortcuts(self):
        """Disconnect shortcut keys when leaving the module to avoid unwanted interactions."""
        try:
            self.shortcutM.activated.disconnect()
            self.shortcutP.activated.disconnect()
            self.shortcutN.activated.disconnect()
            self.shortcutC.activated.disconnect()
            self.shortcutE.activated.disconnect()
            self.shortcutA.activated.disconnect()
        except Exception:
            # If shortcuts were not connected yet, ignore
            pass

    def onExportAndNextButton(self):
        """Helper slot to export the current scan and immediately load the next one (shortcut 'A')."""
        if self.onExportScanButton():
            self.onNextButton()

    def _on_auto_anon_enable_toggled(self, enabled: bool):
        settings = slicer.app.settings()
        settings.setValue(self.AUTO_ANON_ENABLE_SETTING, str(enabled))
        self._apply_auto_anon_visibility(enabled)

    def _apply_auto_anon_visibility(self, enabled: bool):
        if hasattr(self.ui, "autoAnonGroupBox"):
            self.ui.autoAnonGroupBox.setVisible(bool(enabled))


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
        VTKObservationMixin.__init__(self)

        self.dicom_manager = DicomFileManager()
        self.showAutoOverlay = False
        self._autoMaskRGB = None     # 1×H×W×3  uint8, red
        self._manualMaskRGB = None   # 1×H×W×3  uint8, green
        self._parameterNode = self._getOrCreateParameterNode()
        self.transducerMaskCache = {}   # TransducerModel -> mask volume node
        self.currentTransducerModel = 'unknown'
        self._temp_directories = []

    def _getOrCreateParameterNode(self):
        if not hasattr(self, "_parameterNode"):
            self._parameterNode = AnonymizeUltrasoundParameterNode(super().getParameterNode())
        return self._parameterNode

    def getParameterNode(self):
        return self._parameterNode

    def getNumberOfInstances(self):
        """
        Return the number of instances in the current DICOM dataframe.
        """
        return self.dicom_manager.get_number_of_instances()

    def loadPreviousSequence(self, outputDirectory, continueProgress=True, preserve_directory_structure=True):
        if self.dicom_manager.dicom_df is None:
            return None

        if self.dicom_manager.next_index <= 1:
            return None
        else:
            self.dicom_manager.next_index -= 2
            return self.loadNextSequence(outputDirectory=outputDirectory, continueProgress=continueProgress, preserve_directory_structure=preserve_directory_structure)

    def _setup_temp_directory(self) -> str:
        """Setup temporary directory for DICOM files"""
        temp_dir = os.path.join(slicer.app.temporaryPath, 'UltrasoundModules')
        os.makedirs(temp_dir, exist_ok=True)

        # Clean existing files with error handling
        try:
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except OSError as e:
            logging.warning(f"Failed to clean temp directory {temp_dir}: {e}")

        self._temp_directories.append(temp_dir)
        return temp_dir

    def _load_dicom_from_temp(self, temp_dir: str) -> List[str]:
        """Load DICOM files using Slicer's DICOM utilities

        This method creates a temporary DICOM database and loads DICOM files
        from the specified directory into Slicer. It returns a list of node IDs
        for the loaded DICOM files.

        Args:
            temp_dir: Path to the temporary directory containing DICOM files

        Returns:
            List[str]: List of node IDs for the loaded DICOM files
        """
        loaded_node_ids = []
        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(temp_dir, db)
            patient_uids = db.patients()
            for patient_uid in patient_uids:
                loaded_node_ids.extend(DICOMUtils.loadPatientByUID(patient_uid))
        return loaded_node_ids

    def _find_sequence_browser(self, loaded_node_ids: List[str]):
        """Find sequence browser node from loaded nodes"""
        for node_id in loaded_node_ids:
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node and node.IsA("vtkMRMLSequenceBrowserNode"):
                return node
        return None

    def load_sequence(self, parameter_node, output_directory: Optional[str] = None,
                     continue_progress: bool = False, preserve_directory_structure: bool = True):
        """
        Load next DICOM sequence from the dataframe.

        This method loads the next DICOM file in the sequence, creates a temporary directory,
        copies the DICOM file there, and loads it using Slicer's DICOM utilities. It then
        finds the sequence browser node and updates the parameter node.

        Args:
            parameter_node: Parameter node to store the loaded sequence browser
            output_directory: Optional output directory to check for existing files
            continue_progress: If True, skip files that already exist in output directory
            preserve_directory_structure: If True, the output filepath will be the same as the relative path.
        Returns:
            tuple: (current_dicom_df_index, sequence_browser) where:
                - current_dicom_df_index: The index of the current DICOM file in the dataframe
                - sequence_browser: The loaded sequence browser node
                Returns (None, None) if no more sequences available or loading fails.
        """
        if self.dicom_manager.dicom_df is None or self.dicom_manager.next_index is None or self.dicom_manager.next_index >= len(self.dicom_manager.dicom_df):
            return None, None

        next_row = self.dicom_manager.dicom_df.iloc[self.dicom_manager.next_index]
        temp_dicom_dir = self._setup_temp_directory()

        # Copy DICOM file to temporary folder
        shutil.copy(next_row['InputPath'], temp_dicom_dir)

        # Load DICOM using Slicer's DICOM utilities
        loaded_node_ids = self._load_dicom_from_temp(temp_dicom_dir)
        logging.info(f"Loaded DICOM nodes: {loaded_node_ids}")

        sequence_browser = self._find_sequence_browser(loaded_node_ids)

        if sequence_browser:
            parameter_node.ultrasoundSequenceBrowser = sequence_browser
        else:
            logging.error(f"Failed to find sequence browser node in {loaded_node_ids}")
            return None, None

        # Increment index
        next_index_val = self.dicom_manager.increment_dicom_index(output_directory, continue_progress, preserve_directory_structure)

        # Cleanup
        self._cleanup_temp_directory(temp_dicom_dir)

        # Update current DICOM dataframe index
        self.dicom_manager.current_index = self.dicom_manager.next_index - 1 if self.dicom_manager.next_index is not None and self.dicom_manager.next_index > 0 else 0

        if next_index_val or self.dicom_manager.next_index is not None:
            return self.dicom_manager.current_index, sequence_browser

        return None, None

    def _cleanup_temp_directory(self, temp_dir: str):
        """Cleanup temporary directory"""
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if temp_dir in self._temp_directories:
                self._temp_directories.remove(temp_dir)
        except Exception as e:
            logging.warning(f"Failed to cleanup temporary directory {temp_dir}: {e}")

    def loadNextSequence(self, outputDirectory, continueProgress=True, preserve_directory_structure=True):
        """
        Load next sequence in the list of DICOM files.
        Returns the index of the loaded sequence in the dataframe of DICOM files, or None if no more sequences are available.
        """
        self.resetScene()
        parameterNode = self.getParameterNode()

        current_index, sequence_browser = self.load_sequence(parameterNode, outputDirectory, continueProgress, preserve_directory_structure)

        # If no more sequences are available, return None
        if sequence_browser is None:
            return None

        # After loading the DICOM, try to find a cached mask for the transducer model
        # If found, apply it. If not, the user will need to define it manually.
        if self.dicom_manager.dicom_df is not None:
            current_dicom_record = self.dicom_manager.dicom_df.iloc[self.dicom_manager.current_index]
            transducerType = current_dicom_record.get("TransducerModel", "unknown")
            self.currentTransducerModel = self.dicom_manager.get_transducer_model(transducerType)
            cached_mask = self.getCachedMaskForTransducer(self.currentTransducerModel)

            if cached_mask:
                logging.info(f"Found cached mask for transducer {self.currentTransducerModel}")
                if self.applyCachedMask(cached_mask):
                    logging.info("Successfully applied cached mask")
                else:
                    logging.warning("Failed to apply cached mask, will need manual definition")

        # Make this sequence browser node the current one in the toolbar
        slicer.modules.sequences.setToolBarActiveBrowserNode(sequence_browser)

        # Get the proxy node of the master sequence node of the selected sequence browser node
        masterSequenceNode = sequence_browser.GetMasterSequenceNode()
        if masterSequenceNode is None:
            logging.error("Master sequence node of sequence browser node with ID " + sequence_browser.GetID() + " not found")
            return None

        proxyNode = sequence_browser.GetProxyNode(masterSequenceNode)
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

        return current_index

    def resetScene(self):
        """
        Reset the scene by clearing it and setting it up again.
        """
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
        self.dicom_manager.next_index = None
        self.incrementDicomDfIndex(input_folder, output_folder, skip_existing=True)
        return self.dicom_manager.next_index

    def incrementDicomDfIndex(self, input_folder=None, output_directory=None, skip_existing=False):
        """
        Increment the index of the DICOM dataframe. If skipExistingOutput is True, then skip the rows that have already been processed.

        :param skip_existing: If True, skip the rows that have already been processed.
        :param keep_folders: If True, keep the folder structure of the input DICOM files in the output directory.
        :return: None
        """
        if self.dicom_manager.dicom_df is None:
            return None

        listOfIndices = self.dicom_manager.dicom_df.index.tolist()
        listOfIndices.sort()

        if self.dicom_manager.next_index is None:
            nextIndexIndex = 0
        else:
            try:
                nextIndexIndex = listOfIndices.index(self.dicom_manager.next_index)
                nextIndexIndex += 1
            except ValueError:
                nextIndexIndex = 0 # next_index is not in list, so start from beginning

        if skip_existing and output_directory:
            while nextIndexIndex < len(listOfIndices):
                current_dicom_record = self.dicom_manager.dicom_df.iloc[nextIndexIndex]
                output_path = output_directory
                output_filename = current_dicom_record['AnonFilename']
                output_fullpath = os.path.join(output_path, output_filename)

                # Make sure output_fullpath has a .dcm extension
                if not output_fullpath.endswith('.dcm'):
                    output_fullpath += '.dcm'

                if not os.path.exists(output_fullpath):
                    break

                nextIndexIndex += 1

        if nextIndexIndex < len(listOfIndices):
            self.dicom_manager.next_index = listOfIndices[nextIndexIndex]
            logging.info(f"Next DICOM dataframe index: {self.dicom_manager.next_index}")
        else:
            self.dicom_manager.next_index = None
            self.widget.set_processing_mode(False)
            slicer.util.mainWindow().statusBar().showMessage("No more DICOM files to process", 3000)

        return self.dicom_manager.next_index

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

    def sequence_to_numpy(self, masterSequenceNode) -> np.ndarray:
        """
        Convert all frames in the sequence to a numpy array in the format of (N, C, H, W).
        :param masterSequenceNode: the master sequence node
        :param make_copy: whether to make a copy of the array
        :return: numpy array (N, C, H, W)
        """
        n = masterSequenceNode.GetNumberOfDataNodes()
        if n == 0:
            return np.empty((0,))

        firstNode = masterSequenceNode.GetNthDataNode(0)

        # This method assumes the sequence stores volume nodes
        if not (firstNode.IsA('vtkMRMLScalarVolumeNode') or firstNode.IsA('vtkMRMLVectorVolumeNode')):
            raise TypeError(f'Unsupported node type in sequence: {firstNode.GetClassName()}')

        a0 = slicer.util.arrayFromVolume(firstNode).squeeze(axis=0) # (N, H, W, C) -> (H, W, C)
        out = np.empty((n,) + a0.shape, dtype=a0.dtype)
        out[0] = a0
        for i in range(1, n):
            out[i] = slicer.util.arrayFromVolume(masterSequenceNode.GetNthDataNode(i))

        return out.transpose(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

    def getAutoMask(self):
        start = time.time()

        # 1) Model
        try:
            model, device = self._ensure_model(MODEL_PATH)
        except Exception as e:
            logging.error(f"Model prepare failed: {e}")
            return None

        # 2) Frames from current sequence (NCHW)
        frames = self._get_frames_from_current_sequence()
        if frames.size == 0:
            logging.error("No frames in current sequence")
            return None

        # 3) Inference + denorm + optional 3-point merge
        coords4 = self._infer_corners_px(frames, model, device)
        coordsN = self._merge_top_corners_if_close(coords4, threshold_px=15.0)

        logging.info(f"Auto mask infer in {time.time() - start:.3f}s")
        return coordsN

    def downloadAndPrepareModel(self):
        """ Download the AI model and prepare it for inference """
        if not os.path.exists(MODEL_PATH):
            logging.info(f"The AI model does not exist. Starting download...")
            dialog = AnonymizeUltrasoundWidget.createWaitDialog(self, "Downloading AI Model", "The AI model does not exist. Downloading...")
            success = self.download_model(MODEL_URL, MODEL_PATH)
            dialog.close()
            if not success:
                return None, None

        # Check if the model loaded successfully
        try:
            device = get_device()
            model = load_model(MODEL_PATH, device)
        except Exception as e:
            logging.error(f"Failed to load the model: {e}")
            logging.error("Automatic mode is disabled. Please define the mask manually.")
            return None, None

        return model, device

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
            return False

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
            logging.debug(f"Control point {i}: {markupPoint} -> {controlPoints_ijk[i]}")

        if count == 3 and three_point:
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
            # Extract first 4 valid points
            pts = [controlPoints_ijk[i][:3] for i in range(4)]

            # Sort by Y (row), then X (col)
            pts_sorted_by_yx = sorted(pts, key=lambda pt: (pt[1], pt[0]))

            # Split top and bottom by Y position
            top_pts = sorted(pts_sorted_by_yx[:2], key=lambda pt: pt[0])     # left to right
            bottom_pts = sorted(pts_sorted_by_yx[2:], key=lambda pt: pt[0])  # left to right

            topLeft, topRight = top_pts
            bottomLeft, bottomRight = bottom_pts

            if np.array_equal(topLeft, np.zeros(3)) or np.array_equal(topRight, np.zeros(3)) or \
                    np.array_equal(bottomLeft, np.zeros(3)) or np.array_equal(bottomRight, np.zeros(3)):
                logging.debug("Could not determine mask corners")
                return("Mask points should be in a fan or rectangular shape with two points in the top and two points in the bottom."
                    "\nMove points to try again.")

        imageArray = slicer.util.arrayFromVolume(currentVolumeNode)  # (z, y, x, channels)

        # Create mask based on mode
        if count == 3 and three_point:
            # Always create fan mask for 3-point mode
            assert topRight is None, "topRight should be None in 3-point mode"
            logging.debug(f"Creating fan mask for 3-point mode {topLeft}, {None}, {bottomLeft}, {bottomRight}")
            mask_array = self.createFanMask(imageArray, topLeft, None, bottomLeft, bottomRight, value=1, three_point=True)
        else:
            # Detect if the mask is a fan or a rectangle for 4-point mode
            maskHeight = abs(topLeft[1] - bottomLeft[1])
            tolerancePixels = round(0.1 * maskHeight)  #todo: Make this tolerance value a setting
            if abs(topLeft[0] - bottomLeft[0]) < tolerancePixels and abs(topRight[0] - bottomRight[0]) < tolerancePixels:
                # Mask is a rectangle
                logging.debug(f"Creating rectangle mask for 3-point mode {topLeft}, {topRight}, {bottomLeft}, {bottomRight}")
                mask_array = self.createRectangleMask(imageArray, topLeft, topRight, bottomLeft, bottomRight)
            else:
                # 4-point fan
                logging.debug(f"Creating fan mask for 4-point mode {topLeft}, {topRight}, {bottomLeft}, {bottomRight}")
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
        """
        Apply mask to all frames in the ultrasound sequence.

        This method updates the mask volume based on the current mask markups,
        then applies the mask to every frame in the ultrasound sequence by
        multiplying each pixel with the corresponding mask value.

        :param three_point: If True, use three-point fan mask mode. If False, use four-point mask mode.
        :type three_point: bool
        """
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

    def saveDicomFile(self, dicomFilePath, new_patient_name='', new_patient_id='', labels=None):
        """
        Save the current ultrasound sequence as an anonymized DICOM file.
        """
        parameterNode = self.getParameterNode()

        # Collect image data from sequence browser as a numpy array
        image_array = self._collect_image_data_from_sequence(parameterNode)

        self.dicom_manager.save_anonymized_dicom(
            image_array=image_array,
            output_path=dicomFilePath,
            new_patient_name=new_patient_name,
            new_patient_id=new_patient_id,
            labels=labels
        )

    def _collect_image_data_from_sequence(self, parameterNode) -> np.ndarray:
        """Collect all image frames from the sequence browser into a numpy array."""
        currentSequenceBrowser = parameterNode.ultrasoundSequenceBrowser
        masterSequenceNode = currentSequenceBrowser.GetMasterSequenceNode()
        proxyNode = self.getCurrentProxyNode()

        proxyNodeArray = slicer.util.arrayFromVolume(proxyNode)

        imageArray = np.zeros((
            masterSequenceNode.GetNumberOfDataNodes(),
            proxyNodeArray.shape[1],
            proxyNodeArray.shape[2],
            proxyNodeArray.shape[3]
        ), dtype=np.int8)

        for index in range(masterSequenceNode.GetNumberOfDataNodes()):
            currentSequenceBrowser.SetSelectedItemNumber(index)
            currentVolumeNode = masterSequenceNode.GetNthDataNode(index)
            currentVolumeArray = slicer.util.arrayFromVolume(currentVolumeNode)
            imageArray[index, :, :, :] = currentVolumeArray

        return imageArray

    def exportDicom(self,
                    output_directory,
                    output_filename = None,
                    headers_directory = None,
                    labels = None,
                    new_patient_name = "",
                    new_patient_id = "",
                    preserve_directory_structure = True):
        """
        Export the current ultrasound sequence as an anonymized DICOM file with optional annotation labels.

        Args:
            outputDirectory (str): Directory path where the DICOM file will be saved
            outputFilename (str, optional): Custom filename for the output DICOM. If None,
                generates filename from DICOM dataset. Defaults to None.
            headersDirectory (str, optional): Directory path to save original DICOM headers
                as JSON files. If None, headers are not saved. Defaults to None.
            labels (list, optional): List of annotation labels to include in the output
                JSON file. Defaults to None.
            new_patient_name (str, optional): Anonymized patient name to use in the output
                DICOM. If None, uses original name. Defaults to "".
            new_patient_id (str, optional): Anonymized patient ID to use in the output
                DICOM. If None, uses original ID. Defaults to "".
            preserve_directory_structure (bool, optional): Whether to maintain the original
                directory structure in the output path. Defaults to True.

        Returns:
            tuple: (dicomFilePath, jsonFilePath, dicomHeaderFilePath) - Paths to the saved
                DICOM file, annotations JSON file, and DICOM header JSON file respectively.
                Returns (None, None, None) if export fails.

        Note:
            - Collects image data from the sequence browser and saves as anonymized DICOM
            - Creates output directories if they don't exist
            - Saves sequence information and annotations to JSON file
            - Optionally saves original DICOM headers with partial anonymization
        """
        # Record sequence information to a dictionary. This will be saved in the annotations JSON file.
        current_dicom_record = self.dicom_manager.dicom_df.iloc[self.dicom_manager.current_index]
        SOPInstanceUID = current_dicom_record.DICOMDataset.SOPInstanceUID if current_dicom_record is not None else "None"
        if SOPInstanceUID is None:
            SOPInstanceUID = "None"

        sequence_info = {
            'SOPInstanceUID': SOPInstanceUID,
            'GrayscaleConversion': False
        }

        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Save DICOM image file
        if output_filename is None:
            output_filename, _, _ = self.dicom_manager.generate_filename_from_dicom_dataset(current_dicom_record.DICOMDataset)

        if output_filename is None or output_filename == "":
            return None, None, None

        # Generate complete output path with directory structure consideration
        dicom_file_path = self.dicom_manager.generate_output_filepath(
            output_directory, current_dicom_record.OutputPath, preserve_directory_structure)

        self.saveDicomFile(dicom_file_path, new_patient_name, new_patient_id, labels)

        # Save original DICOM header to a json file. This may not be completely anonymized.
        dicom_header_file_path = self.dicom_manager.save_anonymized_dicom_header(current_dicom_record, output_filename, headers_directory)

        # Add mask parameters to sequenceInfo
        for key, value in self.maskParameters.items():
            sequence_info[key] = value

        # Add annotation labels to sequenceInfo
        if labels is not None:
            sequence_info["AnnotationLabels"] = labels

        # Save sequenceInfo to a file
        sequence_info_filename = dicom_file_path.replace(".dcm", ".json")
        sequence_info_file_path = os.path.join(output_directory, sequence_info_filename)

        with open(sequence_info_file_path, 'w') as outfile:
            json.dump(sequence_info, outfile)

        return dicom_file_path, sequence_info_file_path, dicom_header_file_path

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

    def batch_auto_anonymize(
        self,
        input_folder: str,
        output_folder: str,
        headers_folder: str,
        model_path: str = MODEL_PATH,
        device: str = "",
        preserve_directory_structure: bool = True,
        resume_anonymization: bool = False,
        skip_single_frame: bool = False,
        hash_patient_id: bool = True,
        overview_dir: str = "",
        ground_truth_dir: str = "",
        metrics_csv_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        In-process batch anonymization using shared logic:
        - Scans DICOMs into dicom_df
        - Optionally resumes at first missing output
        - For each row: read frames -> AI corners -> mask -> save anonymized DICOM + header + JSON
        - Optionally, compare anonymized DICOMs with ground truth masks and save metrics to CSV
        - Optionally, save overview images of original, mask, and anonymized DICOMs
        - Optionally, save metrics to CSV and PDF
        """
        logging.info(f"Starting batch anonymization...")
        batch_start_time = time.time()

        # Scan directory (filenames reflect hash_patient_id)
        num_files = self.dicom_manager.scan_directory(input_folder, skip_single_frame, hash_patient_id)
        logging.info(f"Found {num_files} DICOM files in input folder")

        # Save keys.csv (drop non-serializable column)
        if self.dicom_manager.dicom_df is not None and headers_folder:
            try:
                df = self.dicom_manager.dicom_df.drop(columns=['DICOMDataset'], inplace=False)
                os.makedirs(headers_folder, exist_ok=True)
                df.to_csv(os.path.join(headers_folder, "keys.csv"), index=False)
            except Exception as e:
                raise Exception(f"Failed to save keys.csv: {e}") from e

        # Determine starting index on resume
        start_index = 0
        if resume_anonymization:
            progressed = self.dicom_manager.update_progress_from_output(output_folder, preserve_directory_structure)
            if progressed is None:
                logging.info("All files already processed; nothing to do.")
                return {"status": "All files already processed; nothing to do."}
            start_index = self.dicom_manager.next_index or 0

        try:
            model, device = self._ensure_model(model_path, device)  # device is the user hint
            logging.info(f"Model loaded on {device}")
        except Exception as e:
            raise Exception(f"Failed to load model: {e}") from e

        # Optional overview
        make_overview = bool(overview_dir)
        if make_overview:
            os.makedirs(overview_dir, exist_ok=True)

        # Metrics CSV
        metrics_fieldnames = [
            "dicom_output_path","dicom_filename","ground_truth_config_path","predicted_config_json",
            "dice_mean","iou_mean","pixel_accuracy_mean","precision_mean","recall_mean",
            "f1_mean","sensitivity_mean","specificity_mean"
        ]
        if not metrics_csv_path:
            metrics_csv_path = os.path.join(overview_dir, "metrics.csv")
        os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
        metrics_file = open(metrics_csv_path, "w", newline="")
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=metrics_fieldnames)
        metrics_writer.writeheader()

        # Index ground truth JSONs recursively for fast lookup by filename
        gt_index = {}
        if ground_truth_dir:
            try:
                for root, _, files in os.walk(ground_truth_dir):
                    for f in files:
                        if f.lower().endswith(".json"):
                            full = os.path.join(root, f)
                            if f not in gt_index:
                                gt_index[f] = full
                            else:
                                logging.warning(f"Duplicate ground truth filename '{f}' found. "
                                                f"Keeping '{gt_index[f]}', ignoring '{full}'.")
            except Exception as e:
                logging.warning(f"Failed to index ground truth directory '{ground_truth_dir}': {e}")

        # Progress dialog
        total = len(self.dicom_manager.dicom_df) if self.dicom_manager.dicom_df is not None else 0
        progress = qt.QProgressDialog("Auto-anonymizing...", "Cancel", 0, total, slicer.util.mainWindow())
        progress.setWindowModality(qt.Qt.WindowModal)
        progress.show()

        success = failed = skipped = 0
        error_messages = []
        overview_manifest = []
        overview_pdf_path = ""

        try:
            for idx in range(start_index, total):
                if progress.wasCanceled:
                    logging.info("Batch auto-anonymize canceled by user.")
                    break
                progress.setValue(idx)
                slicer.app.processEvents()

                self.dicom_manager.current_index = idx
                row = self.dicom_manager.dicom_df.iloc[idx]

                # Build output path and skip if resuming and exists
                final_output_path = self.dicom_manager.generate_output_filepath(
                    output_folder, row.OutputPath, preserve_directory_structure
                )

                # Skip if resuming and output file exists
                if resume_anonymization and os.path.exists(final_output_path):
                    logging.info(f"Skipping {row.InputPath} because it already exists")
                    skipped += 1
                    continue

                try:
                    # Read frames (N,C,H,W) and NHWC view for masking/saving
                    original_nchw = self.dicom_manager.read_frames_from_dicom(row.InputPath)   # (N,C,H,W)
                    H, W = original_nchw.shape[-2], original_nchw.shape[-1]
                    image_nhwc = np.transpose(original_nchw, (0, 2, 3, 1))  # (N,H,W, C)

                    # Inference + 3/4-point merge
                    coords4 = self._infer_corners_px(original_nchw, model, device)
                    coordsN = self._merge_top_corners_if_close(coords4, threshold_px=15.0)

                    # Corners → mask/config (0/1 mask)
                    predicted_corners = self._corners_array_to_dict(coordsN)
                    logging.debug(f"Coords RAS: {predicted_corners}")

                    curvilinear_mask, mask_config = self._mask_from_corners((H, W), predicted_corners)
                    logging.debug(f"Curvilinear mask: {curvilinear_mask}")
                    logging.debug(f"Mask config: {mask_config}")

                    # Apply mask across all frames/channels
                    masked_image_array = self._apply_mask_nhwc(image_nhwc, curvilinear_mask)

                    # Derive anonymized name/id from filename (consistent with CLI)
                    anon_filename = row.AnonFilename
                    new_patient_name = os.path.splitext(anon_filename)[0]
                    new_patient_id = anon_filename.split('_')[0]

                    # Save anonymized DICOM
                    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
                    self.dicom_manager.save_anonymized_dicom(
                        image_array=masked_image_array,
                        output_path=final_output_path,
                        new_patient_name=new_patient_name,
                        new_patient_id=new_patient_id,
                        labels=None # Labels not supported yet.
                    )

                    # Save header JSON
                    try:
                        self.dicom_manager.save_anonymized_dicom_header(
                            current_dicom_record=row,
                            output_filename=anon_filename,
                            headers_directory=headers_folder
                        )
                    except Exception as e:
                        logging.warning(f"Failed to save header for {row.InputPath}: {e}")
                        raise Exception(f"Failed to save header for {row.InputPath}: {e}") from e

                    # Compute metrics
                    gt_config_path = ""
                    metrics_payload = {}
                    if ground_truth_dir:
                        gt_basename = f"{os.path.splitext(anon_filename)[0]}.json"
                        gt_config_path = gt_index.get(gt_basename, "")
                        if gt_config_path:
                            try:
                                gt_mask_config = load_mask_config(gt_config_path)
                                metrics_payload = compare_masks(gt_mask_config, mask_config, (H, W))
                            except Exception as e:
                                logging.warning(f"Failed to compute metrics for {gt_config_path}: {e}")
                                raise Exception(f"Failed to compute metrics for {gt_config_path}: {e}") from e
                        else:
                            logging.warning(f"Ground truth config not found for {anon_filename} (looked for '{gt_basename}') within '{ground_truth_dir}'")

                    row_out = {
                        "dicom_output_path": final_output_path,
                        "dicom_filename": anon_filename,
                        "ground_truth_config_path": gt_config_path,
                        "predicted_config_json": json.dumps(mask_config) if mask_config is not None else "",
                        "dice_mean": metrics_payload.get("dice_mean", ""),
                        "iou_mean": metrics_payload.get("iou_mean", ""),
                        "pixel_accuracy_mean": metrics_payload.get("pixel_accuracy_mean", ""),
                        "precision_mean": metrics_payload.get("precision_mean", ""),
                        "recall_mean": metrics_payload.get("recall_mean", ""),
                        "f1_mean": metrics_payload.get("f1_mean", ""),
                        "sensitivity_mean": metrics_payload.get("sensitivity_mean", ""),
                        "specificity_mean": metrics_payload.get("specificity_mean", ""),
                    }
                    logging.info(f"metrics: {row_out}")
                    metrics_writer.writerow(row_out)

                    # Save sequence JSON (mask config and SOPInstanceUID)
                    try:
                        sequence_info = {
                            'SOPInstanceUID': getattr(row.DICOMDataset, 'SOPInstanceUID', 'None') or 'None',
                            'GrayscaleConversion': False
                        }
                        if mask_config is not None:
                            sequence_info['MaskConfig'] = mask_config
                        json_path = final_output_path.replace(".dcm", ".json")
                        with open(json_path, 'w') as outfile:
                            json.dump(sequence_info, outfile, indent=2)
                    except Exception as e:
                        logging.warning(f"Failed to save sequence info for {final_output_path}: {e}")
                        raise Exception(f"Failed to save sequence info for {final_output_path}: {e}") from e

                    # Overview with metrics if Ground Truth is provided
                    if make_overview and masked_image_array.shape[0] > 0:
                        try:
                            fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
                            fig.patch.set_facecolor('white')
                            for ax in axes:
                                ax.set_facecolor('white')

                            axes[0].set_title('Original'); axes[1].set_title('Mask Outline'); axes[2].set_title('Anonymized')

                            orig_frame = image_nhwc[0].squeeze()
                            masked_frame = masked_image_array[0].squeeze()
                            mask2d = (curvilinear_mask > 0)

                            # Helper to apply white background outside the fan
                            def _with_white_bg(frame, mask2d):
                                if frame.ndim == 2:
                                    out = frame.copy()
                                    out[~mask2d] = 255
                                    return out, 'gray'
                                elif frame.ndim == 3 and frame.shape[2] == 3:
                                    out = frame.copy()
                                    m3 = np.repeat((~mask2d)[..., None], 3, axis=2)
                                    out[m3] = 255
                                    return out, None
                                else:
                                    # single-channel but kept as HxWx1
                                    out = frame[..., 0].copy()
                                    out[~mask2d] = 255
                                    return out, 'gray'

                            masked_disp, cmap2 = _with_white_bg(masked_frame, mask2d)

                            axes[0].imshow(orig_frame, cmap='gray'); axes[0].axis('off')
                            axes[1].imshow(orig_frame, cmap='gray')
                            axes[1].contour(curvilinear_mask, levels=[0.5], colors='lime', linewidths=1.0)
                            axes[1].axis('off')
                            axes[2].imshow(masked_disp, cmap=cmap2); axes[2].axis('off')

                            dice_txt = self._fmt_metric(metrics_payload.get("dice_mean", None))
                            iou_txt  = self._fmt_metric(metrics_payload.get("iou_mean", None))
                            metrics_str = f"Dice: {dice_txt}  IoU: {iou_txt}"
                            axes[1].text(
                                0.02, 0.98, metrics_str,
                                transform=axes[1].transAxes,
                                fontsize=12, color='yellow', ha='left', va='top',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5)
                            )

                            overview_filename = f"{os.path.splitext(anon_filename)[0]}.png"
                            overview_path = os.path.join(overview_dir, overview_filename)
                            plt.tight_layout()
                            plt.savefig(overview_path, dpi=300, bbox_inches='tight', pad_inches=0.05, facecolor='white')
                            plt.close(fig)

                            overview_manifest.append({
                                "path": overview_path,
                                "filename": anon_filename,
                                "dice": metrics_payload.get("dice_mean", None),
                                "iou": metrics_payload.get("iou_mean", None)
                            })
                        except Exception as e:
                            logging.warning(f"Failed to save overview for {row.InputPath}: {e}")

                    success += 1

                except Exception as e:
                    error_messages.append(f"Failed to process {row.InputPath}: {e} {traceback.format_exc()}")
                    logging.error(f"Failed to process {row.InputPath}: {e}")
                    failed += 1

            progress.setValue(total)
            slicer.app.processEvents()
        finally:
            progress.close()
            metrics_file.close()

        # Build high‑quality, low‑whitespace PDF (landscape; filename overlay, no table)
        if make_overview and overview_manifest:
            try:
                from matplotlib.backends.backend_pdf import PdfPages
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                overview_pdf_path = os.path.join(overview_dir, f"overview_{ts}.pdf")

                with PdfPages(overview_pdf_path) as pdf:
                    for item in overview_manifest:
                        img = plt.imread(item["path"])

                        # Full-page image; minimal margins; high DPI (LANDSCAPE)
                        fig = plt.figure(figsize=(11, 8.5), dpi=300)
                        ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))  # tuple to satisfy linter
                        ax.imshow(img, interpolation='nearest')
                        ax.axis('off')

                        # Filename overlay at top-left; no table
                        fig.text(
                            0.012, 0.992,
                            f"{item['filename']}",
                            ha='left', va='top', fontsize=10,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7)
                        )

                        pdf.savefig(fig)  # no bbox to keep full-page image
                        plt.close(fig)

                logging.info(f"Saved overview report: {overview_pdf_path}")
            except Exception as e:
                logging.warning(f"Failed to create overview PDF: {e}")

        batch_end_time = time.time()
        logging.info(f"Batch anonymization complete in {batch_end_time - batch_start_time:.4f} seconds")
        logging.info(f"Batch anonymization complete. Success: {success}, Failed: {failed}, Skipped existing: {skipped}")

        if failed > 0:
            status = f"Batch anonymization complete with errors. Success: {success}, Failed: {failed}, Skipped: {skipped}"
        else:
            status = f"Batch anonymization complete. Success: {success}, Failed: {failed}, Skipped existing: {skipped}"

        return {
            "status": status,
            "success": success, 
            "failed": failed,
            "skipped": skipped,
            "error_messages": error_messages
        }

    # Add metrics overlay (Dice / IoU)
    def _fmt_metric(self, v) -> str:
        try:
            return f"{float(v):.3f}"
        except Exception:
            return "N/A"

    def _ensure_model(self, model_path: str = MODEL_PATH, device_hint: str = ""):
        """
        Ensure a model is available and loaded on a device.
        Returns (model, device). Never shows a widget dialog (logic stays headless).
        """
        if not os.path.exists(model_path):
            logging.info("Model missing; downloading...")
            ok = self.download_model(MODEL_URL, model_path)
            if not ok:
                raise RuntimeError("Model download failed")
        device = get_device(device_hint)
        model = load_model(model_path, device)
        return model, device

    def _get_frames_from_current_sequence(self) -> np.ndarray:
        """
        Return NCHW array from current sequence (pauses rendering for speed).
        Shape: (N, C, H, W), dtype: uint8
        """
        slicer.app.pauseRender()
        try:
            pnode = self.getParameterNode()
            seq = pnode.ultrasoundSequenceBrowser.GetMasterSequenceNode()
            return self.sequence_to_numpy(seq)  # already returns NCHW
        finally:
            slicer.app.resumeRender()

    def _infer_corners_px(self, frames_nchw: np.ndarray, model, device: str) -> np.ndarray:
        """
        Predict corner coordinates (in pixels, x,y) at the original image resolution.
        Returns array of shape (4, 2) in order [UL, UR, LL, LR] BEFORE merging.
        """
        h, w = frames_nchw.shape[-2], frames_nchw.shape[-1]
        with torch.no_grad():
            t = preprocess_image(frames_nchw)            # (1,1,240,320)
            coords_norm = model(t.to(device)).cpu().numpy().reshape(4, 2)
        coords = coords_norm.copy()
        coords[:, 0] *= w  # x
        coords[:, 1] *= h  # y
        return coords

    def _merge_top_corners_if_close(self, coords: np.ndarray, threshold_px: float = 15.0) -> np.ndarray:
        """
        If UL and UR are very close, merge to 3-point fan [apex, LL, LR]; else return 4 corners.
        Input coords must be [UL, UR, LL, LR].
        """
        ul, ur, ll, lr = coords
        if np.linalg.norm(ul - ur) < threshold_px:
            merged_top = (ul + ur) / 2.0
            return np.vstack([merged_top, ll, lr])  # 3-point fan
        return coords  # 4 points

    def _corners_array_to_dict(self, coords: np.ndarray) -> dict:
        """
        Convert array corners to a dictionary format for compute_masks_and_configs.
        """
        return {
            "upper_left": tuple(coords[0]),
            "upper_right": tuple(coords[1]),
            "lower_left": tuple(coords[2]),
            "lower_right": tuple(coords[3]),
        }

    def _mask_from_corners(self, original_dims: tuple[int, int], corners_dict: dict):
        """
        Build curvilinear mask and config from corners (0/1 mask).
        """
        return compute_masks_and_configs(
            original_dims=original_dims,
            predicted_corners=corners_dict
        )

    def _apply_mask_nhwc(self, image_nhwc: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
        """
        Apply a 2D binary mask to a 4D image array to anonymize ultrasound images by masking
        out regions that contain patient information or other sensitive data.
        Multiply each frame/channel by the binary mask (0/1).
        :param image_nhwc: 4D image array (N, H, W, C)
        :param mask_2d: 2D binary mask (H, W)
        :return: 4D image array (N, H, W, C)
        """
        # Validate input shapes
        if len(image_nhwc.shape) != 4:
            raise ValueError(f"Expected 4D image array, got {len(image_nhwc.shape)}D")
        if len(mask_2d.shape) != 2:
            raise ValueError(f"Expected 2D mask array, got {len(mask_2d.shape)}D")
        if image_nhwc.shape[1:3] != mask_2d.shape:
            raise ValueError(f"Mask shape {mask_2d.shape} doesn't match image spatial dims {image_nhwc.shape[1:3]}")

        # Reshape mask to enable broadcasting: (H, W) -> (1, H, W, 1)
        mask_broadcast = mask_2d[np.newaxis, :, :, np.newaxis]
        return image_nhwc.copy() * mask_broadcast

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
