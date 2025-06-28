'''
Useful scripts for debugging

moduleWidget = slicer.modules.annotateultrasound.widgetRepresentation().self()
moduleLogic = moduleWidget.logic
moduleNode = moduleLogic.getParameterNode()

moduleLogic.updateOverlayVolume()
'''


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

#
# AnnotateUltrasound
#

class AnnotateUltrasound(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Annotate ultrasound"
        self.parent.categories = ["Ultrasound"]
        self.parent.dependencies = []
        self.parent.contributors = ["Tamas Ungi (Queen's University)"]

        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = f"""
This module facilitates the process of creating segmentations of B-lines and the pleura in series of B-mode lung ultrasound videos.<br><br>

See more information in <a href="https://github.com/SlicerUltrasound/SlicerUltrasound/blob/main/README.md">README</a> <a href="https://github.com/SlicerUltrasound/SlicerUltrasound/tree/main/AnnotateUltrasound">Source Code</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Tamas Ungi (Queen's University), with support from MLSC Bits to Bytes grant for Point of Care Ultrasound, and NIH grants R21EB034075 and R01EB035679.
"""

        # Additional initialization step after application startup is complete
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
# AnnotateUltrasoundParameterNode
#

@parameterNodeWrapper
class AnnotateUltrasoundParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """
    inputVolume: vtkMRMLScalarVolumeNode
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
    adjudicatorMode: bool = False

#
# AnnotateUltrasoundWidget
#

class AnnotateUltrasoundWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """
    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.notEnteredYet = True
        self._lastFrameIndex = -1

        self.updatingGUI = False

        # Flag to track if this is the first load of DICOM data
        self._isFirstDicomLoad = True

        # Flag to track if the user manually expanded the rater table
        self._userManuallySetRaterTableState = False
        self._lastUserManualCollapsedState = None  # Track the last state the user manually set

        # Flag to prevent rater table state changes during navigation
        self._isNavigating = False

        # Shortcuts
        self.shortcutW = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutW.setKey(qt.QKeySequence('W'))
        self.shortcutS = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutS.setKey(qt.QKeySequence('S'))
        self.shortcutO = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutO.setKey(qt.QKeySequence('O'))

        # Add shortcuts for removing lines
        self.shortcutE = qt.QShortcut(slicer.util.mainWindow())  # "E" for removing last pleura line
        self.shortcutE.setKey(qt.QKeySequence('E'))
        self.shortcutD = qt.QShortcut(slicer.util.mainWindow())  # "D" for removing last B-line
        self.shortcutD.setKey(qt.QKeySequence('D'))

        # shortcut for saving and loading next scan
        self.shortcutA = qt.QShortcut(slicer.util.mainWindow())  # "A" for save and load next scan
        self.shortcutA.setKey(qt.QKeySequence('A'))

        # shortcuts for adjudication actions
        self.shortcutV = qt.QShortcut(slicer.util.mainWindow())  # "V" for validate line
        self.shortcutV.setKey(qt.QKeySequence('V'))
        self.shortcutI = qt.QShortcut(slicer.util.mainWindow())  # "I" for invalidate line
        self.shortcutI.setKey(qt.QKeySequence('I'))
        self.shortcutU = qt.QShortcut(slicer.util.mainWindow())  # "U" for duplicate line
        self.shortcutU.setKey(qt.QKeySequence('U'))
        self.shortcutN = qt.QShortcut(slicer.util.mainWindow())  # "N" for next unadjudicated line
        self.shortcutN.setKey(qt.QKeySequence('N'))
        self.shortcutP = qt.QShortcut(slicer.util.mainWindow())  # "P" for previous unadjudicated line
        self.shortcutP.setKey(qt.QKeySequence('P'))
        # Add Shift+N and Shift+P for next/prev visible annotation
        self.shortcutShiftN = qt.QShortcut(slicer.util.mainWindow())  # "Shift+N" for next visible annotation
        self.shortcutShiftN.setKey(qt.QKeySequence('Shift+N'))
        self.shortcutShiftP = qt.QShortcut(slicer.util.mainWindow())  # "Shift+P" for previous visible annotation
        self.shortcutShiftP.setKey(qt.QKeySequence('Shift+P'))

        # Arrow keys for next/previous frame (Slicer commands)
        self.shortcutRightArrow = qt.QShortcut(slicer.util.mainWindow())  # "Right Arrow" for next frame
        self.shortcutRightArrow.setKey(qt.QKeySequence('Right'))
        self.shortcutLeftArrow = qt.QShortcut(slicer.util.mainWindow())  # "Left Arrow" for previous frame
        self.shortcutLeftArrow.setKey(qt.QKeySequence('Left'))

        # Home/End keys for first/last frame
        self.shortcutHome = qt.QShortcut(slicer.util.mainWindow())  # "Home" for first frame
        self.shortcutHome.setKey(qt.QKeySequence('Home'))
        self.shortcutEnd = qt.QShortcut(slicer.util.mainWindow())  # "End" for last frame
        self.shortcutEnd.setKey(qt.QKeySequence('End'))

        self.raterNameDebounceTimer = qt.QTimer()
        self.raterNameDebounceTimer.setSingleShot(True)
        self.raterNameDebounceTimer.setInterval(300)  # ms of idle time before triggering
        self.raterNameDebounceTimer.timeout.connect(self.onRaterNameChanged)

        # Spacebar for play/pause
        self.shortcutSpace = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutSpace.setKey(qt.QKeySequence('Space'))

        # Page Up/Page Down for previous/next clip
        self.shortcutPageUp = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutPageUp.setKey(qt.QKeySequence('PageUp'))
        self.shortcutPageDown = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutPageDown.setKey(qt.QKeySequence('PageDown'))

        # Shift+Up/Shift+Down for previous/next clip (in case Page Up/Page Down are intercepted by Slicer)
        self.shortcutShiftUp = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutShiftUp.setKey(qt.QKeySequence('Shift+Up'))
        self.shortcutShiftDown = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutShiftDown.setKey(qt.QKeySequence('Shift+Down'))

        self.shortcutL = qt.QShortcut(slicer.util.mainWindow())  # "L" for show/hide lines
        self.shortcutL.setKey(qt.QKeySequence('L'))

        self.shortcutR = qt.QShortcut(slicer.util.mainWindow())  # "R" for reset all adjudication
        self.shortcutR.setKey(qt.QKeySequence('R'))

    def connectKeyboardShortcuts(self):
        # Connect shortcuts to respective actions
        self.shortcutW.connect('activated()', lambda: self.onAddLine("Pleura", not self.ui.addPleuraButton.isChecked()))
        self.shortcutS.connect('activated()', lambda: self.onAddLine("Bline", not self.ui.addBlineButton.isChecked()))
        self.shortcutO.connect('activated()', lambda: self.ui.overlayVisibilityButton.toggle())

        # New shortcuts for removing lines
        self.shortcutE.connect('activated()', lambda: self.onRemoveLine("Pleura"))  # "E" removes the last pleura line
        self.shortcutD.connect('activated()', lambda: self.onRemoveLine("Bline"))   # "D" removes the last B-line

        self.shortcutA.connect('activated()', self.onSaveAndLoadNextButton)  # "A" to save and load next scan

        # shortcuts for adjudication actions
        self.shortcutV.connect('activated()', self.onValidateLine)  # "V" to validate selected line
        self.shortcutI.connect('activated()', self.onInvalidateLine)  # "I" to invalidate selected line
        self.shortcutU.connect('activated()', self.onDuplicateLine)  # "U" to mark selected line as duplicated
        self.shortcutN.connect('activated()', self.selectNextUnadjudicatedLine)
        self.shortcutP.connect('activated()', self.selectPreviousUnadjudicatedLine)
        # Connect Shift+N/P to next/prev visible annotation
        self.shortcutShiftN.connect('activated()', self.selectNextVisibleLine)
        self.shortcutShiftP.connect('activated()', self.selectPreviousVisibleLine)

        # Arrow keys for next/previous frame (Slicer commands)
        self.shortcutRightArrow.connect('activated()', self._nextFrameInSequence)
        self.shortcutLeftArrow.connect('activated()', self._previousFrameInSequence)
        # Home/End keys for first/last frame
        self.shortcutHome.connect('activated()', self._firstFrameInSequence)
        self.shortcutEnd.connect('activated()', self._lastFrameInSequence)
        # Spacebar for play/pause
        self.shortcutSpace.connect('activated()', self._togglePlayPauseSequence)

        # Page Up/Page Down for previous/next clip
        self.shortcutPageUp.connect('activated()', self._onPageUpPressed)
        self.shortcutPageDown.connect('activated()', self._onPageDownPressed)

        # Shift+Up/Shift+Down for previous/next clip
        self.shortcutShiftUp.connect('activated()', self._onPreviousClipPressed)
        self.shortcutShiftDown.connect('activated()', self._onNextClipPressed)

        self.shortcutL.connect('activated()', lambda: self.onShowHideLines(None))  # "L" to show/hide lines
        self.shortcutR.connect('activated()', self.onResetAllAdjudication)

    def disconnectKeyboardShortcuts(self):
        # Disconnect shortcuts to avoid issues when the user leaves the module
        self.shortcutW.activated.disconnect()
        self.shortcutS.activated.disconnect()
        self.shortcutO.activated.disconnect()
        self.shortcutE.activated.disconnect()
        self.shortcutD.activated.disconnect()
        self.shortcutA.activated.disconnect()
        self.shortcutV.activated.disconnect()
        self.shortcutI.activated.disconnect()
        self.shortcutU.activated.disconnect()
        self.shortcutN.activated.disconnect()
        self.shortcutP.activated.disconnect()
        self.shortcutShiftN.activated.disconnect()
        self.shortcutShiftP.activated.disconnect()
        self.shortcutRightArrow.activated.disconnect()
        self.shortcutLeftArrow.activated.disconnect()
        self.shortcutHome.activated.disconnect()
        self.shortcutEnd.activated.disconnect()
        self.shortcutSpace.activated.disconnect()
        self.shortcutPageUp.activated.disconnect()
        self.shortcutPageDown.activated.disconnect()
        self.shortcutShiftUp.activated.disconnect()
        self.shortcutShiftDown.activated.disconnect()
        self.shortcutL.activated.disconnect()
        self.shortcutR.activated.disconnect()

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/AnnotateUltrasound.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        self.ui.currentFileLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.connectKeyboardShortcuts()

        # Set frames table to show 5-6 rows
        rowHeight = 25  # Approximate height of a row
        headerHeight = 25  # Height of the header
        self.ui.framesTableWidget.setFixedHeight(headerHeight + (rowHeight * 6))  # 6 rows + header

        # Set size policy and minimum height for labels scroll area
        self.ui.labelsScrollArea.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Preferred)
        self.ui.labelsScrollAreaWidgetContents.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Preferred)
        self.ui.labelsScrollArea.setMinimumHeight(200)  # Set a reasonable minimum height

        # Set layout spacing to 0 for compact appearance
        self.ui.labelsScrollAreaWidgetContents.layout().setSpacing(0)
        self.ui.labelsScrollAreaWidgetContents.layout().setContentsMargins(0, 0, 0, 0)

        # Ensure labels section starts expanded
        self.ui.labelAnnotationsCollapsibleButton.collapsed = False

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = AnnotateUltrasoundLogic()

        # Update directory button directory from settings
        self.ui.inputDirectoryButton.directory = slicer.app.settings().value("AnnotateUltrasound/InputDirectory", "")

        # Set up frames table
        self.ui.framesTableWidget.setColumnCount(3)
        self.ui.framesTableWidget.setHorizontalHeaderLabels(["Frame index", "Pleura lines (N)", "B-lines (N)"])
        header = self.ui.framesTableWidget.horizontalHeader()
        header.setSectionResizeMode(qt.QHeaderView.Stretch)
        self.ui.framesTableWidget.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.ui.framesTableWidget.setSelectionMode(qt.QAbstractItemView.SingleSelection)

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.inputDirectoryButton.directoryChanged.connect(self.onInputDirectorySelected)
        self.ui.readInputButton.connect('clicked(bool)', self.onReadInputButton)
        self.ui.nextButton.clicked.connect(self.onNextButton)
        self.ui.previousButton.clicked.connect(self.onPreviousButton)
        self.ui.saveButton.clicked.connect(self.onSaveButton)
        self.ui.saveAndLoadNextButton.clicked.connect(self.onSaveAndLoadNextButton)
        self.ui.intensitySlider.valueChanged.connect(self.onIntensitySliderValueChanged)
        self.ui.skipToUnlabeledButton.clicked.connect(self.onSkipToUnlabeledButton)

        self.ui.addPleuraButton.toggled.connect(lambda checked: self.onAddLine("Pleura", checked))
        self.ui.removePleuraButton.clicked.connect(lambda: self.onRemoveLine("Pleura"))
        self.ui.addBlineButton.toggled.connect(lambda checked: self.onAddLine("Bline", checked))
        self.ui.removeBlineButton.clicked.connect(lambda: self.onRemoveLine("Bline"))
        self.ui.overlayVisibilityButton.toggled.connect(self.overlayVisibilityToggled)
        self.ui.clearAllLinesButton.clicked.connect(self.onClearAllLines)
        self.ui.addCurrentFrameButton.clicked.connect(self.onAddCurrentFrame)
        self.ui.removeCurrentFrameButton.clicked.connect(self.onRemoveCurrentFrame)
        self.ui.showHideLinesButton.toggled.connect(self.onShowHideLines)

        # Assign icons to buttons
        self.ui.nextButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueFillNext.png')))
        self.ui.previousButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueFillPrevious.png')))
        self.ui.addPleuraButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueAdd.png')))
        self.ui.addBlineButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueAdd.png')))
        self.ui.saveButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueSave.png')))
        self.ui.saveAndLoadNextButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueSave.png')))
        self.ui.removePleuraButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueRemove.png')))
        self.ui.removeBlineButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueRemove.png')))
        self.ui.overlayVisibilityButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueEye.png')))
        self.ui.clearAllLinesButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueFillTrash.png')))
        self.ui.skipToUnlabeledButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueFastForward.png')))
        self.ui.showHideLinesButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueEye.png')))

        # Frame table
        self.ui.framesTableWidget.itemSelectionChanged.connect(self.onFramesTableSelectionChanged)

        # Settings
        settings = slicer.app.settings()
        showPleuraPercentage = settings.value('AnnotateUltrasound/ShowPleuraPercentage', 'false')
        self.ui.showPleuraPercentageCheckBox.setChecked(showPleuraPercentage.lower() == 'true')
        self.ui.raterName.setText(slicer.app.settings().value("AnnotateUltrasound/Rater", ""))
        self.ui.raterName.returnPressed.connect(self.onRaterNameChanged)
        self.ui.showPleuraPercentageCheckBox.connect('toggled(bool)', self.saveUserSettings)
        self.ui.depthGuideCheckBox.toggled.connect(self.onDepthGuideToggled)

        # Make buttons taller
        buttonHeight = 40  # Set the height you want for the buttons
        self.ui.inputDirectoryButton.setFixedHeight(buttonHeight)
        self.ui.readInputButton.setFixedHeight(buttonHeight)
        self.ui.nextButton.setFixedHeight(buttonHeight)
        self.ui.previousButton.setFixedHeight(buttonHeight)
        self.ui.saveButton.setFixedHeight(buttonHeight)
        self.ui.saveAndLoadNextButton.setFixedHeight(buttonHeight)
        self.ui.addPleuraButton.setFixedHeight(buttonHeight)
        self.ui.removePleuraButton.setFixedHeight(buttonHeight)
        self.ui.addBlineButton.setFixedHeight(buttonHeight)
        self.ui.removeBlineButton.setFixedHeight(buttonHeight)
        self.ui.overlayVisibilityButton.setFixedHeight(buttonHeight)
        self.ui.clearAllLinesButton.setFixedHeight(buttonHeight)
        self.ui.addCurrentFrameButton.setFixedHeight(buttonHeight)
        self.ui.removeCurrentFrameButton.setFixedHeight(buttonHeight)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        if self.logic and self._parameterNode:
            self.logic.parameterNode = self._parameterNode

        # --- Limit raterColorTable visible rows to about 4 programmatically ---
        if hasattr(self.ui, "raterColorTable"):
            vh = self.ui.raterColorTable.verticalHeader()
            self.ui.raterColorTable.setMaximumHeight(vh.defaultSectionSize * 4 + 2)
            self.ui.raterColorTable.cellClicked.connect(self.onRaterColorTableClicked)
            self.ui.raterColorTable.itemChanged.connect(self.onRaterColorSelectionChangedFromUser)

        # --- Adjudication controls ---
        if hasattr(self.ui, 'validateLineButton'):
            self.ui.validateLineButton.clicked.connect(self.onValidateLine)
            self.ui.validateLineButton.setToolTip("Validate the selected line (V)")
        if hasattr(self.ui, 'invalidateLineButton'):
            self.ui.invalidateLineButton.clicked.connect(self.onInvalidateLine)
            self.ui.invalidateLineButton.setToolTip("Invalidate the selected line (I)")
        if hasattr(self.ui, 'duplicateLineButton'):
            self.ui.duplicateLineButton.clicked.connect(self.onDuplicateLine)
            self.ui.duplicateLineButton.setToolTip("Duplicate the selected line (D)")
        if hasattr(self.ui, 'validateAllButton'):
            self.ui.validateAllButton.clicked.connect(self.onValidateAllUnadjudicated)
        if hasattr(self.ui, 'invalidateAllUnadjudicatedButton'):
            self.ui.invalidateAllUnadjudicatedButton.clicked.connect(self.onInvalidateAllUnadjudicated)
        if hasattr(self.ui, 'duplicateAllUnadjudicatedButton'):
            self.ui.duplicateAllUnadjudicatedButton.clicked.connect(self.onDuplicateAllUnadjudicated)
        if hasattr(self.ui, 'resetAllAdjudicationButton'):
            self.ui.resetAllAdjudicationButton.clicked.connect(self.onResetAllAdjudication)
        if hasattr(self.ui, 'showInvalidOrDuplicateCheckBox'):
            self.ui.showInvalidOrDuplicateCheckBox.toggled.connect(self.onShowInvalidOrDuplicateToggled)
            self.ui.showInvalidOrDuplicateCheckBox.setChecked(True)  # Set default to checked
        if hasattr(self.ui, 'adjudicatorModeCheckBox'):
            self.ui.adjudicatorModeCheckBox.toggled.connect(self.onAdjudicatorModeCheckBoxToggled)
        # Set initial state of adjudication controls
        self.onAdjudicatorModeToggled(self.ui.adjudicatorModeCheckBox.isChecked())

        # Add observer to selection node to enforce only visible nodes can be selected
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        if not hasattr(self, 'selectionObserverTag') or self.selectionObserverTag is None:
            self.selectionObserverTag = selectionNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onSelectionChanged)

        self.setupClickObserverOnRedView()

        if hasattr(self.ui, 'nextUnadjudicatedButton'):
            self.ui.nextUnadjudicatedButton.clicked.connect(self.selectNextUnadjudicatedLine)
        if hasattr(self.ui, 'prevUnadjudicatedButton'):
            self.ui.prevUnadjudicatedButton.clicked.connect(self.selectPreviousUnadjudicatedLine)
        if hasattr(self.ui, 'nextVisibleLineButton'):
            self.ui.nextVisibleLineButton.clicked.connect(self.selectNextVisibleLine)
        if hasattr(self.ui, 'prevVisibleLineButton'):
            self.ui.prevVisibleLineButton.clicked.connect(self.selectPreviousVisibleLine)

        # Connect rater table collapsed signal to detect user manual changes
        if hasattr(self.ui, 'raterColorsCollapsibleButton'):
            self.ui.raterColorsCollapsibleButton.connect('collapsedChanged(bool)', self.onRaterColorTableCollapsedChanged)

    # These functions are used to handle clicks on the red view when selecting lines by clicking on the red view
    # near the line to be selected.
    # --- Click observer on red view ---
    def setupClickObserverOnRedView(self):
        sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
        renderWindowInteractor = sliceWidget.sliceView().renderWindow().GetInteractor()
        self._clickObserverTag = renderWindowInteractor.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.onRedViewClick)

    # --- Distance point to segment ---
    def distancePointToSegment(self, p, a, b):
        p = np.array(p)
        a = np.array(a)
        b = np.array(b)
        ab = b - a
        if np.allclose(ab, 0):
            return np.linalg.norm(p - a)
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    # --- On red view click ---
    # This function is called when the user clicks on the red view.
    # It finds the closest visible markup line node (to any segment) and selects it.
    # If the closest node is within 3 mm of the click, it selects the node.
    # If the closest node is more than 3 mm away, it does nothing.
    def onRedViewClick(self, caller, event):
        x, y = caller.GetEventPosition()
        # Get the slice widget and node
        sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
        sliceNode = sliceWidget.mrmlSliceNode()
        xyToRas = sliceNode.GetXYToRAS()
        xy = [x, y, 0, 1]
        ras = [0, 0, 0, 1]
        xyToRas.MultiplyPoint(xy, ras)
        ras = ras[:3]
        # Find the closest visible markup line node (to any segment)
        minDist = float('inf')
        closestNode = None
        threshold = 3.0  # mm
        for node in slicer.util.getNodesByClass('vtkMRMLMarkupsLineNode'):
            displayNode = node.GetDisplayNode()
            nodeRater = node.GetAttribute('rater')
            if not displayNode or not displayNode.GetVisibility():
                continue
            nPoints = node.GetNumberOfControlPoints()
            for i in range(nPoints - 1):
                pt1 = [0, 0, 0]
                pt2 = [0, 0, 0]
                node.GetNthControlPointPosition(i, pt1)
                node.GetNthControlPointPosition(i + 1, pt2)
                dist = self.distancePointToSegment(ras, pt1, pt2)
                if dist < minDist:
                    minDist = dist
                    closestNode = node
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        if closestNode and minDist < threshold:
            selectionNode.SetActivePlaceNodeID(closestNode.GetID())
        else:
            # Clear selection if clicking away from any line
            selectionNode.SetActivePlaceNodeID("")
        # Update line markups to refresh visual appearance (highlighting, etc.)
        self.logic.updateLineMarkups()

    def onSelectionChanged(self, caller, event):
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectedNodeID = selectionNode.GetActivePlaceNodeID()
        if selectedNodeID:
            node = slicer.mrmlScene.GetNodeByID(selectedNodeID)
            if node and node.GetClassName() == "vtkMRMLMarkupsLineNode":
                displayNode = node.GetDisplayNode()
                if not displayNode or not displayNode.GetVisibility():
                    selectionNode.SetActivePlaceNodeID("")  # Clear selection
        # Update line markups to refresh visual appearance (highlighting, etc.)
        self.logic.updateLineMarkups()

    def saveUserSettings(self):
        settings = qt.QSettings()
        settings.setValue('AnnotateUltrasound/ShowPleuraPercentage', self.ui.showPleuraPercentageCheckBox.checked)
        settings.setValue('AnnotateUltrasound/DepthGuide', self.ui.depthGuideCheckBox.checked)
        settings.setValue('AnnotateUltrasound/Rater', self.ui.raterName.text.strip())
        ratio = self.logic.updateOverlayVolume()
        if ratio is not None:
            self._parameterNode.pleuraPercentage = ratio * 100
        self._updateGUIFromParameterNode()

    def onIntensitySliderValueChanged(self, value):
        if self._parameterNode.inputVolume is None:
            logging.warning("No input ultrasound volume found")
            return
        displayNode = self._parameterNode.inputVolume.GetDisplayNode()
        displayNode.SetWindow(255-2*abs(value))
        displayNode.SetLevel(127+value)

    def onClearAllLines(self):
        logging.info('onClearAllLines')
        self.logic.clearAllLines()
        ratio = self.logic.updateOverlayVolume()
        if ratio is not None:
            self._parameterNode.pleuraPercentage = ratio * 100
            self._parameterNode.unsavedChanges = True
        self.updateGuiFromAnnotations()

    def onFramesTableSelectionChanged(self):
        logging.info('onFramesTableSelectionChanged')

        selectedRow = self.ui.framesTableWidget.currentRow()
        if (selectedRow == -1):
            return

        selectedFrameIndex = int(self.ui.framesTableWidget.item(selectedRow, 0).text())

        # Get the current frame index from the sequence browser

        if self.logic.sequenceBrowserNode is None:
            logging.warning("No sequence browser node found")
            return

        currentFrameIndex = self.logic.sequenceBrowserNode.GetSelectedItemNumber()

        if selectedFrameIndex == currentFrameIndex:
            return
        else:
            self.logic.sequenceBrowserNode.SetSelectedItemNumber(selectedFrameIndex)

    def onAddCurrentFrame(self):
        logging.info('onAddCurrentFrame')
        self.logic.updateCurrentFrame()
        self.updateGuiFromAnnotations()

    def _updateMarkupsAndOverlayProgrammatically(self, setUnsavedChanges=False):
        """
        Helper to update line markups and overlay volume programmatically, suppressing unsavedChanges unless specified.
        """
        self.logic._isProgrammaticUpdate = True
        try:
            self.logic.updateLineMarkups()
            ratio = self.logic.updateOverlayVolume()
            if ratio is not None:
                self._parameterNode.pleuraPercentage = ratio * 100
            else:
                self._parameterNode.pleuraPercentage = 0.0
        finally:
            self.logic._isProgrammaticUpdate = False
        # Clear selection if selected node is not visible
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectedNodeID = selectionNode.GetActivePlaceNodeID()
        if selectedNodeID:
            node = slicer.mrmlScene.GetNodeByID(selectedNodeID)
            if node and node.GetClassName() == "vtkMRMLMarkupsLineNode":
                displayNode = node.GetDisplayNode()
                if not displayNode or not displayNode.GetVisibility():
                    selectionNode.SetActivePlaceNodeID("")  # Clear selection
        if setUnsavedChanges:
            self._parameterNode.unsavedChanges = True

    def onRemoveCurrentFrame(self):
        logging.info('removeCurrentFrame')

        # Get the current frame index from the sequence browser
        if self.logic.sequenceBrowserNode is None:
            logging.warning("No sequence browser node found")
            currentFrameIndex = -1
        else:
            currentFrameIndex = self.logic.sequenceBrowserNode.GetSelectedItemNumber()
            self.logic.removeFrame(currentFrameIndex)
            self._updateMarkupsAndOverlayProgrammatically(setUnsavedChanges=True)
            self.updateGuiFromAnnotations()

    def onInputDirectorySelected(self):
        logging.info('onInputDirectorySelected')

        inputDirectory = self.ui.inputDirectoryButton.directory
        if not inputDirectory:
            statusText = '⚠️ Please select an input directory'
            slicer.util.mainWindow().statusBar().showMessage(statusText, 5000)
            self.ui.statusLabel.setText(statusText)
            return

        # Update local settings
        slicer.app.settings().setValue("AnnotateUltrasound/InputDirectory", inputDirectory)

    def extractSeenRaters(self):
        """
        Extracts the set of raters that have contributed lines in the current annotations,
        ensuring the current rater is included even if not present in any frame annotations.
        Sets self.seenRaters to a sorted list of rater names.
        """
        # Use the Logic's centralized method to extract and set up raters
        self.logic.extractAndSetupRaters()
        # Copy the seenRaters from logic to widget for UI purposes
        self.seenRaters = self.logic.seenRaters.copy()

    def onReadInputButton(self):
        """
        Read the input directory and update the dicomDf dataframe, using rater-specific annotation files.

        :return: True if the input directory was read successfully, False otherwise.
        """
        logging.info('onReadInputButton')

        inputDirectory = self.ui.inputDirectoryButton.directory
        if not inputDirectory:
            statusText = '⚠️ Please select an input directory'
            slicer.util.mainWindow().statusBar().showMessage(statusText, 5000)
            self.ui.statusLabel.setText(statusText)
            self._parameterNode.dfLoaded = False
            return

        rater = self._parameterNode.rater
        if not rater:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Missing Rater Name",
                "Please enter a rater name before loading the input directory."
            )
            self.ui.statusLabel.setText("⚠️ Please enter a rater name before loading.")
            self._parameterNode.dfLoaded = False
            return

        numFilesFound, numAnnotationsCreated = self.logic.updateInputDf(rater, inputDirectory)
        logging.info(f"Found {numFilesFound} DICOM files")
        statusText = f"Found {numFilesFound} DICOM files"
        if numAnnotationsCreated > 0:
            self.ui.statusLabel.setText(f"Found {numFilesFound} DICOM files. Created {numAnnotationsCreated} annotations files.")
            statusText += f"\nWARNING: Created {numAnnotationsCreated} annotations files"

        if numFilesFound > 0:
            self._parameterNode.dfLoaded = True
            # Update progress bar
            self.ui.progressBar.minimum = 0
            self.ui.progressBar.maximum = numFilesFound
            self.ui.progressBar.value = 0
            waitDialog = self.createWaitDialog("Loading first sequence", "Loading first sequence...")

            # Set navigation flag to prevent rater table state changes
            self._isNavigating = True

            self.currentDicomDfIndex = self.logic.loadNextSequence()

            # Add observer for the new sequence browser node
            if self.logic.sequenceBrowserNode:
                self.addObserver(self.logic.sequenceBrowserNode, vtk.vtkCommand.ModifiedEvent, self.onSequenceBrowserModified)

            # Update self.ui.currentFileLabel using the DICOM file name
            currentDicomFilepath = self.logic.dicomDf.iloc[self.logic.nextDicomDfIndex - 1]['Filepath']
            currentDicomFilename = os.path.basename(currentDicomFilepath)
            statusText = f"Current file ({self.logic.nextDicomDfIndex}/{len(self.logic.dicomDf)}): {currentDicomFilename}"
            self.ui.currentFileLabel.setText(statusText)
            self.ui.statusLabel.setText('')
            slicer.util.mainWindow().statusBar().showMessage(statusText, 3000)
            self.logic.sequenceBrowserNode.SetSelectedItemNumber(0)
            self.logic.updateCurrentFrame()

            self.ui.intensitySlider.setValue(0)

            # After loading the first sequence, extract seen raters and update checkboxes
            self.extractSeenRaters()
            self.selectedRaters = set(self.seenRaters)

            self._updateRaterColorTableCheckboxes()
            self.updateGuiFromAnnotations()

            # Close the wait dialog
            waitDialog.close()

            self.ui.progressBar.value = self.currentDicomDfIndex

            self.ui.overlayVisibilityButton.setChecked(True)

            # Mark that this is no longer the first DICOM load
            self._isFirstDicomLoad = False
        else:
            statusText = 'Could not find any files to load in input directory!'
            slicer.util.mainWindow().statusBar().showMessage(statusText, 3000)
            self.ui.statusLabel.setText(statusText)

        self._updateGUIFromParameterNode()

        # Clear navigation flag after all operations are complete
        self._isNavigating = False

        # Set focus to red view after initial load to ensure keyboard shortcuts work
        if numFilesFound > 0:
            self._setRedViewFocus()

    def confirmUnsavedChanges(self):
        """
        Asks the user if they want to save unsaved changes before continuing.
        """
        if self._parameterNode.unsavedChanges:
            reply = qt.QMessageBox.question(
                slicer.util.mainWindow(),
                'Unsaved Changes',
                'You have unsaved changes. Do you want to save them before continuing?',
                qt.QMessageBox.Save | qt.QMessageBox.Discard | qt.QMessageBox.Cancel
            )
            if reply == qt.QMessageBox.Save:
                self.saveAnnotations() # saving changes
                self._parameterNode.unsavedChanges = False
                return True
            elif reply == qt.QMessageBox.Discard:
                self._parameterNode.unsavedChanges = False
                logging.info('Discarding changes')
                return True
            else:
                logging.info('Cancelling Next or Previous action')
                return False
        return True

    def onNextButton(self):
        logging.info('onNextButton')

        if self.logic.dicomDf is None:
            self.ui.statusLabel.setText("Please read input directory first")
            return

        if not self.confirmUnsavedChanges():
            return

        if self.logic.nextDicomDfIndex >= len(self.logic.dicomDf):
            # If we are at the last DICOM file, show a message that clears in 5 seconds and return
            slicer.util.mainWindow().statusBar().showMessage('⚠️ No more DICOM files', 5000)
            return

        # Create a dialog to ask the user to wait while the next sequence is loaded.

        waitDialog = self.createWaitDialog("Loading next sequence", "Loading next sequence...")

        # Set navigation flag to prevent rater table state changes
        self._isNavigating = True

        # Saving settings
        showDepthGuide = self._parameterNode.depthGuideVisible

        currentDicomDfIndex = self.logic.loadNextSequence()

        # Add observer for the new sequence browser node
        if self.logic.sequenceBrowserNode:
            self.addObserver(self.logic.sequenceBrowserNode, vtk.vtkCommand.ModifiedEvent, self.onSequenceBrowserModified)

        # After loading the next sequence, extract seen raters and update checkboxes
        self.extractSeenRaters()
        self.selectedRaters = set(self.seenRaters)

        self.populateRaterColorTable()

        # Uncheck all label checkboxes, but prevent them from triggering the onLabelCheckBoxToggled event while we are doing this
        for i in reversed(range(self.ui.labelsScrollAreaWidgetContents.layout().count())):
            widget = self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget()
            if not isinstance(widget, qt.QGroupBox):
                continue
            groupBoxTitle = widget.title
            # Find all checkboxes in groupBox
            for j in reversed(range(widget.layout().count())):
                checkBox = widget.layout().itemAt(j).widget()
                if isinstance(checkBox, qt.QCheckBox):
                    checkBox.blockSignals(True)
                    checkBox.setChecked(False)
                    checkBox.blockSignals(False)

        # Update self.ui.currentFileLabel using the DICOM file name

        currentDicomFilepath = self.logic.dicomDf.iloc[self.logic.nextDicomDfIndex - 1]['Filepath']
        currentDicomFilename = os.path.basename(currentDicomFilepath)
        statusText = f"Current file ({self.logic.nextDicomDfIndex}/{len(self.logic.dicomDf)}): {currentDicomFilename}"
        self.ui.currentFileLabel.setText(statusText)
        slicer.util.mainWindow().statusBar().showMessage(statusText, 3000)

        self.updateGuiFromAnnotations()

        # Restore settings
        self._parameterNode.depthGuideVisible = showDepthGuide

        # Close the wait dialog
        waitDialog.close()

        self.ui.intensitySlider.setValue(0)

        self.ui.progressBar.value = currentDicomDfIndex

        self.ui.overlayVisibilityButton.setChecked(True)

        self.logic.loadAnnotations(self.logic.nextDicomDfIndex - 1, self.ui.adjudicatorModeCheckBox.isChecked())

        # Clear navigation flag after all operations are complete
        self._isNavigating = False

        # Set focus to red view after navigation to ensure keyboard shortcuts work
        self._setRedViewFocus()

    def updateGuiFromAnnotations(self):
        # Check checkboxes in the labels scroll area if the labels are present in the logic.annotations
        if self.logic.annotations is not None and "labels" in self.logic.annotations:
            for i in reversed(range(self.ui.labelsScrollAreaWidgetContents.layout().count())):
                widget = self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget()
                if not isinstance(widget, qt.QGroupBox):
                    continue
                groupBoxTitle = widget.title
                # Find all checkboxes in groupBox
                for j in reversed(range(widget.layout().count())):
                    checkBox = widget.layout().itemAt(j).widget()
                    if isinstance(checkBox, qt.QCheckBox):
                        origCategory = checkBox.property('originalCategory')
                        origLabel = checkBox.property('originalLabel')
                        annotationName = f"{origCategory}/{origLabel}"
                        checkBox.blockSignals(True)
                        if annotationName in self.logic.annotations['labels']:
                            checkBox.setChecked(True)
                        else:
                            checkBox.setChecked(False)
                        checkBox.blockSignals(False)
        else:
            # Uncheck all label checkboxes
            for i in reversed(range(self.ui.labelsScrollAreaWidgetContents.layout().count())):
                widget = self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget()
                if not isinstance(widget, qt.QGroupBox):
                    continue
                # Find all checkboxes in groupBox
                for j in reversed(range(widget.layout().count())):
                    checkBox = widget.layout().itemAt(j).widget()
                    if isinstance(checkBox, qt.QCheckBox):
                        checkBox.setChecked(False)

        # Update frames table, save the current sort column and order to restore after populating
        sort_column = self.ui.framesTableWidget.horizontalHeader().sortIndicatorSection()
        sort_order = self.ui.framesTableWidget.horizontalHeader().sortIndicatorOrder()
        # disable sorting while we populate the table
        self.ui.framesTableWidget.setSortingEnabled(False)

        # Remove all rows from the table
        self.ui.framesTableWidget.setRowCount(0)

        # Add rows to the table
        if self.logic.annotations is not None and "frame_annotations" in self.logic.annotations:
            for frame_annotations in self.logic.annotations["frame_annotations"]:
                row = self.ui.framesTableWidget.rowCount
                self.ui.framesTableWidget.insertRow(row)

                frame_number = int(frame_annotations.get("frame_number", row))
                frame_number_item = qt.QTableWidgetItem()
                frame_number_item.setData(qt.Qt.DisplayRole, frame_number)
                self.ui.framesTableWidget.setItem(row, 0, frame_number_item)

                pleura_count = len([
                    pleura_line for pleura_line in frame_annotations.get("pleura_lines", [])
                    if pleura_line is not None and isinstance(pleura_line.get("line"), dict)
                ])
                pleura_item = qt.QTableWidgetItem()
                pleura_item.setData(qt.Qt.DisplayRole, pleura_count)
                self.ui.framesTableWidget.setItem(row, 1, pleura_item)

                bline_count = len([
                    b_line for b_line in frame_annotations.get("b_lines", [])
                    if b_line is not None and isinstance(b_line.get("line"), dict)
                ])
                bline_item = qt.QTableWidgetItem()
                bline_item.setData(qt.Qt.DisplayRole, bline_count)
                self.ui.framesTableWidget.setItem(row, 2, bline_item)

        # reenable sorting after populating the table
        self.ui.framesTableWidget.setSortingEnabled(True)

        # Restore previous sort state
        self.ui.framesTableWidget.sortItems(sort_column, sort_order)

    def createWaitDialog(self, title, message):
        """
        Create a dialog to ask the user to wait while the sequence is loaded.
        """
        waitDialog = qt.QDialog(slicer.util.mainWindow())
        waitDialog.setWindowTitle(title)
        waitDialogLayout = qt.QVBoxLayout(waitDialog)
        waitDialogLayout.setContentsMargins(20, 14, 20, 14)
        waitDialogLayout.setSpacing(20)
        waitDialogLabel = qt.QLabel(message)
        waitDialogLabel.setWordWrap(False)
        waitDialogLayout.addWidget(waitDialogLabel)
        waitDialog.show()
        return waitDialog

    def onPreviousButton(self):
        logging.info('onPreviousButton')

        if not self.confirmUnsavedChanges():
            return

        if self.logic.dicomDf is None:
            self.ui.statusLabel.setText("Please read input directory first")
            return

        # Create a dialog to ask the user to wait while the next sequence is loaded.
        waitDialog = self.createWaitDialog("Loading previous sequence", "Loading previous sequence...")

        # Set navigation flag to prevent rater table state changes
        self._isNavigating = True

        # Saving settings
        showDepthGuide = self._parameterNode.depthGuideVisible

        savedNextDicomDfIndex = self.logic.nextDicomDfIndex
        currentDicomDfIndex = self.logic.loadPreviousSequence()
        if currentDicomDfIndex is None:
            # Close the wait dialog
            waitDialog.close()
            self.logic.nextDicomDfIndex = savedNextDicomDfIndex
            # show status message for 5 seconds
            slicer.util.mainWindow().statusBar().showMessage('⚠️ First DICOM file reached', 5000)
            return

        # Update self.ui.currentFileLabel using the DICOM file name
        currentDicomFilepath = self.logic.dicomDf.iloc[self.logic.nextDicomDfIndex - 1]['Filepath']
        currentDicomFilename = os.path.basename(currentDicomFilepath)
        statusText = f"Current file ({self.logic.nextDicomDfIndex}/{len(self.logic.dicomDf)}): {currentDicomFilename}"
        self.ui.currentFileLabel.setText(statusText)
        slicer.util.mainWindow().statusBar().showMessage(statusText, 3000)

        # Restore settings
        self._parameterNode.depthGuideVisible = showDepthGuide

        self.ui.intensitySlider.setValue(0)

        # After loading the previous sequence, extract seen raters and update checkboxes
        self.extractSeenRaters()
        self.selectedRaters = set(self.seenRaters)

        self.populateRaterColorTable()

        self.updateGuiFromAnnotations()

        # Close the wait dialog
        waitDialog.close()

        self.ui.progressBar.value = currentDicomDfIndex

        self.logic.loadPreviousSequence()
        self.logic.loadAnnotations(self.logic.nextDicomDfIndex - 1, self.ui.adjudicatorModeCheckBox.isChecked())

        # Clear navigation flag after all operations are complete
        self._isNavigating = False

        # Set focus to red view after navigation to ensure keyboard shortcuts work
        self._setRedViewFocus()

    def saveAnnotations(self):
        """
        Save annotations depending on adjudicator mode.
        Returns True if save was successful, False otherwise.
        """
        try:
            adjudicator_mode = hasattr(self.ui, 'adjudicatorModeCheckBox') and self.ui.adjudicatorModeCheckBox.isChecked()
            rater = self._parameterNode.rater.strip().lower()
            base_file_path = self.logic.dicomDf.iloc[self.logic.nextDicomDfIndex - 1]['Filepath']
            base_prefix = os.path.splitext(base_file_path)[0]
            adjudication_file = f"{base_prefix}.adjudication.json"

            if adjudicator_mode:
                # Save all lines (combined, no filtering) to adjudication file
                save_data = self.logic.convert_ras_to_lps(self.logic.annotations.get("frame_annotations", []))
                with open(adjudication_file, 'w') as f:
                    json.dump(save_data, f)
                logging.info(f"Annotations saved to {adjudication_file}")
                status_message = f"✅ Adjudication annotations saved successfully to {os.path.basename(adjudication_file)}"
                slicer.util.showStatusMessage(status_message, 3000)
                self.ui.statusLabel.setText(status_message)
            else:
                # Save only current rater's lines as before
                filtered_frames = []
                for frame in self.logic.annotations.get("frame_annotations", []):
                    pleura = [line for line in frame.get("pleura_lines", []) if line.get("rater", "").strip().lower() == rater]
                    b_lines = [line for line in frame.get("b_lines", []) if line.get("rater", "").strip().lower() == rater]
                    if pleura or b_lines:
                        filtered_frames.append({
                            "frame_number": frame["frame_number"],
                            "coordinate_space": "RAS",
                            "pleura_lines": pleura,
                            "b_lines": b_lines
                        })
                if filtered_frames or self._parameterNode.unsavedChanges:
                    save_data = self.logic.convert_ras_to_lps(filtered_frames)
                    annotationsFilepath = self.logic.dicomDf.iloc[self.logic.nextDicomDfIndex - 1]['AnnotationsFilepath']
                    base_path, ext = os.path.splitext(annotationsFilepath)
                    if not base_path.endswith(f".{rater}"):
                        annotationsFilepath = f"{base_path}.{rater}.json"
                        self.logic.dicomDf.at[self.logic.nextDicomDfIndex - 1, 'AnnotationsFilepath'] = annotationsFilepath
                    with open(annotationsFilepath, 'w') as f:
                        json.dump(save_data, f)
                    logging.info(f"Annotations saved to {annotationsFilepath}")
                    status_message = f"✅ Annotations saved successfully to {os.path.basename(annotationsFilepath)}"
                    slicer.util.showStatusMessage(status_message, 3000)
                    self.ui.statusLabel.setText(status_message)
                else:
                    # No annotations to save
                    status_message = "ℹ️ No annotations to save"
                    slicer.util.showStatusMessage(status_message, 2000)
                    self.ui.statusLabel.setText(status_message)

            # Clear unsaved changes flag on successful save
            self._parameterNode.unsavedChanges = False
            return True

        except Exception as e:
            error_message = f"❌ Failed to save annotations: {str(e)}"
            logging.error(error_message)
            slicer.util.showStatusMessage(error_message, 5000)
            self.ui.statusLabel.setText(error_message)
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
        else:
            # Error message already shown by saveAnnotations, don't proceed to next
            return

    def onAddLine(self, lineType, checked):
        logging.info(f"onAddLine -- lineType: {lineType}, checked: {checked}")

        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        self.removeObservers(self.onEndPlaceMode)

        if not checked:
            # Return mouse interaction to default
            interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

            if lineType == "Pleura":
                linesList = self.logic.pleuraLines
            elif lineType == "Bline":
                linesList = self.logic.bLines
            else:
                logging.error(f"Unknown line type {lineType}")
                return

            logging.info("Auto-saving frame annotations")
            self.logic.updateCurrentFrame()
            self.updateGuiFromAnnotations()

            # If current line has less than 2 control points, remove it
            if len(linesList) > 0:
                currentLine = linesList[-1]
                if currentLine.GetNumberOfControlPoints() < 2:
                    linesList.pop()
                    slicer.mrmlScene.RemoveNode(currentLine)
            ratio = self.logic.updateOverlayVolume()
            if ratio is not None:
                self._parameterNode.pleuraPercentage = ratio * 100
            return

        # Put interaction model to place line markup
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        interactionNode.SetPlaceModePersistence(0)

        self.addObserver(interactionNode, interactionNode.EndPlacementEvent, self.onEndPlaceMode)

        # Create a new markup fiducial node
        rater = self._parameterNode.rater
        if lineType == "Pleura":
            color_pleura, _ = self.logic.getColorsForRater(rater)
            newLineNode = self.logic.createMarkupLine("Pleura", rater, [], color_pleura)
            self.logic.pleuraLines.append(newLineNode)
        elif lineType == "Bline":
            _, color_blines = self.logic.getColorsForRater(rater)
            newLineNode = self.logic.createMarkupLine("B-line", rater, [], color_blines)
            self.logic.bLines.append(newLineNode)
        else:
            logging.error(f"Unknown line type {lineType}")
            return

        self._parameterNode.lineBeingPlaced = newLineNode
        self._parameterNode.unsavedChanges = True

    def onEndPlaceMode(self, caller, event):
        # Call the next line using qtimer
        lineType = self._parameterNode.lineBeingPlaced.GetName()
        logging.info(f'onEndPlaceMode -- lineType: {lineType}')
        if lineType == "Pleura":
            qt.QTimer.singleShot(0, self.delayedOnEndPlaceMode, "Pleura")
        elif lineType == "B-line":
            qt.QTimer.singleShot(0, self.delayedOnEndPlaceMode, "Bline")
        else:
            logging.error(f"Unknown line type {lineType}")
            return

        logging.info("Auto-saving frame annotations")
        self.logic.updateCurrentFrame()
        self.updateGuiFromAnnotations()

    def delayedOnEndPlaceMode(self, lineType):
        logging.info(f"delayedOnEndPlaceMode -- lineType: {lineType}")
        if lineType == "Pleura":
            self.ui.addPleuraButton.setChecked(False)
        elif lineType == "Bline":
            self.ui.addBlineButton.setChecked(False)
        else:
            logging.error(f"Unknown line type {lineType}")
            return

        logging.info("Auto-saving frame annotations")
        self.logic.updateCurrentFrame()
        self.updateGuiFromAnnotations()

    def onRemovePleuraLine(self):
        logging.info('onRemovePleuraLine')
        self.logic.removeLastPleuraLine()

    def onRemoveLine(self, lineType):
        logging.info(f"onRemoveLine -- lineType: {lineType}")
        if lineType == "Pleura":
            self.logic.removeLastPleuraLine()
        elif lineType == "Bline":
            self.logic.removeLastBline()
        else:
            logging.error(f"Unknown line type {lineType}")
            return

        self.logic.updateCurrentFrame()
        self.updateGuiFromAnnotations()
        self._parameterNode.unsavedChanges = True

    def onLabelsFileSelected(self, labelsFilepath=None):
        # Use self.resourcePath to get the correct path to resources (consistent with other resource usage in this module)
        labelsFilepath = self.resourcePath('annotation_labels.csv')
        logging.info(f"Loading fixed labels file from: {labelsFilepath}")
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
            words = re.findall('[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z]|[^a-zA-Z]|$)|[a-z]+', text)
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

    def onLabelCheckBoxToggled(self, checkBox, checked):
        logging.info(f"onLabelCheckBoxToggled -- checked: {checked}")
        if self.logic.annotations is None:
            logging.error("No annotations loaded")
            return
        if "labels" not in self.logic.annotations:
            self.logic.annotations["labels"] = []

        # Get the sender checkbox and its original category/label
        origCategory = checkBox.property('originalCategory')
        origLabel = checkBox.property('originalLabel')
        annotationName = f"{origCategory}/{origLabel}"

        if checked:
            if annotationName not in self.logic.annotations['labels']:
                self.logic.annotations['labels'].append(annotationName)
        else:
            if annotationName in self.logic.annotations['labels']:
                self.logic.annotations['labels'].remove(annotationName)
        self._parameterNode.unsavedChanges = True

    def onSkipToUnlabeledButton(self):
        """
        Skip to the next unlabeled scan
        """
        if not self.confirmUnsavedChanges():
            return

        if self.logic.dicomDf is None:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Missing Data",
                "Please read input directory first."
            )
            return

        # Find the next unlabeled scan
        nextUnlabeledIndex = self.findNextUnlabeledScan()
        if nextUnlabeledIndex is None:
            qt.QMessageBox.information(
                slicer.util.mainWindow(),
                "Skip to Unlabeled",
                "No unlabeled scans found."
            )
            return

        self.logic.nextDicomDfIndex = nextUnlabeledIndex
        self.onNextButton()

    def findNextUnlabeledScan(self):
        """
        Find the index of the next unlabeled scan in the DICOM dataframe.
        :return: Index of the next unlabeled scan or None if no such scan is found.
        """
        if self.logic.dicomDf is None:
            return None
        for idx in range(self.logic.nextDicomDfIndex, len(self.logic.dicomDf)):
            self.logic.nextDicomDfIndex = idx
            self.logic.loadAnnotations(idx, self.ui.adjudicatorModeCheckBox.isChecked())
            annotations = self.logic.annotations
            if 'frame_annotations' not in annotations or not annotations['frame_annotations']:
                return idx
        return None

    def overlayVisibilityToggled(self, checked):
        logging.info(f"overlayVisibilityToggled -- checked: {checked}")
        if checked:
            # Set overlay volume as foreground in slice viewers
            redSliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceCompositeNode()
            redSliceCompositeNode.SetForegroundVolumeID(self._parameterNode.overlayVolume.GetID())
            redSliceCompositeNode.SetForegroundOpacity(0.12)
            redSliceCompositeNode.SetCompositing(2)
            displayNode = self._parameterNode.overlayVolume.GetDisplayNode()
            displayNode.SetWindow(255)
            displayNode.SetLevel(127)
        else:
            # Set foreground volume to None
            redSliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceCompositeNode()
            redSliceCompositeNode.SetForegroundVolumeID(None)

    def updateZonalOverlay(self):
        # Print the coordinates of all points in all markup nodes
        for markupNode in self.logic.pleuraLines:
            for i in range(markupNode.GetNumberOfControlPoints()):
                coord = [0, 0, 0]
                markupNode.GetNthControlPointPosition(i, coord)

    def onDepthGuideToggled(self, toggled):
        # Save new state in application settings and update overlay volume to show/hide the depth guide
        settings = slicer.app.settings()
        settings.setValue('AnnotateUltrasound/DepthGuide', toggled)
        if toggled:
            self.logic.parameterNode.depthGuideVisible = True
        else:
            self.logic.parameterNode.depthGuideVisible = False
        self.logic.updateOverlayVolume()

    def onRaterNameChanged(self):
        if self._parameterNode:
            self._parameterNode.rater = self.ui.raterName.text.strip()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

        self.disconnectKeyboardShortcuts()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

        # Collapse DataProbe widget
        mw = slicer.util.mainWindow()
        if mw:
            w = slicer.util.findChild(mw, "DataProbeCollapsibleWidget")
            if w:
                w.collapsed = True

        # Switch to red slice only layout if this is the first enter
        if self.notEnteredYet:
            self.notEnteredYet = False
            slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)

        sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode()
        self.compositingModeExit = sliceCompositeNode.GetCompositing()
        sliceCompositeNode.SetCompositing(2)

        # Load fixed labels file
        self.onLabelsFileSelected()

        # Hide slice view annotations to avoid interference with the corner annotation
        sliceAnnotations = slicer.modules.DataProbeInstance.infoWidget.sliceAnnotations
        sliceAnnotations.sliceViewAnnotationsEnabled=False
        sliceAnnotations.updateSliceViewFromGUI()

        # Add sequence browser observer if sequence browser exists
        if self.logic and self.logic.sequenceBrowserNode:
            self.addObserver(self.logic.sequenceBrowserNode, vtk.vtkCommand.ModifiedEvent, self.onSequenceBrowserModified)

        self._updateGUIFromParameterNode()

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
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())
        if self.logic and self._parameterNode:
            self.logic.parameterNode = self._parameterNode

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

        settings = slicer.app.settings()
        showDepthGuide = settings.value('AnnotateUltrasound/DepthGuide', False)
        # be consistent and just read bool, convert if was string
        if isinstance(showDepthGuide, str):
            showDepthGuide = showDepthGuide.lower() == 'true'
        self._parameterNode.rater = settings.value('AnnotateUltrasound/Rater', '')
        self.ui.raterName.setText(self._parameterNode.rater)
        if self._parameterNode.rater != '':
            self.logic.setRater(self._parameterNode.rater)
            self.logic.getColorsForRater(self._parameterNode.rater)
        self.ui.depthGuideCheckBox.setChecked(showDepthGuide)


    def setParameterNode(self, inputParameterNode: AnnotateUltrasoundParameterNode) -> None:
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

            # Update corner annotation if _parameterNode.pleuraPercentage is a non-negative number
            # if we are using multiple raters and have selected more than one, don't show overlay volume
            adjudicator_mode = getattr(self._parameterNode, 'adjudicatorMode', False)
            selectedRaters = self.logic.getSelectedRaters()
            if adjudicator_mode or (selectedRaters is not None and len(selectedRaters) == 1):
                if self.ui.showPleuraPercentageCheckBox.checked and self._parameterNode.pleuraPercentage >= 0:
                    view=slicer.app.layoutManager().sliceWidget("Red").sliceView()
                    view.cornerAnnotation().SetText(vtk.vtkCornerAnnotation.UpperLeft,f"B-line/Pleura = {self._parameterNode.pleuraPercentage:.1f} %")
                    view.cornerAnnotation().GetTextProperty().SetColor(1,1,0)
                    view.forceRender()
                elif self.ui.showPleuraPercentageCheckBox.checked and self._parameterNode.pleuraPercentage == -2.0:
                    view=slicer.app.layoutManager().sliceWidget("Red").sliceView()
                    view.cornerAnnotation().SetText(vtk.vtkCornerAnnotation.UpperLeft,f"No pleura detected")
                    view.cornerAnnotation().GetTextProperty().SetColor(1,1,0)
                    view.forceRender()
                else:
                    view=slicer.app.layoutManager().sliceWidget("Red").sliceView()
                    view.cornerAnnotation().SetText(vtk.vtkCornerAnnotation.UpperLeft,"")
                    view.forceRender()
            else:
                view=slicer.app.layoutManager().sliceWidget("Red").sliceView()
                view.cornerAnnotation().SetText(vtk.vtkCornerAnnotation.UpperLeft,"")
                view.forceRender()

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
                # Only expand sector annotations if not in adjudicator mode
                if not self._parameterNode.adjudicatorMode:
                    self.ui.sectorAnnotationsCollapsibleButton.collapsed = False
                else:
                    self.ui.sectorAnnotationsCollapsibleButton.collapsed = True
                self.ui.labelAnnotationsCollapsibleButton.collapsed = False

                # Handle rater table collapse/expand logic
                if hasattr(self.ui, 'raterColorsCollapsibleButton'):
                    # During navigation, allow content updates but prevent collapse/expand state changes
                    if self._isNavigating:
                        # Skip collapse/expand logic during navigation, but still populate content
                        pass
                    else:
                        # If user manually set state, always restore it
                        if self._userManuallySetRaterTableState:
                            if self._lastUserManualCollapsedState is not None:
                                self._setRaterColorTableCollapsedState(self._lastUserManualCollapsedState)
                        else:
                            should_collapse = self._parameterNode.adjudicatorMode
                            self._setRaterColorTableCollapsedState(should_collapse)

            # Save rater name to settings
            settings = qt.QSettings()
            settings.setValue('AnnotateUltrasound/Rater', self.ui.raterName.text.strip())

            # Only update raterColorTable if present and DICOM is loaded
            if hasattr(self.ui, 'raterColorTable') and self._parameterNode.dfLoaded:
                self.populateRaterColorTable()
        finally:
            self.updatingGUI = False

    def _setRaterColorTableCollapsedState(self, collapsed):
        """
        Set the collapsed state of the rater color table.

        Args:
            collapsed: True to collapse, False to expand
        """
        if not hasattr(self.ui, 'raterColorsCollapsibleButton'):
            return

        self.ui.raterColorsCollapsibleButton.collapsed = collapsed

    def onRaterColorTableCollapsedChanged(self, collapsed):
        """
        Called when the user manually expands/collapses the rater table.
        Sets the flag to respect user's manual state.
        """
        self._userManuallySetRaterTableState = True
        self._lastUserManualCollapsedState = collapsed

    def populateRaterColorTable(self):
        if not hasattr(self.ui, 'raterColorTable'):
            return
        self.ui.raterColorTable.blockSignals(True)
        self.ui.raterColorTable.clearContents()
        colors = list(self.logic.getAllRaterColors())

        # Filter out __selected_node__, __adjudicated_node__ before setting row count, we don't want to show it in the UI
        visible_colors = [(r, (pleura_color, bline_color)) for r, (pleura_color, bline_color) in colors if r != "__selected_node__" and r != "__adjudicated_node__"]

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

    def getSelectedRatersFromTable(self):
        selected = []
        table = self.ui.raterColorTable
        for row in range(table.rowCount):
            item = table.item(row, 0)
            if item is not None and item.checkState() == qt.Qt.Checked:
                selected.append(item.text())
        return selected

    def _updateRaterColorTableCheckboxes(self):
        """
        Helper function to update all checkboxes in the rater color table based on the selectedRaters.

        """

        if not hasattr(self.ui, 'raterColorTable'):
            return
        table = self.ui.raterColorTable
        table.blockSignals(True)
        try:
            for row in range(table.rowCount):
                item = table.item(row, 0)
                if item:
                    if item.text().strip().lower() in self.selectedRaters:
                        item.setCheckState(qt.Qt.Checked)
                    else:
                        item.setCheckState(qt.Qt.Unchecked)
        finally:
            table.blockSignals(False)
        self.ui.raterColorTable.repaint()
        self.ui.raterColorTable.update()

    def onRaterColorSelectionChangedFromUser(self):
        if self.updatingGUI:
            return
        self.updateRatersFromCheckboxes()

    def updateRatersFromCheckboxes(self):
        self.selectedRaters = self.getSelectedRatersFromTable()
        self.logic.setSelectedRaters(self.selectedRaters)
        self._updateMarkupsAndOverlayProgrammatically()
        self._updateGUIFromParameterNode()
        self.ui.raterColorTable.repaint()
        self.ui.raterColorTable.update()

    def onRaterColorTableClicked(self, row, column):
        item = self.ui.raterColorTable.item(row, 0)  # Assume checkbox is in column 0
        if item is not None:
            current_state = item.checkState()
            item.setCheckState(qt.Qt.Unchecked if current_state == qt.Qt.Checked else qt.Qt.Checked)
        self.onRaterColorSelectionChangedFromUser()

    def onValidateLine(self):
        self._setLineValidationStatus("validated", "validate", "validated")

    def onInvalidateLine(self):
        self._setLineValidationStatus("invalidated", "invalidate", "invalidated")

    def onDuplicateLine(self):
        self._setLineValidationStatus("duplicated", "mark as duplicated", "marked as duplicated")
    def _setLineValidationStatus(self, status, action_verb, status_description):
        """
        Set the validation status of the selected line.

        Args:
            status: The validation status to set ("validated", "invalidated", or "duplicated")
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
        # Reset cache to force immediate color updates when validation status changes
        self.logic._resetMarkupCache()
        self.logic.updateCurrentFrame()
        slicer.util.showStatusMessage(f"Line {status_description}.", 3000)

    def onValidateAllUnadjudicated(self):
        self._setAllUnadjudicatedLinesStatus("validated", "Validated", 1.0)

    def onInvalidateAllUnadjudicated(self):
        self._setAllUnadjudicatedLinesStatus("invalidated", "Invalidated", 0.3)

    def onDuplicateAllUnadjudicated(self):
        self._setAllUnadjudicatedLinesStatus("duplicated", "Duplicated", 0.3)

    def _setAllUnadjudicatedLinesStatus(self, status, status_past_tense, opacity):
        """
        Set the validation status of all unadjudicated lines.

        Args:
            status: The validation status to set ("validated", "invalidated", or "duplicated")
            status_past_tense: The past tense form for the success message (e.g., "Validated", "Invalidated")
            opacity: The opacity to set for the lines (1.0 for validated, 0.3 for others)
        """
        # Get current rater (adjudicator)
        adjudicator = self.getCurrentRater()
        count = 0

        # Set programmatic update flag to prevent unsavedChanges from being set
        self.logic._isProgrammaticUpdate = True

        try:
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

            # Reset cache to force immediate color updates when validation status changes
            self.logic._resetMarkupCache()

            # Update the visual appearance of all lines
            self.logic.updateLineMarkups()

            # Update the current frame to save the changes to annotations
            self.logic.updateCurrentFrame()

        finally:
            # Reset programmatic update flag
            self.logic._isProgrammaticUpdate = False

        # Set unsavedChanges since this is a user action that modifies validation status
        self._parameterNode.unsavedChanges = True

        slicer.util.showStatusMessage(f"{status_past_tense} {count} unadjudicated lines.", 3000)

    def onShowInvalidOrDuplicateToggled(self, checked):
        if self._parameterNode:
            self._parameterNode.showInvalidAndDuplicate = checked
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
            if status in ("invalidated", "duplicated"):
                displayNode = markupNode.GetDisplayNode()
                if displayNode:
                    if checked:
                        displayNode.SetVisibility(True)
                        displayNode.SetOpacity(0.3)
                    else:
                        displayNode.SetVisibility(False)
        slicer.util.showStatusMessage(f"{'Showing' if checked else 'Hiding'} invalidated/duplicated lines.", 3000)

    def onAdjudicatorModeToggled(self, enabled):
        if self._parameterNode:
            self._parameterNode.adjudicatorMode = enabled
        # Show/hide the adjudication tools section
        if hasattr(self.ui, 'adjudicationToolsCollapsibleButton'):
            self.ui.adjudicationToolsCollapsibleButton.setVisible(enabled)
            if enabled:
                # Expand adjudication tools when entering adjudicator mode
                self.ui.adjudicationToolsCollapsibleButton.collapsed = False
        # Show/hide the sector annotations section (line creation tools)
        if hasattr(self.ui, 'sectorAnnotationsCollapsibleButton'):
            if enabled:
                # Collapse sector annotations when entering adjudicator mode
                self.ui.sectorAnnotationsCollapsibleButton.collapsed = True
            else:
                # Expand sector annotations when exiting adjudicator mode
                self.ui.sectorAnnotationsCollapsibleButton.collapsed = False
        if self.logic:
            self._updateMarkupsAndOverlayProgrammatically()

    def onAdjudicatorModeCheckBoxToggled(self, enabled):
        """
        Called when the user manually toggles the adjudicator mode checkbox.
        This resets the user manual state to allow programmatic control.
        """
        # Reset user manual state flag and last manual state when user manually changes adjudicator mode
        self._userManuallySetRaterTableState = False
        self._lastUserManualCollapsedState = None
        # Call the regular toggle method
        self.onAdjudicatorModeToggled(enabled)

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
        self.logic.updateLineMarkups()
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

    def selectNextUnadjudicatedLine(self):
        self._selectLineByFilter(self._isUnadjudicated, "next", "No unadjudicated lines in this frame.", "Selected unadjudicated line")

    def selectPreviousUnadjudicatedLine(self):
        self._selectLineByFilter(self._isUnadjudicated, "previous", "No unadjudicated lines in this frame.", "Selected unadjudicated line")

    def selectNextVisibleLine(self):
        self._selectLineByFilter(lambda node: True, "next", "No visible lines in this frame.", "Selected visible line")

    def selectPreviousVisibleLine(self):
        self._selectLineByFilter(lambda node: True, "previous", "No visible lines in this frame.", "Selected visible line")

    def _getActiveSequenceBrowserNode(self):
        """Return the active sequence browser node, even if the toolbar is focused on a sequence node."""
        node = slicer.modules.sequences.toolBar().activeBrowserNode()
        if node is None:
            return None
        # If it's already a browser node, return it
        if isinstance(node, slicer.vtkMRMLSequenceBrowserNode):
            return node
        # Otherwise, find the browser node that references this sequence node
        sequenceBrowsers = slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode")
        for browser in sequenceBrowsers:
            collection = vtk.vtkCollection()
            browser.GetSynchronizedSequenceNodes(collection, True)
            if collection.IsItemPresent(node):
                return browser
        return None

    def _navigateToFrameInSequence(self, target_frame_calculator, already_at_message):
        """
        Generic frame navigation method that eliminates code duplication.

        Args:
            target_frame_calculator: Function that takes (current_index, max_index) and returns target_index
            already_at_message: Status message to show when already at the target position
        """
        activeBrowserNode = self._getActiveSequenceBrowserNode()
        if activeBrowserNode:
            currentIndex = activeBrowserNode.GetSelectedItemNumber()
            maxIndex = activeBrowserNode.GetNumberOfItems() - 1

            targetIndex = target_frame_calculator(currentIndex, maxIndex)

            if targetIndex != currentIndex:
                activeBrowserNode.SetSelectedItemNumber(targetIndex)
                # Reset selected node ID when changing frames
                selectionNode = slicer.app.applicationLogic().GetSelectionNode()
                if selectionNode:
                    selectionNode.SetActivePlaceNodeID("")
            else:
                slicer.util.mainWindow().statusBar().showMessage(already_at_message, 3000)

    def _nextFrameInSequence(self):
        """Go to next frame in the current sequence using Slicer's built-in sequence browser."""
        def next_target(current, max_index):
            return current + 1 if current < max_index else current

        self._navigateToFrameInSequence(next_target, '⚠️ Already at last frame')

    def _previousFrameInSequence(self):
        """Go to previous frame in the current sequence using Slicer's built-in sequence browser."""
        def previous_target(current, max_index):
            return current - 1 if current > 0 else current

        self._navigateToFrameInSequence(previous_target, '⚠️ Already at first frame')

    def _firstFrameInSequence(self):
        """Go to the first frame in the current sequence."""
        def first_target(current, max_index):
            return 0 if current > 0 else current

        self._navigateToFrameInSequence(first_target, '⚠️ Already at first frame')

    def _lastFrameInSequence(self):
        """Go to the last frame in the current sequence."""
        def last_target(current, max_index):
            return max_index if current < max_index else current

        self._navigateToFrameInSequence(last_target, '⚠️ Already at last frame')

    def _togglePlayPauseSequence(self):
        """Toggle play/pause for the current sequence browser."""
        activeBrowserNode = self._getActiveSequenceBrowserNode()
        if activeBrowserNode:
            isPlaying = activeBrowserNode.GetPlaybackActive()
            activeBrowserNode.SetPlaybackActive(not isPlaying)

    def _setRedViewFocus(self):
        """Set focus to the red view to ensure keyboard shortcuts work immediately."""
        # Use a timer to delay focus setting to ensure all UI updates are complete
        qt.QTimer.singleShot(200, self._delayedSetRedViewFocus)

    def _delayedSetRedViewFocus(self):
        """Delayed focus setting to ensure all UI updates are complete."""
        try:
            # Since shortcuts are connected to main window, focus on that
            mainWindow = slicer.util.mainWindow()
            if mainWindow:
                mainWindow.activateWindow()
                mainWindow.setFocus()
                mainWindow.raise_()

            # Also try setting focus to the module widget itself
            if hasattr(self, 'parent') and self.parent:
                self.parent.setFocus()

        except Exception as e:
            logging.warning(f"Could not set focus: {e}")

    def _forceShortcutsActive(self):
        """Force keyboard shortcuts to be active by temporarily disconnecting and reconnecting them."""
        try:
            # Temporarily disconnect and reconnect shortcuts to force them to be active
            self.disconnectKeyboardShortcuts()
            qt.QTimer.singleShot(50, self.connectKeyboardShortcuts)
        except Exception as e:
            logging.warning(f"Could not force shortcuts active: {e}")

    def _navigateToClip(self, direction):
        """Helper method to navigate to previous or next clip."""
        direction_text = "previous" if direction == "previous" else "next"
        slicer.util.mainWindow().statusBar().showMessage(f'Loading {direction_text} clip...', 2000)
        if direction == "previous":
            self.onPreviousButton()
        else:
            self.onNextButton()

    def _onPageUpPressed(self):
        """Handle Page Up press for previous clip."""
        self._navigateToClip("previous")

    def _onPageDownPressed(self):
        """Handle Page Down press for next clip."""
        self._navigateToClip("next")

    def _onPreviousClipPressed(self):
        """Handle Shift+Up or Ctrl+Up press for previous clip."""
        self._navigateToClip("previous")

    def _onNextClipPressed(self):
        """Handle Shift+Down or Ctrl+Down press for next clip."""
        self._navigateToClip("next")

    def onSequenceBrowserModified(self, caller, event):
        """Handle sequence browser modifications (e.g., frame navigation via Slicer UI)."""
        # Reset selected node ID when sequence browser changes (UI state management)
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        if selectionNode:
            selectionNode.SetActivePlaceNodeID("")

        # Call Logic class to update markups (data processing)
        if self.logic:
            self.logic._updateMarkupsAndOverlayProgrammatically(None)

    def onShowHideLines(self, checked=None):
        """Toggle visibility of all lines and overlays."""
        if checked is None:
            # Toggle button state
            self.ui.showHideLinesButton.setChecked(not self.ui.showHideLinesButton.isChecked())
            checked = self.ui.showHideLinesButton.isChecked()
        # Set visibility of all lines
        for node in self.logic.pleuraLines + self.logic.bLines:
            displayNode = node.GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility(checked)
        # Also toggle overlay visibility
        self.ui.overlayVisibilityButton.setChecked(checked)

    def onResetAllAdjudication(self):
        """Reset all lines in the current frame to unadjudicated status."""
        for node in self.logic.pleuraLines + self.logic.bLines:
            displayNode = node.GetDisplayNode()
            if displayNode and displayNode.GetVisibility():
                node.SetAttribute("validation", json.dumps({"status": "unadjudicated"}))
                displayNode.SetOpacity(1.0)
        self.logic.updateCurrentFrame()
        self._parameterNode.unsavedChanges = True
        slicer.util.showStatusMessage("All lines reset to unadjudicated.", 3000)

#
# AnnotateUltrasoundLogic
#

class AnnotateUltrasoundLogic(ScriptedLoadableModuleLogic, VTKObservationMixin):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

        # These variables keep their values when the scene is cleared
        self.dicomDf = None
        self.nextDicomDfIndex = 0

        # These variables need to be reset when the scene is cleared
        self.annotations = None
        self.pleuraLines = []
        self.bLines = []
        self.sequenceBrowserNode = None
        self.depthGuideMode = 1
        logging.debug(f"Initialized depthGuideMode to {self.depthGuideMode}")
        self.parameterNode = self._getOrCreateParameterNode()

        # Flag to track when we're doing programmatic updates (to avoid setting unsavedChanges)
        self._isProgrammaticUpdate = False

    # Static variable to track seen raters and their order
    seenRaters = []

    def _getOrCreateParameterNode(self):
        if not hasattr(self, "parameterNode"):
            self.parameterNode = AnnotateUltrasoundParameterNode(super().getParameterNode())
        return self.parameterNode

    def getParameterNode(self):
        return self.parameterNode

    def getColorsForRater(self, rater: str):
        """
        Assign unique, visually distinct colors for pleura and b-lines per rater.
        Each rater gets completely unique colors that are distinct both within the rater and between raters.
        Uses golden ratio distribution starting from green (pleura) and blue (b-lines).
        """

        rater = rater.strip().lower()
        current_rater = self.getParameterNode().rater.strip().lower()

        if rater not in self.seenRaters and rater != '':
            self.seenRaters.append(rater)
            self.seenRaters.sort()

        if current_rater not in self.seenRaters:
            self.seenRaters.append(current_rater)
            self.seenRaters.sort()

        raters = self.seenRaters

        if rater not in raters:
            return [1.0, 0.0, 0.0], [1.0, 0.5, 0.0]  # fallback red/orange

        # Current rater gets green + blue
        if rater == current_rater:
            pleura_rgb = colorsys.hsv_to_rgb(0.33, 0.95, 1.0)  # bright green
            bline_rgb = colorsys.hsv_to_rgb(0.66, 0.95, 1.0)   # bright blue
            return list(pleura_rgb), list(bline_rgb)

        # Count all raters (excluding current rater)
        regular_raters = [r for r in raters if r != current_rater]
        N = len(regular_raters)

        if N == 0:
            return [1.0, 0.0, 0.0], [1.0, 0.5, 0.0]  # fallback red/orange

        # Find the index of this rater among all non-current raters
        rater_index = regular_raters.index(rater)

        # Use golden ratio for non-repeating distribution
        # φ = (1 + √5) / 2 ≈ 1.618033988749895
        golden_ratio = (1 + 5**0.5) / 2

        # Start from positions after green and blue to avoid conflicts with current rater
        pleura_start = (0.33 + golden_ratio) % 1.0  # Start after green
        bline_start = (0.66 + golden_ratio) % 1.0   # Start after blue

        # Generate hues by adding golden ratio steps from the starting points
        pleura_hue = (pleura_start + rater_index * golden_ratio) % 1.0
        bline_hue = (bline_start + rater_index * golden_ratio) % 1.0

        # Ensure pleura and b-line colors are distinct by adjusting saturation and value
        pleura_rgb = colorsys.hsv_to_rgb(pleura_hue, 0.95, 1.0)  # bright colors
        bline_rgb = colorsys.hsv_to_rgb(bline_hue, 0.95, 1.0)   # bright colors
        return list(pleura_rgb), list(bline_rgb)

    def getAllRaterColors(self):
        """
        Returns a list of (rater, (pleura_color, bline_color)) for all seen raters.
        """
        colors = []
        for r in self.seenRaters:
            pleura_color, bline_color = self.getColorsForRater(r)
            colors.append((r, (pleura_color, bline_color)))
        return colors

    def setSelectedRaters(self, raters: set):
        """
        Store the selected raters and filter visuals accordingly.
        """
        self.selectedRaters = set(raters)

    def getSelectedRaters(self):
        if hasattr(self, "selectedRaters"):
            return self.selectedRaters
        return None

    def setRater(self, value):
        node = self.getParameterNode()
        wasModifying = node.StartModify()
        node.rater = value.strip().lower()
        node.EndModify(wasModifying)

    def updateInputDf(self, rater, input_folder):
        """
        Update the dicomDf dataframe with the DICOM files in the input folder.

        :param input_folder: The input folder to search for DICOM files.
        :return: The number of rows in the dataframe and the number of annotations files created.
        """
        dicom_data = []

        # Get the total number of files
        total_files = sum([len(files) for root, dirs, files in os.walk(input_folder)])

        # Create a QProgressDialog
        progress_dialog = qt.QProgressDialog("Parsing DICOM files...", "Cancel", 0, total_files)
        progress_dialog.setWindowModality(qt.Qt.WindowModal)
        progress_dialog.show()

        # Recursively walk through the input folder
        file_count = 0
        annotations_created_count = 0
        for root, dirs, files in os.walk(input_folder):
            files.sort()
            for file in files:
                progress_dialog.setValue(file_count)
                slicer.app.processEvents()

                # Construct the full file path
                file_path = os.path.join(root, file)
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

                    base_filename = os.path.splitext(os.path.join(root, file))[0]
                    candidates = [
                        f"{base_filename}.{rater}.json",
                        f"{base_filename}.json"
                    ]
                    annotation_path = None
                    for candidate in candidates:
                        if os.path.exists(candidate):
                            annotation_path = candidate
                            break
                    # Now: select which file to use or create/copy
                    if annotation_path is None:
                        annotations_file_path = f"{base_filename}.{rater}.json"
                    elif annotation_path.endswith(f".{rater}.json"):
                        annotations_file_path = annotation_path
                    else:
                        # this will just be ${base_filename}.json
                        annotations_file_path = annotation_path

                    # Append the information to the list, if PatientID, StudyInstanceUID, and SeriesInstanceUID are present
                    if patient_uid and study_uid and series_uid and instance_uid:
                        dicom_data.append([file_path, annotations_file_path, patient_uid, study_uid, series_uid, instance_uid])
                except Exception as e:
                    # If the file is not a valid DICOM file, continue to the next file
                    continue

        # Update dicomDf
        self.dicomDf = pd.DataFrame(dicom_data, columns=['Filepath', 'AnnotationsFilepath', 'PatientUID', 'StudyUID', 'SeriesUID', 'InstanceUID'])
        self.nextDicomDfIndex = 0

        # Close the progress dialog
        progress_dialog.setValue(total_files)
        progress_dialog.close()

        # Return the number of rows in the dataframe and the number of annotations files created
        return len(self.dicomDf), annotations_created_count

    def updateCurrentFrame(self):
        if self.sequenceBrowserNode is None:
            logging.warning("No sequence browser node found, cannot update current frame.")
            return

        currentFrameIndex = self.sequenceBrowserNode.GetSelectedItemNumber()
        if self.annotations is None:
            self.annotations = {"frame_annotations": []}

        # Find or create the frame entry in annotations
        frame_data = next((item for item in self.annotations['frame_annotations'] if int(item.get("frame_number", -1)) == currentFrameIndex), None)
        if frame_data is None:
            frame_data = {"frame_number": currentFrameIndex, "pleura_lines": [], "b_lines": []}
            self.annotations['frame_annotations'].append(frame_data)

        # This set will keep track of which raters we have updated from the scene
        processed_raters = set()

        # Update lists directly from visible markups
        new_pleura_lines = []
        for node in self.pleuraLines:
            if node.GetDisplayNode() and node.GetDisplayNode().GetVisibility() and node.GetNumberOfControlPoints() > 0:
                rater = node.GetAttribute("rater")
                processed_raters.add(rater)
                points = []
                for i in range(node.GetNumberOfControlPoints()):
                    coord = [0, 0, 0]
                    node.GetNthControlPointPosition(i, coord)
                    points.append(coord)

                validation_json = node.GetAttribute("validation")
                try:
                    validation = json.loads(validation_json) if validation_json else {"status": "unadjudicated"}
                except json.JSONDecodeError:
                    validation = {"status": "unadjudicated"}

                new_pleura_lines.append({
                    "rater": rater,
                    "line": {"points": points},
                    "validation": validation,
                })

        new_b_lines = []
        for node in self.bLines:
            if node.GetDisplayNode() and node.GetDisplayNode().GetVisibility() and node.GetNumberOfControlPoints() > 0:
                rater = node.GetAttribute("rater")
                processed_raters.add(rater)
                points = []
                for i in range(node.GetNumberOfControlPoints()):
                    coord = [0, 0, 0]
                    node.GetNthControlPointPosition(i, coord)
                    points.append(coord)

                validation_json = node.GetAttribute("validation")
                try:
                    validation = json.loads(validation_json) if validation_json else {"status": "unadjudicated"}
                except json.JSONDecodeError:
                    validation = {"status": "unadjudicated"}

                new_b_lines.append({
                    "rater": rater,
                    "line": {"points": points},
                    "validation": validation,
                })

        # Now, add back the lines from raters that were not visible
        for line in frame_data.get('pleura_lines', []):
            if line.get('rater') not in processed_raters:
                new_pleura_lines.append(line)

        for line in frame_data.get('b_lines', []):
            if line.get('rater') not in processed_raters:
                new_b_lines.append(line)

        # Replace the old lists with the new, combined lists
        frame_data['pleura_lines'] = new_pleura_lines
        frame_data['b_lines'] = new_b_lines

        # Update the markups in the scene to match the annotation data
        self.updateLineMarkups()

        # At the end of updateCurrentFrame, always update overlays
        self.updateOverlayVolume()

    def removeFrame(self, frameIndex):
        logging.info(f"removeFrame -- frameIndex: {frameIndex}")
        if self.annotations is None:
            logging.warning("No annotations loaded")
            return

        # Remove the frame index from the list of frame annotations
        self.annotations["frame_annotations"] = [
            fa for fa in self.annotations.get("frame_annotations", [])
            if int(fa.get("frame_number", -1)) != frameIndex
        ]


    def loadPreviousSequence(self):
        if self.dicomDf is None:
            return None

        if self.nextDicomDfIndex <= 1:
            return None
        else:
            self.nextDicomDfIndex -= 2
            return self.loadNextSequence()

    def clearScene(self):
        slicer.mrmlScene.Clear(0)
        self.annotations = None
        self.pleuraLines = []
        self.bLines = []
        self.sequenceBrowserNode = None

    def convert_lps_to_ras(self, annotations: list):
        for frame in annotations:
            if frame.get("coordinate_space", "RAS") == "LPS":
                for line_group in ["pleura_lines", "b_lines"]:
                    for entry in frame.get(line_group, []):
                        points = entry["line"]["points"]
                        for point in points:
                            point[0] = -point[0]  # Negate X (Left → Right)
                            point[1] = -point[1]  # Negate Y (Posterior → Anterior)
                frame["coordinate_space"] = "RAS"  # Update coordinate_space

    # Use deepcopy because in-memory data should not be changed to LPS
    def convert_ras_to_lps(self, annotated_frames: list):
        # deepcopy so that modifications do not affect self.annotations
        save_data = copy.deepcopy(self.annotations)
        # deepcopy so that changes to the annotated frames don't actually affect the frames passed in
        copied_frames = copy.deepcopy(annotated_frames)
        for frame in copied_frames:
            if frame.get("coordinate_space", "RAS") == "RAS":
                for line_group in ["pleura_lines", "b_lines"]:
                    for entry in frame.get(line_group, []):
                        points = entry["line"]["points"]
                        for point in points:
                            point[0] = -point[0]  # Negate X (Right → Left)
                            point[1] = -point[1]  # Negate Y (Anterior → Posterior)
                frame["coordinate_space"] = "LPS"  # Update coordinate_space
        save_data['frame_annotations'] = copied_frames
        return save_data # a copy of the data, so caller has to save

    def _updateMarkupsAndOverlayProgrammatically(self, parameterNode=None):
        if parameterNode is None:
            parameterNode = self.getParameterNode()
        self._isProgrammaticUpdate = True
        try:
            self.updateLineMarkups()
            ratio = self.updateOverlayVolume()
            if ratio is not None:
                parameterNode.pleuraPercentage = ratio * 100
            else:
                parameterNode.pleuraPercentage = 0.0
        finally:
            self._isProgrammaticUpdate = False

    def loadNextSequence(self):
        """
        Load the next sequence in the dataframe.
        Returns the index of the loaded sequence in the dataframe or None if no more sequences are available.
        """
        # Save current depth guide and adjudicator modes
        currentDepthGuideMode = self.depthGuideMode
        parameterNode = self.getParameterNode()
        currentAdjudicatorMode = parameterNode.adjudicatorMode
        logging.debug(f"Saving depthGuideMode {currentDepthGuideMode} and adjudicatorMode {currentAdjudicatorMode} before loading next sequence")

        # Clear the scene
        self.clearScene()
        if not parameterNode:
            logging.error("No parameter node found, cannot load next sequence.")
            return None

        if self.dicomDf is None:
            parameterNode.dfLoaded = False
            return None
        else:
            parameterNode.dfLoaded = True

        if self.nextDicomDfIndex >= len(self.dicomDf):
            return None

        # We advance the index here, so for loading annotations we use `self.nextDicomDfIndex` which was just incremented
        self.nextDicomDfIndex += 1

        # Make sure a temporary folder for the DICOM files exists
        tempDicomDir = slicer.app.temporaryPath + '/AnonymizeUltrasound'
        if not os.path.exists(tempDicomDir):
            os.makedirs(tempDicomDir)

        # Delete all files in the temporary folder
        for file in os.listdir(tempDicomDir):
            os.remove(os.path.join(tempDicomDir, file))

        # Get the filepath for the current index
        nextDicomFilepath = self.dicomDf.iloc[self.nextDicomDfIndex - 1]['Filepath']
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

        # Restore depth guide and adjudicator modes
        self.depthGuideMode = currentDepthGuideMode
        parameterNode.adjudicatorMode = currentAdjudicatorMode
        logging.debug(f"Restored depthGuideMode to {self.depthGuideMode} and adjudicatorMode to {currentAdjudicatorMode} after loading sequence")

        ultrasoundArray = slicer.util.arrayFromVolume(inputUltrasoundNode)
        # Mask array should be the same size as the ultrasound array, but with 3 channels
        maskArray = np.zeros([1, ultrasoundArray.shape[1], ultrasoundArray.shape[2], 3], dtype=np.uint8)

        # Initialize the mask volume to be the same size as the ultrasound volume but with all voxels set to 0
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
        slicer.util.updateVolumeFromArray(overlayVolume, maskArray)
        parameterNode.overlayVolume = overlayVolume

        # Load annotations using the dedicated method. The adjudicator mode flag comes from the widget's parameter node.
        self.loadAnnotations(self.nextDicomDfIndex - 1, self.parameterNode.adjudicatorMode)

        parameterNode.EndModify(previousNodeState)

        # Set overlay volume as foreground in slice viewers
        redSliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceCompositeNode()
        redSliceCompositeNode.SetForegroundVolumeID(overlayVolume.GetID())
        redSliceCompositeNode.SetForegroundOpacity(0.12)
        redSliceCompositeNode.SetCompositing(2)

        displayNode = overlayVolume.GetDisplayNode()
        displayNode.SetWindow(255)
        displayNode.SetLevel(127)

        # Return the index of the loaded sequence in the dataframe
        return self.nextDicomDfIndex

    def loadAnnotations(self, dicom_index, adjudicator_mode):
        """
        Load annotations for the given dicom_index.
        In adjudicator_mode, it loads the '.adjudication.json' file if it exists.
        Otherwise, it merges all rater files.
        Sets self.annotations.
        """
        base_file_path = self.dicomDf.iloc[dicom_index]['Filepath']
        dir_path = os.path.dirname(base_file_path)
        base_prefix = os.path.splitext(base_file_path)[0]

        adjudication_file = None
        adjudication_file_correct = f"{base_prefix}.adjudication.json"
        adjudication_file_typo = f"{base_prefix}.adjucator.json" # Common typo

        # A more robust check for the file's existence, bypassing os.path.exists which can fail in some environments.
        try:
            dir_contents = os.listdir(dir_path)
            base_correct = os.path.basename(adjudication_file_correct)
            base_typo = os.path.basename(adjudication_file_typo)

            if base_correct in dir_contents:
                adjudication_file = adjudication_file_correct
            elif base_typo in dir_contents:
                adjudication_file = adjudication_file_typo
                logging.warning(f"Found misspelled adjudication file: {adjudication_file_typo}. Using it, but please consider renaming.")
        except OSError as e:
            logging.error(f"Could not read directory {dir_path}: {e}")
            # Fallback to the original check, just in case
            if os.path.exists(adjudication_file_correct):
                 adjudication_file = adjudication_file_correct
            elif os.path.exists(adjudication_file_typo):
                 adjudication_file = adjudication_file_typo

        # Always clear existing annotations before loading new ones
        self.annotations = {"frame_annotations": []}

        if adjudicator_mode and adjudication_file:
            logging.info(f"Adjudicator mode: Loading single adjudication file: {adjudication_file}")
            try:
                with open(adjudication_file, 'r') as f:
                    # Directly load the adjudicated data, which includes fan mask info
                    self.annotations = json.load(f)

                    if "frame_annotations" in self.annotations:
                        self.convert_lps_to_ras(self.annotations.get("frame_annotations", []))
                    else:
                        # ensure frame_annotations key exists even if file is empty or malformed
                        self.annotations["frame_annotations"] = []
            except Exception as e:
                logging.error(f"Failed to load adjudication file {adjudication_file}: {e}")
        else:
            logging.info("Rater mode or no adjudication file: Merging all rater annotation files.")
            merged_data = {"frame_annotations": []}
            is_first_file = True
            # Find all json files for this scan, excluding any adjudication files
            # Reverted to a simpler filter that was known to work, but handles both spellings.
            glob_files = glob.glob(f"{base_prefix}*.json")
            filepaths = [f for f in glob_files if not f.endswith('.adjudication.json') and not f.endswith('.adjucator.json')]

            for filepath in filepaths:
                try:
                    with open(filepath, 'r') as f:
                        ann = json.load(f)

                        # Copy top-level keys (like fan mask info) from the first file that has them
                        if is_first_file:
                            for key, value in ann.items():
                                if key != "frame_annotations":
                                    merged_data[key] = value
                            if any(k in ann for k in ["mask_type", "radius1"]): # Heuristic to check if fan info was found
                                is_first_file = False

                        # Individual rater files are dicts with a "frame_annotations" key
                        frame_annotations = ann.get("frame_annotations", [])
                        self.convert_lps_to_ras(frame_annotations)
                        for frame in frame_annotations:
                            frame_number = frame.get("frame_number")
                            if frame_number is None:
                                continue

                            matched_frame = next((f for f in merged_data["frame_annotations"] if f.get("frame_number") == frame_number), None)
                            if matched_frame:
                                matched_frame["pleura_lines"].extend(frame.get("pleura_lines", []))
                                matched_frame["b_lines"].extend(frame.get("b_lines", []))
                            else:
                                merged_data["frame_annotations"].append(frame)
                except Exception as e:
                    logging.warning(f"Failed to load or merge annotation file {filepath}: {e}")
            self.annotations = merged_data

        # Extract and set up raters using the centralized method
        self.extractAndSetupRaters()

        # Preallocate markup nodes based on loaded annotations
        self.preallocateMarkupNodesFromAnnotations()
        # Set programmatic update flag to prevent unsavedChanges from being set
        self._updateMarkupsAndOverlayProgrammatically(None)

    def createMarkupLine(self, name, rater, coordinates, color=[1, 1, 0], validation=None):
        markupNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
        markupNode.CreateDefaultDisplayNodes()
        markupNode.SetName(name)
        # Always set rater attribute
        markupNode.SetAttribute("rater", rater)
        if validation is not None:
            markupNode.SetAttribute("validation", json.dumps(validation))
        markupNode.GetDisplayNode().SetPropertiesLabelVisibility(False)
        markupNode.GetDisplayNode().SetSelectedColor(color)
        for coord in coordinates:
            markupNode.AddControlPointWorld(coord[0], coord[1], coord[2])
        self.addObserver(markupNode, markupNode.PointModifiedEvent, self.onPointModified)
        self.addObserver(markupNode, markupNode.PointPositionDefinedEvent, self.onPointPositionDefined)
        return markupNode

    def clearSceneLines(self):
        """
        Remove all pleura lines and B-lines from the scene and from the list of lines.
        """
        # Remove all pleura lines
        while len(self.pleuraLines) > 0:
            self.removeLastPleuraLine()

        # Remove all B-lines
        while len(self.bLines) > 0:
            self.removeLastBline()

    def clearAllLines(self):
        """
        Remove all pleura lines and B-lines from the scene and from the list of lines.
        Only updates the annotation if the current frame is already in the annotations.
        """
        self.clearSceneLines()
        # Only update annotation if current frame is already present
        if self.sequenceBrowserNode is not None and self.annotations is not None and 'frame_annotations' in self.annotations:
            currentFrameIndex = max(0, self.sequenceBrowserNode.GetSelectedItemNumber())
            if any(int(f.get("frame_number", -1)) == currentFrameIndex for f in self.annotations["frame_annotations"]):
                self.updateCurrentFrame()

    def removeLastPleuraLine(self):
        """
        Remove the last pleura line from the scene and from the list of pleura lines.
        """
        if len(self.pleuraLines) > 0:
            currentLine = self.pleuraLines.pop()
            self.removeObserver(currentLine, currentLine.PointModifiedEvent, self.onPointModified)
            if self.hasObserver(currentLine, currentLine.PointPositionDefinedEvent, self.onPointPositionDefined):
                self.removeObserver(currentLine, currentLine.PointPositionDefinedEvent, self.onPointPositionDefined)
            slicer.mrmlScene.RemoveNode(currentLine)
            ratio = self.updateOverlayVolume()
            if ratio is not None:
                parameterNode = self.getParameterNode()
                parameterNode.pleuraPercentage = ratio * 100

    def removeLastBline(self):
        """
        Remove the last B-line from the scene and from the list of B-lines.
        """
        if len(self.bLines) > 0:
            currentLine = self.bLines.pop()
            self.removeObserver(currentLine, currentLine.PointModifiedEvent, self.onPointModified)
            if self.hasObserver(currentLine, currentLine.PointPositionDefinedEvent, self.onPointPositionDefined):
                self.removeObserver(currentLine, currentLine.PointPositionDefinedEvent, self.onPointPositionDefined)
            slicer.mrmlScene.RemoveNode(currentLine)
            ratio = self.updateOverlayVolume()
            if ratio is not None:
                parameterNode = self.getParameterNode()
                parameterNode.pleuraPercentage = ratio * 100

    def onPointModified(self, caller, event):
        parameterNode = self.getParameterNode()
        ratio = self.updateOverlayVolume()
        if ratio is not None:
            parameterNode.pleuraPercentage = ratio * 100

        # Only set unsavedChanges if this is a user-initiated modification
        if not self._isProgrammaticUpdate:
            parameterNode.unsavedChanges = True

    def onPointPositionDefined(self, caller, event):
        parameterNode = self.getParameterNode()
        numControlPoints = caller.GetNumberOfControlPoints()
        if numControlPoints >= 2:
            parameterNode.lineBeingPlaced = None
            self.removeObserver(caller, caller.PointPositionDefinedEvent, self.onPointPositionDefined)

        ratio = self.updateOverlayVolume()
        if ratio is not None:
            parameterNode.pleuraPercentage = ratio * 100

        # Set unsavedChanges when user finishes placing a line (only if not programmatic)
        if not self._isProgrammaticUpdate:
            parameterNode.unsavedChanges = True

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

    def _updateMarkupNode(self, node, entry, selectedNodeID):
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
        status = validation.get("status", "unadjudicated")
        adjudicator_mode = getattr(self.parameterNode, 'adjudicatorMode', False)

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

        if adjudicator_mode and isSelected:
            # Apply selected highlighting using the __selected_node__ color
            selected_node_color, _ = self.getColorsForRater("__selected_node__")
            displayNode.SetSelectedColor(selected_node_color)
        else:
            # Apply colors based on mode and validation status
            if adjudicator_mode:
                if status == "unadjudicated":
                    # In adjudicator mode, unadjudicated lines use current rater's colors
                    current_rater = self.getCurrentRater()
                    if node in self.pleuraLines:
                        color_pleura, _ = self.getColorsForRater(current_rater)
                        displayNode.SetSelectedColor(color_pleura)
                    else:
                        _, color_bline = self.getColorsForRater(current_rater)
                        displayNode.SetSelectedColor(color_bline)
                else:
                    # In adjudicator mode, validated/invalidated/duplicated lines use __adjudicated_node__ color
                    if node in self.pleuraLines:
                        color_pleura, _ = self.getColorsForRater("__adjudicated_node__")
                        displayNode.SetSelectedColor(color_pleura)
                    else:
                        _, color_bline = self.getColorsForRater("__adjudicated_node__")
                        displayNode.SetSelectedColor(color_bline)
            else:
                # Use original rater's colors for validated/invalidated/duplicated lines or non-adjudicator mode
                if node in self.pleuraLines:
                    color_pleura, _ = self.getColorsForRater(rater)
                    displayNode.SetSelectedColor(color_pleura)
                else:
                    _, color_bline = self.getColorsForRater(rater)
                    displayNode.SetSelectedColor(color_bline)

        # Set visibility, opacity, and line style based on validation status and adjudicator mode
        status = validation.get("status", "unadjudicated")
        adjudicator_mode = getattr(self.parameterNode, 'adjudicatorMode', False)
        displayNode.SetGlyphTypeFromString("Circle2D")
        displayNode.SetGlyphScale(2.0)
        displayNode.SetLineThickness(0.25)
        if not adjudicator_mode:
            displayNode.SetVisibility(True)
            displayNode.SetOpacity(1.0)
        else:
            if status == "invalidated":
                showInvalidAndDuplicate = self.parameterNode.showInvalidAndDuplicate if self.parameterNode else True
                displayNode.SetVisibility(showInvalidAndDuplicate)
                displayNode.SetGlyphTypeFromString("Cross2D")
                displayNode.SetOpacity(0.40)
            elif status == "duplicated":
                showInvalidAndDuplicate = self.parameterNode.showInvalidAndDuplicate if self.parameterNode else True
                displayNode.SetVisibility(showInvalidAndDuplicate)
                displayNode.SetGlyphTypeFromString("Diamond2D")
                displayNode.SetOpacity(0.50)
            elif status == "unadjudicated":
                displayNode.SetVisibility(True)
                if not isSelected:
                    displayNode.SetOpacity(0.65)
            else:  # validated
                displayNode.SetVisibility(True)
                if not isSelected:
                    displayNode.SetOpacity(0.85)

            # Selected node highlighting is independent of validation status
            if isSelected:
                displayNode.SetGlyphScale(3.0)
                displayNode.SetOpacity(1.0)

        node.RemoveAllControlPoints()
        for pt in coordinates:
            node.AddControlPointWorld(*pt)

    def _updateMarkupNodesForFrame(self, frame):
        """
        Update markup nodes for pleura and b-lines for the given frame.
        """
        # Check if scene is valid before proceeding
        if not slicer.mrmlScene:
            return

        # Always filter by selectedRaters, even in adjudicator mode
        pleura_entries = [entry for entry in frame.get("pleura_lines", []) if entry.get("rater", "").strip().lower() in {r.strip().lower() for r in self.selectedRaters}]
        bline_entries = [entry for entry in frame.get("b_lines", []) if entry.get("rater", "").strip().lower() in {r.strip().lower() for r in self.selectedRaters}]

        # Get currently selected node ID for highlighting
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectedNodeID = selectionNode.GetActivePlaceNodeID() if selectionNode else None

        # Ensure enough markup nodes exist
        while len(self.pleuraLines) < len(pleura_entries):
            self.pleuraLines.append(self.createMarkupLine("Pleura", "", [], [1,1,0]))
        while len(self.bLines) < len(bline_entries):
            self.bLines.append(self.createMarkupLine("B-line", "", [], [0,1,1]))

        # Update pleura markups
        for i, entry in enumerate(pleura_entries):
            node = self.pleuraLines[i]
            self._updateMarkupNode(node, entry, selectedNodeID)
        # Hide unused pleura markups
        for i in range(len(pleura_entries), len(self.pleuraLines)):
            displayNode = self.pleuraLines[i].GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility(False)

        # Hide pleura markups for non-selected raters
        for node in self.pleuraLines:
            nodeRater = node.GetAttribute("rater")
            if nodeRater not in self.selectedRaters:
                displayNode = node.GetDisplayNode()
                if displayNode:
                    displayNode.SetVisibility(False)

        # Update b-line markups
        for i, entry in enumerate(bline_entries):
            node = self.bLines[i]
            self._updateMarkupNode(node, entry, selectedNodeID)
        # Hide unused b-line markups
        for i in range(len(bline_entries), len(self.bLines)):
            displayNode = self.bLines[i].GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility(False)

        # Hide b-line markups for non-selected raters
        for node in self.bLines:
            nodeRater = node.GetAttribute("rater")
            if nodeRater not in self.selectedRaters:
                displayNode = node.GetDisplayNode()
                if displayNode:
                    displayNode.SetVisibility(False)

    def updateLineMarkups(self):
        """
        Update the line markups to match the annotations at the current frame index. Reuse markups instead of deleting/creating them every frame. Only update if frame or annotation data has changed.
        """
        # Check if scene is valid before proceeding
        if not slicer.mrmlScene:
            return

        # Initialize cache attributes if not present
        if not hasattr(self, '_lastMarkupFrameIndex'):
            self._lastMarkupFrameIndex = None
        if not hasattr(self, '_lastMarkupFrameHash'):
            self._lastMarkupFrameHash = None

        if self.annotations is None:
            logging.warning("No annotations loaded")
            return

        if self.sequenceBrowserNode is None:
            logging.warning("No sequence browser node found")
            return

        currentFrameIndex = max(0, self.sequenceBrowserNode.GetSelectedItemNumber())

        if 'frame_annotations' not in self.annotations:
            logging.debug("No frame annotations found")
            return

        frame = next((item for item in self.annotations['frame_annotations'] if str(item.get("frame_number")) == str(currentFrameIndex)), None)
        # Compute a hash of the frame annotation data and selected raters for caching
        frame_hash = None
        if frame is not None:
            try:
                selectionNode = slicer.app.applicationLogic().GetSelectionNode()
                selectedNodeID = selectionNode.GetActivePlaceNodeID() if selectionNode else None
                # Include selected raters and selected node in the hash so selection changes trigger updates
                cache_data = {
                    'frame': frame,
                    'selectedRaters': sorted(list(self.selectedRaters)) if hasattr(self, 'selectedRaters') else [],
                    'selectedNodeID': selectedNodeID,
                    'adjudicatorMode': getattr(self.parameterNode, 'adjudicatorMode', False)
                }
                frame_hash = hash(json.dumps(cache_data, sort_keys=True))
            except Exception:
                frame_hash = None

        # Early return if frame index, annotation data, and selected raters are unchanged
        if self._lastMarkupFrameIndex == currentFrameIndex and self._lastMarkupFrameHash == frame_hash:
            return

        # Update cache after check
        self._lastMarkupFrameIndex = currentFrameIndex
        self._lastMarkupFrameHash = frame_hash

        # Set programmatic update flag to prevent unsavedChanges from being set
        self._isProgrammaticUpdate = True

        # Batch scene updates using StartState/EndState
        slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
        try:
            if frame is None:
                # Hide all markups if no frame data
                for node in self.pleuraLines:
                    displayNode = node.GetDisplayNode()
                    if displayNode:
                        displayNode.SetVisibility(False)
                for node in self.bLines:
                    displayNode = node.GetDisplayNode()
                    if displayNode:
                        displayNode.SetVisibility(False)
                return

            self._updateMarkupNodesForFrame(frame)
        finally:
            slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)
            # Reset programmatic update flag
            self._isProgrammaticUpdate = False

    def drawDepthGuideLine(self, image_size_rows, image_size_cols, depth_ratio=0.5, color=(0, 255, 255), thickness=4, dash_length=20, dash_gap=16):
        """
        Main function to handle different visualization modes for the depth guide.
        """
        # Extract fan parameters from annotations
        if "mask_type" not in self.annotations or self.annotations["mask_type"] != "fan":
            logging.error("No fan mask information available in annotations.")
            return np.zeros((image_size_rows, image_size_cols, 3), dtype=np.uint8)

        radius1 = self.annotations["radius1"]
        radius2 = self.annotations["radius2"]
        center_rows_px = self.annotations["center_rows_px"]
        center_cols_px = self.annotations["center_cols_px"]
        angle1 = self.annotations["angle1"]
        angle2 = self.annotations["angle2"]

        # Calculate the depth position
        depth_radius = radius1 + depth_ratio * (radius2 - radius1)

        # Scale dash parameters based on radius
        scale_factor = depth_radius / 500.0  # Use 500 as reference radius
        scaled_thickness = max(1, int(thickness * scale_factor))
        scaled_dash_length = int(dash_length * scale_factor)
        scaled_dash_gap = int(dash_gap * scale_factor)

        # Choose visualization based on depthGuideMode
        if self.depthGuideMode == 1:
            # Mode 1: Default dashed line
            return self._drawDashedLine(image_size_rows, image_size_cols, center_cols_px, center_rows_px,
                                        depth_radius, angle1, angle2, color, thickness=scaled_thickness,
                                        dash_length=scaled_dash_length, dash_gap=scaled_dash_gap)
        elif self.depthGuideMode == 2:
            # Mode 2: Thinner, more spaced dashed line
            return self._drawDashedLine(image_size_rows, image_size_cols, center_cols_px, center_rows_px,
                                        depth_radius, angle1, angle2, color, thickness=2, dash_length=10, dash_gap=24)
        elif self.depthGuideMode == 3:
            # Mode 3: Arrows at 50% depth
            return self._drawArrows(image_size_rows, image_size_cols, depth_radius, center_cols_px, center_rows_px, angle1, angle2)
        elif self.depthGuideMode == 4:
            # Mode 4: Translucent band
            return self._drawTranslucentBand(image_size_rows, image_size_cols, depth_radius, center_cols_px, center_rows_px, angle1, angle2, color)

        # Return blank image if no valid mode
        return np.zeros((image_size_rows, image_size_cols, 3), dtype=np.uint8)

    def _drawDashedLine(self, image_size_rows, image_size_cols, center_cols_px, center_rows_px,
                        depth_radius, angle1, angle2, color, thickness, dash_length, dash_gap):
        line_img = np.zeros((image_size_rows, image_size_cols, 3), dtype=np.uint8)

        # Ensure angle1 is always less than angle2
        if angle1 > angle2:
            angle1, angle2 = angle2, angle1

        theta_start = angle1
        theta_end = angle2
        theta_range = theta_end - theta_start
        arc_length = (theta_range * depth_radius * math.pi / 180)
        num_dashes = int(arc_length / (dash_length + dash_gap))

        if num_dashes <= 0:
            num_dashes = max(15, int(theta_range / 5))

        theta_step = theta_range / num_dashes

        for i in range(num_dashes):
            theta1 = math.radians(theta_start + i * theta_step)
            theta2 = math.radians(theta_start + (i * theta_step + dash_length / depth_radius * 180 / math.pi))
            start_point = (
                int(center_cols_px + depth_radius * math.cos(theta1)),
                int(center_rows_px + depth_radius * math.sin(theta1))
            )
            end_point = (
                int(center_cols_px + depth_radius * math.cos(theta2)),
                int(center_rows_px + depth_radius * math.sin(theta2))
            )
            line_img = cv2.line(line_img, start_point, end_point, color, thickness)

        return line_img

    def _drawArrows(self, image_size_rows, image_size_cols, depth_radius, center_cols_px, center_rows_px, angle1, angle2, thickness=4, color=(200, 255, 255)):
        """
        Draws two arrows outside the left and right edges of the ultrasound fan at the specified depth mark,
        aligned with the tangent of the fan's curve at the 50% depth mark.
        """
        line_img = np.zeros((image_size_rows, image_size_cols, 3), dtype=np.uint8)
        theta1 = math.radians(angle1)
        theta2 = math.radians(angle2)

        # Calculate tip positions at the specified depth
        left_tip_point = (
            int(center_cols_px + depth_radius * math.cos(theta1)),
            int(center_rows_px + depth_radius * math.sin(theta1))
        )
        right_tip_point = (
            int(center_cols_px + depth_radius * math.cos(theta2)),
            int(center_rows_px + depth_radius * math.sin(theta2))
        )

        # Calculate tangent vectors for arrow alignment
        tangent_offset = 40  # Adjust to control the distance for the start point
        tangent_angle_left = theta1 + math.pi / 2
        tangent_angle_right = theta2 - math.pi / 2

        left_start_point = (
            int(left_tip_point[0] + tangent_offset * math.cos(tangent_angle_left)),
            int(left_tip_point[1] + tangent_offset * math.sin(tangent_angle_left))
        )
        right_start_point = (
            int(right_tip_point[0] + tangent_offset * math.cos(tangent_angle_right)),
            int(right_tip_point[1] + tangent_offset * math.sin(tangent_angle_right))
        )

        # Draw the arrows aligned with the fan's curve
        line_img = cv2.arrowedLine(line_img, left_start_point, left_tip_point, color, thickness, tipLength=0.5)
        line_img = cv2.arrowedLine(line_img, right_start_point, right_tip_point, color, thickness, tipLength=0.5)

        return line_img

    def _drawTranslucentBand(self, image_size_rows, image_size_cols, depth_radius, center_cols_px, center_rows_px, angle1, angle2, color=(100, 255, 255)):
        """
        Draws a thin translucent band around the 50% depth line on the ultrasound fan.
        """
        # Create a blank image for the band
        band_img = np.zeros((image_size_rows, image_size_cols, 3), dtype=np.uint8)

        # Define a thin band around the 50% depth line by setting a small thickness
        band_thickness = 10  # Adjust to make the band thicker or thinner as desired
        inner_radius = int(depth_radius - band_thickness / 2)
        outer_radius = int(depth_radius + band_thickness / 2)

        # Draw the translucent band as an ellipse segment between angle1 and angle2
        cv2.ellipse(band_img, (center_cols_px, center_rows_px), (outer_radius, outer_radius),
                    0, angle1, angle2, color, -1)
        cv2.ellipse(band_img, (center_cols_px, center_rows_px), (inner_radius, inner_radius),
                    0, angle1, angle2, (0, 0, 0), -1)

        # Make the band semi-transparent by blending it with a background
        alpha = 0.4  # Opacity of the band; adjust as needed
        translucent_band = cv2.addWeighted(band_img, alpha, np.zeros_like(band_img), 1 - alpha, 0)

        return translucent_band

    def _applyDepthGuideToMask(self, maskArray, parameterNode=None):
        """
        Helper function to apply depth guide to a mask array if enabled.

        Args:
            maskArray: The mask or overlay array to apply depth guide to
            parameterNode: Optional parameter node. If not provided, will use self.getParameterNode()

                    Returns:
            The updated mask array with depth guide applied (if enabled)
        """
        if parameterNode is None:
            parameterNode = self.getParameterNode()

        if parameterNode.depthGuideVisible:
            ultrasoundArray = slicer.util.arrayFromVolume(parameterNode.inputVolume)
            image_size_rows = ultrasoundArray.shape[1]
            image_size_cols = ultrasoundArray.shape[2]
            depth_guide = self.drawDepthGuideLine(image_size_rows, image_size_cols)
            maskArray[0, :, :, :] = np.maximum(maskArray[0, :, :, :], depth_guide)

        return maskArray

    def updateOverlayVolume(self):
        """
        Update the overlay volume based on the annotations.

        :return: The ratio of green pixels to blue pixels in the overlay volume. None if inputs not defined yet.
        """
        parameterNode = self.getParameterNode()
        if parameterNode.overlayVolume is None:
            logging.debug("updateOverlayVolume: No overlay volume found! Cannot update overlay volume.")
            return None

        adjudicator_mode = getattr(parameterNode, 'adjudicatorMode', False)

        # If not in adjudicator mode and more than one rater is selected, hide overlay and pleura percentage
        if not adjudicator_mode and hasattr(self, "selectedRaters") and len(self.selectedRaters) > 1:
            overlayArray = slicer.util.arrayFromVolume(parameterNode.overlayVolume)
            overlayArray[:] = 0
            overlayArray = self._applyDepthGuideToMask(overlayArray, parameterNode)
            slicer.util.updateVolumeFromArray(parameterNode.overlayVolume, overlayArray)
            slicer.util.showStatusMessage("Overlay hidden: multiple raters selected", 3000)
            parameterNode.pleuraPercentage = -1.0  # Hide pleura percentage
            return None

        if self.annotations is None:
            logging.warning("updateOverlayVolume: No annotations loaded")
            # Make sure all voxels are set to 0
            parameterNode.overlayVolume.GetImageData().GetPointData().GetScalars().Fill(0)
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

        # Add pleura lines to mask array using full RGB overlay
        for markupNode in self.pleuraLines:
            nodeRater = markupNode.GetAttribute("rater") if markupNode else None

            # Only check validation status in adjudicator mode
            if adjudicator_mode:
                validation_json = markupNode.GetAttribute("validation")
                status = None
                if validation_json:
                    try:
                        validation = json.loads(validation_json)
                        status = validation.get("status", None)
                    except Exception:
                        status = None
                if status != "validated":
                    continue

            # Always filter by selectedRaters, even in adjudicator mode
            if not adjudicator_mode and hasattr(self, "selectedRaters") and self.selectedRaters and nodeRater not in self.selectedRaters:
                continue
            if not markupNode.GetDisplayNode().GetVisibility():
                continue
            for i in range(markupNode.GetNumberOfControlPoints() - 1):
                coord1 = [0, 0, 0]
                coord2 = [0, 0, 0]
                markupNode.GetNthControlPointPosition(i, coord1)
                markupNode.GetNthControlPointPosition(i + 1, coord2)
                coord1 = rasToIjk.MultiplyPoint(coord1 + [1])
                coord2 = rasToIjk.MultiplyPoint(coord2 + [1])
                coord1 = [int(round(coord1[0])), int(round(coord1[1])), int(round(coord1[2]))]
                coord2 = [int(round(coord2[0])), int(round(coord2[1])), int(round(coord2[2]))]
                # Draw mask fan between coord1 and coord2
                sectorArray = self.createSectorMaskBetweenPoints(ultrasoundArray, coord1, coord2, value=255)
                # Add sectorArray to maskArray by maximum compounding
                maskArray[0, :, :, 2] = np.maximum(maskArray[0, :, :, 2], sectorArray)

        # Add B-lines to mask array using full RGB overlay
        for markupNode in self.bLines:
            nodeRater = markupNode.GetAttribute("rater") if markupNode else None

            # Only check validation status in adjudicator mode
            if adjudicator_mode:
                validation_json = markupNode.GetAttribute("validation")
                status = None
                if validation_json:
                    try:
                        validation = json.loads(validation_json)
                        status = validation.get("status", None)
                    except Exception:
                        status = None
                if status != "validated":
                    continue

            # Always filter by selectedRaters, even in adjudicator mode
            if hasattr(self, "selectedRaters") and self.selectedRaters and nodeRater not in self.selectedRaters:
                continue
            if not markupNode.GetDisplayNode().GetVisibility():
                continue
            for i in range(markupNode.GetNumberOfControlPoints() - 1):
                coord1 = [0, 0, 0]
                coord2 = [0, 0, 0]
                markupNode.GetNthControlPointPosition(i, coord1)
                markupNode.GetNthControlPointPosition(i + 1, coord2)
                coord1 = rasToIjk.MultiplyPoint(coord1 + [1])
                coord2 = rasToIjk.MultiplyPoint(coord2 + [1])
                coord1 = [int(round(coord1[0])), int(round(coord1[1])), int(round(coord1[2]))]
                coord2 = [int(round(coord2[0])), int(round(coord2[1])), int(round(coord2[2]))]
                # Draw mask fan between coord1 and coord2
                sectorArray = self.createSectorMaskBetweenPoints(ultrasoundArray, coord1, coord2)
                # Add sectorArray to maskArray by maximum compounding
                maskArray[0, :, :, 1] = np.maximum(maskArray[0, :, :, 1], sectorArray)

        # Erase all B-lines pixels where there is no pleura line
        maskArray[0, :, :, 1] = np.where(maskArray[0, :, :, 2] == 0, 0, maskArray[0, :, :, 1])

        # Calculate the amount of blue pixels in maskArray and green pixels in maskArray
        bluePixels = np.count_nonzero(maskArray[0, :, :, 2])
        greenPixels = np.count_nonzero(maskArray[0, :, :, 1])

        # apply depthGuide if enabled
        maskArray = self._applyDepthGuideToMask(maskArray, parameterNode)

        # Compute a hash of the mask array for fast comparison
        mask_hash = zlib.crc32(maskArray.tobytes())
        if hasattr(self, '_lastOverlayMaskHash') and self._lastOverlayMaskHash == mask_hash:
            # No change, skip update
            if bluePixels == 0:
                parameterNode.pleuraPercentage = 0.0
                return 0.0
            else:
                parameterNode.pleuraPercentage = greenPixels / bluePixels * 100
                return greenPixels / bluePixels
        # Update the overlay volume
        slicer.util.updateVolumeFromArray(parameterNode.overlayVolume, maskArray)
        self._lastOverlayMaskHash = mask_hash

        # Return the ratio of green pixels to blue pixels
        if bluePixels == 0:
            parameterNode.pleuraPercentage = 0.0
            return 0.0
        else:
            parameterNode.pleuraPercentage = greenPixels / bluePixels * 100
            return greenPixels / bluePixels

    def dicomHeaderDictForBrowserNode(self, browserNode):
        """Return DICOM header for the given browser node"""
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

    def preallocateMarkupNodesFromAnnotations(self):
        """
        Preallocate markup nodes for pleura and B-lines based on the maximum needed in the loaded annotations.
        """
        if not hasattr(self, 'pleuraLines'):
            self.pleuraLines = []
        if not hasattr(self, 'bLines'):
            self.bLines = []
        max_pleura = 0
        max_blines = 0
        if self.annotations and 'frame_annotations' in self.annotations:
            for frame in self.annotations['frame_annotations']:
                pleura_count = len(frame.get('pleura_lines', []))
                bline_count = len(frame.get('b_lines', []))
                if pleura_count > max_pleura:
                    max_pleura = pleura_count
                if bline_count > max_blines:
                    max_blines = bline_count

        # Set programmatic update flag to prevent unsavedChanges from being set
        self._isProgrammaticUpdate = True
        try:
            # Preallocate pleura markup nodes
            while len(self.pleuraLines) < max_pleura:
                self.pleuraLines.append(self.createMarkupLine("Pleura", "", [], [1,1,0]))
            # Preallocate b-line markup nodes
            while len(self.bLines) < max_blines:
                self.bLines.append(self.createMarkupLine("B-line", "", [], [0,1,1]))
        finally:
            # Reset programmatic update flag
            self._isProgrammaticUpdate = False

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

    def extractAndSetupRaters(self):
        """
        Extract all unique raters from the loaded annotations and set up selectedRaters.
        This centralizes the rater extraction logic to avoid duplication.
        """
        # Extract all unique raters from the loaded annotations
        seenRaters = []
        if self.annotations and 'frame_annotations' in self.annotations:
            for frame in self.annotations['frame_annotations']:
                for line_type in ['pleura_lines', 'b_lines']:
                    for line in frame.get(line_type, []):
                        rater = line.get('rater')
                        if rater and rater not in seenRaters:
                            seenRaters.append(rater)

        parameterNode = self.getParameterNode()
        current_rater = parameterNode.rater.strip().lower()
        if current_rater in seenRaters:
            seenRaters.remove(current_rater)
        # Remove __selected_node__ if it exists (to avoid duplicates)
        if "__selected_node__" in seenRaters:
            seenRaters.remove("__selected_node__")
        # Remove __adjudicated_node__ if it exists (to avoid duplicates)
        if "__adjudicated_node__" in seenRaters:
            seenRaters.remove("__adjudicated_node__")
        # Remove current_rater if it exists (to avoid duplicates)
        if current_rater in seenRaters:
            seenRaters.remove(current_rater)
        # Now build the list: current_rater, __selected_node__, __adjudicated_node__then sorted rest
        # we need to add __selected_node__, __adjudicated_node__ to the list to ensure that the selected line is always visible and
        # uses a different color than the other lines
        self.seenRaters = [current_rater, "__selected_node__", "__adjudicated_node__"] + sorted(seenRaters)
        self.setSelectedRaters(set(self.seenRaters))

    def getCurrentRater(self):
        parameterNode = self.getParameterNode()
        if parameterNode and hasattr(parameterNode, 'rater'):
            return parameterNode.rater.strip().lower()
        return None

    def _resetMarkupCache(self):
        """
        Reset the markup cache to force immediate updates when validation status changes.
        """
        if hasattr(self, '_lastMarkupFrameIndex'):
            self._lastMarkupFrameIndex = None
        if hasattr(self, '_lastMarkupFrameHash'):
            self._lastMarkupFrameHash = None
