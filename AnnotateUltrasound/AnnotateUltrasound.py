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

class SliceViewEventFilter(qt.QObject):
    def __init__(self, parentWidget, sliceView):
        super().__init__()
        self.parentWidget = parentWidget
        self.sliceView = sliceView

    def eventFilter(self, obj, event):
        if type(self.parentWidget) is not AnnotateUltrasoundWidget:
            return False

        if event.type() == qt.QEvent.MouseButtonPress:
            self.sliceView.setFocus()
            modifiers = qt.QApplication.keyboardModifiers()
            if modifiers & (qt.Qt.MetaModifier | qt.Qt.ControlModifier):
                interactor = self.parentWidget.sliceView.interactor()
                x, y = interactor.GetEventPosition()
                if self.parentWidget.pickClosestLineNodeInSlice(x, y):
                    return True
                else:
                    return False
        return False

    def _isChildOf(self, child, parent):
        while child:
            if child is parent:
                return True
            child = child.parent()
        return False

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

#
# AnnotateUltrasoundWidget
#
# global singleton instance of the widget
annotateUltrasoundWidgetInstance = None
def getAnnotateUltrasoundWidget():
    """
    Get the singleton instance of the AnnotateUltrasoundWidget.
    """
    global annotateUltrasoundWidgetInstance
    if annotateUltrasoundWidgetInstance is None:
        raise RuntimeError("AnnotateUltrasoundWidget instance is not initialized")
    return annotateUltrasoundWidgetInstance

class AnnotateUltrasoundWidget(ScriptedLoadableModuleWidget, CustomObserverMixin, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """
    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        self.updatingGUI = False
        self._parameterNode = None

        self._parameterNodeGuiTag = None
        self.notEnteredYet = True
        self._lastFrameIndex = -1

        # Safeguard to preserve logic if set by subclass before super().__init__()
        if not hasattr(self, 'logic'):
            self.logic = None

        # Flag to track if this is the first load of DICOM data
        self._isFirstDicomLoad = True
        # Flag to prevent multiple updateCurrentFrame calls during line placement
        self._isUpdatingCurrentFrame = False

        # Flag to track if the user manually expanded the rater table
        self._userManuallySetRaterTableState = False
        self._lastUserManualCollapsedState = None  # Track the last state the user manually set

        # Flag to prevent rater table state changes during navigation
        self._isNavigating = False

        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

        # Shortcuts will be initialized in initializeShortcuts()

    def enableRedViewFocus(self):
        layoutManager = slicer.app.layoutManager()
        redWidget = layoutManager.sliceWidget("Red")
        redView = redWidget.sliceView()

        # Ensure the slice view can accept focus
        redView.setFocusPolicy(qt.Qt.StrongFocus)

        # Try to set the same on its child (render widget)
        renderWidget = redView.findChild(qt.QWidget, "qt_viewport")
        if renderWidget:
            renderWidget.setFocusPolicy(qt.Qt.StrongFocus)

        # Now set focus
        redView.setFocus()
        redView.activateWindow()

    def initializeShortcuts(self):
        self.shortcutW = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutW.setKey(qt.QKeySequence('W'))
        self.shortcutW.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutS = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutS.setKey(qt.QKeySequence('S'))
        self.shortcutS.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutSpace = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutSpace.setKey(qt.QKeySequence('Space'))
        self.shortcutSpace.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutE = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutE.setKey(qt.QKeySequence('E'))
        self.shortcutE.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutD = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutD.setKey(qt.QKeySequence('D'))
        self.shortcutD.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutA = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutA.setKey(qt.QKeySequence('A'))
        self.shortcutA.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutRightArrow = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutRightArrow.setKey(qt.QKeySequence('Right'))
        self.shortcutRightArrow.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutLeftArrow = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutLeftArrow.setKey(qt.QKeySequence('Left'))
        self.shortcutLeftArrow.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutHome = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutHome.setKey(qt.QKeySequence('Home'))
        self.shortcutHome.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutEnd = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutEnd.setKey(qt.QKeySequence('End'))
        self.shortcutEnd.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutC = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutC.setKey(qt.QKeySequence('C'))
        self.shortcutC.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutPageUp = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutPageUp.setKey(qt.QKeySequence('PageUp'))
        self.shortcutPageUp.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutPageDown = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutPageDown.setKey(qt.QKeySequence('PageDown'))
        self.shortcutPageDown.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutShiftUp = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutShiftUp.setKey(qt.QKeySequence('Shift+Up'))
        self.shortcutShiftUp.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutShiftDown = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutShiftDown.setKey(qt.QKeySequence('Shift+Down'))
        self.shortcutShiftDown.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutL = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutL.setKey(qt.QKeySequence('L'))
        self.shortcutL.setContext(qt.Qt.ApplicationShortcut)
        # Add SelectAll/Copy/Paste shortcuts to mainWindow with ApplicationShortcut context
        mainWindow = slicer.util.mainWindow()
        self.shortcutSelectAll = qt.QShortcut(qt.QKeySequence(qt.QKeySequence.SelectAll), mainWindow)
        self.shortcutSelectAll.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutEscape = qt.QShortcut(qt.QKeySequence(qt.Qt.Key_Escape), mainWindow)
        self.shortcutEscape.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutCopy = qt.QShortcut(qt.QKeySequence(qt.QKeySequence.Copy), mainWindow)
        self.shortcutCopy.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutPaste = qt.QShortcut(qt.QKeySequence(qt.QKeySequence.Paste), mainWindow)
        self.shortcutPaste.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutDelete = qt.QShortcut(qt.QKeySequence(qt.Qt.Key_Delete), mainWindow)
        self.shortcutDelete.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutBackspace = qt.QShortcut(qt.QKeySequence(qt.Qt.Key_Backspace), mainWindow)
        self.shortcutBackspace.setContext(qt.Qt.ApplicationShortcut)
        self.shortcutCut = qt.QShortcut(qt.QKeySequence(qt.QKeySequence.Cut), mainWindow)
        self.shortcutCut.setContext(qt.Qt.ApplicationShortcut)

    def connectDrawingShortcuts(self):
        self.shortcutW.connect('activated()', lambda: self.onAddLine("Pleura", not self.ui.addPleuraButton.isChecked()))
        self.shortcutS.connect('activated()', lambda: self.onAddLine("B-line", not self.ui.addBlineButton.isChecked()))
        self.shortcutE.connect('activated()', lambda: self.onRemoveLine("Pleura", not self.ui.removePleuraButton.isChecked()))  # "E" removes the last pleura line
        self.shortcutD.connect('activated()', lambda: self.onRemoveLine("B-line", not self.ui.removeBlineButton.isChecked()))   # "D" removes the last B-line
        # Add SelectAll/Copy/Paste/Cut shortcut connections using helpers that check for Red view focus
        self.shortcutSelectAll.activated.connect(self._selectAllIfRedViewFocused)
        self.shortcutEscape.activated.connect(self.onDeselectAllLines)
        self.shortcutCut.activated.connect(self._cutIfRedViewFocused)
        self.shortcutCopy.activated.connect(self._copyIfRedViewFocused)
        self.shortcutPaste.activated.connect(self._pasteIfRedViewFocused)
        self.shortcutDelete.activated.connect(self._deleteIfRedViewFocused)
        self.shortcutBackspace.activated.connect(self._deleteIfRedViewFocused)

    def _redViewHasFocus(self):
        focused = qt.QApplication.focusWidget()
        return self._isChildOf(focused, self.sliceView)

    def _isChildOf(self, child, parent):
        while child:
            if child is parent:
                return True
            child = child.parent()
        return False

    def _pasteIfRedViewFocused(self):
        if self._redViewHasFocus():
            self.onPasteLines()

    def _copyIfRedViewFocused(self):
        if self._redViewHasFocus():
            self.onCopyLines()

    def _selectAllIfRedViewFocused(self):
        if self._redViewHasFocus():
            self.onSelectAllLines()

    def _deleteIfRedViewFocused(self):
        if self._redViewHasFocus():
            self.onDeleteSelectedLines()

    def _cutIfRedViewFocused(self):
        if self._redViewHasFocus():
            selected = list(self.logic.selectedLineIDs) # Save current selection
            self.onCopyLines()
            self.logic.selectedLineIDs = selected # Restore for delete, clipboard has the nodes we might paste later
            self.onDeleteSelectedLines(force=True) # don't ask for confirmation
            self.logic.selectedLineIDs = []

    def connectKeyboardShortcuts(self):
        # Disconnect any existing connections first to avoid duplicates
        self.disconnectKeyboardShortcuts()

        # Connect shortcuts to respective actions
        self.connectDrawingShortcuts()
        self.shortcutSpace.connect('activated()', lambda: self.ui.overlayVisibilityButton.toggle())

        self.shortcutA.connect('activated()', self.onSaveAndLoadNextButton)  # "A" to save and load next scan

        # Arrow keys for next/previous frame (Slicer commands)
        self.shortcutRightArrow.connect('activated()', self._nextFrameInSequence)
        self.shortcutLeftArrow.connect('activated()', self._previousFrameInSequence)
        # Home/End keys for first/last frame
        self.shortcutHome.connect('activated()', self._firstFrameInSequence)
        self.shortcutEnd.connect('activated()', self._lastFrameInSequence)
        self.shortcutC.connect('activated()', self._togglePlayPauseSequence)

        # Page Up/Page Down for previous/next clip
        self.shortcutPageUp.connect('activated()', self._onPageUpPressed)
        self.shortcutPageDown.connect('activated()', self._onPageDownPressed)

        # Shift+Up/Shift+Down for next/previous clip
        self.shortcutShiftUp.connect('activated()', self._onNextClipPressed)
        self.shortcutShiftDown.connect('activated()', self._onPreviousClipPressed)

        self.shortcutL.connect('activated()', lambda: self.onShowHideLines(None))  # "L" to show/hide lines

    def disconnectDrawingShortcuts(self):
        try:
            self.shortcutW.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutS.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutE.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutD.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected

        # Disconnect Copy/Paste shortcuts if they exist
        try:
            self.shortcutSelectAll.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutEscape.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutCut.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutCopy.activated.disconnect()
        except RuntimeError:
            pass
        try:
            self.shortcutPaste.activated.disconnect()
        except RuntimeError:
            pass
        try:
            self.shortcutDelete.activated.disconnect()
        except RuntimeError:
            pass
        try:
            self.shortcutBackspace.activated.disconnect()
        except RuntimeError:
            pass

    def disconnectKeyboardShortcuts(self):
        # Disconnect shortcuts to avoid issues when the user leaves the module
        self.disconnectDrawingShortcuts()
        try:
            self.shortcutSpace.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutA.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutRightArrow.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutLeftArrow.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutHome.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutEnd.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutC.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutPageUp.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutPageDown.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutShiftUp.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutShiftDown.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected
        try:
            self.shortcutL.activated.disconnect()
        except RuntimeError:
            pass  # Already disconnected

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        global annotateUltrasoundWidgetInstance
        annotateUltrasoundWidgetInstance = self

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

        self.enableRedViewFocus()

        self.initializeShortcuts()
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
        # in batch mode, without a graphical user interface. We check for None because
        # a subclass might have set it already.
        if self.logic is None:
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
        self.ui.removePleuraButton.clicked.connect(lambda checked: self.onRemoveLine("Pleura", checked))
        self.ui.addBlineButton.toggled.connect(lambda checked: self.onAddLine("B-line", checked))
        self.ui.removeBlineButton.clicked.connect(lambda checked:  self.onRemoveLine("B-line", checked))
        self.ui.overlayVisibilityButton.toggled.connect(self.overlayVisibilityToggled)

        # Set up dynamic layout adjustment for overlay button
        self._setupOverlayButtonLayout()
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
        self.raterNameDebounceTimer = qt.QTimer()
        self.raterNameDebounceTimer.setSingleShot(True)
        self.raterNameDebounceTimer.setInterval(300)  # ms of idle time before triggering
        self.raterNameDebounceTimer.timeout.connect(self.onRaterNameChanged)
        self.ui.raterName.textChanged.connect(lambda: self.raterNameDebounceTimer.start())
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
            if self.logic is not None:
                self.logic.extractAndSetupRaters()
                self.selectedRaters = self.logic.getSelectedRaters()
                self.populateRaterColorTable()

        # Connect rater table collapsed signal to detect user manual changes
        if hasattr(self.ui, 'raterColorsCollapsibleButton'):
            self.ui.raterColorsCollapsibleButton.toggled.connect(self.onRaterColorTableCollapsedChanged)
            # Do not set collapsed state here; let subclass or user decide.
        # Guard flag for programmatic collapse/expand of raterColorsCollapsibleButton
        self._ignoreCollapsedChangedSignal = False
        sliceView = slicer.app.layoutManager().sliceWidget("Red").sliceView()
        self.sliceViewEventFilter = SliceViewEventFilter(self, sliceView)
        sliceView.installEventFilter(self.sliceViewEventFilter)
        self.sliceView = sliceView  # Save reference so it's not GC'ed

        # Customize sequence browser toolbar button tooltips
        self._customizeSequenceBrowserTooltips()

    def distanceToLineSegment2D(self, p, a, b):
        """Squared distance from point p to segment ab in 2D (XY only)."""
        px, py = p[0], p[1]
        ax, ay = a[0], a[1]
        bx, by = b[0], b[1]

        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            # a and b are the same point
            return (px - ax)**2 + (py - ay)**2

        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))

        closest_x = ax + t * dx
        closest_y = ay + t * dy

        return (px - closest_x)**2 + (py - closest_y)**2

    def pickClosestLineNodeInSlice(self, x, y):
        sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
        sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
        sliceNode = sliceWidget.mrmlSliceNode()
        xyToRas = sliceNode.GetXYToRAS()
        xy = [x, y, 0, 1]
        ras = [0, 0, 0, 1]
        xyToRas.MultiplyPoint(xy, ras)
        ras = ras[:3]

        minDist2 = float("inf")
        closestNode = None

        for node in self.logic.pleuraLines + self.logic.bLines:
            if node.GetNumberOfControlPoints() < 2:
                continue
            a = [0, 0, 0]
            b = [0, 0, 0]
            node.GetNthControlPointPosition(0, a)
            node.GetNthControlPointPosition(1, b)

            # Use only XY for 2D slice
            dist2 = self.distanceToLineSegment2D(ras[:2], a[:2], b[:2])
            if dist2 < minDist2:
                minDist2 = dist2
                closestNode = node

        if closestNode and minDist2 < 4.0:  # ~2mm threshold
            self.toggleLineSelection(closestNode)
            return True
        else:
            return False

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
        logging.debug('onClearAllLines')
        self.logic.clearAllLines()
        ratio = self.logic.updateOverlayVolume()
        if ratio is not None:
            self._parameterNode.pleuraPercentage = ratio * 100
            self._parameterNode.unsavedChanges = True
        self.updateGuiFromAnnotations()

    def onFramesTableSelectionChanged(self):
        logging.debug('onFramesTableSelectionChanged')

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
        logging.debug('onAddCurrentFrame')
        self.logic.syncMarkupsToAnnotations()
        self.logic.refreshDisplay(updateOverlay=True, updateGui=True)
        self.updateGuiFromAnnotations()

    def onRemoveCurrentFrame(self):
        logging.debug('removeCurrentFrame')

        # Get the current frame index from the sequence browser
        if self.logic.sequenceBrowserNode is None:
            logging.warning("No sequence browser node found")
            currentFrameIndex = -1
        else:
            currentFrameIndex = self.logic.sequenceBrowserNode.GetSelectedItemNumber()
            self.logic.removeFrame(currentFrameIndex)
            self.logic.refreshDisplay(updateOverlay=True, updateGui=True)
            self.updateGuiFromAnnotations()

    def onInputDirectorySelected(self):
        logging.debug('onInputDirectorySelected')

        inputDirectory = self.ui.inputDirectoryButton.directory
        if not inputDirectory:
            statusText = '⚠️ Please select an input directory'
            slicer.util.mainWindow().statusBar().showMessage(statusText, 5000)
            self.ui.statusLabel.setText(statusText)
            return

        # Update local settings
        slicer.app.settings().setValue("AnnotateUltrasound/InputDirectory", inputDirectory)

    def extractSeenAndSelectedRaters(self):
        """
        Extracts the set of raters that have contributed lines in the current annotations,
        ensuring the current rater is included even if not present in any frame annotations.
        Sets self.seenRaters to a sorted list of rater names.
        """
        # Use the Logic's centralized method to extract and set up raters
        self.logic.extractAndSetupRaters()
        # Copy the seenRaters from logic to widget for UI purposes
        self.seenRaters = self.logic.seenRaters.copy()
        self.realRaters = self.logic.realRaters.copy()
        self.selectedRaters = self.logic.selectedRaters.copy()

    def refocusAndRestoreShortcuts(self, delay: int = 300):
        qt.QTimer.singleShot(delay + 100, self._restoreFocusAndShortcuts)

    def onReadInputButton(self):
        """
        Read the input directory and update the dicomDf dataframe, using rater-specific annotation files.

        :return: True if the input directory was read successfully, False otherwise.
        """
        logging.debug('onReadInputButton')

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

        # Remove existing sequence browser observer before reloading
        if self.logic.sequenceBrowserNode:
            if self.hasObserver(self.logic.sequenceBrowserNode, vtk.vtkCommand.ModifiedEvent, self.logic.onSequenceBrowserModified):
                self.removeObserver(self.logic.sequenceBrowserNode, vtk.vtkCommand.ModifiedEvent, self.logic.onSequenceBrowserModified)

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

            # Reset the DICOM index to start from the beginning when reloading
            self.logic.nextDicomDfIndex = 0

            self.currentDicomDfIndex = self.logic.loadNextSequence()
            self.logic.clearClipboard()

            # Add observer for the new sequence browser node
            if self.logic.sequenceBrowserNode:
                self.addObserver(self.logic.sequenceBrowserNode, vtk.vtkCommand.ModifiedEvent, self.logic.onSequenceBrowserModified)

            # Update self.ui.currentFileLabel using the DICOM file name
            currentDicomFilepath = self.logic.dicomDf.iloc[self.logic.nextDicomDfIndex - 1]['Filepath']
            currentDicomFilename = os.path.basename(currentDicomFilepath)
            statusText = f"Current file ({self.logic.nextDicomDfIndex}/{len(self.logic.dicomDf)}): {currentDicomFilename}"
            self.ui.currentFileLabel.setText(statusText)
            self.ui.statusLabel.setText('')
            slicer.util.mainWindow().statusBar().showMessage(statusText, 3000)
            self.logic.sequenceBrowserNode.SetSelectedItemNumber(0)
            self.logic.refreshDisplay(updateOverlay=True, updateGui=True)

            self.ui.intensitySlider.setValue(0)

            # After loading the first sequence, extract seen raters and update checkboxes
            self.extractSeenAndSelectedRaters()

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

        # Restore focus and shortcuts after loading input
        self.refocusAndRestoreShortcuts()

        # Clear navigation flag after all operations are complete
        self._isNavigating = False

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
        logging.debug('onNextButton')

        if self.logic.dicomDf is None:
            self.ui.statusLabel.setText("Please read input directory first")
            return

        if not self.confirmUnsavedChanges():
            return

        if self.logic.nextDicomDfIndex >= len(self.logic.dicomDf):
            # If we are at the last DICOM file, show a message that clears in 5 seconds and return
            slicer.util.mainWindow().statusBar().showMessage('⚠️ No more DICOM files', 5000)
            return

        # Reset user rater table collapse state for new clip
        self._userManuallySetRaterTableState = False
        self._lastUserManualCollapsedState = None

        # Create a dialog to ask the user to wait while the next sequence is loaded.

        waitDialog = self.createWaitDialog("Loading next sequence", "Loading next sequence...")

        # Set navigation flag to prevent rater table state changes
        self._isNavigating = True

        # Saving settings
        showDepthGuide = self._parameterNode.depthGuideVisible

        # Reset back to showing all lines on new clip
        self.onShowHideLines(True)

        currentDicomDfIndex = self.logic.loadNextSequence()
        self.logic.clearClipboard()

        # Add observer for the new sequence browser node
        if self.logic.sequenceBrowserNode:
            self.addObserver(self.logic.sequenceBrowserNode, vtk.vtkCommand.ModifiedEvent, self.logic.onSequenceBrowserModified)

        # Set sequence browser to first frame
        self.logic.sequenceBrowserNode.SetSelectedItemNumber(0)
        self.logic.refreshDisplay(updateOverlay=True, updateGui=True)

        # After loading the next sequence, extract seen raters and update checkboxes
        self.extractSeenAndSelectedRaters()
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

        # Clear navigation flag after all operations are complete
        self._isNavigating = False
        self.refocusAndRestoreShortcuts()

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
        logging.debug('onPreviousButton')

        if not self.confirmUnsavedChanges():
            return

        if self.logic.dicomDf is None:
            self.ui.statusLabel.setText("Please read input directory first")
            return

        # Reset user rater table collapse state for new clip
        self._userManuallySetRaterTableState = False
        self._lastUserManualCollapsedState = None

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

        self.logic.clearClipboard()
        # Add observer for the new sequence browser node
        if self.logic.sequenceBrowserNode:
            self.addObserver(self.logic.sequenceBrowserNode, vtk.vtkCommand.ModifiedEvent, self.logic.onSequenceBrowserModified)

        # Set sequence browser to first frame
        self.logic.sequenceBrowserNode.SetSelectedItemNumber(0)
        self.logic.refreshDisplay(updateOverlay=True, updateGui=True)

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
        self.extractSeenAndSelectedRaters()
        self.populateRaterColorTable()

        self.updateGuiFromAnnotations()

        # Close the wait dialog
        waitDialog.close()

        self.ui.progressBar.value = currentDicomDfIndex

        # Clear navigation flag after all operations are complete
        self._isNavigating = False
        self.refocusAndRestoreShortcuts()

    def saveAnnotations(self):
        """
        Saves current annotations to rater-specific json file.
        Returns True if save was successful, False otherwise.
        """
        try:
            # Add annotation line control points to the annotations dictionary and save it to file
            if self.logic.annotations is None:
                logging.error("saveAnnotations: No annotations loaded")
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

            # Filter annotations to include only current rater's lines
            rater = self._parameterNode.rater.strip().lower()
            filtered_frames = []
            total_pleura_lines = 0
            total_b_lines = 0

            for frame in self.logic.annotations.get("frame_annotations", []):
                pleura = [line for line in frame.get("pleura_lines", []) if line.get("rater", "").strip().lower() == rater]
                b_lines = [line for line in frame.get("b_lines", []) if line.get("rater", "").strip().lower() == rater]
                total_pleura_lines += len(pleura)
                total_b_lines += len(b_lines)

                if pleura or b_lines:
                    filtered_frames.append({
                        "frame_number": frame["frame_number"],
                        "coordinate_space": "RAS",
                        "pleura_lines": pleura,
                        "b_lines": b_lines
                    })

            logging.info(f"Saving {len(filtered_frames)} frames with {total_pleura_lines} pleura lines and {total_b_lines} b-lines for rater '{rater}'")

            # if we have frames from the current rater or we deleted all lines so unsavedChanges is true
            if filtered_frames or self._parameterNode.unsavedChanges:
                # Convert RAS to LPS before saving
                save_data = self.logic.convert_ras_to_lps(filtered_frames)

                # Save annotations to file (use rater-specific filename from dicomDf)
                annotationsFilepath = self.logic.dicomDf.iloc[self.logic.nextDicomDfIndex - 1]['AnnotationsFilepath']
                # Check if the filepath already ends with .{rater}.json
                if not annotationsFilepath.endswith(f".{rater}.json"):
                    # Get base path (everything before the first dot)
                    if '.' in annotationsFilepath:
                        base_path = annotationsFilepath.split('.', 1)[0]
                    else:
                        base_path = annotationsFilepath
                    annotationsFilepath = f"{base_path}.{rater}.json"
                    self.logic.dicomDf.at[self.logic.nextDicomDfIndex - 1, 'AnnotationsFilepath'] = annotationsFilepath

                with open(annotationsFilepath, 'w') as f:
                    json.dump(save_data, f)

                waitDialog.close()

                self._parameterNode.unsavedChanges = False

                statusText = f"✅ Annotations saved successfully to {os.path.basename(annotationsFilepath)}"
                slicer.util.mainWindow().statusBar().showMessage(statusText, 3000)
                self.ui.statusLabel.setText(statusText)
                logging.info(f"Annotations saved to {annotationsFilepath}")
                return True
            else:
                waitDialog.close()
                statusText = "ℹ️ No annotations to save"
                slicer.util.mainWindow().statusBar().showMessage(statusText, 3000)
                self.ui.statusLabel.setText(statusText)
                logging.info(f"No annotations to save for current rater")
                return True

        except Exception as e:
            statusText = f"❌ Failed to save annotations: {str(e)}"
            slicer.util.mainWindow().statusBar().showMessage(statusText, 5000)
            self.ui.statusLabel.setText(statusText)
            logging.error(f"Error saving annotations: {e}")
            return False

    def onSaveButton(self):
        """
        Saves current annotations to json file only
        """
        logging.debug('onSaveButton (save)')
        success = self.saveAnnotations()
        self._setRedViewFocus()
        if not success:
            # Error message already shown by saveAnnotations
            return

    def onSaveAndLoadNextButton(self):
        """
        Saves current annotations to json file and loads next sequence.
        """
        logging.debug('onSaveAndLoadNextButton (save and load next scan)')

        success = self.saveAnnotations()
        if success:
            self.onNextButton()
        else:
            # Error message already shown by saveAnnotations, don't proceed to next
            return

    def onAddLine(self, lineType, checked):
        logging.debug(f"onAddLine -- lineType: {lineType}, checked: {checked}")

        if not checked:
            self.removeObservers(self.onEndPlaceMode)

            # Return mouse interaction to default
            interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

            if lineType == "Pleura":
                linesList = self.logic.pleuraLines
            elif lineType == "B-line":
                linesList = self.logic.bLines
            else:
                logging.error(f"Unknown line type {lineType}")
                return

            # If current line has less than 2 control points, remove it
            if len(linesList) > 0:
                currentLine = linesList[-1]
                if currentLine.GetNumberOfControlPoints() < 2:
                    linesList.pop()
                    self.logic._freeMarkupNode(currentLine)

            # Only update if we're not already updating (prevents duplicate calls)
            if not self._isUpdatingCurrentFrame:
                self._isUpdatingCurrentFrame = True
                try:
                    logging.info("Auto-saving frame annotations")
                    self.logic.syncMarkupsToAnnotations()
                    self.logic.refreshDisplay(updateOverlay=True, updateGui=True)
                finally:
                    self._isUpdatingCurrentFrame = False
            return

        # if we are already placing a line, don't place another one - we are being called from an unrelated callback
        if self._parameterNode.lineBeingPlaced:
            logging.debug(f"Already placing a line, ignoring add line")
            return

        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        if hasattr(self.logic, "_suppressSync"):
            self.logic._suppressSync = True
        selectionNode.SetActivePlaceNodeID("")
        if hasattr(self.logic, "_suppressSync"):
            self.logic._suppressSync = False

        # Create a new markup fiducial node
        rater = self._parameterNode.rater
        if lineType == "Pleura":
            color_pleura, _ = self.logic.getColorsForRater(rater)
            newLineNode = self.logic.createMarkupLine("Pleura", rater, [], color_pleura)
            self.logic.pleuraLines.append(newLineNode)
        elif lineType == "B-line":
            _, color_blines = self.logic.getColorsForRater(rater)
            newLineNode = self.logic.createMarkupLine("B-line", rater, [], color_blines)
            self.logic.bLines.append(newLineNode)
        else:
            logging.error(f"Unknown line type {lineType}")
            return

        # Suppress sync to avoid having selection node based handlers run
        # while we are adding a line
        if hasattr(self.logic, "_suppressSync"):
            self.logic._suppressSync = True
        selectionNode.SetActivePlaceNodeID(newLineNode.GetID())
        if hasattr(self.logic, "_suppressSync"):
            self.logic._suppressSync = False
        # Put interaction model to place line markup
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        interactionNode.SetPlaceModePersistence(0)

        self.addObserver(interactionNode, interactionNode.EndPlacementEvent, self.onEndPlaceMode)

        self._parameterNode.lineBeingPlaced = newLineNode
        self._parameterNode.unsavedChanges = True

    def onEndPlaceMode(self, caller, event):
        if self._parameterNode is None or self._parameterNode.lineBeingPlaced is None:
            logging.error(f"No line being placed")
            return
        lineType = self._parameterNode.lineBeingPlaced.GetName()
        logging.debug(f'onEndPlaceMode -- lineType: {lineType}')
        # Call the next line using qtimer
        if lineType == "Pleura":
            qt.QTimer.singleShot(0, lambda: self.delayedOnEndPlaceMode("Pleura"))
        elif lineType == "B-line":
            qt.QTimer.singleShot(0, lambda: self.delayedOnEndPlaceMode("B-line"))
        else:
            logging.error(f"Unknown line type {lineType}")
            return

    def delayedOnEndPlaceMode(self, lineType):
        logging.debug(f"delayedOnEndPlaceMode -- lineType: {lineType}")
        if lineType == "Pleura":
            self.ui.addPleuraButton.setChecked(False)
        elif lineType == "B-line":
            self.ui.addBlineButton.setChecked(False)
        else:
            logging.error(f"Unknown line type {lineType}")
            return

        # Restore focus and ensure shortcuts are active
        self._restoreFocusAndShortcuts()

        # Only update if we're not already updating (prevents duplicate calls)
        if not self._isUpdatingCurrentFrame:
            self._isUpdatingCurrentFrame = True
            try:
                logging.info("Auto-saving frame annotations")
                self.logic.syncMarkupsToAnnotations()
                self.logic.refreshDisplay(updateOverlay=True, updateGui=True)
            finally:
                self._isUpdatingCurrentFrame = False

    def onRemovePleuraLine(self):
        logging.debug('onRemovePleuraLine')
        self.logic.removeLastPleuraLine()

    def onRemoveLine(self, lineType, _):
        logging.debug(f"onRemoveLine -- lineType: {lineType}")
        if lineType == "Pleura":
            self.logic.removeLastPleuraLine()
        elif lineType == "B-line":
            self.logic.removeLastBline()
        else:
            logging.error(f"Unknown line type {lineType}")
            return

        self.logic.syncMarkupsToAnnotations()
        self.logic.refreshDisplay(updateOverlay=True, updateGui=True)
        self._parameterNode.unsavedChanges = True

        # Restore focus and ensure shortcuts are active
        self._restoreFocusAndShortcuts()

    def onSelectAllLines(self):
        logging.debug('onSelectAllLines')
        self.logic.selectedLineIDs = []

        # Just highlight the nodes in the current frame (already tracked in pleuraLines + bLines)
        for node in self.logic.pleuraLines + self.logic.bLines:
            if slicer.mrmlScene.IsNodePresent(node):
                self.logic.selectedLineIDs.append(node.GetID())
                self.logic.updateMarkupNodeAppearance(node)

        slicer.util.mainWindow().statusBar().showMessage(f"Selected {len(self.logic.selectedLineIDs)} lines", 2000)
        self._setRedViewFocus()

    def onDeselectAllLines(self):
        logging.debug("onDeselectAllLines")

        # Clear selection
        self.logic.selectedLineIDs = []

        # Reset appearance for all lines
        for node in self.logic.pleuraLines + self.logic.bLines:
            if slicer.mrmlScene.IsNodePresent(node):
                self.logic.updateMarkupNodeAppearance(node)
        self._setRedViewFocus()

    def toggleLineSelection(self, lineNode):
        if not lineNode or not slicer.mrmlScene.IsNodePresent(lineNode):
            return

        nodeID = lineNode.GetID()
        selected_lines_before = len(self.logic.selectedLineIDs)
        if nodeID in self.logic.selectedLineIDs:
            self.logic.selectedLineIDs.remove(nodeID)
        else:
            self.logic.selectedLineIDs.append(nodeID)
        selected_lines_after = len(self.logic.selectedLineIDs)

        self.logic.updateMarkupNodeAppearance(lineNode)
        slicer.util.mainWindow().statusBar().showMessage(f"Selected lines changed from {selected_lines_before} to {selected_lines_after}", 2000)
        self._setRedViewFocus()

    def onCopyLines(self):
        logging.debug('onCopyLines')

        self.logic.clearClipboard()

        if not hasattr(self.logic, "selectedLineIDs") or not self.logic.selectedLineIDs:
            logging.info("No lines selected to copy")
            return

        for nodeID in self.logic.selectedLineIDs:
            node = slicer.mrmlScene.GetNodeByID(nodeID)
            if not node or not slicer.mrmlScene.IsNodePresent(node):
                continue
            if node.GetNumberOfControlPoints() < 2:
                continue

            pt1 = [0, 0, 0]
            pt2 = [0, 0, 0]
            node.GetNthControlPointPositionWorld(0, pt1)
            node.GetNthControlPointPositionWorld(1, pt2)

            # create clipboard line using rater's color and normal thickness/glyph scale
            lineType = node.GetName()
            rater = self._parameterNode.rater
            if lineType == "Pleura":
                color_pleura, _ = self.logic.getColorsForRater(rater)
                newLineNode = self.logic.createMarkupLine("Pleura", rater, [pt1, pt2], color_pleura)
            elif lineType == "B-line":
                _, color_blines = self.logic.getColorsForRater(rater)
                newLineNode = self.logic.createMarkupLine("B-line", rater, [pt1, pt2], color_blines)
            else:
                logging.error(f"Unknown line type {lineType}")
                return

            displayNode = newLineNode.GetDisplayNode()
            if displayNode:
                displayNode.SetLineThickness(0.25)
                displayNode.SetGlyphScale(2.0)
            newLineNode.SetDisplayVisibility(False)
            self.logic.clipboardLines.append(newLineNode)

        self.logic.selectedLineIDs = []  # clear after copying
        slicer.util.mainWindow().statusBar().showMessage(f"Copied {len(self.logic.clipboardLines)} lines", 2000)
        self._setRedViewFocus()

    def onPasteLines(self, force=False):
        logging.debug(f'onPasteLines -- force: {force}')

        if not getattr(self.logic, "clipboardLines", None):
            logging.info("Clipboard is empty")
            return

        pastedLines = []

        for node in self.logic.clipboardLines:
            if not node or not slicer.mrmlScene.IsNodePresent(node):
                continue
            if node.GetNumberOfControlPoints() < 2:
                continue

            pt1 = [0, 0, 0]
            pt2 = [0, 0, 0]
            node.GetNthControlPointPositionWorld(0, pt1)
            node.GetNthControlPointPositionWorld(1, pt2)

            lineType = node.GetName()
            targetList = self.logic.pleuraLines if lineType == "Pleura" else self.logic.bLines
            alreadyExists = False
            for existingNode in targetList:
                if existingNode.GetDisplayNode() is None or existingNode.GetDisplayNode().GetVisibility() == False:
                    continue

                e1 = [0, 0, 0]
                e2 = [0, 0, 0]
                existingNode.GetNthControlPointPosition(0, e1)
                existingNode.GetNthControlPointPosition(1, e2)

                sameLine = (
                    vtk.vtkMath.Distance2BetweenPoints(pt1, e1) < 1e-3 and
                    vtk.vtkMath.Distance2BetweenPoints(pt2, e2) < 1e-3
                ) or (
                    vtk.vtkMath.Distance2BetweenPoints(pt1, e2) < 1e-3 and
                    vtk.vtkMath.Distance2BetweenPoints(pt2, e1) < 1e-3
                )

                if sameLine:
                    alreadyExists = True
                    break

            if alreadyExists and not force:
                continue

            rater = self._parameterNode.rater
            color = node.GetDisplayNode().GetSelectedColor()

            # Create a new line in the current frame by copying the clipboard line
            newNode = self.logic.createMarkupLine(lineType, rater, [pt1, pt2], color, addPointRemovedObserver=True)

            # Append to the appropriate list so syncMarkupsToAnnotations includes it
            if lineType == "Pleura":
                self.logic.pleuraLines.append(newNode)
            elif lineType == "B-line":
                self.logic.bLines.append(newNode)
            else:
                logging.error(f"Unknown line type {lineType}")
                return
            newNode.SetDisplayVisibility(True)
            pastedLines.append(newNode)

        # Sync to internal annotation structure
        self.logic.syncMarkupsToAnnotations()
        self.logic.refreshDisplay(updateOverlay=True, updateGui=True)
        self._parameterNode.unsavedChanges = True

        if len(pastedLines) > 0:
            slicer.util.mainWindow().statusBar().showMessage(f"Pasted {len(pastedLines)} unique lines, and skipped {len(self.logic.clipboardLines) - len(pastedLines)} duplicate lines", 2000)
        else:
            slicer.util.mainWindow().statusBar().showMessage(f"Skipped {len(self.logic.clipboardLines)} duplicate lines", 2000)
        self._setRedViewFocus()

    def onDeleteSelectedLines(self, force=False):
        logging.debug(f"onDeleteSelectedLines -- force: {force}")

        selectedIDs = self.logic.selectedLineIDs
        if not selectedIDs:
            return

        raterNodes = []
        for nodeID in selectedIDs:
            node = slicer.mrmlScene.GetNodeByID(nodeID)
            if node and (node.GetAttribute("rater") == self._parameterNode.rater):
                raterNodes.append(node)

        if len(raterNodes) == 0:
            slicer.util.mainWindow().statusBar().showMessage("No lines belonging to you were selected for deletion", 2000)
            return

        # Confirm if more than one line is selected
        if not force and len(raterNodes) > 1:
            confirmed = qt.QMessageBox.question(
                slicer.util.mainWindow(),
                "Delete selected lines",
                f"Are you sure you want to delete {len(raterNodes)} of your selected lines?",
                qt.QMessageBox.Yes | qt.QMessageBox.No,
            )
            if confirmed != qt.QMessageBox.Yes:
                return

        deletedLines = 0
        for node in raterNodes:
            if node in self.logic.pleuraLines:
                self.logic.pleuraLines.remove(node)
            if node in self.logic.bLines:
                self.logic.bLines.remove(node)
            self.logic._freeMarkupNode(node)
            deletedLines += 1
        if deletedLines > 0:
            self.onDeselectAllLines() # clear selection as we deleted lines
            self.logic.syncMarkupsToAnnotations()
            self.logic.refreshDisplay(updateOverlay=True, updateGui=True)
            self._parameterNode.unsavedChanges = True
            slicer.util.mainWindow().statusBar().showMessage(f"Deleted {deletedLines} line(s) you created", 2000)
        else:
            slicer.util.mainWindow().statusBar().showMessage("No lines belonging to you were deleted", 2000)
        self._setRedViewFocus()

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
            words = re.findall(r'[A-Z][a-z\d]+|[A-Z\d]+(?=[A-Z][a-z\d]|[^a-zA-Z\d]|$)|[a-z\d]+', text)
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
        logging.debug(f"onLabelCheckBoxToggled -- checked: {checked}")
        if self.logic.annotations is None:
            logging.error("onLabelCheckBoxToggled: No annotations loaded")
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
            annotationsFilepath = self.logic.dicomDf.iloc[idx]['AnnotationsFilepath']

            # Check if the annotation file exists
            if not os.path.exists(annotationsFilepath):
                # File doesn't exist, so this scan is unlabeled
                return idx

            try:
                with open(annotationsFilepath, 'r') as f:
                    annotations = json.load(f)
                    # Check if frame annotations exist and are empty
                    if 'frame_annotations' not in annotations or not annotations['frame_annotations']:
                        return idx
            except Exception as e:
                logging.error(f"Error reading annotations file {annotationsFilepath}: {e}")
                # If there's an error reading the file, treat it as unlabeled
                return idx

        return None

    def _setupOverlayButtonLayout(self):
        """Set up dynamic layout adjustment for the overlay button based on clearAllLinesButton visibility"""
        # Initial adjustment
        self._adjustOverlayButtonLayout()

        # Connect to the sector annotations collapsible button state changes
        self.ui.sectorAnnotationsCollapsibleButton.toggled.connect(self._adjustOverlayButtonLayout)

    def _adjustOverlayButtonLayout(self):
        """Adjust the overlay button layout based on visibility of clearAllLinesButton"""
        # Check if clearAllLinesButton is visible
        clearButtonVisible = self.ui.clearAllLinesButton.isVisible()

        # Get the grid layout
        gridLayout = self.ui.overlayVisibilityButton.parent().layout()

        if not isinstance(gridLayout, qt.QGridLayout):
            return  # Not a grid layout, can't adjust

        # Find and remove the overlay button item from the layout
        overlayItem = None
        for i in range(gridLayout.count()):
            item = gridLayout.itemAt(i)
            if item.widget() == self.ui.overlayVisibilityButton:
                overlayItem = item
                break

        if overlayItem is None:
            return  # Overlay button not found in layout

        # Remove the item from the layout
        gridLayout.removeItem(overlayItem)

        # Add it back with the appropriate column span
        # We know the overlay button is in row 3, column 0 based on the UI file
        if not clearButtonVisible:
            # Clear button not visible, make overlay button span full width (colspan=2)
            gridLayout.addWidget(self.ui.overlayVisibilityButton, 3, 0, 1, 2)
        else:
            # Clear button visible, keep overlay button in normal position (colspan=1)
            gridLayout.addWidget(self.ui.overlayVisibilityButton, 3, 0, 1, 1)

    def overlayVisibilityToggled(self, checked):
        logging.debug(f"overlayVisibilityToggled -- checked: {checked}")
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

    def onDepthGuideToggled(self, toggled):
        # Save new state in application settings and update depth guide volume to show/hide the depth guide
        settings = slicer.app.settings()
        settings.setValue('AnnotateUltrasound/DepthGuide', toggled)
        if toggled:
            self.logic.parameterNode.depthGuideVisible = True
        else:
            self.logic.parameterNode.depthGuideVisible = False
        self.logic.updateDepthGuideVolume()

    def onRaterNameChanged(self):
        if self._parameterNode:
            self._parameterNode.rater = self.ui.raterName.text.strip().lower()
            statusText = f"Rater name changed to {self._parameterNode.rater}"
            slicer.util.mainWindow().statusBar().showMessage(statusText, 3000)
            self.logic.extractAndSetupRaters()
            self.selectedRaters = self.logic.getSelectedRaters()
            self.populateRaterColorTable()
            self.logic.syncAnnotationsToMarkups()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

        self.disconnectKeyboardShortcuts()
        self.logic.clearScene()
        self.logic.clearClipboard()
        global annotateUltrasoundWidgetInstance
        annotateUltrasoundWidgetInstance = None

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
            self.addObserver(self.logic.sequenceBrowserNode, vtk.vtkCommand.ModifiedEvent, self.logic.onSequenceBrowserModified)

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
        widget = slicer.modules.annotateultrasound.widgetRepresentation()
        if widget and widget.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

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
            if self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateGUIFromParameterNode):
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

            # Update corner annotation if _parameterNode.pleuraPercentage is a non-negative number
            # if we are using multiple raters and have selected more than one, don't show overlay volume
            selectedRaters = self.logic.getSelectedRaters()
            if selectedRaters is not None:
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
        self._ignoreCollapsedChangedSignal = True
        self.ui.raterColorsCollapsibleButton.collapsed = collapsed
        self._ignoreCollapsedChangedSignal = False

    def onRaterColorTableCollapsedChanged(self, _):
        """
        Called when the user manually expands/collapses the rater table.
        Sets the flag to respect user's manual state.
        """
        if getattr(self, "_ignoreCollapsedChangedSignal", False):
            return

        # Use actual state, not signal value
        actual = self.ui.raterColorsCollapsibleButton.collapsed

        self._userManuallySetRaterTableState = True
        self._lastUserManualCollapsedState = actual

    def populateRaterColorTable(self):
        if not hasattr(self.ui, 'raterColorTable'):
            return
        self.ui.raterColorTable.blockSignals(True)
        self.ui.raterColorTable.clearContents()
        colors = list(self.logic.getAllRaterColors())

        # Filter out __selected_node__ and __adjudicated_node__ before setting row count, we don't want to show it in the UI.
        # Note: populateRaterColorTable is shared with AdjudicatedUltrasound, it is leaking __adjudicated_node__ knowledge
        # but better than copying it to AdjudicatedUltrasound for this and we do not want to have the __selected_node__ "red"
        # and __adjudicated_node__ "blue"/"magenta" colors used by raters in any module.
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

        if getattr(self, "_userManuallySetRaterTableState", False):
            if hasattr(self.ui, "raterColorsCollapsibleButton") and hasattr(self, "_lastUserManualCollapsedState"):
                self._ignoreCollapsedChangedSignal = True
                self.ui.raterColorsCollapsibleButton.collapsed = self._lastUserManualCollapsedState
                self._ignoreCollapsedChangedSignal = False

    def updateRatersFromCheckboxes(self):
        self.selectedRaters = self.getSelectedRatersFromTable()
        self.logic.setSelectedRaters(self.selectedRaters)
        self.logic.updateDisplayForRaterChange()
        self._updateGUIFromParameterNode()
        self.ui.raterColorTable.repaint()
        self.ui.raterColorTable.update()

    def onRaterColorTableClicked(self, row, column):
        item = self.ui.raterColorTable.item(row, 0)  # Assume checkbox is in column 0
        if item is not None:
            current_state = item.checkState()
            item.setCheckState(qt.Qt.Unchecked if current_state == qt.Qt.Checked else qt.Qt.Checked)
        self.onRaterColorSelectionChangedFromUser()

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
        self._setRedViewFocus()

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
        self._setRedViewFocus()

    def _setRedViewFocus(self):
        """Force focus to the Red slice view's render widget after delay."""
        def forceFocus():
            mainWin = slicer.util.mainWindow()
            mainWin.raise_()
            mainWin.activateWindow()

            redSliceView = slicer.app.layoutManager().sliceWidget("Red").sliceView()
            renderWidget = None

            for child in redSliceView.findChildren(qt.QWidget):
                className = type(child).__name__
                if "vtk" in className.lower() or "render" in className.lower():
                    renderWidget = child
                    break

            if not renderWidget:
                # Fallback: try by object name
                renderWidget = redSliceView.findChild(qt.QWidget, "qt_viewport")

            if renderWidget:
                renderWidget.setFocus()
                renderWidget.raise_()
                renderWidget.activateWindow()
                qt.QApplication.setActiveWindow(renderWidget)
            else:
                logging.warning("Could not find render widget in Red slice view")

        qt.QTimer.singleShot(100, forceFocus)

    def _forceShortcutsActive(self):
        """Force keyboard shortcuts to be active by temporarily disconnecting and reconnecting them."""
        try:
            # Temporarily disconnect and reconnect shortcuts to force them to be active
            self.disconnectKeyboardShortcuts()
            qt.QTimer.singleShot(50, self.connectKeyboardShortcuts)
        except Exception as e:
            logging.warning(f"Could not force shortcuts active: {e}")

    def _restoreFocusAndShortcuts(self):
        """Restore focus to main window and ensure shortcuts are active."""
        try:
            # Reset interaction mode to ensure keyboard shortcuts work
            interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
            if interactionNode:
                interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

            # Set focus back red view
            self._setRedViewFocus()

            # Force shortcuts to be active
            self._forceShortcutsActive()
        except Exception as e:
            logging.warning(f"Could not restore focus and shortcuts: {e}")

    def _navigateToClip(self, direction):
        """Helper method to navigate to previous or next clip."""
        direction_text = "previous" if direction == "previous" else "next"
        slicer.util.mainWindow().statusBar().showMessage(f'Loading {direction_text} clip...', 2000)
        if direction == "previous":
            self.onPreviousButton()
        else:
            self.onNextButton()
        self._setRedViewFocus()

    def _onPageUpPressed(self):
        """Handle Page Up press for next clip."""
        self._navigateToClip("next")

    def _onPageDownPressed(self):
        """Handle Page Down press for previous clip."""
        self._navigateToClip("previous")

    def _onPreviousClipPressed(self):
        """Handle Shift+Up or Ctrl+Up press for previous clip."""
        self._navigateToClip("previous")

    def _onNextClipPressed(self):
        """Handle Shift+Down or Ctrl+Down press for next clip."""
        self._navigateToClip("next")

    def onShowHideLines(self, checked=None):
        """Toggle visibility of all lines and overlays."""
        if checked is None:
            # Toggle button state
            self.ui.showHideLinesButton.setChecked(not self.ui.showHideLinesButton.isChecked())
            checked = self.ui.showHideLinesButton.isChecked()
        # Toggle visibility of all lines from the logic class
        self.logic.onShowHideLines(checked)

        self.logic.refreshDisplay(updateOverlay=True, updateGui=True)
        self._setRedViewFocus()

    def _customizeSequenceBrowserTooltips(self):
        """Customize the tooltips of sequence browser toolbar buttons to reflect custom shortcuts."""
        try:
            logging.info("Customizing sequence browser tooltips")
            # Get the sequence browser toolbar
            sequenceToolBar = slicer.modules.sequences.toolBar()

            # Find all buttons in the toolbar
            buttons = sequenceToolBar.findChildren(qt.QPushButton)

            for button in buttons:
                # Look for the next frame button by checking its tooltip or object name
                currentTooltip = button.toolTip

                # Update the "Next Frame" button tooltip
                if "Next frame" in currentTooltip and "Ctrl+Shift+Right" in currentTooltip:
                    button.setToolTip("Next Frame (Right)")
                    logging.info("Updated Next Frame button tooltip to show 'Right' shortcut")

                # Optionally, also update the "Previous Frame" button
                elif "Previous frame" in currentTooltip and "Ctrl+Shift+Left" in currentTooltip:
                    button.setToolTip("Previous Frame (Left)")
                    logging.info("Updated Previous Frame button tooltip to show 'Left' shortcut")

        except Exception as e:
            logging.warning(f"Could not customize sequence browser tooltips: {e}")

#
# AnnotateUltrasoundLogic
#

class AnnotateUltrasoundLogic(ScriptedLoadableModuleLogic, CustomObserverMixin, VTKObservationMixin):
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
        VTKObservationMixin.__init__(self)

        # These variables keep their values when the scene is cleared
        self.dicomDf = None
        self.nextDicomDfIndex = 0

        # These variables need to be reset when the scene is cleared
        self.annotations = None
        self.freeMarkupNodes = []
        self.pleuraLines = []
        self.bLines = []
        self.selectedLineIDs = []
        self.sequenceBrowserNode = None
        self.depthGuideMode = 1
        logging.debug(f"Initialized depthGuideMode to {self.depthGuideMode}")
        self.parameterNode = self._getOrCreateParameterNode()
        self.showHideLines = True
        self.useFreeList = False

        # Flag to track when we're doing programmatic updates (to avoid setting unsavedChanges)
        self._isProgrammaticUpdate = False

        self.seenRaters = []
        self.realRaters = []
        self.selectedRaters = []

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
        current_rater = self.getRater()

        if rater not in self.seenRaters and rater != '':
            self.seenRaters.append(rater)
            self.seenRaters.sort()

        if current_rater not in self.seenRaters:
            self.seenRaters.append(current_rater)
            self.seenRaters.sort()

        raters = self.seenRaters

        if rater not in raters:
            return [1.0, 0.0, 0.0], [1.0, 0.5, 0.0]  # fallback red/orange

        N = len(raters)

        if N == 0:
            return [1.0, 0.0, 0.0], [1.0, 0.5, 0.0]  # fallback red/orange

        # Find the index of this rater among all non-current raters
        rater_index = raters.index(rater)

        # Use golden ratio for non-repeating distribution
        # φ = (1 + √5) / 2 ≈ 1.618033988749895
        golden_ratio = (1 + 5**0.5) / 2

        # Start from positions after green and blue to avoid conflicts with current rater
        pleura_start = 0.66 # Start at blue
        bline_start = 0.33  # Start at green

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

    def getRater(self):
        return self.getParameterNode().rater.strip().lower()

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

    def removeFrame(self, frameIndex):
        logging.debug(f"removeFrame -- frameIndex: {frameIndex}")
        if self.annotations is None:
            logging.warning("removeFrame: No annotations loaded")
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
        if self.hasObserver(slicer.mrmlScene, slicer.mrmlScene.NodeRemovedEvent, self.onMarkupNodeRemoved):
            self.removeObserver(slicer.mrmlScene, slicer.mrmlScene.NodeRemovedEvent, self.onMarkupNodeRemoved)
        self.selectedLineIDs = []
        self.annotations = None
        for node in self.pleuraLines:
            self.pleuraLines.remove(node)
            self._freeMarkupNode(node)
        self.pleuraLines = []
        for node in self.bLines:
            self.bLines.remove(node)
            self._freeMarkupNode(node)
        self.bLines = []
        self._freeAllMarkupNodes()
        parameterNode = self.getParameterNode()

        if self.sequenceBrowserNode:
            slicer.mrmlScene.RemoveNode(self.sequenceBrowserNode)
            self.sequenceBrowserNode = None
        if parameterNode and parameterNode.overlayVolume:
            slicer.mrmlScene.RemoveNode(parameterNode.overlayVolume)
            parameterNode.overlayVolume = None
        if parameterNode and parameterNode.depthGuideVolume:
            slicer.mrmlScene.RemoveNode(parameterNode.depthGuideVolume)
            parameterNode.depthGuideVolume = None
        slicer.mrmlScene.Clear(0)

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

    def clearClipboard(self):
        if getattr(self, "clipboardLines", None):
            for node in self.clipboardLines:
                self._freeMarkupNode(node)
            self.clipboardLines.clear()
        else:
            self.clipboardLines = []

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

        # --- Begin: Custom annotation file selection logic ---
        base_file_path = self.dicomDf.iloc[self.nextDicomDfIndex]['Filepath']
        base_prefix = os.path.splitext(base_file_path)[0]
        current_rater = self.getParameterNode().rater.strip().lower()
        candidates = [
            f"{base_prefix}.{current_rater}.json",
            f"{base_prefix}.json"
        ]
        nextAnnotationsFilepath = None
        for candidate in candidates:
            if os.path.exists(candidate):
                nextAnnotationsFilepath = candidate
                break
        if nextAnnotationsFilepath is None:
            nextAnnotationsFilepath = f"{base_prefix}.{current_rater}.json"
        self.nextDicomDfIndex += 1

        # Make sure a temporary folder for the DICOM files exists
        tempDicomDir = slicer.app.temporaryPath + '/AnnotateUltrasound'
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

        logging.debug(f"Loaded {len(loadedNodeIDs)} nodes")

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
            parameterNode.overlayVolume = overlayVolume

        # Load all annotations with the same base prefix and deeply merge frame_annotations by frame_number
        self.seenRaters = []
        merged_data = {}
        merged_data["frame_annotations"] = []
        filepaths = glob.glob(f"{base_prefix}.json") + glob.glob(f"{base_prefix}.*.json")
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
                            # Add new raters from pleura_lines
                            for entry in frame.get("pleura_lines", []):
                                rater = entry.get("rater")
                                if rater and rater not in self.seenRaters:
                                    self.seenRaters.append(rater)
                            # Add new raters from b_lines
                            for entry in frame.get("b_lines", []):
                                rater = entry.get("rater")
                                if rater and rater not in self.seenRaters:
                                    self.seenRaters.append(rater)
                        else:
                            merged_data["frame_annotations"].append({
                                "frame_number": frame["frame_number"],
                                "coordinate_space": frame.get("coordinate_space", "RAS"),
                                "pleura_lines": frame.get("pleura_lines", []),
                                "b_lines": frame.get("b_lines", [])
                            })
                            # Add new raters from pleura_lines
                            for entry in frame.get("pleura_lines", []):
                                rater = entry.get("rater")
                                if rater and rater not in self.seenRaters:
                                    self.seenRaters.append(rater)
                            # Add new raters from b_lines
                            for entry in frame.get("b_lines", []):
                                rater = entry.get("rater")
                                if rater and rater not in self.seenRaters:
                                    self.seenRaters.append(rater)
            except Exception as e:
                logging.warning(f"Failed to load annotation file {filepath}: {e}")

        self.annotations = merged_data

        # Initialize markup nodes based on loaded annotations
        if self.useFreeList:
            self.initializeMarkupNodesFromAnnotations()

        if current_rater in self.seenRaters:
            self.seenRaters.remove(current_rater)
        # put current rater at the top and add __selected_node__ and __adjudicated_node__ after so that those colors
        # all stay constant regardless of which module we are in.
        self.seenRaters = [current_rater, "__selected_node__", "__adjudicated_node__"] + sorted(self.seenRaters)
        self.realRaters = [r for r in self.seenRaters if r != "__selected_node__" and r != "__adjudicated_node__"]
        self.setSelectedRaters(self.realRaters)

        # Set programmatic update flag to prevent unsavedChanges from being set
        self.syncAnnotationsToMarkups()
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

    def onSequenceBrowserModified(self, caller, event):
        """Handle sequence browser modifications (e.g., frame navigation via Slicer UI)."""
        # Reset selected node ID when sequence browser changes (UI state management)
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        if selectionNode:
            selectionNode.SetActivePlaceNodeID("")

        # Update display for new frame
        self.syncAnnotationsToMarkups()
        self.refreshDisplay(updateOverlay=True, updateGui=True)

    def _getUnusedMarkupNode(self):
        for node in self.freeMarkupNodes:
            self.freeMarkupNodes.remove(node)
            return node
        return None

    def _freeAllMarkupNodes(self):
        for node in self.freeMarkupNodes:
            self.freeMarkupNodes.remove(node)
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
            slicer.mrmlScene.RemoveNode(node)
        self.freeMarkupNodes = []

    def _allocateNewMarkupNode(self):
        markupNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
        markupNode.CreateDefaultDisplayNodes()
        markupNode.GetDisplayNode().SetPropertiesLabelVisibility(False)
        markupNode.GetDisplayNode().SetSelectedColor([1, 1, 0])
        return markupNode

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

    def createMarkupLine(self, name, rater, coordinates, color=[1, 1, 0], addPointRemovedObserver=False):
        if self.useFreeList:
            markupNode = self._getUnusedMarkupNode()
            if markupNode is None:
                markupNode = self._allocateNewMarkupNode()
        else:
            markupNode = self._allocateNewMarkupNode()

        markupNode.SetName(name)
        markupNode.SetAttribute("rater", rater)
        displayNode = markupNode.GetDisplayNode()
        if displayNode is None:
            logging.debug(f"createMarkupLine: Creating default display nodes for {markupNode.GetName()}")
            markupNode.CreateDefaultDisplayNodes()
            displayNode = markupNode.GetDisplayNode()
            displayNode.SetPropertiesLabelVisibility(False)
        displayNode.SetSelectedColor(color)
        displayNode.SetGlyphTypeFromString("Circle2D")
        displayNode.SetGlyphScale(2.0)
        displayNode.SetLineThickness(0.25)
        for coord in coordinates:
            markupNode.AddControlPointWorld(coord[0], coord[1], coord[2])

        self.addObserver(markupNode, markupNode.PointModifiedEvent, self.onPointModified)
        self.addObserver(markupNode, markupNode.PointPositionDefinedEvent, self.onPointPositionDefined)
        if addPointRemovedObserver and not self.hasObserver(markupNode, markupNode.PointRemovedEvent, self.onPointRemoved):
            self.addObserver(markupNode, markupNode.PointRemovedEvent, self.onPointRemoved)
        if not self.hasObserver(slicer.mrmlScene, slicer.mrmlScene.NodeRemovedEvent, self.onMarkupNodeRemoved):
            self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeRemovedEvent, self.onMarkupNodeRemoved)

        return markupNode

    def onShowHideLines(self, checked=None):
        """Toggle visibility of all lines."""
        self.showHideLines = checked
        for node in self.pleuraLines + self.bLines:
            displayNode = node.GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility(checked)

    def updateMarkupNodeAppearance(self, node):
        displayNode = node.GetDisplayNode()
        if not displayNode:
            return

        if hasattr(self, "selectedLineIDs") and node.GetID() in self.selectedLineIDs:
            displayNode.SetLineThickness(0.35)
            displayNode.SetGlyphScale(2.5)
        else:
            displayNode.SetLineThickness(0.25)
            displayNode.SetGlyphScale(2.0)

    def _updateMarkupNode(self, node, entry):
        """
        Update a markup node with the given entry.
        """
        # Check if node is still valid
        if not node or not slicer.mrmlScene.IsNodePresent(node):
            return

        current_rater = self.getParameterNode().rater.strip().lower()
        coordinates = entry.get("line", {}).get("points", [])
        rater = entry.get("rater", "")
        color_pleura, _ = self.getColorsForRater(rater)
        node.SetAttribute("rater", rater)
        if rater != current_rater:
            node.SetLocked(True)
        else:
            node.SetLocked(False)

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

        displayNode.SetGlyphTypeFromString("Circle2D")
        self.updateMarkupNodeAppearance(node)

        if node in self.pleuraLines:
            color_pleura, _ = self.getColorsForRater(rater)
            displayNode.SetSelectedColor(color_pleura)
        else:
            _, color_bline = self.getColorsForRater(rater)
            displayNode.SetSelectedColor(color_bline)

        # Set visibility of the line
        displayNode.SetVisibility(self.showHideLines)

        # Update control points
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

    def onPointModified(self, caller, event):
        numControlPoints = caller.GetNumberOfControlPoints()
        parameterNode = self.getParameterNode()
        if numControlPoints >= 2:
            # Save current markup state to annotations
            self.syncMarkupsToAnnotations()
            # Update overlay display
            self.refreshDisplay(updateOverlay=True, updateGui=True)
        elif parameterNode.lineBeingPlaced is None:
            # Remove the line from the scene
            if caller in self.pleuraLines:
                self.pleuraLines.remove(caller)
            elif caller in self.bLines:
                self.bLines.remove(caller)
            self._freeMarkupNode(caller)

            # Only set unsavedChanges if this is a user-initiated modification
            if not self._isProgrammaticUpdate:
                parameterNode.unsavedChanges = True

    def onPointPositionDefined(self, caller, event):
        parameterNode = self.getParameterNode()
        numControlPoints = caller.GetNumberOfControlPoints()
        if numControlPoints >= 2:
            parameterNode.lineBeingPlaced = None
            if self.hasObserver(caller, caller.PointPositionDefinedEvent, self.onPointPositionDefined):
                self.removeObserver(caller, caller.PointPositionDefinedEvent, self.onPointPositionDefined)

            # Save current markup state to annotations
            self.syncMarkupsToAnnotations()
            # Update overlay display
            self.refreshDisplay(updateOverlay=True, updateGui=True)

            self.addObserver(caller, caller.PointRemovedEvent, self.onPointRemoved)

            # Set unsavedChanges when user finishes placing a line (only if not programmatic)
            if not self._isProgrammaticUpdate:
                parameterNode.unsavedChanges = True

    def onPointRemoved(self, caller, event):
        numControlPoints = caller.GetNumberOfControlPoints()
        if numControlPoints < 2:
            # Remove the line from the scene
            if caller in self.pleuraLines:
                self.pleuraLines.remove(caller)
            elif caller in self.bLines:
                self.bLines.remove(caller)
            qt.QTimer.singleShot(0, lambda: self._freeMarkupNode(caller))

        parameterNode = self.getParameterNode()
        parameterNode.unsavedChanges = True
        self.syncMarkupsToAnnotations()
        self.refreshDisplay(updateOverlay=True, updateGui=True)

    def onMarkupNodeRemoved(self, caller, event):
        removedNodes = {
            node
            for node in self.pleuraLines + self.bLines
            if slicer.mrmlScene.GetNodeByID(node.GetID()) is None
        }

        if not removedNodes:
            return

        didRemove = False
        for node in removedNodes:
            if isinstance(node, slicer.vtkMRMLMarkupsLineNode):
                if node in self.pleuraLines:
                    self.pleuraLines.remove(node)
                elif node in self.bLines:
                    self.bLines.remove(node)
                logging.debug(f"onMarkupNodeRemoved: {node.GetName()} was removed")
                # we do not call _freeMarkupNode as this node is already Removed from the scene and is gone
                didRemove = True
        if didRemove:
            parameterNode = self.getParameterNode()
            parameterNode.unsavedChanges = True
            self.syncMarkupsToAnnotations()
            self.refreshDisplay(updateOverlay=True, updateGui=True)

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

    def _updateMarkupNodesForFrame(self, frame):
        """
        Update markup nodes for pleura and b-lines for the given frame.
        """

        # Check if scene is valid before proceeding
        if not slicer.mrmlScene:
            return

        pleura_entries = [entry for entry in frame.get("pleura_lines", []) if entry.get("rater") in self.selectedRaters]
        bline_entries = [entry for entry in frame.get("b_lines", []) if entry.get("rater") in self.selectedRaters]

        # Update pleura markups
        for i, entry in enumerate(pleura_entries):
            if i >= len(self.pleuraLines):
                node = self.createMarkupLine("Pleura", entry.get("rater", ""), entry.get("coordinates", []), [1,1,0])
                self.pleuraLines.append(node)
            else:
                node = self.pleuraLines[i]
            self._updateMarkupNode(node, entry)

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
            self._updateMarkupNode(node, entry)

        # free unused b-line markups
        unused_b_lines = len(self.bLines) - len(bline_entries)
        for i in range(unused_b_lines):
            node = self.bLines.pop()
            self._freeMarkupNode(node)

    def drawDepthGuideLine(self, image_size_rows, image_size_cols, depth_ratio=0.5, color=(0, 255, 255), thickness=4, dash_length=20, dash_gap=16):
        """
        Main function to handle different visualization modes for the depth guide.
        """
        # Extract fan parameters from annotations
        if self.annotations is None:
            return np.zeros((image_size_rows, image_size_cols, 3), dtype=np.uint8)

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

    def updateDepthGuideVolume(self):
        """
        Update the depth guide volume based on the current visibility setting.
        This is separate from the overlay volume and can be toggled independently.
        """
        parameterNode = self.getParameterNode()

        if parameterNode is None or parameterNode.depthGuideVolume is None:
            return

        if parameterNode.inputVolume is None:
            return

        ultrasoundArray = slicer.util.arrayFromVolume(parameterNode.inputVolume)
        image_size_rows = ultrasoundArray.shape[1]
        image_size_cols = ultrasoundArray.shape[2]

        if parameterNode.depthGuideVisible:
            # Create depth guide and update the volume
            depth_guide = self.drawDepthGuideLine(image_size_rows, image_size_cols)

            # For scalar volume, use the maximum value across all channels (depth_guide is already 3D: rows, cols, channels)
            # This ensures we capture the depth guide regardless of which color channel it's drawn in
            depth_guide_scalar = np.max(depth_guide, axis=2)  # Take maximum across channels
            slicer.util.updateVolumeFromArray(parameterNode.depthGuideVolume, depth_guide_scalar)

            # Make sure the depth guide is visible in the slice viewer as a separate layer
            redSliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceCompositeNode()
            # Set depth guide as label volume (appears below foreground, no blending)
            redSliceCompositeNode.SetLabelVolumeID(parameterNode.depthGuideVolume.GetID())
            redSliceCompositeNode.SetLabelOpacity(0.3)

            # Set up display properties for the depth guide
            displayNode = parameterNode.depthGuideVolume.GetDisplayNode()
            if displayNode:
                displayNode.SetWindow(255)
                displayNode.SetLevel(127)
                                # For label volumes, use a simple color map that shows cyan for non-zero values
                # Create a custom color table that maps non-zero values to cyan
                colorNode = slicer.vtkMRMLColorTableNode()
                colorNode.SetTypeToUser()
                colorNode.SetNumberOfColors(256)
                colorNode.SetNamesInitialised(True)

                # Set all colors to transparent except for non-zero values which will be cyan
                for i in range(256):
                    if i == 0:
                        colorNode.SetColor(i, 0, 0, 0, 0)  # Transparent for background
                    else:
                        colorNode.SetColor(i, 0, 1, 1, 1)  # Cyan for depth guide

                slicer.mrmlScene.AddNode(colorNode)
                displayNode.SetAndObserveColorNodeID(colorNode.GetID())
                displayNode.SetAutoWindowLevel(False)
            else:
                logging.warning("updateDepthGuideVolume: No display node found for depth guide volume")

            # Force slice viewer to refresh
            slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceNode().Modified()
        else:
            # Clear the depth guide volume
            depthGuideArray = slicer.util.arrayFromVolume(parameterNode.depthGuideVolume)
            depthGuideArray[:] = 0
            slicer.util.updateVolumeFromArray(parameterNode.depthGuideVolume, depthGuideArray)

            # Remove depth guide from slice viewer
            redSliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceCompositeNode()
            redSliceCompositeNode.SetLabelVolumeID(None)

    def updateOverlayVolume(self):
        """
        Update the overlay volume based on the annotations.

        :return: The ratio of green pixels to blue pixels in the overlay volume. None if inputs not defined yet.
        """
        parameterNode = self.getParameterNode()

        if parameterNode is None or parameterNode.overlayVolume is None:
            logging.debug("updateOverlayVolume: No overlay volume found! Cannot update overlay volume.")
            return None

        if self.annotations is None:
            logging.warning("updateOverlayVolume: No annotations loaded")
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
            slicer.util.updateVolumeFromArray(parameterNode.overlayVolume, overlayArray)
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

        # Add pleura lines to mask array using full RGB overlay
        for markupNode in self.pleuraLines:
            nodeRater = markupNode.GetAttribute("rater") if markupNode else None
            if hasattr(self, "selectedRaters") and self.selectedRaters and nodeRater not in self.selectedRaters:
                continue
            if not markupNode.GetDisplayNode().GetVisibility():
                continue

            for i in range(markupNode.GetNumberOfControlPoints() - 1):
                coord1 = [0, 0, 0]
                coord2 = [0, 0, 0]
                markupNode.GetNthControlPointPosition(i, coord1)
                markupNode.GetNthControlPointPosition(i + 1, coord2)
                # Skip if the two control points are the same, this sometimes happens when we start placing a line
                if coord1 == coord2:
                    continue
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
            if hasattr(self, "selectedRaters") and self.selectedRaters and nodeRater not in self.selectedRaters:
                continue
            if not markupNode.GetDisplayNode().GetVisibility():
                continue

            for i in range(markupNode.GetNumberOfControlPoints() - 1):
                coord1 = [0, 0, 0]
                coord2 = [0, 0, 0]
                markupNode.GetNthControlPointPosition(i, coord1)
                markupNode.GetNthControlPointPosition(i + 1, coord2)
                # Skip if the two control points are the same, this sometimes happens when we start placing a line
                if coord1 == coord2:
                    continue
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

        # Update the overlay volume (depth guide is handled separately)
        slicer.util.updateVolumeFromArray(parameterNode.overlayVolume, maskArray)

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

        # Update depth guide visibility
        self.updateDepthGuideVolume()

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
        logging.debug('Processing started')

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
        logging.debug(f'Processing completed in {stopTime-startTime:.2f} seconds')

    def extractAndSetupRaters(self):
        """
        Extract all unique raters from the loaded annotations and set up selectedRaters.
        This centralizes the rater extraction logic to avoid duplication.
        """
        # Extract all unique raters from the loaded annotations
        self.seenRaters = []
        if self.annotations and 'frame_annotations' in self.annotations:
            for frame in self.annotations['frame_annotations']:
                for line_type in ['pleura_lines', 'b_lines']:
                    for line in frame.get(line_type, []):
                        rater = line.get('rater')
                        if rater and rater not in self.seenRaters:
                            self.seenRaters.append(rater)

        parameterNode = self.getParameterNode()
        current_rater = parameterNode.rater.strip().lower()
        if current_rater in self.seenRaters:
            self.seenRaters.remove(current_rater)
        # Remove __selected_node__ if it exists (to avoid duplicates)
        if "__selected_node__" in self.seenRaters:
            self.seenRaters.remove("__selected_node__")
        # Remove __adjudicated_node__ if it exists (to avoid duplicates)
        if "__adjudicated_node__" in self.seenRaters:
            self.seenRaters.remove("__adjudicated_node__")
        # Remove current_rater if it exists (to avoid duplicates)
        if current_rater in self.seenRaters:
            self.seenRaters.remove(current_rater)
        # Now build the list: current_rater, __selected_node__, __adjudicated_node__, then sorted rest
        # we need to add __selected_node__ and __adjudicated_node__ to the list to ensure that the selected line is always visible and
        # uses a different color than the other lines and that the adjudicated_node colors aren't taken by the raters regardless of which module we are in.
        self.seenRaters = [current_rater, "__selected_node__", "__adjudicated_node__"] + sorted(self.seenRaters)
        # Select all real raters by default (exclude __selected_node__ and __adjudicated_node__)
        self.realRaters = [r for r in self.seenRaters if r != "__selected_node__" and r != "__adjudicated_node__"]
        self.setSelectedRaters(set(self.realRaters))

    def initializeMarkupNodesFromAnnotations(self):
        """
        Initialize markup nodes based on the maximum number needed across all frames.
        This ensures we start with the right number of nodes and don't need to create/remove them constantly.
        """
        if not self.annotations or 'frame_annotations' not in self.annotations:
            return

        # Calculate maximum number of lines needed across all frames
        max_pleura_lines = 0
        max_blines = 0

        for frame in self.annotations['frame_annotations']:
            pleura_count = len(frame.get('pleura_lines', []))
            bline_count = len(frame.get('b_lines', []))
            max_pleura_lines = max(max_pleura_lines, pleura_count)
            max_blines = max(max_blines, bline_count)

        # Add some buffer for new lines (at least 2 of each type, but cap at reasonable limits)
        max_pleura_lines = max(max_pleura_lines, 2)
        max_blines = max(max_blines, 2)

        # Cap the maximum to prevent excessive node creation
        max_pleura_lines = min(max_pleura_lines, 10)  # Cap at 10 pleura nodes
        max_blines = min(max_blines, 10)              # Cap at 10 B-line nodes

        # Clear existing nodes
        self.clearSceneLines()

        # Create the required number of nodes
        if self.useFreeList:
            for i in range(len(self.freeMarkupNodes), max_pleura_lines):
                node = self._allocateNewMarkupNode()
                self.freeMarkupNodes.append(node)
            for i in range(len(self.freeMarkupNodes), max_blines):
                node = self._allocateNewMarkupNode()
                self.freeMarkupNodes.append(node)

    def syncMarkupsToAnnotations(self):
        """
        One-way sync: Save current markup nodes to annotations for the current frame.
        This is the single source of truth for persisting markup changes.
        """
        if self.sequenceBrowserNode is None:
            logging.warning("No sequence browser node found, cannot sync markups to annotations.")
            return

        currentFrameIndex = max(0, self.sequenceBrowserNode.GetSelectedItemNumber())
        logging.debug(f"Syncing markups to annotations for frame {currentFrameIndex}")

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

        # Get current rater
        current_rater = self.getParameterNode().rater.strip().lower()

        # Remove only current rater's lines from existing annotations
        existing['pleura_lines'] = [
            line for line in existing['pleura_lines']
            if line.get("rater", "").strip().lower() != current_rater
        ]
        existing['b_lines'] = [
            line for line in existing['b_lines']
            if line.get("rater", "").strip().lower() != current_rater
        ]

        # Add current rater's pleura lines to annotations
        for markupNode in self.pleuraLines:
            nodeRater = markupNode.GetAttribute("rater") if markupNode else ""
            displayNode = markupNode.GetDisplayNode() if markupNode else None
            is_visible = displayNode.GetVisibility() if displayNode else False
            num_points = markupNode.GetNumberOfControlPoints() if markupNode else 0

            if nodeRater.strip().lower() != current_rater:
                continue  # Skip lines from other raters

            # Only save visible nodes with valid coordinates
            if not is_visible:
                continue  # Skip hidden nodes

            coordinates = []
            for j in range(num_points):
                coord = [0, 0, 0]
                markupNode.GetNthControlPointPosition(j, coord)
                coordinates.append(coord)

            if coordinates and len(coordinates) >= 2:  # Only save lines with at least 2 points
                line_data = {
                    "rater": markupNode.GetAttribute("rater"),
                    "line": {"points": coordinates}
                }
                existing['pleura_lines'].append(line_data)

        # Add current rater's B-lines to annotations
        for markupNode in self.bLines:
            nodeRater = markupNode.GetAttribute("rater") if markupNode else ""
            displayNode = markupNode.GetDisplayNode() if markupNode else None
            is_visible = displayNode.GetVisibility() if displayNode else False
            num_points = markupNode.GetNumberOfControlPoints() if markupNode else 0

            if nodeRater.strip().lower() != current_rater:
                continue  # Skip lines from other raters

            # Only save visible nodes with valid coordinates
            if not is_visible:
                continue  # Skip hidden nodes

            coordinates = []
            for j in range(num_points):
                coord = [0, 0, 0]
                markupNode.GetNthControlPointPosition(j, coord)
                coordinates.append(coord)

            if coordinates and len(coordinates) >= 2:  # Only save lines with at least 2 points
                line_data = {
                    "rater": markupNode.GetAttribute("rater"),
                    "line": {"points": coordinates}
                }
                existing['b_lines'].append(line_data)

    def syncAnnotationsToMarkups(self):
        """
        One-way sync: Update markup nodes from annotations for the current frame.
        This is the single source of truth for displaying annotations.
        """
        if not slicer.mrmlScene:
            return

        if self.annotations is None or 'frame_annotations' not in self.annotations:
            logging.debug("No annotations loaded")
            # remove all markup nodes
            for node in self.pleuraLines:
                node.GetDisplayNode().SetVisibility(False)
                self.pleuraLines.remove(node)
                self._freeMarkupNode(node)
            for node in self.bLines:
                node.GetDisplayNode().SetVisibility(False)
                self.bLines.remove(node)
                self._freeMarkupNode(node)
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
            # remove all markup nodes as we will recreate them in updateMarkupNodesForFrame
            for node in self.pleuraLines:
                node.GetDisplayNode().SetVisibility(False)
                self.pleuraLines.remove(node)
                self._freeMarkupNode(node)
            for node in self.bLines:
                node.GetDisplayNode().SetVisibility(False)
                self.bLines.remove(node)
                self._freeMarkupNode(node)

            if frame is not None:
                self._updateMarkupNodesForFrame(frame)
        finally:
            slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)
            # Reset programmatic update flag
            self._isProgrammaticUpdate = False

    def refreshDisplay(self, updateOverlay=True, updateGui=True):
        """
        Central method to refresh the display after any changes.
        This ensures consistent updates across all UI elements.
        """
        parameterNode = self.getParameterNode()

        # Update overlay volume if requested
        if updateOverlay:
            ratio = self.updateOverlayVolume()
            if ratio is not None:
                parameterNode.pleuraPercentage = ratio * 100
            else:
                parameterNode.pleuraPercentage = 0.0

        # Update GUI if requested and we have a widget
        if updateGui:
            try:
                widget = getAnnotateUltrasoundWidget()
                if widget:
                    widget.updateGuiFromAnnotations()
            except RuntimeError:
                # Widget not initialized yet, skip GUI update
                pass

    def updateDisplayForRaterChange(self):
        """
        Specialized method for when rater selection changes.
        This ensures the display is updated to show/hide the correct lines.
        """
        # Sync annotations to markups (this respects selectedRaters)
        self.syncAnnotationsToMarkups()

        # Refresh display
        self.refreshDisplay(updateOverlay=True, updateGui=True)

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
    import AnnotateUltrasound
    slicer.modules.annotateultrasound = AnnotateUltrasound.AnnotateUltrasound(slicer.qSlicerApplication().moduleManager())
