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
import pydicom
import qt
import shutil
import slicer
import vtk

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
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#AnnotateUltrasound">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
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
    depthGuideVisible: bool = False

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

        # Shortcuts
        self.shortcutW = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutW.setKey(qt.QKeySequence('W'))
        self.shortcutS = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutS.setKey(qt.QKeySequence('S'))
        self.shortcutSpace = qt.QShortcut(slicer.util.mainWindow())
        self.shortcutSpace.setKey(qt.QKeySequence('Space'))

        # Add shortcuts for removing lines
        self.shortcutE = qt.QShortcut(slicer.util.mainWindow())  # "E" for removing last pleura line
        self.shortcutE.setKey(qt.QKeySequence('E'))
        self.shortcutD = qt.QShortcut(slicer.util.mainWindow())  # "D" for removing last B-line
        self.shortcutD.setKey(qt.QKeySequence('D'))       

        # shortcut for saving and loading next scan
        self.shortcutA = qt.QShortcut(slicer.util.mainWindow())  # "A" for save and load next scan
        self.shortcutA.setKey(qt.QKeySequence('A'))

    def connectKeyboardShortcuts(self):
        # Connect shortcuts to respective actions
        self.shortcutW.connect('activated()', lambda: self.onAddLine("Pleura", not self.ui.addPleuraButton.isChecked()))
        self.shortcutS.connect('activated()', lambda: self.onAddLine("Bline", not self.ui.addBlineButton.isChecked()))
        self.shortcutSpace.connect('activated()', lambda: self.ui.overlayVisibilityButton.toggle())

        # New shortcuts for removing lines
        self.shortcutE.connect('activated()', lambda: self.onRemoveLine("Pleura"))  # "E" removes the last pleura line
        self.shortcutD.connect('activated()', lambda: self.onRemoveLine("Bline"))   # "D" removes the last B-line

        self.shortcutA.connect('activated()', self.onSaveButton)  # "A" to save and load next scan

    def disconnectKeyboardShortcuts(self):
        # Disconnect shortcuts to avoid issues when the user leaves the module
        self.shortcutW.activated.disconnect()
        self.shortcutS.activated.disconnect()
        self.shortcutSpace.activated.disconnect()      
        self.shortcutE.activated.disconnect()
        self.shortcutD.activated.disconnect()
        self.shortcutA.activated.disconnect()

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

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.connectKeyboardShortcuts() 


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
        self.ui.intensitySlider.valueChanged.connect(self.onIntensitySliderValueChanged)
        self.ui.skipToUnlabelledButton.clicked.connect(self.onSkipToUnlabelledButton)
        
        self.ui.addPleuraButton.toggled.connect(lambda checked: self.onAddLine("Pleura", checked))
        self.ui.removePleuraButton.clicked.connect(lambda: self.onRemoveLine("Pleura"))
        self.ui.addBlineButton.toggled.connect(lambda checked: self.onAddLine("Bline", checked))
        self.ui.removeBlineButton.clicked.connect(lambda: self.onRemoveLine("Bline"))
        self.ui.overlayVisibilityButton.toggled.connect(self.overlayVisibilityToggled)
        self.ui.clearAllLinesButton.clicked.connect(self.onClearAllLines)
        self.ui.addCurrentFrameButton.clicked.connect(self.onAddCurrentFrame)
        self.ui.removeCurrentFrameButton.clicked.connect(self.onRemoveCurrentFrame)

        # Assign icons to buttons
        self.ui.nextButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueFillNext.png')))
        self.ui.previousButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueFillPrevious.png')))
        self.ui.addPleuraButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueAdd.png')))
        self.ui.addBlineButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueAdd.png')))
        self.ui.saveButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueSave.png')))
        self.ui.removePleuraButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueRemove.png')))
        self.ui.removeBlineButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueRemove.png')))
        self.ui.overlayVisibilityButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueEye.png')))
        self.ui.clearAllLinesButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueFillTrash.png')))
        self.ui.skipToUnlabelledButton.setIcon(qt.QIcon(self.resourcePath('Icons/blueFastForward.png')))

        # Frame table
        self.ui.framesTableWidget.itemSelectionChanged.connect(self.onFramesTableSelectionChanged) 

        # Settings
        settings = slicer.app.settings()
        autoSave = settings.value('AnnotateUltrasound/AutoSave', 'true')
        self.ui.autoSaveCheckBox.setChecked(autoSave.lower() == 'true')
        showPleuraPercentage = settings.value('AnnotateUltrasound/ShowPleuraPercentage', 'false')
        self.ui.showPleuraPercentageCheckBox.setChecked(showPleuraPercentage.lower() == 'true')
        
        self.ui.autoSaveCheckBox.connect('toggled(bool)', self.saveUserSettings)
        self.ui.showPleuraPercentageCheckBox.connect('toggled(bool)', self.saveUserSettings)
        self.ui.depthGuideCheckBox.toggled.connect(self.onDepthGuideToggled)

        # Make buttons taller
        buttonHeight = 40  # Set the height you want for the buttons
        self.ui.inputDirectoryButton.setFixedHeight(buttonHeight)
        self.ui.readInputButton.setFixedHeight(buttonHeight)
        self.ui.nextButton.setFixedHeight(buttonHeight)
        self.ui.previousButton.setFixedHeight(buttonHeight)
        self.ui.saveButton.setFixedHeight(buttonHeight)
        self.ui.addPleuraButton.setFixedHeight(buttonHeight)
        self.ui.removePleuraButton.setFixedHeight(buttonHeight)
        self.ui.addBlineButton.setFixedHeight(buttonHeight)
        self.ui.removeBlineButton.setFixedHeight(buttonHeight)
        self.ui.overlayVisibilityButton.setFixedHeight(buttonHeight)
        self.ui.clearAllLinesButton.setFixedHeight(buttonHeight)
        self.ui.addCurrentFrameButton.setFixedHeight(buttonHeight)
        self.ui.removeCurrentFrameButton.setFixedHeight(buttonHeight)
        
        self.ui.labelsFileSelector.connect('currentPathChanged(QString)', self.onLabelsFileSelected)
        
        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
    
    def saveUserSettings(self):
        settings = qt.QSettings()
        settings.setValue('AnnotateUltrasound/AutoSave', 'true' if self.ui.autoSaveCheckBox.checked else 'false')
        settings.setValue('AnnotateUltrasound/ShowPleuraPercentage', self.ui.showPleuraPercentageCheckBox.checked)
        settings.setValue('AnnotateUltrasound/DepthGuide', self.ui.depthGuideCheckBox.checked)
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
        self.logic.clearAllLines()
        ratio = self.logic.updateOverlayVolume()
        self._parameterNode.pleuraPercentage = ratio * 100
        self._parameterNode.unsavedChanges = True

    def onFramesTableSelectionChanged(self):
        logging.info('onFramesTableSelectionChanged')

        selectedRow = self.ui.framesTableWidget.currentRow()
        if selectedRow == -1:
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
        self.logic.addCurrentFrame()
        self.updateGuiFromAnnotations()

    def onRemoveCurrentFrame(self):
        logging.info('removeCurrentFrame')

        # Get the current frame index from the sequence browser
        if self.logic.sequenceBrowserNode is None:
            logging.warning("No sequence browser node found")
            currentFrameIndex = -1
        else:
            currentFrameIndex = self.logic.sequenceBrowserNode.GetSelectedItemNumber()
            self.logic.removeFrame(currentFrameIndex)
            self.logic.updateLineMarkups()
            ratio = self.logic.updateOverlayVolume()
            self._parameterNode.pleuraPercentage = ratio * 100
            self.updateGuiFromAnnotations()

    def onInputDirectorySelected(self):
        logging.info('onInputDirectorySelected')
        
        inputDirectory = self.ui.inputDirectoryButton.directory
        if not inputDirectory:
            statusText = "Please select an input directory"
            self.ui.statusLabel.setText(statusText)
            return
        
        # Update local settings
        slicer.app.settings().setValue("AnnotateUltrasound/InputDirectory", inputDirectory)
        
    def onReadInputButton(self):
        """
        Read the input directory and update the dicomDf dataframe.
        
        :return: True if the input directory was read successfully, False otherwise.
        """
        logging.info('onReadInputButton')
        
        inputDirectory = self.ui.inputDirectoryButton.directory
        if not inputDirectory:
            statusText = "Please select an input directory"
            self.ui.statusLabel.setText(statusText)
            self._parameterNode.dfLoaded = False
            return
        
        numFilesFound, numAnnotationsCreated = self.logic.updateInputDf(inputDirectory)
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
        else:
            self.ui.statusLabel.setText("Could not find any files to load in input directory!")
            
        self._updateGUIFromParameterNode()

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

        # Create a dialog to ask the user to wait while the next sequence is loaded.

        waitDialog = self.createWaitDialog("Loading next sequence", "Loading next sequence...")
        
        # Saving settings
        showDepthGuide = self._parameterNode.depthGuideVisible

        currentDicomDfIndex = self.logic.loadNextSequence()

        # Uncheck all label checkboxes, but prevent them from triggering the onLabelCheckBoxToggled event while we are doing this
        for i in reversed(range(self.ui.labelsScrollAreaWidgetContents.layout().count())):
            groupBox = self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget()
            # Find all checkboxes in groupBox
            for j in reversed(range(groupBox.layout().count())):
                checkBox = groupBox.layout().itemAt(j).widget()
                if isinstance(checkBox, qt.QCheckBox):
                    checkBox.blockSignals(True)
                    checkBox.setChecked(False)
                    checkBox.blockSignals(False)
        
        # Update self.ui.currentFileLabel using the DICOM file name

        currentDicomFilepath = self.logic.dicomDf.iloc[self.logic.nextDicomDfIndex - 1]['Filepath']
        currentDicomFilename = os.path.basename(currentDicomFilepath)
        self.ui.currentFileLabel.setText(f"Current file ({self.logic.nextDicomDfIndex}/{len(self.logic.dicomDf)}): {currentDicomFilename}")
        
        self.updateGuiFromAnnotations()
        
        # Restore settings
        self._parameterNode.depthGuideVisible = showDepthGuide
        
        # Close the wait dialog
        waitDialog.close()

        self.ui.intensitySlider.setValue(0)

        if currentDicomDfIndex is None:
            self.ui.statusLabel.setText("No more DICOM files")
            return
        
        self.ui.progressBar.value = currentDicomDfIndex
        
        self.ui.overlayVisibilityButton.setChecked(True)

    def updateGuiFromAnnotations(self):
        # Check checkboxes in the labels scroll area if the labels are present in the logic.annotations
        if self.logic.annotations is not None and "labels" in self.logic.annotations:
            for i in reversed(range(self.ui.labelsScrollAreaWidgetContents.layout().count())): 
                groupBox = self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget()
                groupBoxTitle = groupBox.title
                # Find all checkboxes in groupBox
                for j in reversed(range(groupBox.layout().count())): 
                    checkBox = groupBox.layout().itemAt(j).widget()
                    if isinstance(checkBox, qt.QCheckBox):
                        annotationName = f"{groupBoxTitle}/{checkBox.text}"
                        checkBox.blockSignals(True)
                        if annotationName in self.logic.annotations['labels']:
                            checkBox.setChecked(True)
                        else:
                            checkBox.setChecked(False)
                        checkBox.blockSignals(False)
        else:
            # Uncheck all label checkboxes
            for i in reversed(range(self.ui.labelsScrollAreaWidgetContents.layout().count())): 
                groupBox = self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget()
                # Find all checkboxes in groupBox
                for j in reversed(range(groupBox.layout().count())): 
                    checkBox = groupBox.layout().itemAt(j).widget()
                    if isinstance(checkBox, qt.QCheckBox):
                        checkBox.setChecked(False)
        
        # Update frames table

        # Remove all rows from the table
        self.ui.framesTableWidget.setRowCount(0)

        # Add rows to the table
        if self.logic.annotations is not None and "frame_annotations" in self.logic.annotations:
            for frame_index, frame_annotations in self.logic.annotations["frame_annotations"].items():
                self.ui.framesTableWidget.insertRow(self.ui.framesTableWidget.rowCount)
                self.ui.framesTableWidget.setItem(self.ui.framesTableWidget.rowCount - 1, 0, qt.QTableWidgetItem(str(frame_index)))
                self.ui.framesTableWidget.setItem(self.ui.framesTableWidget.rowCount - 1, 1, qt.QTableWidgetItem(str(len(frame_annotations["pleura_lines"]))))
                self.ui.framesTableWidget.setItem(self.ui.framesTableWidget.rowCount - 1, 2, qt.QTableWidgetItem(str(len(frame_annotations["b_lines"]))))

    
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
        
        currentDicomDfIndex = self.logic.loadPreviousSequence()
        
        # Update self.ui.currentFileLabel using the DICOM file name
        currentDicomFilepath = self.logic.dicomDf.iloc[self.logic.nextDicomDfIndex - 1]['Filepath']
        currentDicomFilename = os.path.basename(currentDicomFilepath)
        self.ui.currentFileLabel.setText(f"Current file ({self.logic.nextDicomDfIndex}/{len(self.logic.dicomDf)}): {currentDicomFilename}")
        
        self.updateGuiFromAnnotations()

        self.ui.intensitySlider.setValue(0)
        
        # Close the wait dialog
        waitDialog.close()
        
        if currentDicomDfIndex is None:
            self.ui.statusLabel.setText("First DICOM file reached")
            return
        else:
            self.ui.progressBar.value = currentDicomDfIndex
    
    def saveAnnotations(self):
        """
        Saves current annotations to json file.
        """
        # Add annotation line control points to the annotations dictionary and save it to file
        if self.logic.annotations is None:
            logging.error("saveAnnotations: No annotations loaded")
            return
        
        waitDialog = self.createWaitDialog("Saving annotations", "Saving annotations...")

        # Check if any labels are checked
        annotationLabels = []
        for i in reversed(range(self.ui.labelsScrollAreaWidgetContents.layout().count())): 
            groupBox = self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget()
            groupBoxTitle = groupBox.title
            # Find all checkboxes in groupBox
            for j in reversed(range(groupBox.layout().count())):
                checkBox = groupBox.layout().itemAt(j).widget()
                if isinstance(checkBox, qt.QCheckBox) and checkBox.isChecked():
                    annotationLabels.append(f"{groupBoxTitle}/{checkBox.text}")
        self.logic.annotations['labels'] = annotationLabels
        
        # Save annotations to file
        annotationsFilepath = self.logic.dicomDf.iloc[self.logic.nextDicomDfIndex - 1]['AnnotationsFilepath']
        with open(annotationsFilepath, 'w') as f:
            json.dump(self.logic.annotations, f)
        
        waitDialog.close()

        self._parameterNode.unsavedChanges = False
        
        logging.info(f"Annotations saved to {annotationsFilepath}")

        return True

    def onSaveButton(self):
        """
        Saves current annotations to json file and loads next sequence.
        """
        logging.info('onSaveButton (save and load next scan)')

        if self.saveAnnotations():
            self.onNextButton()
        
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
            
            if self.ui.autoSaveCheckBox.checked:
                logging.info("Auto-saving frame annotations")
                self.logic.addCurrentFrame()
                self.updateGuiFromAnnotations()

            # If current line has less than 2 control points, remove it
            if len(linesList) > 0:
                currentLine = linesList[-1]
                if currentLine.GetNumberOfControlPoints() < 2:
                    linesList.pop()
                    slicer.mrmlScene.RemoveNode(currentLine)
            ratio = self.logic.updateOverlayVolume()
            self._parameterNode.pleuraPercentage = ratio * 100
            return
        
        # Put interaction model to place line markup
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        interactionNode.SetPlaceModePersistence(0)
        
        self.addObserver(interactionNode, interactionNode.EndPlacementEvent, self.onEndPlaceMode)
        
        # Create a new markup fiducial node
        if lineType == "Pleura":
            newLineNode = self.logic.createMarkupLine("Pleura", [], [0, 0.2, 1])
            self.logic.pleuraLines.append(newLineNode)
        elif lineType == "Bline":
            newLineNode = self.logic.createMarkupLine("B-line", [], [0, 1, 0.2])
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
    
    def delayedOnEndPlaceMode(self, lineType):
        logging.info(f"delayedOnEndPlaceMode -- lineType: {lineType}")
        if lineType == "Pleura":
            self.ui.addPleuraButton.setChecked(False)
        elif lineType == "Bline":
            self.ui.addBlineButton.setChecked(False)
        else:
            logging.error(f"Unknown line type {lineType}")
            return
        
        if self.ui.autoSaveCheckBox.checked:
            logging.info("Auto-saving frame annotations")
            self.logic.addCurrentFrame()
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
        
        self.logic.addCurrentFrame()

        self._parameterNode.unsavedChanges = True

    def onLabelsFileSelected(self, labelsFilepath):
        logging.info(f"onLabelsFileSelected -- labelsFilepath: {labelsFilepath}")
        settings = qt.QSettings()
        settings.setValue('AnnotateUltrasound/LabelsPath', labelsFilepath)
        
        categories = defaultdict(list)

        try:
            with open(labelsFilepath, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    category, label = map(str.strip, row)
                    categories[category].append(label)
        except (FileNotFoundError, PermissionError) as e:
            logging.error(f"Cannot read labels file: {labelsFilepath}, error: {e}")

        # Remove all existing labels from the scroll area
        for i in reversed(range(self.ui.labelsScrollAreaWidgetContents.layout().count())): 
            # self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget().setParent(None)
            self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget().deleteLater()

        # Populate labels scroll area
        for category, labels in categories.items():
            categoryGroupBox = qt.QGroupBox(category, self.ui.labelsScrollAreaWidgetContents)
            categoryLayout = qt.QVBoxLayout(categoryGroupBox)
            for label in labels:
                checkBox = qt.QCheckBox(label, categoryGroupBox)
                checkBox.toggled.connect(self.onLabelCheckBoxToggled)
                categoryLayout.addWidget(checkBox)
            categoryGroupBox.setLayout(categoryLayout)
            self.ui.labelsScrollAreaWidgetContents.layout().addWidget(categoryGroupBox)
    
    def onLabelCheckBoxToggled(self, checked):
        logging.info(f"onLabelCheckBoxToggled -- checked: {checked}")
        if self.logic.annotations is None:
            logging.error("No annotations loaded")
            return
        if "labels" not in self.logic.annotations:
            self.logic.annotations["labels"] = []
        
        # Update all labels in the annotations dictionary
        annotationLabels = []
        for i in reversed(range(self.ui.labelsScrollAreaWidgetContents.layout().count())): 
            groupBox = self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget()
            groupBoxTitle = groupBox.title
            # Find all checkboxes in groupBox
            for j in reversed(range(groupBox.layout().count())):
                checkBox = groupBox.layout().itemAt(j).widget()
                if isinstance(checkBox, qt.QCheckBox) and checkBox.isChecked():
                    annotationLabels.append(f"{groupBoxTitle}/{checkBox.text}")
        self.logic.annotations['labels'] = annotationLabels

    def onSkipToUnlabelledButton(self):
        """ 
        Skip to the next unlabelled scan
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

        # Find the next unlabelled scan
        nextUnlabelledIndex = self.findNextUnlabelledScan()
        if nextUnlabelledIndex is None:
            qt.QMessageBox.information(
                slicer.util.mainWindow(),
                "Skip to Unlabelled",
                "No unlabelled scans found."
            )
            return

        self.logic.nextDicomDfIndex = nextUnlabelledIndex
        self.onNextButton()

    def findNextUnlabelledScan(self):
        """
        Find the index of the next unlabelled scan in the DICOM dataframe.
        :return: Index of the next unlabelled scan or None if no such scan is found.
        """
        if self.logic.dicomDf is None:
            return None

        for idx in range(self.logic.nextDicomDfIndex, len(self.logic.dicomDf)):
            annotationsFilepath = self.logic.dicomDf.iloc[idx]['AnnotationsFilepath']
            try:
                with open(annotationsFilepath, 'r') as f:
                    annotations = json.load(f)
                    # Check if frame annotations exist and are empty
                    if 'frame_annotations' not in annotations or not annotations['frame_annotations']:
                        return idx
            except Exception as e:
                logging.error(f"Error reading annotations file {annotationsFilepath}: {e}")
        
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
        # Save new state in application settings
        settings = slicer.app.settings()
        if toggled:
            settings.setValue('AnnotateUltrasound/DepthGuide', "True")
        else:
            settings.setValue('AnnotateUltrasound/DepthGuide', "False")
        
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

        # Load labels for annotations
        
        moduleWidget = slicer.modules.annotateultrasound.widgetRepresentation().self()
        settings = qt.QSettings()

        labelsPath = settings.value('AnnotateUltrasound/LabelsPath', '')
        if labelsPath == '':
            labelsPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Resources/default_labels.csv')
        moduleWidget.ui.labelsFileSelector.currentPath = labelsPath
        self.onLabelsFileSelected(labelsPath)
        
        # Hide slice view annotations to avoid interference with the corner annotation
        
        sliceAnnotations = slicer.modules.DataProbeInstance.infoWidget.sliceAnnotations
        sliceAnnotations.sliceViewAnnotationsEnabled=False
        sliceAnnotations.updateSliceViewFromGUI()

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

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode
        
        settings = slicer.app.settings()
        showDepthGuide = settings.value('AnnotateUltrasound/DepthGuide', 'false')
        self.ui.depthGuideCheckBox.setChecked(showDepthGuide.lower() == 'true')
        

    def setParameterNode(self, inputParameterNode: AnnotateUltrasoundParameterNode) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateGUIFromParameterNode)
            self._updateGUIFromParameterNode()

    def _updateGUIFromParameterNode(self, caller=None, event=None) -> None:
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
        
        self.logic.updateOverlayVolume()

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

    def getParameterNode(self):
        return AnnotateUltrasoundParameterNode(super().getParameterNode())
    
    def updateInputDf(self, input_folder):
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
            for file in files:
                progress_dialog.setValue(file_count)
                slicer.app.processEvents()

                # Construct the full file path
                file_path = os.path.join(root, file)

                try:
                    # Try to read the file as a DICOM file
                    dicom_file = pydicom.dcmread(file_path)

                    # Extract required information
                    patient_uid = dicom_file.PatientID if 'PatientID' in dicom_file else None
                    study_uid = dicom_file.StudyInstanceUID if 'StudyInstanceUID' in dicom_file else None
                    series_uid = dicom_file.SeriesInstanceUID if 'SeriesInstanceUID' in dicom_file else None
                    instance_uid = dicom_file.SOPInstanceUID if 'SOPInstanceUID' in dicom_file else None
                    
                    # Check if there is an annotations json file in the same folder with the same filename
                    annotations_file_path = os.path.join(root, file.replace('.dcm', '.json'))
                    if not os.path.exists(annotations_file_path):
                        logging.warning(f"Annotations file not found for {file_path}. A blank annotations file will be created.")
                        # Create a blank annotations file
                        with open(annotations_file_path, 'w') as f:
                            f.write('{}')
                        annotations_created_count += 1

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
    
    def addCurrentFrame(self):
        logging.info('addCurrentFrame')
        if self.sequenceBrowserNode is None:
            logging.warning("No sequence browser node found")
            return
        
        # Get the current frame index from the sequence browser
        currentFrameIndexStr = str(self.sequenceBrowserNode.GetSelectedItemNumber())
        
        # Check if annotations already has a list of frame annotations. Create it if it doesn't exist.

        if 'frame_annotations' not in self.annotations:
            self.annotations['frame_annotations'] = {}

        # Check if the current frame index is already in the list of frame annotations. Create it if it doesn't exist.

        if currentFrameIndexStr not in self.annotations['frame_annotations']:
            self.annotations['frame_annotations'][currentFrameIndexStr] = {
                "pleura_lines": [],
                "b_lines": []
            }
        
        # Add pleura lines to annotations. Organize the coordinates in a list of lists.

        self.annotations['frame_annotations'][currentFrameIndexStr]['pleura_lines'] = []  # Reset the list of pleura lines
        for markupNode in self.pleuraLines:
            coordinates = []
            for i in range(markupNode.GetNumberOfControlPoints()):
                coord = [0, 0, 0]
                markupNode.GetNthControlPointPosition(i, coord)
                coordinates.append(coord)
            self.annotations['frame_annotations'][currentFrameIndexStr]['pleura_lines'].append(coordinates)
            
        # Add B-lines to annotations. Organize the coordinates in a list of lists.

        self.annotations['frame_annotations'][currentFrameIndexStr]['b_lines'] = []  # Reset the list of B-lines
        for markupNode in self.bLines:
            coordinates = []
            for i in range(markupNode.GetNumberOfControlPoints()):
                coord = [0, 0, 0]
                markupNode.GetNthControlPointPosition(i, coord)
                coordinates.append(coord)
            self.annotations['frame_annotations'][currentFrameIndexStr]['b_lines'].append(coordinates)

    def removeFrame(self, frameIndex):
        logging.info(f"removeFrame -- frameIndex: {frameIndex}")
        if self.annotations is None:
            logging.warning("No annotations loaded")
            return
        
        # Remove the frame index from the list of frame annotations
        if str(frameIndex) in self.annotations['frame_annotations']:
            self.annotations['frame_annotations'].pop(str(frameIndex))
        

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
        
    def loadNextSequence(self):
        """
        Load the next sequence in the dataframe.
        Returns the index of the loaded sequence in the dataframe or None if no more sequences are available.
        """
        # Clear the scene
        self.clearScene()
        parameterNode = self.getParameterNode()

        if self.dicomDf is None:
            parameterNode.dfLoaded = False
            return None
        else:
            parameterNode.dfLoaded = True
        
        if self.nextDicomDfIndex >= len(self.dicomDf):
            return None
        
        nextDicomFilepath = self.dicomDf.iloc[self.nextDicomDfIndex]['Filepath']
        nextAnnotationsFilepath = self.dicomDf.iloc[self.nextDicomDfIndex]['AnnotationsFilepath']
        self.nextDicomDfIndex += 1
        
        # Make sure a temporary folder for the DICOM files exists
        tempDicomDir = slicer.app.temporaryPath + '/AnonymizeUltrasound'
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
        parameterNode.inputVolume = inputUltrasoundNode
        
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
        # overlayVolume.CreateDefaultDisplayNodes()
        slicer.util.updateVolumeFromArray(overlayVolume, maskArray)
        parameterNode.overlayVolume = overlayVolume

        # Load annotations file
        with open(nextAnnotationsFilepath, 'r') as f:
            self.annotations = json.load(f)
        
        self.updateLineMarkups()
        ratio = self.updateOverlayVolume()
        parameterNode = self.getParameterNode()
        parameterNode.pleuraPercentage = ratio * 100
        
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
        self.updateLineMarkups()
        ratio = self.updateOverlayVolume()
        parameterNode = self.getParameterNode()
        parameterNode.pleuraPercentage = ratio * 100

    def createMarkupLine(self, name, coordinates, color=[1, 1, 0]):
        markupNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
        markupNode.CreateDefaultDisplayNodes()
        markupNode.SetName(name)
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
        Remove all pleura lines and B-lines from the scene and from the list of lines. Adds current frame annotations to dictionary.
        """
        self.clearSceneLines()
        self.addCurrentFrame()

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
            self.updateOverlayVolume()
    
    def onPointModified(self, caller, event):
        ratio = self.updateOverlayVolume()
        self.addCurrentFrame()
        parameterNode = self.getParameterNode()
        parameterNode.pleuraPercentage = ratio * 100
        parameterNode.unsavedChanges = True
    
    def onPointPositionDefined(self, caller, event):
        parameterNode = self.getParameterNode()
        numControlPoints = caller.GetNumberOfControlPoints()
        if numControlPoints >= 2:
            parameterNode.lineBeingPlaced = None
            self.removeObserver(caller, caller.PointPositionDefinedEvent, self.onPointPositionDefined)
        
        ratio = self.updateOverlayVolume()
        parameterNode = self.getParameterNode()
        parameterNode.pleuraPercentage = ratio * 100
    
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
        # Check if ultrasound is fan shape or rectangular
        # If self.annotations["mask_type"] doesn't exist, then assume it's rectangular
        if "mask_type" not in self.annotations:
            logging.error("No mask type found in annotations. Assuming rectangular mask.")

        if "mask_type" not in self.annotations or self.annotations["mask_type"] != "fan":
            # Create a rectangular mask
            maskArray = np.zeros(imageArray.shape, dtype=np.uint8)
            maskArray[:, point1[1]:point2[1], point1[0]:point2[0]] = value
            return maskArray
        else:
            radius1 = self.annotations["radius1"]
            radius2 = self.annotations["radius2"]
            center_rows_px = self.annotations["center_rows_px"]
            center_cols_px = self.annotations["center_cols_px"]
            a, b, c, d = self.fanCornersFromSectorLine(point1[:2], point2[:2],
                                                       (center_cols_px, center_rows_px),
                                                       radius1, radius2)
            maskArray = self.createFanMask(imageArray, a, b, c, d, value)
        
        return maskArray
    
    def updateLineMarkups(self):
        """
        Update the line markups to match the annotations at the current frame index. Clear markups if current frame index not in annotations.
        """
        self.clearSceneLines()

        if self.annotations is None:
            logging.warning("No annotations loaded")
            return
        
        if self.sequenceBrowserNode is None:
            logging.warning("No sequence browser node found")
            return
        
        currentFrameIndex = self.sequenceBrowserNode.GetSelectedItemNumber()
        
        if 'frame_annotations' not in self.annotations:
            logging.debug("No frame annotations found")
            return
        
        if str(currentFrameIndex) not in self.annotations['frame_annotations']:  # No annotations for current frame
            return
        
        # Add pleura lines
        for coordinates in self.annotations['frame_annotations'][str(currentFrameIndex)]['pleura_lines']:
            self.pleuraLines.append(self.createMarkupLine("Pleura", coordinates, [0, 0.2, 1]))
        
        # Add B-lines
        for coordinates in self.annotations['frame_annotations'][str(currentFrameIndex)]['b_lines']:
            self.bLines.append(self.createMarkupLine("B-line", coordinates, [0, 1, 0.2]))

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

        # Choose visualization based on depthGuideMode
        if self.depthGuideMode == 1:
            # Mode 1: Default dashed line
            return self._drawDashedLine(image_size_rows, image_size_cols, center_cols_px, center_rows_px, 
                                        depth_radius, angle1, angle2, color, thickness=4, dash_length=20, dash_gap=16)
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

    def updateOverlayVolume(self):
        """
        Update the overlay volume based on the annotations.

        :return: The ratio of green pixels to blue pixels in the overlay volume. None if inputs not defined yet.
        """
        # # TODO remove (debugging)
        # import inspect 
        # caller = inspect.stack()[1].function
        # print(f"updateOverlayVolume called by {caller}") 
        # print(f"Overlay volume updated with depth guide: {self.depthGuideEnabled}") 
        parameterNode = self.getParameterNode()
        
        if parameterNode.overlayVolume is None:
            logging.debug("updateOverlayVolume: No overlay volume found! Cannot update overlay volume.")
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
        
        if parameterNode.depthGuideVisible:
            # draw the depth guide lines
            depth_guide = self.drawDepthGuideLine(ultrasoundArray.shape[1], ultrasoundArray.shape[2])
            maskArray[0, :, :, :] = np.maximum(maskArray[0, :, :, :], depth_guide)
        
        ijkToRas = vtk.vtkMatrix4x4()
        parameterNode.inputVolume.GetIJKToRASMatrix(ijkToRas)
        rasToIjk = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(ijkToRas, rasToIjk)
        
        # Add pleura lines to mask array
        for markupNode in self.pleuraLines:
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
                
        # Add B-lines to mask array
        for markupNode in self.bLines:
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
        
        # Update the overlay volume
        slicer.util.updateVolumeFromArray(parameterNode.overlayVolume, maskArray)

        # Return the ratio of green pixels to blue pixels
        if bluePixels == 0:
            return 0.0
        else:
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


#
# AnnotateUltrasoundTest
#

class AnnotateUltrasoundTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_AnnotateUltrasound1()

    def test_AnnotateUltrasound1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
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
        postModuleDiscoveryTasks()
        inputVolume = SampleData.downloadSample('AnnotateUltrasound1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = AnnotateUltrasoundLogic()

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

        self.delayDisplay('Test passed')
