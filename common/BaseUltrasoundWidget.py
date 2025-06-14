#!/usr/bin/env python3

import os

import qt
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleWidget
from slicer.util import VTKObservationMixin
from typing import Optional


from .ErrorHandler import ErrorHandler
from .LabelsManager import LabelsManager
from .SettingsManager import SettingsManager
from .UIHelpers import UIHelpers

class BaseUltrasoundWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Base class for ultrasound module widgets with common functionality"""

    def __init__(self, parent=None, module_name: Optional[str] = None):
        super().__init__(parent)
        VTKObservationMixin.__init__(self)

        self.module_name = module_name or self.__class__.__name__.replace('Widget', '')
        self.settings_manager = SettingsManager(self.module_name)
        self.labels_manager = None
        self.logic = None
        self.parameterNode = None
        self.parameterNodeGuiTag = None
        self.compositing_mode_exit = None

    def setup_common_ui(self, ui_file_path: str):
        """Setup common UI elements"""
        # Load UI file
        ui_widget = slicer.util.loadUI(ui_file_path)
        self.layout.addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        ui_widget.setMRMLScene(slicer.mrmlScene)

        # Setup labels manager if labels UI exists
        if hasattr(self.ui, 'labelsScrollAreaWidgetContents'):
            self.labels_manager = LabelsManager(self.ui.labelsScrollAreaWidgetContents)

    def setup_directory_connections(self):
        """Setup directory button connections"""
        directory_mappings = [
            ('inputDirectoryButton', 'INPUT_DIRECTORY'),
            ('outputDirectoryButton', 'OUTPUT_DIRECTORY'),
            ('headersDirectoryButton', 'HEADERS_DIRECTORY'),
        ]

        for button_name, setting_name in directory_mappings:
            if hasattr(self.ui, button_name):
                button = getattr(self.ui, button_name)

                # Load saved directory
                saved_dir = self.settings_manager.get_value(setting_name)
                if saved_dir and os.path.exists(saved_dir):
                    button.directory = saved_dir

                # Connect to save changes
                button.connect("directoryChanged(QString)",
                             lambda new_value, sn=setting_name: self.settings_manager.set_value(sn, new_value))

    def setup_labels_ui(self, default_labels_file: Optional[str] = None):
        """Setup labels UI"""
        if not self.labels_manager:
            return

        if hasattr(self.ui, 'labelsFileSelector'):
            # Load saved labels path or use default
            labels_path = self.settings_manager.get_labels_path()
            if not labels_path and default_labels_file:
                labels_path = default_labels_file

            if labels_path:
                self.ui.labelsFileSelector.currentPath = labels_path
                self.on_labels_path_changed(labels_path)

            # Connect file selector
            self.ui.labelsFileSelector.connect('currentPathChanged(QString)',
                                             self.on_labels_path_changed)

    def on_labels_path_changed(self, file_path: str):
        """Handle labels file path change"""
        self.settings_manager.set_labels_path(file_path)
        if self.labels_manager.load_labels_from_file(file_path):
            self.labels_manager.populate_ui(self.on_label_checkbox_toggled)

    def on_label_checkbox_toggled(self, checked: bool):
        """Handle label checkbox toggle - to be implemented by subclasses"""
        pass

    def setup_collapsible_sections(self):
        """Setup collapsible section behavior"""
        # Collapse settings by default
        if hasattr(self.ui, 'settingsCollapsibleButton'):
            self.ui.settingsCollapsibleButton.collapsed = True

        # Setup other collapsible sections based on data state
        self.update_collapsible_sections()

    def update_collapsible_sections(self):
        """Update collapsible sections based on data state - to be implemented by subclasses"""
        pass

    def enter_module(self):
        """Common enter module functionality"""
        # Collapse DataProbe widget
        mw = slicer.util.mainWindow()
        if mw:
            w = slicer.util.findChild(mw, "DataProbeCollapsibleWidget")
            if w:
                w.collapsed = True

        # Save and set compositing mode
        slice_composite_node = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode()
        self.compositing_mode_exit = slice_composite_node.GetCompositing()
        if self.compositing_mode_exit != 2:
            slice_composite_node.SetCompositing(2)

    def exit_module(self):
        """Common exit module functionality"""
        # Restore compositing mode
        if self.compositing_mode_exit is not None:
            slice_composite_node = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode()
            slice_composite_node.SetCompositing(self.compositing_mode_exit)

    def validate_directories(self, required_dirs: list) -> bool:
        """Validate required directories exist"""
        directory_map = {
            'input': ('inputDirectoryButton', 'input directory'),
            'output': ('outputDirectoryButton', 'output directory'),
            'headers': ('headersDirectoryButton', 'headers directory'),
        }

        for dir_type in required_dirs:
            if dir_type in directory_map:
                button_name, dir_name = directory_map[dir_type]
                if hasattr(self.ui, button_name):
                    directory = getattr(self.ui, button_name).directory
                    if not ErrorHandler.validate_directory_exists(directory, dir_name):
                        return False
        return True

    def create_wait_dialog(self, title: str, message: str) -> qt.QDialog:
        """Create wait dialog"""
        return UIHelpers.create_wait_dialog(title, message)
