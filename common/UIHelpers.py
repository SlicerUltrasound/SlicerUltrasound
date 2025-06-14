#!/usr/bin/env python3

import os
import qt
import slicer
from typing import List, Dict, Optional, Callable
from collections import defaultdict
import csv
import logging

class UIHelpers:
    """Common UI helper functions for ultrasound modules"""

    @staticmethod
    def create_wait_dialog(title: str, message: str, parent=None) -> qt.QDialog:
        """Create a wait dialog"""
        if parent is None:
            parent = slicer.util.mainWindow()

        dialog = qt.QDialog(parent)
        dialog.setWindowTitle(title)
        layout = qt.QVBoxLayout(dialog)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(4)
        layout.addStretch(1)

        label = qt.QLabel(message)
        label.setAlignment(qt.Qt.AlignCenter)
        layout.addWidget(label)
        layout.addStretch(1)

        dialog.show()
        slicer.app.processEvents()
        return dialog

    @staticmethod
    def setup_button_styling(buttons_config: Dict[qt.QPushButton, str],
                           resource_path_func: Callable, height: int = 40):
        """Setup button icons and heights"""
        for button, icon_name in buttons_config.items():
            # Set height
            button.setFixedHeight(height)

            # Set icon if provided
            if icon_name:
                icon_path = resource_path_func(f'Icons/{icon_name}')
                if os.path.exists(icon_path):
                    button.setIcon(qt.QIcon(icon_path))

    @staticmethod
    def show_confirmation_dialog(title: str, message: str,
                               buttons: int = qt.QMessageBox.Save | qt.QMessageBox.Discard | qt.QMessageBox.Cancel,
                               parent=None) -> int:
        """Show confirmation dialog"""
        if parent is None:
            parent = slicer.util.mainWindow()
        return qt.QMessageBox.question(parent, title, message, buttons)

    @staticmethod
    def show_error_dialog(title: str, message: str, parent=None):
        """Show error dialog"""
        if parent is None:
            parent = slicer.util.mainWindow()
        qt.QMessageBox.critical(parent, title, message)

    @staticmethod
    def show_warning_dialog(title: str, message: str, parent=None):
        """Show warning dialog"""
        if parent is None:
            parent = slicer.util.mainWindow()
        qt.QMessageBox.warning(parent, title, message)

    @staticmethod
    def show_info_dialog(title: str, message: str, parent=None):
        """Show info dialog"""
        if parent is None:
            parent = slicer.util.mainWindow()
        qt.QMessageBox.information(parent, title, message)
