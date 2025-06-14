#!/usr/bin/env python3

import logging
import qt
import slicer
from typing import Optional

class ErrorHandler:
    """Centralized error handling for ultrasound modules"""

    @staticmethod
    def handle_file_error(operation: str, filepath: str, error: Exception,
                         show_user_message: bool = True) -> bool:
        """Handle file operation errors"""
        error_msg = f"Failed to {operation} file {filepath}: {error}"
        logging.error(error_msg)

        if show_user_message:
            ErrorHandler.show_error_message(
                f"File {operation.title()} Error",
                f"Could not {operation} file:\n{filepath}\n\nError: {str(error)}"
            )
        return False

    @staticmethod
    def show_error_message(title: str, message: str, parent=None):
        """Show error dialog to user"""
        if parent is None:
            parent = slicer.util.mainWindow()
        qt.QMessageBox.critical(parent, title, message)

    @staticmethod
    def show_warning_message(title: str, message: str, parent=None):
        """Show warning dialog to user"""
        if parent is None:
            parent = slicer.util.mainWindow()
        qt.QMessageBox.warning(parent, title, message)

    @staticmethod
    def validate_required_field(field_name: str, value: str,
                               show_user_message: bool = True) -> bool:
        """Validate that required field is not empty"""
        if not value or not value.strip():
            if show_user_message:
                ErrorHandler.show_warning_message(
                    f"Missing {field_name}",
                    f"Please enter a {field_name.lower()} before continuing."
                )
            return False
        return True

    @staticmethod
    def validate_directory_exists(directory: str, directory_type: str = "directory") -> bool:
        """Validate that directory exists"""
        import os
        if not directory:
            ErrorHandler.show_error_message(
                "Missing Directory",
                f"Please select a {directory_type}"
            )
            return False

        if not os.path.exists(directory):
            ErrorHandler.show_error_message(
                "Directory Not Found",
                f"The {directory_type} does not exist:\n{directory}"
            )
            return False

        return True
