#!/usr/bin/env python3

import qt
from typing import Dict, Any, Optional

class SettingsManager:
    """Centralized settings management for ultrasound modules"""

    # Common settings keys
    COMMON_SETTINGS = {
        'INPUT_DIRECTORY': 'InputDirectory',
        'OUTPUT_DIRECTORY': 'OutputDirectory',
        'HEADERS_DIRECTORY': 'HeadersDirectory',
        'LABELS_PATH': 'LabelsPath',
        'CONTINUE_PROGRESS': 'ContinueProgress',
        'SKIP_SINGLE_FRAME': 'SkipSingleFrame',
    }

    # Module-specific settings
    ANNOTATE_SETTINGS = {
        'RATER_NAME': 'Rater',
        'SHOW_PLEURA_PERCENTAGE': 'ShowPleuraPercentage',
        'DEPTH_GUIDE': 'DepthGuide',
    }

    ANONYMIZE_SETTINGS = {
        'AUTO_MASK': 'AutoMask',
        'HASH_PATIENT_ID': 'HashPatientId',
        'FILENAME_PREFIX': 'FilenamePrefix',
    }

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.settings = qt.QSettings()

        # Combine settings based on module
        self.all_settings = self.COMMON_SETTINGS.copy()
        if module_name == 'AnnotateUltrasound':
            self.all_settings.update(self.ANNOTATE_SETTINGS)
        elif module_name == 'AnonymizeUltrasound':
            self.all_settings.update(self.ANONYMIZE_SETTINGS)

    def get_setting_key(self, setting_name: str) -> str:
        """Get full setting key with module prefix"""
        return f"{self.module_name}/{self.all_settings.get(setting_name, setting_name)}"

    def get_value(self, setting_name: str, default_value: Any = "") -> Any:
        """Get setting value"""
        key = self.get_setting_key(setting_name)
        return self.settings.value(key, default_value)

    def set_value(self, setting_name: str, value: Any) -> None:
        """Set setting value"""
        key = self.get_setting_key(setting_name)
        if value is not None and value != "":
            self.settings.setValue(key, value)
        else:
            self.settings.remove(key)

    def get_bool_value(self, setting_name: str, default_value: bool = False) -> bool:
        """Get boolean setting value"""
        value = self.get_value(setting_name, str(default_value))
        if isinstance(value, str):
            return value.lower() == 'true'
        return bool(value)

    def set_bool_value(self, setting_name: str, value: bool) -> None:
        """Set boolean setting value"""
        self.set_value(setting_name, str(value))

    # Convenience methods for common settings
    def get_input_directory(self) -> str:
        return self.get_value('INPUT_DIRECTORY')

    def set_input_directory(self, path: str):
        self.set_value('INPUT_DIRECTORY', path)

    def get_output_directory(self) -> str:
        return self.get_value('OUTPUT_DIRECTORY')

    def set_output_directory(self, path: str):
        self.set_value('OUTPUT_DIRECTORY', path)

    def get_labels_path(self) -> str:
        return self.get_value('LABELS_PATH')

    def set_labels_path(self, path: str):
        self.set_value('LABELS_PATH', path)
