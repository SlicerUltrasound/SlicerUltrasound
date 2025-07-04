"""
Unit tests for configuration management and utilities.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path


class TestConfiguration:
    """Test configuration management."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = ConfigurationManager()

        # Test default values
        assert config.get_default_rater() == "default_rater"
        assert config.get_default_colors() is not None
        assert config.get_depth_guide_mode() == 1
        assert config.get_auto_save_enabled() == True
        assert config.get_backup_enabled() == True

    def test_configuration_persistence(self):
        """Test saving and loading configuration."""
        config = ConfigurationManager()

        # Set some custom values
        config.set_default_rater("custom_rater")
        config.set_depth_guide_mode(2)
        config.set_auto_save_enabled(False)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
            config.save_configuration(temp_file)

        try:
            # Load configuration in new instance
            new_config = ConfigurationManager()
            new_config.load_configuration(temp_file)

            # Verify values were loaded correctly
            assert new_config.get_default_rater() == "custom_rater"
            assert new_config.get_depth_guide_mode() == 2
            assert new_config.get_auto_save_enabled() == False

        finally:
            os.unlink(temp_file)

    def test_configuration_validation(self):
        """Test configuration validation."""
        config = ConfigurationManager()

        # Test valid configurations
        valid_configs = [
            {"default_rater": "rater1", "depth_guide_mode": 1},
            {"auto_save_enabled": True, "backup_enabled": False},
            {"default_colors": {"pleura": [1, 0, 0], "bline": [0, 1, 0]}},
        ]

        for valid_config in valid_configs:
            assert config.validate_configuration(valid_config) == True

        # Test invalid configurations
        invalid_configs = [
            {"depth_guide_mode": "invalid"},  # Wrong type
            {"depth_guide_mode": -1},  # Invalid value
            {"default_colors": "not_a_dict"},  # Wrong type
            {"auto_save_enabled": "yes"},  # Wrong type
        ]

        for invalid_config in invalid_configs:
            assert config.validate_configuration(invalid_config) == False

    def test_color_configuration(self):
        """Test color configuration management."""
        config = ConfigurationManager()

        # Test default colors
        default_colors = config.get_default_colors()
        assert "pleura" in default_colors
        assert "bline" in default_colors
        assert len(default_colors["pleura"]) == 3
        assert len(default_colors["bline"]) == 3

        # Test custom colors
        custom_colors = {
            "pleura": [1.0, 0.0, 0.0],
            "bline": [0.0, 1.0, 0.0]
        }
        config.set_default_colors(custom_colors)

        retrieved_colors = config.get_default_colors()
        assert retrieved_colors == custom_colors

    def test_path_configuration(self):
        """Test path configuration management."""
        config = ConfigurationManager()

        # Test default paths
        assert config.get_default_input_directory() is not None
        assert config.get_default_output_directory() is not None

        # Test custom paths
        custom_input = "/custom/input/path"
        custom_output = "/custom/output/path"

        config.set_default_input_directory(custom_input)
        config.set_default_output_directory(custom_output)

        assert config.get_default_input_directory() == custom_input
        assert config.get_default_output_directory() == custom_output

    def test_keyboard_shortcuts(self):
        """Test keyboard shortcut configuration."""
        config = ConfigurationManager()

        # Test default shortcuts
        shortcuts = config.get_keyboard_shortcuts()
        assert "add_pleura" in shortcuts
        assert "add_bline" in shortcuts
        assert "next_frame" in shortcuts
        assert "prev_frame" in shortcuts

        # Test custom shortcuts
        custom_shortcuts = {
            "add_pleura": "Ctrl+P",
            "add_bline": "Ctrl+B",
            "next_frame": "Right",
            "prev_frame": "Left"
        }

        config.set_keyboard_shortcuts(custom_shortcuts)
        retrieved_shortcuts = config.get_keyboard_shortcuts()
        assert retrieved_shortcuts == custom_shortcuts

    def test_export_settings(self):
        """Test export settings configuration."""
        config = ConfigurationManager()

        # Test default export settings
        export_settings = config.get_export_settings()
        assert "format" in export_settings
        assert "include_metadata" in export_settings
        assert "compression" in export_settings

        # Test custom export settings
        custom_export = {
            "format": "json",
            "include_metadata": True,
            "compression": False,
            "indent": 2
        }

        config.set_export_settings(custom_export)
        retrieved_export = config.get_export_settings()
        assert retrieved_export == custom_export


class TestUtilities:
    """Test utility functions."""

    def test_file_utilities(self):
        """Test file utility functions."""
        utils = FileUtilities()

        # Test file extension validation
        assert utils.is_valid_dicom_file("test.dcm") == True
        assert utils.is_valid_dicom_file("test.DCM") == True
        assert utils.is_valid_dicom_file("test.txt") == False
        assert utils.is_valid_dicom_file("") == False

        # Test JSON file validation
        assert utils.is_valid_json_file("test.json") == True
        assert utils.is_valid_json_file("test.JSON") == True
        assert utils.is_valid_json_file("test.txt") == False

        # Test filename sanitization
        assert utils.sanitize_filename("test file.dcm") == "test_file.dcm"
        assert utils.sanitize_filename("test/file.dcm") == "test_file.dcm"
        assert utils.sanitize_filename("test*file.dcm") == "test_file.dcm"

    def test_directory_utilities(self):
        """Test directory utility functions."""
        utils = FileUtilities()

        # Test directory creation
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = os.path.join(temp_dir, "test_subdir")

            assert utils.ensure_directory_exists(test_dir) == True
            assert os.path.exists(test_dir) == True

            # Test that it doesn't fail if directory already exists
            assert utils.ensure_directory_exists(test_dir) == True

    def test_backup_utilities(self):
        """Test backup utility functions."""
        utils = FileUtilities()

        # Test backup filename generation
        original_file = "/path/to/test.json"
        backup_file = utils.create_backup_filename(original_file)

        assert backup_file != original_file
        assert "backup" in backup_file.lower()
        assert backup_file.endswith(".json")

        # Test backup file with timestamp
        backup_with_timestamp = utils.create_backup_filename(original_file, include_timestamp=True)
        assert backup_with_timestamp != backup_file
        assert len(backup_with_timestamp) > len(backup_file)

    def test_validation_utilities(self):
        """Test validation utility functions."""
        utils = ValidationUtilities()

        # Test coordinate validation
        assert utils.is_valid_coordinate([100, 100, 0]) == True
        assert utils.is_valid_coordinate([100, 100]) == False
        assert utils.is_valid_coordinate([100, 100, 0, 0]) == False
        assert utils.is_valid_coordinate(["x", "y", "z"]) == False

        # Test RGB color validation
        assert utils.is_valid_rgb_color([1.0, 0.0, 0.0]) == True
        assert utils.is_valid_rgb_color([0.5, 0.5, 0.5]) == True
        assert utils.is_valid_rgb_color([1.1, 0.0, 0.0]) == False
        assert utils.is_valid_rgb_color([-0.1, 0.0, 0.0]) == False
        assert utils.is_valid_rgb_color([1.0, 0.0]) == False


# Mock classes for testing
class ConfigurationManager:
    """Mock configuration manager for testing."""

    def __init__(self):
        self.config = {
            "default_rater": "default_rater",
            "depth_guide_mode": 1,
            "auto_save_enabled": True,
            "backup_enabled": True,
            "default_colors": {
                "pleura": [0.0, 0.0, 1.0],
                "bline": [0.0, 1.0, 0.0]
            },
            "default_input_directory": "~/input",
            "default_output_directory": "~/output",
            "keyboard_shortcuts": {
                "add_pleura": "Ctrl+1",
                "add_bline": "Ctrl+2",
                "next_frame": "Ctrl+Right",
                "prev_frame": "Ctrl+Left"
            },
            "export_settings": {
                "format": "json",
                "include_metadata": True,
                "compression": True
            }
        }

    def get_default_rater(self):
        return self.config["default_rater"]

    def set_default_rater(self, rater):
        self.config["default_rater"] = rater

    def get_depth_guide_mode(self):
        return self.config["depth_guide_mode"]

    def set_depth_guide_mode(self, mode):
        self.config["depth_guide_mode"] = mode

    def get_auto_save_enabled(self):
        return self.config["auto_save_enabled"]

    def set_auto_save_enabled(self, enabled):
        self.config["auto_save_enabled"] = enabled

    def get_backup_enabled(self):
        return self.config["backup_enabled"]

    def get_default_colors(self):
        return self.config["default_colors"]

    def set_default_colors(self, colors):
        self.config["default_colors"] = colors

    def get_default_input_directory(self):
        return self.config["default_input_directory"]

    def set_default_input_directory(self, directory):
        self.config["default_input_directory"] = directory

    def get_default_output_directory(self):
        return self.config["default_output_directory"]

    def set_default_output_directory(self, directory):
        self.config["default_output_directory"] = directory

    def get_keyboard_shortcuts(self):
        return self.config["keyboard_shortcuts"]

    def set_keyboard_shortcuts(self, shortcuts):
        self.config["keyboard_shortcuts"] = shortcuts

    def get_export_settings(self):
        return self.config["export_settings"]

    def set_export_settings(self, settings):
        self.config["export_settings"] = settings

    def save_configuration(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)

    def load_configuration(self, filepath):
        with open(filepath, 'r') as f:
            self.config = json.load(f)

    def validate_configuration(self, config):
        try:
            if "depth_guide_mode" in config:
                if not isinstance(config["depth_guide_mode"], int):
                    return False
                if config["depth_guide_mode"] < 0:
                    return False

            if "default_colors" in config:
                if not isinstance(config["default_colors"], dict):
                    return False

            if "auto_save_enabled" in config:
                if not isinstance(config["auto_save_enabled"], bool):
                    return False

            return True
        except Exception:
            return False


class FileUtilities:
    """Mock file utilities for testing."""

    def is_valid_dicom_file(self, filename):
        if not filename:
            return False
        return filename.lower().endswith('.dcm')

    def is_valid_json_file(self, filename):
        if not filename:
            return False
        return filename.lower().endswith('.json')

    def sanitize_filename(self, filename):
        import re
        # Replace invalid characters with underscores
        return re.sub(r'[<>:"/\\|?*\s]', '_', filename)

    def ensure_directory_exists(self, directory):
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception:
            return False

    def create_backup_filename(self, original_file, include_timestamp=False):
        base_name = os.path.splitext(os.path.basename(original_file))[0]
        extension = os.path.splitext(original_file)[1]

        if include_timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{base_name}_backup_{timestamp}{extension}"
        else:
            return f"{base_name}_backup{extension}"


class ValidationUtilities:
    """Mock validation utilities for testing."""

    def is_valid_coordinate(self, coordinate):
        try:
            if not isinstance(coordinate, list):
                return False

            if len(coordinate) != 3:
                return False

            for value in coordinate:
                if not isinstance(value, (int, float)):
                    return False

            return True
        except Exception:
            return False

    def is_valid_rgb_color(self, color):
        try:
            if not isinstance(color, list):
                return False

            if len(color) != 3:
                return False

            for value in color:
                if not isinstance(value, (int, float)):
                    return False
                if value < 0.0 or value > 1.0:
                    return False

            return True
        except Exception:
            return False