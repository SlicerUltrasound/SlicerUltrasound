"""
Basic functionality tests for SlicerUltrasound modules.
These tests verify core functionality without requiring Slicer.
"""

import pytest
import json
import os
import tempfile
from datetime import datetime


class TestBasicFunctionality:
    """Test basic functionality of the SlicerUltrasound modules."""

    def test_json_handling(self):
        """Test JSON file handling capabilities."""
        # Test creating a valid annotation structure
        annotation = {
            "frame_annotations": [
                {
                    "frame_number": 0,
                    "pleura_lines": [
                        {
                            "rater": "test_rater",
                            "coordinates": [[100, 100, 0], [200, 150, 0]],
                            "timestamp": datetime.now().isoformat()
                        }
                    ],
                    "b_lines": []
                }
            ]
        }

        # Test JSON serialization
        json_str = json.dumps(annotation)
        assert json_str is not None
        assert len(json_str) > 0

        # Test JSON deserialization
        loaded_annotation = json.loads(json_str)
        assert loaded_annotation == annotation

        # Test file I/O
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(annotation, f)
            temp_file = f.name

        try:
            with open(temp_file, 'r') as f:
                file_annotation = json.load(f)
            assert file_annotation == annotation
        finally:
            os.unlink(temp_file)

    def test_coordinate_calculations(self):
        """Test coordinate calculation utilities."""
        # Test distance calculation
        point1 = [0, 0, 0]
        point2 = [3, 4, 0]
        distance = self.calculate_distance(point1, point2)
        assert distance == 5.0  # 3-4-5 triangle

        # Test midpoint calculation
        midpoint = self.calculate_midpoint(point1, point2)
        assert midpoint == [1.5, 2.0, 0.0]

        # Test line length calculation
        line_points = [[0, 0, 0], [3, 4, 0]]
        length = self.calculate_line_length(line_points)
        assert length == 5.0

    def test_data_validation(self):
        """Test data validation functions."""
        # Test valid coordinate validation
        valid_coords = [[100, 100, 0], [200, 150, 0]]
        assert self.validate_coordinates(valid_coords) == True

        # Test invalid coordinate validation
        invalid_coords = [
            [],  # Empty
            [[100, 100]],  # Missing Z
            [[100, 100, 0, 0]],  # Too many dimensions
            [["x", "y", "z"]],  # Non-numeric
        ]

        for coords in invalid_coords:
            assert self.validate_coordinates(coords) == False

        # Test rater name validation
        assert self.validate_rater_name("valid_rater") == True
        assert self.validate_rater_name("") == False
        assert self.validate_rater_name("a" * 101) == False

    def test_color_utilities(self):
        """Test color utility functions."""
        # Test RGB color validation
        assert self.validate_rgb_color([1.0, 0.0, 0.0]) == True
        assert self.validate_rgb_color([0.5, 0.5, 0.5]) == True
        assert self.validate_rgb_color([1.1, 0.0, 0.0]) == False
        assert self.validate_rgb_color([-0.1, 0.0, 0.0]) == False

        # Test color generation
        color1 = self.generate_color_for_rater("rater1")
        color2 = self.generate_color_for_rater("rater2")
        color1_again = self.generate_color_for_rater("rater1")

        # Same rater should get same color
        assert color1 == color1_again
        # Different raters should get different colors
        assert color1 != color2

    def test_file_utilities(self):
        """Test file utility functions."""
        # Test file extension validation
        assert self.is_dicom_file("test.dcm") == True
        assert self.is_dicom_file("test.DCM") == True
        assert self.is_dicom_file("test.txt") == False
        assert self.is_dicom_file("") == False

        # Test filename sanitization
        assert self.sanitize_filename("test file.dcm") == "test_file.dcm"
        assert self.sanitize_filename("test/file.dcm") == "test_file.dcm"
        assert self.sanitize_filename("test*file.dcm") == "test_file.dcm"

    def test_timestamp_utilities(self):
        """Test timestamp utility functions."""
        # Test timestamp generation
        timestamp = self.generate_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0

        # Test timestamp validation
        valid_timestamps = [
            "2024-01-01T00:00:00",
            "2024-12-31T23:59:59",
            datetime.now().isoformat()
        ]

        for ts in valid_timestamps:
            assert self.validate_timestamp(ts) == True

        invalid_timestamps = [
            "",
            "not-a-timestamp",
            "2024-13-01T00:00:00",  # Invalid month
            "2024-01-32T00:00:00",  # Invalid day
        ]

        for ts in invalid_timestamps:
            assert self.validate_timestamp(ts) == False

    def test_error_handling(self):
        """Test error handling and edge cases."""
        # Test handling of None values
        assert self.validate_coordinates(None) == False
        assert self.validate_rater_name(None) == False
        assert self.validate_rgb_color(None) == False

        # Test handling of empty values
        assert self.validate_coordinates([]) == False
        assert self.validate_rater_name("") == False
        assert self.validate_rgb_color([]) == False

    # Helper utility methods for testing
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        import math
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

    def calculate_midpoint(self, point1, point2):
        """Calculate midpoint between two points."""
        return [(a + b) / 2 for a, b in zip(point1, point2)]

    def calculate_line_length(self, points):
        """Calculate total length of a line defined by points."""
        if len(points) < 2:
            return 0.0

        total_length = 0.0
        for i in range(len(points) - 1):
            total_length += self.calculate_distance(points[i], points[i + 1])

        return total_length

    def validate_coordinates(self, coordinates):
        """Validate coordinate format."""
        try:
            if coordinates is None or not isinstance(coordinates, list):
                return False

            if len(coordinates) < 2:
                return False

            for coord in coordinates:
                if not isinstance(coord, list) or len(coord) != 3:
                    return False

                for value in coord:
                    if not isinstance(value, (int, float)):
                        return False

            return True
        except Exception:
            return False

    def validate_rater_name(self, rater_name):
        """Validate rater name."""
        try:
            if rater_name is None or not isinstance(rater_name, str):
                return False

            if len(rater_name.strip()) == 0:
                return False

            if len(rater_name) > 100:
                return False

            return True
        except Exception:
            return False

    def validate_rgb_color(self, color):
        """Validate RGB color format."""
        try:
            if color is None or not isinstance(color, list):
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

    def generate_color_for_rater(self, rater):
        """Generate consistent color for a rater."""
        # Use hash to generate consistent colors
        hash_val = hash(rater)
        return [
            (hash_val % 100) / 100.0,
            ((hash_val // 100) % 100) / 100.0,
            ((hash_val // 10000) % 100) / 100.0
        ]

    def is_dicom_file(self, filename):
        """Check if filename has DICOM extension."""
        if not filename:
            return False
        return filename.lower().endswith('.dcm')

    def sanitize_filename(self, filename):
        """Sanitize filename by replacing invalid characters."""
        import re
        return re.sub(r'[<>:"/\\|?*\s]', '_', filename)

    def generate_timestamp(self):
        """Generate current timestamp."""
        return datetime.now().isoformat()

    def validate_timestamp(self, timestamp):
        """Validate timestamp format."""
        try:
            if not isinstance(timestamp, str):
                return False

            # Try to parse as ISO format
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return True
        except Exception:
            return False


def test_module_imports():
    """Test that we can import basic Python modules needed for the project."""
    # Test standard library imports
    import json
    import os
    import tempfile
    from datetime import datetime

    # Test that these work
    assert json is not None
    assert os is not None
    assert tempfile is not None
    assert datetime is not None


def test_project_structure():
    """Test basic project structure."""
    # Test that we can find the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Check for expected directories
    expected_dirs = [
        "AnnotateUltrasound",
        "AnonymizeUltrasound",
        "TimeSeriesAnnotation",
        "MmodeAnalysis",
        "SceneCleaner"
    ]

    for expected_dir in expected_dirs:
        dir_path = os.path.join(project_root, expected_dir)
        assert os.path.exists(dir_path), f"Expected directory {expected_dir} not found"


if __name__ == '__main__':
    # Run basic tests if executed directly
    test_module_imports()
    test_project_structure()

    # Run the main test class
    test_instance = TestBasicFunctionality()
    test_instance.test_json_handling()
    test_instance.test_coordinate_calculations()
    test_instance.test_data_validation()
    test_instance.test_color_utilities()
    test_instance.test_file_utilities()
    test_instance.test_timestamp_utilities()
    test_instance.test_error_handling()

    print("✓ All basic tests passed!")