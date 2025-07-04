"""
Unit tests for data validation utilities.
These tests validate JSON annotation formats and data structures.
"""

import pytest
import json
import tempfile
import os
from jsonschema import validate, ValidationError


class TestDataValidation:
    """Test data validation utilities."""

    def test_annotation_schema_validation(self):
        """Test annotation schema validation."""
        # Valid annotation structure
        valid_annotation = {
            "frame_annotations": [
                {
                    "frame_number": 0,
                    "pleura_lines": [
                        {
                            "rater": "test_rater",
                            "coordinates": [[100, 100, 0], [200, 150, 0]],
                            "timestamp": "2024-01-01T00:00:00"
                        }
                    ],
                    "b_lines": [
                        {
                            "rater": "test_rater",
                            "coordinates": [[150, 120, 0], [180, 180, 0]],
                            "timestamp": "2024-01-01T00:00:00"
                        }
                    ]
                }
            ]
        }

        # Test that valid annotation passes validation
        assert self.validate_annotation_structure(valid_annotation) == True

        # Test invalid annotation structures
        invalid_annotations = [
            {},  # Empty annotation
            {"frame_annotations": "not_a_list"},  # Wrong type
            {"frame_annotations": [{"frame_number": "not_a_number"}]},  # Wrong frame number type
            {"frame_annotations": [{"frame_number": 0, "pleura_lines": "not_a_list"}]},  # Wrong pleura_lines type
        ]

        for invalid_annotation in invalid_annotations:
            assert self.validate_annotation_structure(invalid_annotation) == False

    def test_coordinate_validation(self):
        """Test coordinate validation."""
        # Valid coordinates
        valid_coords = [
            [[100, 100, 0], [200, 150, 0]],  # Standard 2-point line
            [[0, 0, 0], [100, 100, 0], [200, 200, 0]],  # Multi-point line
        ]

        for coords in valid_coords:
            assert self.validate_coordinates(coords) == True

        # Invalid coordinates
        invalid_coords = [
            [],  # Empty coordinates
            [[100, 100]],  # Missing Z coordinate
            [[100, 100, 0, 0]],  # Too many dimensions
            [["x", "y", "z"]],  # Non-numeric coordinates
            [[100, 100, 0]],  # Single point (need at least 2 for a line)
        ]

        for coords in invalid_coords:
            assert self.validate_coordinates(coords) == False

    def test_rater_name_validation(self):
        """Test rater name validation."""
        # Valid rater names
        valid_names = [
            "rater1",
            "test_rater",
            "Dr_Smith",
            "rater-123",
            "user@example.com"
        ]

        for name in valid_names:
            assert self.validate_rater_name(name) == True

        # Invalid rater names
        invalid_names = [
            "",  # Empty name
            " ",  # Whitespace only
            "a" * 101,  # Too long (over 100 characters)
            None,  # None value
        ]

        for name in invalid_names:
            assert self.validate_rater_name(name) == False

    def test_timestamp_validation(self):
        """Test timestamp validation."""
        # Valid timestamps
        valid_timestamps = [
            "2024-01-01T00:00:00",
            "2024-12-31T23:59:59",
            "2024-01-01T00:00:00.123456",
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00+00:00"
        ]

        for timestamp in valid_timestamps:
            assert self.validate_timestamp(timestamp) == True

        # Invalid timestamps
        invalid_timestamps = [
            "",  # Empty timestamp
            "not-a-timestamp",  # Invalid format
            "2024-13-01T00:00:00",  # Invalid month
            "2024-01-32T00:00:00",  # Invalid day
            "2024-01-01T25:00:00",  # Invalid hour
        ]

        for timestamp in invalid_timestamps:
            assert self.validate_timestamp(timestamp) == False

    def test_frame_number_validation(self):
        """Test frame number validation."""
        # Valid frame numbers
        valid_frames = [0, 1, 100, 999]

        for frame in valid_frames:
            assert self.validate_frame_number(frame) == True

        # Invalid frame numbers
        invalid_frames = [-1, -100, "0", None, 1.5]

        for frame in invalid_frames:
            assert self.validate_frame_number(frame) == False

    def test_annotation_file_validation(self):
        """Test complete annotation file validation."""
        # Create a valid annotation file
        valid_annotation = {
            "frame_annotations": [
                {
                    "frame_number": 0,
                    "pleura_lines": [
                        {
                            "rater": "test_rater",
                            "coordinates": [[100, 100, 0], [200, 150, 0]],
                            "timestamp": "2024-01-01T00:00:00"
                        }
                    ],
                    "b_lines": []
                }
            ]
        }

        # Test saving and validating file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_annotation, f)
            temp_file = f.name

        try:
            assert self.validate_annotation_file(temp_file) == True
        finally:
            os.unlink(temp_file)

        # Test non-existent file
        assert self.validate_annotation_file("non_existent.json") == False

    def test_data_consistency_validation(self):
        """Test data consistency validation."""
        # Test consistent annotation data
        consistent_annotation = {
            "frame_annotations": [
                {
                    "frame_number": 0,
                    "pleura_lines": [
                        {
                            "rater": "rater1",
                            "coordinates": [[100, 100, 0], [200, 150, 0]],
                            "timestamp": "2024-01-01T00:00:00"
                        }
                    ],
                    "b_lines": [
                        {
                            "rater": "rater1",
                            "coordinates": [[150, 120, 0], [180, 180, 0]],
                            "timestamp": "2024-01-01T00:00:01"
                        }
                    ]
                }
            ]
        }

        assert self.validate_data_consistency(consistent_annotation) == True

        # Test inconsistent data (duplicate frame numbers)
        inconsistent_annotation = {
            "frame_annotations": [
                {
                    "frame_number": 0,
                    "pleura_lines": [],
                    "b_lines": []
                },
                {
                    "frame_number": 0,  # Duplicate frame number
                    "pleura_lines": [],
                    "b_lines": []
                }
            ]
        }

        assert self.validate_data_consistency(inconsistent_annotation) == False

    # Helper validation methods
    def validate_annotation_structure(self, annotation):
        """Validate annotation structure."""
        try:
            if not isinstance(annotation, dict):
                return False

            if "frame_annotations" not in annotation:
                return False

            frame_annotations = annotation["frame_annotations"]
            if not isinstance(frame_annotations, list):
                return False

            for frame_annotation in frame_annotations:
                if not isinstance(frame_annotation, dict):
                    return False

                if "frame_number" not in frame_annotation:
                    return False

                if not isinstance(frame_annotation["frame_number"], int):
                    return False

                if "pleura_lines" in frame_annotation:
                    if not isinstance(frame_annotation["pleura_lines"], list):
                        return False

                if "b_lines" in frame_annotation:
                    if not isinstance(frame_annotation["b_lines"], list):
                        return False

            return True
        except Exception:
            return False

    def validate_coordinates(self, coordinates):
        """Validate coordinate format."""
        try:
            if not isinstance(coordinates, list):
                return False

            if len(coordinates) < 2:  # Need at least 2 points for a line
                return False

            for coord in coordinates:
                if not isinstance(coord, list):
                    return False

                if len(coord) != 3:  # Must be 3D coordinates
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
            if not isinstance(rater_name, str):
                return False

            if len(rater_name.strip()) == 0:
                return False

            if len(rater_name) > 100:  # Reasonable length limit
                return False

            return True
        except Exception:
            return False

    def validate_timestamp(self, timestamp):
        """Validate timestamp format."""
        try:
            if not isinstance(timestamp, str):
                return False

            # Try to parse as ISO format
            from datetime import datetime
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return True
        except Exception:
            return False

    def validate_frame_number(self, frame_number):
        """Validate frame number."""
        try:
            if not isinstance(frame_number, int):
                return False

            if frame_number < 0:
                return False

            return True
        except Exception:
            return False

    def validate_annotation_file(self, filepath):
        """Validate annotation file."""
        try:
            if not os.path.exists(filepath):
                return False

            with open(filepath, 'r') as f:
                annotation = json.load(f)

            return self.validate_annotation_structure(annotation)
        except Exception:
            return False

    def validate_data_consistency(self, annotation):
        """Validate data consistency."""
        try:
            if not self.validate_annotation_structure(annotation):
                return False

            # Check for duplicate frame numbers
            frame_numbers = set()
            for frame_annotation in annotation["frame_annotations"]:
                frame_number = frame_annotation["frame_number"]
                if frame_number in frame_numbers:
                    return False
                frame_numbers.add(frame_number)

            return True
        except Exception:
            return False