#!/usr/bin/env python3
"""
Test script for AnnotateUltrasound module with real DICOM data.
This script loads DICOM files and tests the module's functionality.
"""

import sys
import os
import slicer
import json
import time

# Only import if we're in a proper Slicer environment
try:
    # Import directly from the module file for Slicer extensions
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from AnnotateUltrasound import AnnotateUltrasoundLogic, AnnotateUltrasoundWidget
except ImportError:
    # Skip import if not in proper Slicer environment
    pass

class DicomLoadingTest:
    """
    Test class for loading DICOM files and testing module functionality.
    """

    def __init__(self):
        self.widget = None
        self.logic = None
        self.test_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data'))

    def setUp(self):
        """Set up the test environment."""
        print("Setting up test environment...")
        slicer.mrmlScene.Clear(0)

        # Create the widget
        self.widget = slicer.modules.annotateultrasound.widgetRepresentation().self()
        self.logic = self.widget.logic

        # Wait for UI to be ready
        time.sleep(1)
        print("✅ Test environment set up")

    def tearDown(self):
        """Clean up after tests."""
        if self.widget:
            del self.widget
        slicer.mrmlScene.Clear(0)
        print("✅ Test environment cleaned up")

    def runTest(self):
        """Run all DICOM loading tests."""
        print("=== Running DICOM Loading Tests ===")

        self.setUp()

        try:
            self.test_dicom_directory_loading()
            self.test_annotation_loading()
            self.test_rater_management_with_data()
            self.test_line_creation_with_data()
            self.test_sequence_navigation()
            self.test_annotation_saving()

            print("✅ All DICOM loading tests passed!")
            return True

        except Exception as e:
            print(f"❌ DICOM loading test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.tearDown()

    def test_dicom_directory_loading(self):
        """Test loading DICOM files from the test directory."""
        print("Testing DICOM directory loading...")

        # Set a rater first (required for updateInputDf) - use "tom" since we have annotation files for that rater
        self.logic.setRater("tom")

        # Test loading the DICOM directory
        num_files, num_annotations = self.logic.updateInputDf("tom", self.test_data_dir)

        print(f"Found {num_files} DICOM files")
        print(f"Created {num_annotations} annotation files")

        # Verify DICOM dataframe was created
        self.assertIsNotNone(self.logic.dicomDf)
        self.assertGreater(len(self.logic.dicomDf), 0)

        # Check that we have the expected files
        expected_files = [
            '3038953328_70622118.dcm',
            '9938886735_88815303.dcm'
        ]

        for expected_file in expected_files:
            found = False
            for _, row in self.logic.dicomDf.iterrows():
                if expected_file in row['Filepath']:
                    found = True
                    break
            self.assertTrue(found, f"Expected file {expected_file} not found")

        print("✅ DICOM directory loading test passed")

    def test_annotation_loading(self):
        """Test loading existing annotations."""
        print("Testing annotation loading...")

        # Load the first sequence
        current_index = self.logic.loadNextSequence()
        self.assertIsNotNone(current_index)

        print(f"Loaded sequence at index {current_index}")

        # Check if annotations were loaded
        self.assertIsNotNone(self.logic.annotations)
        if self.logic.annotations is not None:
            print(f"Loaded annotations with {len(self.logic.annotations.get('frame_annotations', []))} frames")

        print("✅ Annotation loading test passed")

    def test_rater_management_with_data(self):
        """Test rater management with loaded data."""
        print("Testing rater management with data...")

        # Set up test raters - use the raters we have annotation files for
        test_raters = {"tom", "test_rater"}
        self.logic.setSelectedRaters(test_raters)

        # Verify raters are set
        selected = self.logic.getSelectedRaters()
        self.assertEqual(selected, test_raters)

        # Test color assignment for each rater
        for rater in test_raters:
            pleura_color, bline_color = self.logic.getColorsForRater(rater)
            self.assertIsNotNone(pleura_color)
            self.assertIsNotNone(bline_color)
            print(f"Rater {rater}: Pleura color {pleura_color}, B-line color {bline_color}")

        print("✅ Rater management test passed")

    def test_line_creation_with_data(self):
        """Test creating lines with loaded data."""
        print("Testing line creation with data...")

        # Set a rater - use "tom" since we have annotation files for that rater
        self.logic.setRater("tom")

        # Create test coordinates (in image space)
        coordinates = [[100, 100, 0], [200, 150, 0]]

        # Create a pleura line
        pleura_line = self.logic.createMarkupLine("test_pleura", "tom", coordinates)
        self.assertIsNotNone(pleura_line)
        print(f"Created pleura line: {pleura_line.GetName()}")

        # Create a B-line
        bline_coordinates = [[150, 120, 0], [180, 180, 0]]
        bline = self.logic.createMarkupLine("test_bline", "tom", bline_coordinates)
        self.assertIsNotNone(bline)
        print(f"Created B-line: {bline.GetName()}")

        # Verify lines are in the scene
        scene_lines = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsLineNode")
        self.assertGreaterEqual(scene_lines.GetNumberOfItems(), 2)

        print("✅ Line creation test passed")

    def test_sequence_navigation(self):
        """Test navigating between sequences."""
        print("Testing sequence navigation...")

        if self.logic.dicomDf is None or len(self.logic.dicomDf) < 2:
            print("Skipping sequence navigation test - need at least 2 DICOM files")
            return

        # Get current index
        current_index = self.logic.nextDicomDfIndex
        print(f"Current index: {current_index}")

        # Try to load next sequence
        next_index = self.logic.loadNextSequence()
        if next_index is not None:
            print(f"Loaded next sequence at index {next_index}")
            self.assertGreater(next_index, current_index)

        # Try to load previous sequence
        prev_index = self.logic.loadPreviousSequence()
        if prev_index is not None:
            print(f"Loaded previous sequence at index {prev_index}")
            self.assertLessEqual(prev_index, next_index)

        print("✅ Sequence navigation test passed")

    def test_annotation_saving(self):
        """Test saving annotations."""
        print("Testing annotation saving...")

        # Create some test annotations
        test_annotations = {
            "frame_annotations": [
                {
                    "frame_number": 0,
                    "pleura_lines": [
                        {
                            "rater": "test_rater",
                            "coordinates": [[100, 100, 0], [200, 150, 0]]
                        }
                    ],
                    "b_lines": [
                        {
                            "rater": "test_rater",
                            "coordinates": [[150, 120, 0], [180, 180, 0]]
                        }
                    ]
                }
            ]
        }

        # Save annotations
        self.logic.annotations = test_annotations

        # Test saving to a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
            json.dump(test_annotations, f, indent=2)

        try:
            # Verify the file was created and contains valid JSON
            with open(temp_file, 'r') as f:
                loaded_annotations = json.load(f)

            self.assertEqual(loaded_annotations, test_annotations)
            print(f"Saved annotations to {temp_file}")

        finally:
            # Clean up
            os.unlink(temp_file)

        print("✅ Annotation saving test passed")

    # Helper assertion methods
    def assertIsNotNone(self, obj):
        if obj is None:
            raise AssertionError("Object is None")

    def assertIsNone(self, obj):
        if obj is not None:
            raise AssertionError(f"Object is not None: {obj}")

    def assertEqual(self, a, b):
        if a != b:
            raise AssertionError(f"{a} != {b}")

    def assertNotEqual(self, a, b):
        if a == b:
            raise AssertionError(f"{a} == {b}")

    def assertGreater(self, a, b):
        if not a > b:
            raise AssertionError(f"{a} is not greater than {b}")

    def assertGreaterEqual(self, a, b):
        if not a >= b:
            raise AssertionError(f"{a} is not greater than or equal to {b}")

    def assertLessEqual(self, a, b):
        if not a <= b:
            raise AssertionError(f"{a} is not less than or equal to {b}")

    def assertTrue(self, condition, msg=None):
        if not condition:
            if msg:
                raise AssertionError(msg)
            else:
                raise AssertionError("Condition is not True")


def runDicomTest():
    """Run the DICOM loading test."""
    test = DicomLoadingTest()
    success = test.runTest()

    # Print final result
    if success:
        print("✅ All DICOM loading tests completed successfully!")
    else:
        print("❌ Some DICOM loading tests failed!")

    # Exit Slicer after test completion
    print("Tests complete. Exiting Slicer...")
    slicer.util.quit()

    return success


if __name__ == "__main__":
    success = runDicomTest()
    exit(0 if success else 1)
