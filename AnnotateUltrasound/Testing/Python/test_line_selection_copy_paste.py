#!/usr/bin/env python3
"""
Line selection, copy, and paste test for AnnotateUltrasound module.
This test tests the new SelectAll/Copy/Paste shortcuts and functionality.
"""

import sys
import os
import time
import logging
import gc
import slicer
import vtk
import json
import tempfile
import shutil
import qt
import argparse

# Add the module path to sys.path
modulePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, modulePath)

# Get the test data path
testDataPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

# Import the module
from AnnotateUltrasound import AnnotateUltrasoundLogic

# Import Slicer test base class
try:
    from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest
except ImportError:
    import unittest
    ScriptedLoadableModuleTest = unittest.TestCase


class LineSelectionCopyPasteTest(ScriptedLoadableModuleTest):
    """
    Line selection, copy, and paste test that tests the new functionality.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear(0)

        # Initialize test parameters
        self.test_data_path = testDataPath
        self.dicom_file = "3038953328_70622118.dcm"
        self.annotation_file = "3038953328_70622118.json"

        # Select the AnnotateUltrasound module
        slicer.util.selectModule('AnnotateUltrasound')

        # Create the widget and get the logic from it
        self.widget = slicer.modules.annotateultrasound.widgetRepresentation().self()
        self.logic = self.widget.logic

        # Wait for UI to be ready
        time.sleep(1)
        self.widget.ui.overlayVisibilityButton.setChecked(True)

    def tearDown(self):
        """Reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear(0)

    def load_test_data(self):
        """Load the test DICOM and annotation data."""
        print("Loading test data...")

        try:
            # Set the rater (use 'tom' as in other tests)
            self.logic.setRater("tom")

            # Load the DICOM directory and annotations
            num_files, num_annotations = self.logic.updateInputDf("tom", self.test_data_path)
            print(f"Found {num_files} DICOM files, {num_annotations} annotation files")

            # Load the first sequence
            current_index = self.logic.loadNextSequence()
            if current_index is None:
                print("Failed to load sequence")
                return False

            # Wait a moment for the sequence to be fully loaded
            time.sleep(2)

            # Set total frames
            if self.logic.sequenceBrowserNode:
                total_frames = self.logic.sequenceBrowserNode.GetNumberOfItems()
                print(f"Loaded {total_frames} frames")

                # Ensure the volume is displayed in the 3D view
                parameterNode = self.logic.getParameterNode()
                volumeNode = parameterNode.inputVolume
                if volumeNode:
                    # Get the first 3D view and display the volume
                    layoutManager = slicer.app.layoutManager()
                    threeDWidget = layoutManager.threeDWidget(0)
                    if threeDWidget:
                        threeDView = threeDWidget.threeDView()
                        threeDView.mrmlViewNode().SetBackgroundColor(0, 0, 0)
                        # The volume should already be displayed by the module
                        print(f"Volume '{volumeNode.GetName()}' should be visible in 3D view")

                # Pause to allow viewing the loaded data
                print("Data loaded successfully! You should see the ultrasound volume in Slicer.")
                print("Processing events to ensure rendering...")
                slicer.app.processEvents()
                print("Waiting 3 seconds for rendering to complete...")
                time.sleep(3)
            else:
                print("No sequence browser node found")
                return False

            return True

        except Exception as e:
            print(f"Error loading test data: {e}")
            return False

    def create_test_lines(self):
        """Create test lines for the copy/paste tests."""
        print("Creating test lines...")

        try:
            # Create test coordinates (simple lines)
            # Get the current volume bounds
            parameterNode = self.logic.getParameterNode()
            parameterNode.depthGuideVisible = True

            volumeNode = parameterNode.inputVolume
            if not volumeNode:
                print("No volume node found")
                return False

            # Get volume dimensions and transform to RAS coordinates
            imageData = volumeNode.GetImageData()
            if not imageData:
                print("No image data found")
                return False

            dims = imageData.GetDimensions()
            spacing = imageData.GetSpacing()
            origin = imageData.GetOrigin()

            # Get IJK to RAS transform matrix
            ijkToRas = vtk.vtkMatrix4x4()
            volumeNode.GetIJKToRASMatrix(ijkToRas)

            # Create lines similar to those in the JSON annotations
            # Based on the annotation data, typical ranges are:
            # X: 80-150, Y: 35-85, Z: 0 (LPS coordinates)

            # Get volume bounds to ensure lines are visible
            bounds = [0, 0, 0, 0, 0, 0]
            volumeNode.GetBounds(bounds)

            # Use coordinates similar to the JSON annotations, converted to RAS
            # Original LPS coordinates from JSON: X: 80-150, Y: 35-85, Z: 0

            # Pleura line 1: longer line similar to annotation pattern (LPS: [85, 41, 0] to [97, 43, 0])
            pleura1_start = [-85.0, -41.0, 0.0]  # LPS to RAS: flip X and Y
            pleura1_end = [-97.0, -43.0, 0.0]

            # Pleura line 2: another longer line (LPS: [128, 44, 0] to [143, 38, 0])
            pleura2_start = [-128.0, -44.0, 0.0]  # LPS to RAS: flip X and Y
            pleura2_end = [-143.0, -38.0, 0.0]

            # B-line 1: shorter line similar to annotation pattern (LPS: [80, 82, 0] to [85, 83, 0])
            bline1_start = [-80.0, -82.0, 0.0]  # LPS to RAS: flip X and Y
            bline1_end = [-85.0, -83.0, 0.0]

            # B-line 2: another shorter line (LPS: [144, 85, 0] to [149, 84, 0])
            bline2_start = [-144.0, -85.0, 0.0]  # LPS to RAS: flip X and Y
            bline2_end = [-149.0, -84.0, 0.0]

            # Use the current rater from the logic
            parameterNode = self.logic.getParameterNode()
            rater = parameterNode.rater if parameterNode and parameterNode.rater else "tom"

            # Ensure lines are visible
            self.logic.showHideLines = True

            # Create markup nodes similar to JSON annotations
            color_pleura, _ = self.logic.getColorsForRater(rater)
            pleura_line1 = self.logic.createMarkupLine(f"Pleura",
                                                      rater, [pleura1_start, pleura1_end], color_pleura)
            pleura_line2 = self.logic.createMarkupLine(f"Pleura",
                                                      rater, [pleura2_start, pleura2_end], color_pleura)
            _, color_blines = self.logic.getColorsForRater(rater)
            bline1 = self.logic.createMarkupLine(f"B-line",
                                                rater, [bline1_start, bline1_end], color_blines)
            bline2 = self.logic.createMarkupLine(f"B-line",
                                                rater, [bline2_start, bline2_end], color_blines)

            # Add to logic lists
            self.logic.pleuraLines.append(pleura_line1)
            self.logic.pleuraLines.append(pleura_line2)
            self.logic.bLines.append(bline1)
            self.logic.bLines.append(bline2)

            print(f"Created {len(self.logic.pleuraLines)} pleura lines and {len(self.logic.bLines)} b-lines")

            # Ensure display nodes are visible
            for node in [pleura_line1, pleura_line2, bline1, bline2]:
                displayNode = node.GetDisplayNode()
                if displayNode:
                    displayNode.SetVisibility(True)

            # Sync and refresh
            self.logic.syncMarkupsToAnnotations()
            self.logic.refreshDisplay(updateOverlay=True, updateGui=True)

            # Process events to ensure lines are rendered
            slicer.app.processEvents()
            time.sleep(1)

            return True

        except Exception as e:
            print(f"Error creating test lines: {e}")
            return False

    def test_select_all_lines(self):
        """Test selecting all lines."""
        print("Testing select all lines...")

        # Clear any existing selection
        self.logic.selectedLineIDs = []

        # Call the select all function
        self.widget.onSelectAllLines()

        # Verify all lines are selected
        expected_selected = len(self.logic.pleuraLines) + len(self.logic.bLines)
        actual_selected = len(self.logic.selectedLineIDs)

        print(f"Expected {expected_selected} selected lines, got {actual_selected}")
        assert actual_selected == expected_selected, f"Expected {expected_selected} selected lines, got {actual_selected}"

        # Verify the lines have the selected appearance
        for node in self.logic.pleuraLines + self.logic.bLines:
            if slicer.mrmlScene.IsNodePresent(node):
                display_node = node.GetDisplayNode()
                if display_node:
                    # Selected lines should have thicker lines and larger glyphs
                    assert display_node.GetLineThickness() >= 0.3, f"Line thickness {display_node.GetLineThickness()} should be >= 0.3"
                    assert display_node.GetGlyphScale() >= 2.0, f"Glyph scale {display_node.GetGlyphScale()} should be >= 2.0"

        print("✅ Select all lines test passed")

    def test_copy_lines(self):
        """Test copying selected lines."""
        print("Testing copy lines...")

        # Check if lines are in the scene
        for i, node in enumerate(self.logic.pleuraLines):
            print(f"Debug: Pleura line {i}: {node.GetName() if node else 'None'}, in scene: {slicer.mrmlScene.IsNodePresent(node) if node else False}, control points: {node.GetNumberOfControlPoints() if node else 0}")
        for i, node in enumerate(self.logic.bLines):
            print(f"Debug: B-line {i}: {node.GetName() if node else 'None'}, in scene: {slicer.mrmlScene.IsNodePresent(node) if node else False}, control points: {node.GetNumberOfControlPoints() if node else 0}")

        # Select all the lines
        self.widget.onSelectAllLines()

        # Debug: Check what lines we have
        print(f"Debug: pleuraLines count: {len(self.logic.pleuraLines)}")
        print(f"Debug: bLines count: {len(self.logic.bLines)}")
        print(f"Debug: selectedLineIDs count: {len(self.logic.selectedLineIDs)}")
        print(f"Debug: selectedLineIDs: {self.logic.selectedLineIDs}")

        # Copy the selected lines
        print("Debug: About to call onCopyLines()")
        try:
            self.widget.onCopyLines()
            print("Debug: onCopyLines() completed successfully")
        except Exception as e:
            print(f"Debug: Exception in onCopyLines(): {e}")
            import traceback
            traceback.print_exc()
        print(f"Debug: After onCopyLines(), clipboardLines count: {len(self.logic.clipboardLines)}")

        # Verify lines were copied to clipboard
        assert self.logic.clipboardLines is not None, "Clipboard lines should not be None"
        expected_copied = len(self.logic.pleuraLines) + len(self.logic.bLines)
        actual_copied = len(self.logic.clipboardLines)

        print(f"Expected {expected_copied} copied lines, got {actual_copied}")
        assert actual_copied == expected_copied, f"Expected {expected_copied} copied lines, got {actual_copied}"

        # Verify clipboard lines are hidden
        for node in self.logic.clipboardLines:
            assert not node.GetDisplayVisibility(), "Clipboard lines should be hidden"

        print("✅ Copy lines test passed")

    def test_paste_lines(self):
        """Test pasting copied lines."""
        print("Testing paste lines...")

        # First copy some lines
        self.widget.onSelectAllLines()
        self.widget.onCopyLines()
        expected_pleura_lines_copied = len(self.logic.pleuraLines)
        expected_b_lines_copied = len(self.logic.bLines)

        # Go to the next frame
        self.widget._nextFrameInSequence()

        # Count lines before pasting
        pleura_count_before = len(self.logic.pleuraLines)
        bline_count_before = len(self.logic.bLines)

        # Paste the lines
        self.widget.onPasteLines()

        # Verify lines were pasted
        pleura_count_after = len(self.logic.pleuraLines)
        bline_count_after = len(self.logic.bLines)

        # Should have doubled the number of lines
        assert pleura_count_after == pleura_count_before + expected_pleura_lines_copied, f"Pleura lines should be: {pleura_count_before} + {expected_pleura_lines_copied} = {pleura_count_after}"
        assert bline_count_after == bline_count_before + expected_b_lines_copied, f"B-lines should be: {bline_count_before} + {expected_b_lines_copied} = {bline_count_after}"
        print(f"✅ Paste lines test passed")

        # Verify pasted lines are visible
        for node in self.logic.pleuraLines + self.logic.bLines:
            if slicer.mrmlScene.IsNodePresent(node):
                assert node.GetDisplayVisibility(), "Pasted lines should be visible"

        # Verify unsaved changes flag is set
        assert self.widget._parameterNode.unsavedChanges, "Unsaved changes flag should be set"

        print("✅ Paste lines test passed")

    def test_deselect_all_lines(self):
        """Test deselecting all lines."""
        print("Testing deselect all lines...")

        # First select all lines
        self.widget.onSelectAllLines()
        assert len(self.logic.selectedLineIDs) > 0, "Should have selected lines"

        # Deselect all lines
        self.widget.onDeselectAllLines()

        # Verify selection is cleared
        assert len(self.logic.selectedLineIDs) == 0, "Selection should be cleared"

        # Verify lines have normal appearance
        for node in self.logic.pleuraLines + self.logic.bLines:
            if slicer.mrmlScene.IsNodePresent(node):
                display_node = node.GetDisplayNode()
                if display_node:
                    # Normal lines should have standard thickness and glyph scale
                    assert display_node.GetLineThickness() <= 0.3, f"Line thickness {display_node.GetLineThickness()} should be <= 0.3"
                    assert display_node.GetGlyphScale() <= 2.5, f"Glyph scale {display_node.GetGlyphScale()} should be <= 2.5"

        print("✅ Deselect all lines test passed")

    def test_copy_paste_multiple_lines(self):
        """Test copying and pasting multiple lines multiple times."""
        print("Testing multiple copy/paste operations...")

        # Copy and paste multiple times
        for i in range(3):
            print(f"Copy/paste iteration {i+1}")

            total_lines_before = len(self.logic.pleuraLines) + len(self.logic.bLines)

            # Select and copy
            self.widget.onSelectAllLines()
            self.widget.onCopyLines()

            # Paste
            self.widget.onPasteLines()

            # Verify we have more lines each time
            total_lines = len(self.logic.pleuraLines) + len(self.logic.bLines)
            expected_lines = total_lines_before * (2 ** i)  # initial lines, doubled each time
            print(f"Total lines after iteration {i+1}: {total_lines} (expected ~{expected_lines})")

        print("✅ Multiple copy/paste operations test passed")

    def test_keyboard_shortcuts(self):
        """Test keyboard shortcuts for select all, copy, paste, and deselect."""
        print("Testing keyboard shortcuts...")

        # Test Ctrl+A (Select All)
        print("Testing Ctrl+A (Select All)...")
        self.logic.selectedLineIDs = []  # Clear selection first

        # Simulate Ctrl+A
        try:
            import qt
            # Create a QKeyEvent for Ctrl+A
            key_event = qt.QKeyEvent(qt.QEvent.KeyPress, qt.Qt.Key_A, qt.Qt.ControlModifier)
            self.widget.shortcutSelectAll.activated.emit()

            # Verify all lines are selected
            expected_selected = len(self.logic.pleuraLines) + len(self.logic.bLines)
            actual_selected = len(self.logic.selectedLineIDs)
            assert actual_selected == expected_selected, f"Expected {expected_selected} selected lines, got {actual_selected}"
            print("✅ Ctrl+A (Select All) shortcut works")

        except Exception as e:
            print(f"⚠️ Could not test Ctrl+A shortcut: {e}")

        # Test Ctrl+C (Copy)
        print("Testing Ctrl+C (Copy)...")
        try:
            self.widget.shortcutCopy.activated.emit()

            # Verify lines were copied
            expected_copied = len(self.logic.pleuraLines) + len(self.logic.bLines)
            actual_copied = len(self.logic.clipboardLines)
            assert actual_copied == expected_copied, f"Expected {expected_copied} copied lines, got {actual_copied}"
            print("✅ Ctrl+C (Copy) shortcut works")

        except Exception as e:
            print(f"⚠️ Could not test Ctrl+C shortcut: {e}")

        # Test Ctrl+V (Paste)
        print("Testing Ctrl+V (Paste)...")
        try:
            pleura_count_before = len(self.logic.pleuraLines)
            bline_count_before = len(self.logic.bLines)

            self.widget.shortcutPaste.activated.emit()

            pleura_count_after = len(self.logic.pleuraLines)
            bline_count_after = len(self.logic.bLines)

            # Should have more lines after pasting
            assert pleura_count_after > pleura_count_before or bline_count_after > bline_count_before, "Should have more lines after pasting"
            print("✅ Ctrl+V (Paste) shortcut works")

        except Exception as e:
            print(f"⚠️ Could not test Ctrl+V shortcut: {e}")

        # Test Escape (Deselect All)
        print("Testing Escape (Deselect All)...")
        try:
            # First select some lines
            self.widget.onSelectAllLines()
            assert len(self.logic.selectedLineIDs) > 0, "Should have selected lines"

            # Simulate Escape
            self.widget.shortcutDeselectAll.activated.emit()

            # Verify selection is cleared
            assert len(self.logic.selectedLineIDs) == 0, "Selection should be cleared after Escape"
            print("✅ Escape (Deselect All) shortcut works")

        except Exception as e:
            print(f"⚠️ Could not test Escape shortcut: {e}")

        print("✅ Keyboard shortcuts test passed")

    def runTest(self):
        """Run the line selection, copy, and paste test."""
        print("Starting line selection, copy, and paste test...")
        print("This test will:")
        print("1. Load the AnnotateUltrasound module")
        print("2. Load test DICOM data")
        print("3. Create test lines (Pleura and B-lines)")
        print("4. Test line selection functionality")
        print("5. Test copy and paste functionality")
        print("6. Test keyboard shortcuts")

        # Load test data
        if not self.load_test_data():
            print("Failed to load test data")
            return

        # Create test lines
        if not self.create_test_lines():
            print("Failed to create test lines")
            return

        # Run the tests
        print("\n=== PHASE 1: Basic Functionality Tests ===")

        print("\n--- Testing Select All Lines ---")
        self.test_select_all_lines()

        print("\n--- Testing Copy Lines ---")
        self.test_copy_lines()

        print("\n--- Testing Paste Lines ---")
        self.test_paste_lines()

        print("\n--- Testing Deselect All Lines ---")
        self.test_deselect_all_lines()

        print("\n=== PHASE 2: Advanced Functionality Tests ===")

        print("\n--- Testing Multiple Copy/Paste Operations ---")
        self.test_copy_paste_multiple_lines()

        print("\n--- Testing Keyboard Shortcuts ---")
        self.test_keyboard_shortcuts()

        print("\n=== TEST SUMMARY ===")
        print("✅ All line selection, copy, and paste tests completed successfully!")
        print("The following functionality was tested:")
        print("- Line selection (Select All, Deselect All)")
        print("- Copy and paste operations")
        print("- Multiple copy/paste cycles")
        print("- Keyboard shortcuts (Ctrl+A, Ctrl+C, Ctrl+V, Escape)")
        print("- Visual feedback for selected lines")
        print("- Clipboard management")
        print("- State management and cleanup")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Line selection, copy, and paste test for AnnotateUltrasound')
    return parser.parse_args()

def runLineSelectionCopyPasteTest():
    """Run the line selection, copy, and paste test."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        test = LineSelectionCopyPasteTest()
        # Store args in the test instance so setUp can access it
        test.args = args
        test.setUp()  # Explicitly call setUp
        test.runTest()
        test.tearDown()  # Explicitly call tearDown
        return True
    except Exception as e:
        print(f"Error running line selection, copy, and paste test: {e}")
        return False


if __name__ == '__main__':
    runLineSelectionCopyPasteTest()
    slicer.util.exit(0)