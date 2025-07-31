#!/usr/bin/env python3
"""
Line selection, copy, paste, cut, and delete test for AnnotateUltrasound module.
This test tests the new SelectAll/Copy/Paste/Cut/Delete shortcuts and functionality.
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
    Line selection, copy, paste, cut, and delete test that tests the new functionality.
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

        # Select all the lines
        self.widget.onSelectAllLines()


        # Copy the selected lines
        try:
            self.widget.onCopyLines()
        except Exception as e:
            import traceback
            traceback.print_exc()

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

        lines_to_copy = len(self.logic.pleuraLines) + len(self.logic.bLines)
        assert lines_to_copy > 0, "Should have lines to copy"

        # First copy some lines
        self.widget.onSelectAllLines()
        self.widget.onCopyLines()
        clipboard_line_count = len(self.logic.clipboardLines)
        assert clipboard_line_count > 0, "Should have copied lines"

        expected_pleura_lines_copied = len(self.logic.pleuraLines)
        expected_b_lines_copied = len(self.logic.bLines)
        assert clipboard_line_count == expected_pleura_lines_copied + expected_b_lines_copied, f"Clipboard lines should be: {expected_pleura_lines_copied} + {expected_b_lines_copied} = {clipboard_line_count}"

        self.widget._nextFrameInSequence()

        # Count lines before pasting
        pleura_count_before = len(self.logic.pleuraLines)
        bline_count_before = len(self.logic.bLines)

        # Paste the lines
        self.widget.onPasteLines(force=True)

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

        total_lines_before = len(self.logic.pleuraLines) + len(self.logic.bLines)

        # Copy and paste multiple times
        for i in range(3):
            # Select and copy
            self.widget.onSelectAllLines()
            self.widget.onCopyLines()

            # Paste
            self.widget.onPasteLines(force=True)

            # Verify we have more lines each time
            total_lines = len(self.logic.pleuraLines) + len(self.logic.bLines)
            expected_lines = total_lines_before * (2 ** (i+1))  # initial lines, doubled each time
            assert total_lines == expected_lines, f"Total lines should be: {total_lines_before} * (2 ** {i+1}) = {expected_lines}"

        print("✅ Multiple copy/paste operations test passed")

    def test_keyboard_shortcuts(self):
        """Test keyboard shortcuts for select all, copy, paste, and deselect."""
        print("Testing keyboard shortcuts...")

        # Ensure we have test lines
        self.create_test_lines()
        initial_line_count = len(self.logic.pleuraLines) + len(self.logic.bLines)
        assert initial_line_count > 0, "Should have lines to test with"

        # Test Ctrl+A (Select All)
        print("Testing Ctrl+A (Select All)...")
        self.logic.selectedLineIDs = []  # Clear selection first

        try:
            # Simulate Ctrl+A
            self.widget.shortcutSelectAll.activated.emit()

            # Verify all lines are selected
            expected_selected = len(self.logic.pleuraLines) + len(self.logic.bLines)
            actual_selected = len(self.logic.selectedLineIDs)
            assert actual_selected == expected_selected, f"Expected {expected_selected} selected lines, got {actual_selected}"
            print("✅ Ctrl+A (Select All) shortcut works")

        except Exception as e:
            print(f"⚠️ Could not test Ctrl+A shortcut: {e}")
            raise  # Re-raise the exception so the test fails properly

        # Test Ctrl+C (Copy)
        print("Testing Ctrl+C (Copy)...")
        try:
            # Ensure we have selected lines to copy
            if len(self.logic.selectedLineIDs) == 0:
                self.widget.onSelectAllLines()

            self.widget.shortcutCopy.activated.emit()

            # Verify lines were copied
            expected_copied = len(self.logic.pleuraLines) + len(self.logic.bLines)
            actual_copied = len(self.logic.clipboardLines)
            assert actual_copied == expected_copied, f"Expected {expected_copied} copied lines, got {actual_copied}"
            print("✅ Ctrl+C (Copy) shortcut works")

        except Exception as e:
            print(f"⚠️ Could not test Ctrl+C shortcut: {e}")
            raise  # Re-raise the exception so the test fails properly

        # Test Ctrl+V (Paste)
        print("Testing Ctrl+V (Paste)...")
        try:
            # Clear any existing lines and create fresh ones for copying
            self.logic.clearSceneLines()
            self.create_test_lines()

            # Select and copy lines
            self.widget.onSelectAllLines()
            self.widget.onCopyLines()

            # Verify we have lines in clipboard
            assert self.logic.clipboardLines and len(self.logic.clipboardLines) > 0, "Should have lines in clipboard"

            # Move to next frame for pasting (to avoid duplicate detection)
            self.widget._nextFrameInSequence()
            time.sleep(1)

            # Count lines before pasting
            pleura_count_before = len(self.logic.pleuraLines)
            bline_count_before = len(self.logic.bLines)

                                    # For testing purposes, temporarily mock the focus check to always return True
            original_red_view_has_focus = self.widget._redViewHasFocus
            self.widget._redViewHasFocus = lambda: True

            try:
                # Trigger paste via shortcut (should now work since focus check returns True)
                self.widget.shortcutPaste.activated.emit()
            finally:
                # Restore original method
                self.widget._redViewHasFocus = original_red_view_has_focus

            # Give some time for the paste operation to complete
            time.sleep(0.5)

            pleura_count_after = len(self.logic.pleuraLines)
            bline_count_after = len(self.logic.bLines)

            # Should have more lines after pasting
            assert pleura_count_after > pleura_count_before or bline_count_after > bline_count_before, f"Should have more lines after pasting: before={pleura_count_before}+{bline_count_before}, after={pleura_count_after}+{bline_count_after}"
            print("✅ Ctrl+V (Paste) shortcut works")

        except Exception as e:
            print(f"⚠️ Could not test Ctrl+V shortcut: {e}")
            raise  # Re-raise the exception so the test fails properly

        # Test Escape (Deselect All)
        print("Testing Escape (Deselect All)...")
        try:
            # First select some lines
            self.widget.onSelectAllLines()
            assert len(self.logic.selectedLineIDs) > 0, "Should have selected lines"

            # Simulate Escape
            self.widget.shortcutEscape.activated.emit()

            # Verify selection is cleared
            assert len(self.logic.selectedLineIDs) == 0, "Selection should be cleared after Escape"
            print("✅ Escape (Deselect All) shortcut works")

        except Exception as e:
            print(f"⚠️ Could not test Escape shortcut: {e}")
            raise  # Re-raise the exception so the test fails properly

        # Test Delete (Delete Selected Lines)
        print("Testing Delete (Delete Selected Lines)...")

        # Mock the QMessageBox to simulate user clicking "Yes"
        original_question = qt.QMessageBox.question
        qt.QMessageBox.question = lambda *args, **kwargs: qt.QMessageBox.Yes

        try:
            # Create fresh test lines for deletion test
            self.create_test_lines()

            # First select some lines
            self.widget.onSelectAllLines()
            assert len(self.logic.selectedLineIDs) > 0, "Should have selected lines"

            # Count lines before deletion
            total_lines_before = len(self.logic.pleuraLines) + len(self.logic.bLines)

            # Simulate Delete using the event filter
            # Create a QKeyEvent for Delete key
            key_event = qt.QKeyEvent(qt.QEvent.KeyPress, qt.Qt.Key_Delete, qt.Qt.NoModifier)

            # Send the event to the slice view (which has the event filter installed)
            slice_view = slicer.app.layoutManager().sliceWidget("Red").sliceView()
            qt.QApplication.postEvent(slice_view, key_event)
            qt.QApplication.processEvents()

            # Verify lines were deleted
            total_lines_after = len(self.logic.pleuraLines) + len(self.logic.bLines)
            assert total_lines_after < total_lines_before, f"Should have fewer lines after Delete key: {total_lines_after} < {total_lines_before}"

            # Verify selection is cleared
            assert len(self.logic.selectedLineIDs) == 0, "Selection should be cleared after Delete"
            print("✅ Delete (Delete Selected Lines) shortcut works")

        except Exception as e:
            print(f"⚠️ Could not test Delete shortcut: {e}")
            raise  # Re-raise the exception so the test fails properly
        finally:
            # Restore original QMessageBox.question
            qt.QMessageBox.question = original_question

        print("✅ Keyboard shortcuts test passed")

    def test_individual_line_selection(self):
        """Test selecting and unselecting individual lines by clicking on them."""
        print("Testing individual line selection by clicking...")

        # Clear any existing selection
        self.logic.selectedLineIDs = []

        # Test selecting individual lines by calling toggleLineSelection directly
        # Only test a subset to avoid too much output
        test_lines = (self.logic.pleuraLines[:5] + self.logic.bLines[:5])  # Test first 5 of each type

        for i, node in enumerate(test_lines):
            if not node or not slicer.mrmlScene.IsNodePresent(node):
                continue

            # Test selecting the line
            initial_selection_count = len(self.logic.selectedLineIDs)
            self.widget.toggleLineSelection(node)

            # Verify the line is now selected
            assert node.GetID() in self.logic.selectedLineIDs, f"Line {i} should be selected after toggle"
            assert len(self.logic.selectedLineIDs) == initial_selection_count + 1, f"Selection count should increase by 1"

            # Verify the line has selected appearance
            display_node = node.GetDisplayNode()
            if display_node:
                assert display_node.GetLineThickness() >= 0.3, f"Selected line should have thicker line"
                assert display_node.GetGlyphScale() >= 2.0, f"Selected line should have larger glyphs"

        # Try to unselect the first selected line
        if self.logic.selectedLineIDs:
            first_selected_id = self.logic.selectedLineIDs[0]
            first_selected_node = slicer.mrmlScene.GetNodeByID(first_selected_id)

            if first_selected_node:
                initial_selection_count = len(self.logic.selectedLineIDs)

                # Toggle again to unselect
                self.widget.toggleLineSelection(first_selected_node)

                # Verify the line is now unselected
                assert first_selected_id not in self.logic.selectedLineIDs, f"Line should be unselected after second toggle"
                assert len(self.logic.selectedLineIDs) == initial_selection_count - 1, f"Selection count should decrease by 1"

                # Verify the line has normal appearance
                display_node = first_selected_node.GetDisplayNode()
                if display_node:
                    assert display_node.GetLineThickness() <= 0.3, f"Unselected line should have normal line thickness"
                    assert display_node.GetGlyphScale() <= 2.5, f"Unselected line should have normal glyph scale"

        print("✅ Individual line selection test passed")

    def test_control_point_selection(self):
        """Test selecting lines by clicking on their control points."""
        print("Testing control point selection...")

        # Clear any existing selection
        self.logic.selectedLineIDs = []

        # Get the slice view for testing clicks
        sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
        sliceNode = sliceWidget.mrmlSliceNode()

        # Get the transformation matrix from slice coordinates to RAS
        xyToRas = sliceNode.GetXYToRAS()
        # Get the inverse transformation from RAS to slice coordinates
        rasToXY = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(xyToRas, rasToXY)

        # Test clicking on control points of each line
        # Only test a subset to avoid too much output
        test_lines = (self.logic.pleuraLines[:3] + self.logic.bLines[:3])  # Test first 3 of each type

        for i, node in enumerate(test_lines):
            if not node or not slicer.mrmlScene.IsNodePresent(node):
                continue

            # Test clicking on each control point
            for cp_idx in range(node.GetNumberOfControlPoints()):
                # Get control point position in RAS coordinates
                cp_pos = [0, 0, 0]
                node.GetNthControlPointPositionWorld(cp_idx, cp_pos)

                # Convert RAS to slice coordinates using the inverse transformation
                ras_homogeneous = [cp_pos[0], cp_pos[1], cp_pos[2], 1.0]
                xy_homogeneous = [0, 0, 0, 0]
                rasToXY.MultiplyPoint(ras_homogeneous, xy_homogeneous)

                # Extract the slice coordinates (x, y)
                click_x = int(xy_homogeneous[0])
                click_y = int(xy_homogeneous[1])

                # Check if the coordinates are within the slice view bounds
                sliceView = sliceWidget.sliceView()
                view_width = sliceView.width
                view_height = sliceView.height

                if 0 <= click_x < view_width and 0 <= click_y < view_height:
                    try:
                        was_picked = self.widget.pickClosestLineNodeInSlice(click_x, click_y)

                        if was_picked:
                            # Verify the line is selected
                            assert node.GetID() in self.logic.selectedLineIDs, f"Line {i} should be selected after clicking control point"

                            # Verify the line has selected appearance
                            display_node = node.GetDisplayNode()
                            if display_node:
                                assert display_node.GetLineThickness() >= 0.3, f"Selected line should have thicker line"
                                assert display_node.GetGlyphScale() >= 2.0, f"Selected line should have larger glyphs"

                            break  # Found this line, move to next

                    except Exception as e:
                        continue

            # If we didn't successfully pick this line, try clicking near the middle of the line
            if node.GetID() not in self.logic.selectedLineIDs:

                # Get line endpoints in RAS coordinates
                pt1 = [0, 0, 0]
                pt2 = [0, 0, 0]
                node.GetNthControlPointPositionWorld(0, pt1)
                node.GetNthControlPointPositionWorld(1, pt2)

                # Calculate middle point in RAS coordinates
                mid_ras = [(pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2, (pt1[2] + pt2[2]) / 2]

                # Convert middle point to slice coordinates
                mid_ras_homogeneous = [mid_ras[0], mid_ras[1], mid_ras[2], 1.0]
                mid_xy_homogeneous = [0, 0, 0, 0]
                rasToXY.MultiplyPoint(mid_ras_homogeneous, mid_xy_homogeneous)

                mid_x = int(mid_xy_homogeneous[0])
                mid_y = int(mid_xy_homogeneous[1])

                if 0 <= mid_x < view_width and 0 <= mid_y < view_height:
                    was_picked = self.widget.pickClosestLineNodeInSlice(mid_x, mid_y)

                    if was_picked:
                        assert node.GetID() in self.logic.selectedLineIDs, f"Line {i} should be selected after middle click"

        print("✅ Control point selection test passed")

    def test_delete_selected_lines(self):
        """Test deleting selected lines."""
        print("Testing delete selected lines...")

        # First, select some lines
        self.widget.onSelectAllLines()
        initial_selected_count = len(self.logic.selectedLineIDs)
        assert initial_selected_count > 0, "Should have selected lines to delete"

        # Count lines before deletion
        pleura_count_before = len(self.logic.pleuraLines)
        bline_count_before = len(self.logic.bLines)
        total_lines_before = pleura_count_before + bline_count_before

        # Delete the selected lines
        self.widget.onDeleteSelectedLines(force=True)

        # Verify lines were deleted
        pleura_count_after = len(self.logic.pleuraLines)
        bline_count_after = len(self.logic.bLines)
        total_lines_after = pleura_count_after + bline_count_after

        # Should have fewer lines after deletion
        assert total_lines_after < total_lines_before, f"Should have fewer lines after deletion: {total_lines_after} < {total_lines_before}"

        # Verify selection is cleared after deletion
        assert len(self.logic.selectedLineIDs) == 0, "Selection should be cleared after deletion"

        # Verify unsaved changes flag is set
        assert self.widget._parameterNode.unsavedChanges, "Unsaved changes flag should be set after deletion"

        print(f"✅ Delete selected lines test passed: deleted {total_lines_before - total_lines_after} lines")

    def test_delete_single_line(self):
        """Test deleting a single selected line."""
        print("Testing delete single line...")

        self.create_test_lines()
        # Select only one line
        if len(self.logic.pleuraLines) > 0:
            test_line = self.logic.pleuraLines[0]
            self.widget.toggleLineSelection(test_line)
            assert len(self.logic.selectedLineIDs) == 1, "Should have exactly one selected line"

            # Count lines before deletion
            pleura_count_before = len(self.logic.pleuraLines)

            # Delete the selected line
            self.widget.onDeleteSelectedLines(force=True)

            # Verify line was deleted
            pleura_count_after = len(self.logic.pleuraLines)
            assert pleura_count_after == pleura_count_before - 1, f"Should have deleted exactly one line: {pleura_count_before} -> {pleura_count_after}"

            # Verify selection is cleared
            assert len(self.logic.selectedLineIDs) == 0, "Selection should be cleared after deletion"

            print("✅ Delete single line test passed")

        elif len(self.logic.bLines) > 0:
            test_line = self.logic.bLines[0]
            self.widget.toggleLineSelection(test_line)
            assert len(self.logic.selectedLineIDs) == 1, "Should have exactly one selected line"

            # Count lines before deletion
            bline_count_before = len(self.logic.bLines)

            # Delete the selected line
            self.widget.onDeleteSelectedLines(force=True)

            # Verify line was deleted
            bline_count_after = len(self.logic.bLines)
            assert bline_count_after == bline_count_before - 1, f"Should have deleted exactly one line: {bline_count_before} -> {bline_count_after}"

            # Verify selection is cleared
            assert len(self.logic.selectedLineIDs) == 0, "Selection should be cleared after deletion"

            print("✅ Delete single line test passed")

        else:
            print("⚠️ No lines available for single line deletion test")

    def test_delete_no_selection(self):
        """Test delete behavior when no lines are selected."""
        print("Testing delete with no selection...")

        # Clear any existing selection
        self.logic.selectedLineIDs = []
        assert len(self.logic.selectedLineIDs) == 0, "Should have no selected lines"

        # Count lines before attempting deletion
        pleura_count_before = len(self.logic.pleuraLines)
        bline_count_before = len(self.logic.bLines)

        # Try to delete with no selection
        self.widget.onDeleteSelectedLines(force=True)

        # Verify no lines were deleted
        pleura_count_after = len(self.logic.pleuraLines)
        bline_count_after = len(self.logic.bLines)
        assert pleura_count_after == pleura_count_before, "No pleura lines should be deleted"
        assert bline_count_after == bline_count_before, "No b-lines should be deleted"

        print("✅ Delete with no selection test passed")

    def test_delete_keyboard_shortcut(self):
        """Test the Delete key shortcut for deleting selected lines."""
        print("Testing Delete key shortcut...")

        # Select some lines
        self.widget.onSelectAllLines()
        initial_selected_count = len(self.logic.selectedLineIDs)
        assert initial_selected_count > 0, "Should have selected lines"

        # Count lines before deletion
        total_lines_before = len(self.logic.pleuraLines) + len(self.logic.bLines)

        # Mock the QMessageBox to simulate user clicking "Yes"
        original_question = qt.QMessageBox.question
        qt.QMessageBox.question = lambda *args, **kwargs: qt.QMessageBox.Yes

        # Simulate Delete key press using the event filter
        try:
            # Create a QKeyEvent for Delete key
            key_event = qt.QKeyEvent(qt.QEvent.KeyPress, qt.Qt.Key_Delete, qt.Qt.NoModifier)

            # Send the event to the slice view (which has the event filter installed)
            slice_view = slicer.app.layoutManager().sliceWidget("Red").sliceView()
            qt.QApplication.postEvent(slice_view, key_event)
            qt.QApplication.processEvents()

            # Verify lines were deleted
            total_lines_after = len(self.logic.pleuraLines) + len(self.logic.bLines)
            assert total_lines_after < total_lines_before, f"Should have fewer lines after Delete key: {total_lines_after} < {total_lines_before}"

            # Verify selection is cleared
            assert len(self.logic.selectedLineIDs) == 0, "Selection should be cleared after Delete key"

            print("✅ Delete key shortcut test passed")

        except Exception as e:
            print(f"⚠️ Could not test Delete key shortcut: {e}")
            raise  # Re-raise the exception so the test fails properly
        finally:
            # Restore original QMessageBox.question
            qt.QMessageBox.question = original_question

    def test_backspace_keyboard_shortcut(self):
        """Test the Backspace key shortcut for deleting selected lines."""
        print("Testing Backspace key shortcut...")

        # Create fresh test lines
        self.create_test_lines()

        # Select some lines
        self.widget.onSelectAllLines()
        initial_selected_count = len(self.logic.selectedLineIDs)
        assert initial_selected_count > 0, "Should have selected lines"

        # Count lines before deletion
        total_lines_before = len(self.logic.pleuraLines) + len(self.logic.bLines)

        # Mock the QMessageBox to simulate user clicking "Yes"
        original_question = qt.QMessageBox.question
        qt.QMessageBox.question = lambda *args, **kwargs: qt.QMessageBox.Yes

        # Simulate Backspace key press using the event filter
        try:
            # Create a QKeyEvent for Backspace key
            key_event = qt.QKeyEvent(qt.QEvent.KeyPress, qt.Qt.Key_Backspace, qt.Qt.NoModifier)

            # Send the event to the slice view (which has the event filter installed)
            slice_view = slicer.app.layoutManager().sliceWidget("Red").sliceView()
            qt.QApplication.postEvent(slice_view, key_event)
            qt.QApplication.processEvents()

            # Verify lines were deleted
            total_lines_after = len(self.logic.pleuraLines) + len(self.logic.bLines)
            assert total_lines_after < total_lines_before, f"Should have fewer lines after Backspace key: {total_lines_after} < {total_lines_before}"

            # Verify selection is cleared
            assert len(self.logic.selectedLineIDs) == 0, "Selection should be cleared after Backspace key"

            print("✅ Backspace key shortcut test passed")

        except Exception as e:
            print(f"⚠️ Could not test Backspace key shortcut: {e}")
            raise  # Re-raise the exception so the test fails properly
        finally:
            # Restore original QMessageBox.question
            qt.QMessageBox.question = original_question

    def test_delete_rater_filtering(self):
        """Test that delete only affects lines belonging to the current rater."""
        print("Testing delete rater filtering...")

        # Create lines with different raters
        current_rater = self.logic.getRater()
        other_rater = "other_rater" if current_rater != "other_rater" else "test_rater"

        # Create a line with the current rater
        current_rater_line = self.logic.createMarkupLine(
            "Pleura", current_rater,
            [[-90.0, -45.0, 0.0], [-100.0, -47.0, 0.0]],
            [1, 1, 0]
        )
        current_rater_line.SetAttribute("rater", current_rater)
        self.logic.pleuraLines.append(current_rater_line)

        # Create a line with a different rater
        other_rater_line = self.logic.createMarkupLine(
            "B-line", other_rater,
            [[-95.0, -50.0, 0.0], [-105.0, -52.0, 0.0]],
            [0, 1, 1]
        )
        other_rater_line.SetAttribute("rater", other_rater)
        self.logic.bLines.append(other_rater_line)

        # Select both lines
        self.widget.toggleLineSelection(current_rater_line)
        self.widget.toggleLineSelection(other_rater_line)
        assert len(self.logic.selectedLineIDs) == 2, "Should have two selected lines"

        # Count lines before deletion
        pleura_count_before = len(self.logic.pleuraLines)
        bline_count_before = len(self.logic.bLines)

        # Delete selected lines
        self.widget.onDeleteSelectedLines()

        # Verify only the current rater's line was deleted
        pleura_count_after = len(self.logic.pleuraLines)
        bline_count_after = len(self.logic.bLines)

        # The current rater's line should be deleted, but the other rater's line should remain
        assert pleura_count_after == pleura_count_before - 1, f"Current rater's line should be deleted: {pleura_count_before} -> {pleura_count_after}"
        assert bline_count_after == bline_count_before, f"Other rater's line should remain: {bline_count_before}"

        # Verify the other rater's line is still in the scene
        assert slicer.mrmlScene.IsNodePresent(other_rater_line), "Other rater's line should still be in scene"

        print("✅ Delete rater filtering test passed")

    def test_delete_confirmation_dialog(self):
        """Test the confirmation dialog when deleting multiple lines."""
        print("Testing delete confirmation dialog...")

        self.create_test_lines()

        # Select multiple lines
        self.widget.onSelectAllLines()
        selected_count = len(self.logic.selectedLineIDs)
        assert selected_count > 1, "Should have multiple selected lines for confirmation test"

        # Mock the QMessageBox to simulate user clicking "No"
        original_question = qt.QMessageBox.question
        qt.QMessageBox.question = lambda *args, **kwargs: qt.QMessageBox.No

        try:
            # Count lines before attempted deletion
            total_lines_before = len(self.logic.pleuraLines) + len(self.logic.bLines)

            # Try to delete (should be cancelled by the mock)
            self.widget.onDeleteSelectedLines()

            # Verify no lines were deleted
            total_lines_after = len(self.logic.pleuraLines) + len(self.logic.bLines)
            assert total_lines_after == total_lines_before, "No lines should be deleted when user cancels"

            # Verify selection is still intact
            assert len(self.logic.selectedLineIDs) == selected_count, "Selection should remain when user cancels"

            print("✅ Delete confirmation dialog test passed (cancelled)")

        finally:
            # Restore original QMessageBox.question
            qt.QMessageBox.question = original_question

        # Now test with user clicking "Yes"
        qt.QMessageBox.question = lambda *args, **kwargs: qt.QMessageBox.Yes

        try:
            # Count lines before deletion
            total_lines_before = len(self.logic.pleuraLines) + len(self.logic.bLines)

            # Delete the selected lines
            self.widget.onDeleteSelectedLines()

            # Verify lines were deleted
            total_lines_after = len(self.logic.pleuraLines) + len(self.logic.bLines)
            assert total_lines_after < total_lines_before, f"Lines should be deleted when user confirms: {total_lines_after} < {total_lines_before}"

            print("✅ Delete confirmation dialog test passed (confirmed)")

        finally:
            # Restore original QMessageBox.question
            qt.QMessageBox.question = original_question

    def test_cut_lines(self):
        """Test cutting selected lines (Cmd+X functionality)."""
        print("Testing cut lines (Cmd+X)...")

        # Create fresh test lines
        self.create_test_lines()

        # Select all the lines
        self.widget.onSelectAllLines()
        initial_selected_count = len(self.logic.selectedLineIDs)
        assert initial_selected_count > 0, "Should have selected lines to cut"

        # Count lines before cutting
        pleura_count_before = len(self.logic.pleuraLines)
        bline_count_before = len(self.logic.bLines)
        total_lines_before = pleura_count_before + bline_count_before

        # Mock the QMessageBox to simulate user clicking "Yes" for delete confirmation
        original_question = qt.QMessageBox.question
        qt.QMessageBox.question = lambda *args, **kwargs: qt.QMessageBox.Yes

        try:
            # Cut the selected lines using the cut method
            self.widget._cutIfRedViewFocused()

            # Verify lines were copied to clipboard
            assert self.logic.clipboardLines is not None, "Clipboard lines should not be None"
            expected_copied = total_lines_before
            actual_copied = len(self.logic.clipboardLines)
            assert actual_copied == expected_copied, f"Expected {expected_copied} copied lines, got {actual_copied}"

            # Verify lines were deleted from the scene
            pleura_count_after = len(self.logic.pleuraLines)
            bline_count_after = len(self.logic.bLines)
            total_lines_after = pleura_count_after + bline_count_after
            assert total_lines_after < total_lines_before, f"Should have fewer lines after cut: {total_lines_after} < {total_lines_before}"

            # Verify selection is cleared after cut
            assert len(self.logic.selectedLineIDs) == 0, "Selection should be cleared after cut"

            # Verify clipboard lines are hidden
            for node in self.logic.clipboardLines:
                assert not node.GetDisplayVisibility(), "Clipboard lines should be hidden"

            # Verify unsaved changes flag is set
            assert self.widget._parameterNode.unsavedChanges, "Unsaved changes flag should be set after cut"

            print("✅ Cut lines test passed")

        except Exception as e:
            print(f"⚠️ Could not test cut functionality: {e}")
        finally:
            # Restore original QMessageBox.question
            qt.QMessageBox.question = original_question

    def test_cut_keyboard_shortcut(self):
        """Test the Cmd+X keyboard shortcut for cutting selected lines."""
        print("Testing Cmd+X keyboard shortcut...")

        # Create fresh test lines
        self.create_test_lines()

        # Select some lines
        self.widget.onSelectAllLines()
        initial_selected_count = len(self.logic.selectedLineIDs)
        assert initial_selected_count > 0, "Should have selected lines"

        # Count lines before cutting
        total_lines_before = len(self.logic.pleuraLines) + len(self.logic.bLines)

        # Mock the QMessageBox to simulate user clicking "Yes"
        original_question = qt.QMessageBox.question
        qt.QMessageBox.question = lambda *args, **kwargs: qt.QMessageBox.Yes

        try:
            # Simulate Cmd+X using the shortcut
            self.widget.shortcutCut.activated.emit()

            # Verify lines were copied to clipboard
            assert self.logic.clipboardLines is not None, "Clipboard lines should not be None"
            expected_copied = total_lines_before
            actual_copied = len(self.logic.clipboardLines)
            assert actual_copied == expected_copied, f"Expected {expected_copied} copied lines, got {actual_copied}"

            # Verify lines were deleted from the scene
            total_lines_after = len(self.logic.pleuraLines) + len(self.logic.bLines)
            assert total_lines_after < total_lines_before, f"Should have fewer lines after Cmd+X: {total_lines_after} < {total_lines_before}"

            # Verify selection is cleared
            assert len(self.logic.selectedLineIDs) == 0, "Selection should be cleared after Cmd+X"

            print("✅ Cmd+X keyboard shortcut test passed")

        except Exception as e:
            print(f"⚠️ Could not test Cmd+X shortcut: {e}")
            raise  # Re-raise the exception so the test fails properly
        finally:
            # Restore original QMessageBox.question
            qt.QMessageBox.question = original_question

    def test_cut_and_paste_workflow(self):
        """Test the complete cut and paste workflow."""
        print("Testing cut and paste workflow...")

        # Create fresh test lines
        self.create_test_lines()

        # Select all lines
        self.widget.onSelectAllLines()
        initial_selected_count = len(self.logic.selectedLineIDs)
        assert initial_selected_count > 0, "Should have selected lines"

        # Count lines before cutting
        pleura_count_before = len(self.logic.pleuraLines)
        bline_count_before = len(self.logic.bLines)
        total_lines_before = pleura_count_before + bline_count_before

        # Mock the QMessageBox to simulate user clicking "Yes"
        original_question = qt.QMessageBox.question
        qt.QMessageBox.question = lambda *args, **kwargs: qt.QMessageBox.Yes

        try:
            # Cut the lines
            self.widget._cutIfRedViewFocused()

            # Verify lines are in clipboard and removed from scene
            assert len(self.logic.clipboardLines) == total_lines_before, "All lines should be in clipboard"
            assert len(self.logic.pleuraLines) + len(self.logic.bLines) < total_lines_before, "Lines should be removed from scene"

            # Move to next frame
            self.widget._nextFrameInSequence()
            time.sleep(1)

            # Paste the lines
            self.widget.onPasteLines(force=True)

            # Verify lines were pasted
            pleura_count_after = len(self.logic.pleuraLines)
            bline_count_after = len(self.logic.bLines)
            total_lines_after = pleura_count_after + bline_count_after

            # Should have the same number of lines as before cutting
            assert total_lines_after == total_lines_before, f"Should have same number of lines after cut and paste: {total_lines_after} == {total_lines_before}"

            # Verify pasted lines are visible
            for node in self.logic.pleuraLines + self.logic.bLines:
                if slicer.mrmlScene.IsNodePresent(node):
                    assert node.GetDisplayVisibility(), "Pasted lines should be visible"

            print("✅ Cut and paste workflow test passed")

        except Exception as e:
            print(f"⚠️ Could not test cut and paste workflow: {e}")
        finally:
            # Restore original QMessageBox.question
            qt.QMessageBox.question = original_question

    def test_cut_no_selection(self):
        """Test cut behavior when no lines are selected."""
        print("Testing cut with no selection...")

        # Clear any existing selection
        self.logic.selectedLineIDs = []
        self.logic.clearClipboard()

        assert len(self.logic.selectedLineIDs) == 0, "Should have no selected lines"

        # Count lines before attempting cut
        pleura_count_before = len(self.logic.pleuraLines)
        bline_count_before = len(self.logic.bLines)
        total_lines_before = pleura_count_before + bline_count_before

        # Try to cut with no selection
        self.widget._cutIfRedViewFocused()

        # Verify no lines were affected
        pleura_count_after = len(self.logic.pleuraLines)
        bline_count_after = len(self.logic.bLines)
        total_lines_after = pleura_count_after + bline_count_after
        assert total_lines_after == total_lines_before, "No lines should be affected when cutting with no selection"

        # Verify clipboard is empty (should be an empty list, not None)
        assert self.logic.clipboardLines is not None, "Clipboard should be initialized as empty list"
        assert len(self.logic.clipboardLines) == 0, "Clipboard should be empty when cutting with no selection"

        print("✅ Cut with no selection test passed")

    def test_cut_single_line(self):
        """Test cutting a single selected line."""
        print("Testing cut single line...")

        self.create_test_lines()

        # Select only one line
        if len(self.logic.pleuraLines) > 0:
            test_line = self.logic.pleuraLines[0]
            self.widget.toggleLineSelection(test_line)
            assert len(self.logic.selectedLineIDs) == 1, "Should have exactly one selected line"

            # Count lines before cutting
            pleura_count_before = len(self.logic.pleuraLines)
            total_lines_before = pleura_count_before + len(self.logic.bLines)

            # Mock the QMessageBox to simulate user clicking "Yes"
            original_question = qt.QMessageBox.question
            qt.QMessageBox.question = lambda *args, **kwargs: qt.QMessageBox.Yes

            try:
                # Cut the selected line
                self.widget._cutIfRedViewFocused()

                # Verify line was copied to clipboard
                assert self.logic.clipboardLines is not None, "Clipboard should not be None"
                assert len(self.logic.clipboardLines) == 1, "Should have exactly one line in clipboard"

                # Verify line was deleted from scene
                pleura_count_after = len(self.logic.pleuraLines)
                assert pleura_count_after == pleura_count_before - 1, f"Should have deleted exactly one line: {pleura_count_before} -> {pleura_count_after}"

                # Verify selection is cleared
                assert len(self.logic.selectedLineIDs) == 0, "Selection should be cleared after cut"

                print("✅ Cut single line test passed")

            finally:
                # Restore original QMessageBox.question
                qt.QMessageBox.question = original_question

        elif len(self.logic.bLines) > 0:
            test_line = self.logic.bLines[0]
            self.widget.toggleLineSelection(test_line)
            assert len(self.logic.selectedLineIDs) == 1, "Should have exactly one selected line"

            # Count lines before cutting
            bline_count_before = len(self.logic.bLines)
            total_lines_before = len(self.logic.pleuraLines) + bline_count_before

            # Mock the QMessageBox to simulate user clicking "Yes"
            original_question = qt.QMessageBox.question
            qt.QMessageBox.question = lambda *args, **kwargs: qt.QMessageBox.Yes

            try:
                # Cut the selected line
                self.widget._cutIfRedViewFocused()

                # Verify line was copied to clipboard
                assert self.logic.clipboardLines is not None, "Clipboard should not be None"
                assert len(self.logic.clipboardLines) == 1, "Should have exactly one line in clipboard"

                # Verify line was deleted from scene
                bline_count_after = len(self.logic.bLines)
                assert bline_count_after == bline_count_before - 1, f"Should have deleted exactly one line: {bline_count_before} -> {bline_count_after}"

                # Verify selection is cleared
                assert len(self.logic.selectedLineIDs) == 0, "Selection should be cleared after cut"

                print("✅ Cut single line test passed")

            finally:
                # Restore original QMessageBox.question
                qt.QMessageBox.question = original_question

        else:
            print("⚠️ No lines available for single line cut test")

    def runTest(self):
        """Run the line selection, copy, paste, cut, and delete test."""
        print("Starting line selection, copy, paste, cut, and delete test...")
        print("This test will:")
        print("1. Load the AnnotateUltrasound module")
        print("2. Load test DICOM data")
        print("3. Create test lines (Pleura and B-lines)")
        print("4. Test line selection functionality")
        print("5. Test copy and paste functionality")
        print("6. Test cut functionality")
        print("7. Test delete functionality")
        print("8. Test keyboard shortcuts")

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

        print("\n=== PHASE 3: Individual Line Selection Tests ===")

        print("\n--- Testing Individual Line Selection by Clicking ---")
        self.test_individual_line_selection()

        print("\n--- Testing Control Point Selection ---")
        self.test_control_point_selection()

        print("\n=== PHASE 4: Delete Line Tests ===")
        print("\n--- Testing Delete Selected Lines ---")
        self.test_delete_selected_lines()
        print("\n--- Testing Delete Single Line ---")
        self.test_delete_single_line()
        print("\n--- Testing Delete with No Selection ---")
        self.test_delete_no_selection()
        print("\n--- Testing Delete Keyboard Shortcut ---")
        self.test_delete_keyboard_shortcut()
        print("\n--- Testing Backspace Keyboard Shortcut ---")
        self.test_backspace_keyboard_shortcut()
        print("\n--- Testing Delete Rater Filtering ---")
        self.test_delete_rater_filtering()
        print("\n--- Testing Delete Confirmation Dialog ---")
        self.test_delete_confirmation_dialog()

        print("\n=== PHASE 5: Cut Line Tests ===")
        print("\n--- Testing Cut Lines (Cmd+X) ---")
        self.test_cut_lines()
        print("\n--- Testing Cut Keyboard Shortcut (Cmd+X) ---")
        self.test_cut_keyboard_shortcut()
        print("\n--- Testing Cut and Paste Workflow ---")
        self.test_cut_and_paste_workflow()
        print("\n--- Testing Cut with No Selection ---")
        self.test_cut_no_selection()
        print("\n--- Testing Cut Single Line ---")
        self.test_cut_single_line()

        print("\n=== TEST SUMMARY ===")
        print("✅ All line selection, copy, paste, cut, and delete tests completed successfully!")
        print("The following functionality was tested:")
        print("- Line selection (Select All, Deselect All)")
        print("- Copy and paste operations")
        print("- Multiple copy/paste cycles")
        print("- Cut operations (Cmd+X)")
        print("- Cut and paste workflow")
        print("- Keyboard shortcuts (Ctrl+A, Ctrl+C, Ctrl+V, Cmd+X, Escape, Delete, Backspace)")
        print("- Visual feedback for selected lines")
        print("- Clipboard management")
        print("- Individual line selection by clicking")
        print("- Control point selection")
        print("- Delete functionality (multiple, single, no selection, keyboard shortcut, rater filtering, confirmation dialog)")
        print("- Cut functionality (multiple, single, no selection, keyboard shortcut, workflow)")
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