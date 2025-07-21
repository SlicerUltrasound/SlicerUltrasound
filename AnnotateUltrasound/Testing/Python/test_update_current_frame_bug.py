#!/usr/bin/env python3
"""
Test for updateCurrentFrame bug in AnnotateUltrasound module.
Standalone test that doesn't rely on test framework classes.
"""

import sys
import os

def test_updateCurrentFrame_bug(test_data_dir=None):
    """
    Test the specific bug in updateCurrentFrame.

    Args:
        test_data_dir: Optional path to test data directory. If None, will try to find it.
    """
    print("=== Testing updateCurrentFrame bug ===")

    # Import slicer first and ensure it's initialized
    try:
        import slicer

        # Check if slicer app is available
        if not hasattr(slicer, 'app') or slicer.app is None:
            print("ERROR: Slicer app not available - cannot run test")
            return False

        # Clear scene first
        slicer.mrmlScene.Clear(0)
        print("SUCCESS: Slicer initialized and scene cleared")

        # Add the module path to sys.path - try multiple approaches
        possible_paths = [
            os.path.join(os.getcwd(), 'AnnotateUltrasound'),
            os.getcwd()
        ]

        module_imported = False
        for path in possible_paths:
            if path not in sys.path:
                sys.path.insert(0, path)
            try:
                from AnnotateUltrasound import AnnotateUltrasoundLogic
                print("SUCCESS: AnnotateUltrasound module imported from: {}".format(path))
                module_imported = True
                break
            except ImportError:
                continue

        if not module_imported:
            print("ERROR: Could not import AnnotateUltrasound module")
            print("Tried paths: {}".format(possible_paths))
            return False

    except Exception as e:
        print("ERROR: Error during setup: {}".format(e))
        return False

    try:
        # Set up test data directory - try multiple locations
        if test_data_dir is None:
            possible_test_dirs = [
                os.path.join(os.getcwd(), 'AnnotateUltrasound', 'Testing', 'Python', 'test_data'),
                os.path.join(os.getcwd(), 'test_data'),
                'AnnotateUltrasound/Testing/Python/test_data'
            ]

            test_data_dir = None
            for test_dir in possible_test_dirs:
                if os.path.exists(test_dir):
                    test_data_dir = test_dir
                    break

        if test_data_dir is None or not os.path.exists(test_data_dir):
            print("ERROR: Could not find test data directory")
            print("Tried: {}".format(possible_test_dirs))
            print("Current working directory: {}".format(os.getcwd()))
            return False

        print("SUCCESS: Test data directory: {}".format(test_data_dir))

        # List files in test data directory
        try:
            files = os.listdir(test_data_dir)
            print("Test data files: {}".format(files))
        except Exception as e:
            print("ERROR: Cannot list test data files: {}".format(e))
            return False

        # Create logic instance
        logic = AnnotateUltrasoundLogic()
        print("SUCCESS: Logic instance created")

        # Set rater (required for loading)
        logic.setRater("test_rater")
        print("SUCCESS: Rater set to 'test_rater'")

        # Load DICOM directory
        print("Loading DICOM directory...")
        num_files, num_annotations = logic.updateInputDf("test_rater", test_data_dir)
        print("SUCCESS: Found {} DICOM files, {} annotations".format(num_files, num_annotations))

        if num_files == 0:
            print("ERROR: No DICOM files found in test data")
            return False

        # Load first sequence
        print("Loading first sequence...")
        current_index = logic.loadNextSequence()
        if current_index is None:
            print("ERROR: Failed to load first sequence")
            return False
        print("SUCCESS: Loaded sequence at index {}".format(current_index))

        # Verify sequence browser is available
        if logic.sequenceBrowserNode is None:
            print("ERROR: No sequence browser node found")
            return False

        # Get number of frames in sequence
        num_frames = logic.sequenceBrowserNode.GetNumberOfItems()
        print("SUCCESS: Sequence has {} frames".format(num_frames))
        if num_frames <= 1:
            print("ERROR: Need at least 2 frames for this test")
            return False

        # Start at frame 0
        logic.sequenceBrowserNode.SetSelectedItemNumber(0)
        current_frame = logic.sequenceBrowserNode.GetSelectedItemNumber()
        print("SUCCESS: Starting at frame {}".format(current_frame))
        frame0_visible_pleura_count = countVisiblePleuraLines(logic)
        print("Got {} visible pleura lines at frame 0".format(frame0_visible_pleura_count))

        # Step 1: Add a pleura line at frame 0
        print("\n--- Step 1: Adding pleura line at frame 0 ---")
        pleura_coords = [[100, 100, 0], [200, 150, 0]]
        pleura_line_1 = logic.createMarkupLine("Pleura", "test_rater", pleura_coords)
        logic.pleuraLines.append(pleura_line_1)
        print("SUCCESS: Created pleura line")

        # Update current frame to save the annotation
        logic.syncMarkupsToAnnotations()
        logic.refreshDisplay(updateOverlay=True, updateGui=True)
        print("SUCCESS: Updated current frame")

        # Verify we have 1 pleura line
        visible_pleura_count = countVisiblePleuraLines(logic)
        new_visible_pleura_count = visible_pleura_count - frame0_visible_pleura_count
        print("SUCCESS: Frame 0: {} new visible pleura lines".format(new_visible_pleura_count))
        if new_visible_pleura_count != 1:
            print("ERROR: Should have exactly 1 visible pleura line at frame 0, got {}".format(new_visible_pleura_count))
            return False

        # Step 2: Navigate to frame 1
        print("\n--- Step 2: Navigating to frame 1 ---")
        logic.sequenceBrowserNode.SetSelectedItemNumber(1)
        current_frame = logic.sequenceBrowserNode.GetSelectedItemNumber()
        print("SUCCESS: Now at frame {}".format(current_frame))

        # Trigger update (this should hide lines from frame 0)
        logic.syncAnnotationsToMarkups()
        logic.refreshDisplay(updateOverlay=True, updateGui=True)
        print("SUCCESS: Updated line markups")
        frame1_visible_pleura_count = countVisiblePleuraLines(logic)
        print("Got {} visible pleura lines at frame 1".format(frame1_visible_pleura_count))

        # Step 3: Add another pleura line at frame 1
        print("\n--- Step 3: Adding pleura line at frame 1 ---")
        pleura_coords_2 = [[150, 120, 0], [250, 180, 0]]
        pleura_line_2 = logic.createMarkupLine("Pleura", "test_rater", pleura_coords_2)
        point1 = [0, 0, 0]
        point2 = [0, 0, 0]
        pleura_line_2.GetNthControlPointPosition(0, point1)
        pleura_line_2.GetNthControlPointPosition(1, point2)
        print("SUCCESS: Created second pleura line with coordinates: {}".format(pleura_coords_2))
        print("SUCCESS: Second pleura line point 1: {}".format(point1))
        print("SUCCESS: Second pleura line point 2: {}".format(point2))
        logic.pleuraLines.append(pleura_line_2)
        print("SUCCESS: Created second pleura line")

        # Update current frame to save the annotation
        logic.syncMarkupsToAnnotations()
        logic.refreshDisplay(updateOverlay=True, updateGui=True)
        print("SUCCESS: Updated current frame")

        # Verify we have 1 pleura line visible at frame 1
        visible_pleura_count = countVisiblePleuraLines(logic)
        new_visible_pleura_count = visible_pleura_count - frame1_visible_pleura_count
        print("SUCCESS: Frame 1: {} new visible pleura lines".format(new_visible_pleura_count))
        if new_visible_pleura_count != 1:
            print("ERROR: Should have exactly 1 visible pleura line at frame 1, got {}".format(new_visible_pleura_count))
            return False

        # Step 4: Navigate back to frame 0
        print("\n--- Step 4: Navigating back to frame 0 ---")
        logic.sequenceBrowserNode.SetSelectedItemNumber(0)
        current_frame = logic.sequenceBrowserNode.GetSelectedItemNumber()
        print("SUCCESS: Back to frame {}".format(current_frame))

        # Trigger update (this should show only frame 0 lines)
        logic.syncAnnotationsToMarkups()
        logic.refreshDisplay(updateOverlay=True, updateGui=True)
        print("SUCCESS: Updated line markups")

        # Step 5: Verify only 1 pleura line is visible at frame 0 (the bug would show multiple)
        print("\n--- Step 5: Final verification at frame 0 ---")
        visible_pleura_count = countVisiblePleuraLines(logic)
        total_pleura_nodes = len(logic.pleuraLines)

        print("Frame 0 final check:")
        print("  - Total pleura nodes: {}".format(total_pleura_nodes))
        print("  - Visible pleura lines: {}".format(visible_pleura_count))

        # Show details of each line
        for i, line in enumerate(logic.pleuraLines):
            visible = line.GetDisplayNode().GetVisibility()
            rater = line.GetAttribute("rater")
            print("  - Line {}: visible={}, rater='{}'".format(i, visible, rater))

        # The bug test: frame 0 should have exactly 1 new visible pleura line
        new_visible_pleura_count = visible_pleura_count - frame0_visible_pleura_count
        if new_visible_pleura_count != 1:
            print("\nBUG DETECTED: Should have exactly 1 new visible pleura line at frame 0, "
                  "but found {}".format(new_visible_pleura_count))
            print("This indicates the updateCurrentFrame bug is present!")
            return False

        print("SUCCESS: Frame 0 check passed - exactly 1 new visible pleura line")

        # Step 6: Go back to frame 1 and verify it still has exactly 1 visible pleura line
        print("\n--- Step 6: Final verification at frame 1 ---")
        logic.sequenceBrowserNode.SetSelectedItemNumber(1)
        current_frame = logic.sequenceBrowserNode.GetSelectedItemNumber()
        print("SUCCESS: Back to frame {}".format(current_frame))

        # Trigger update (this should show only frame 1 lines)
        logic.syncAnnotationsToMarkups()
        logic.refreshDisplay(updateOverlay=True, updateGui=True)
        print("SUCCESS: Updated line markups")

        # Check visibility at frame 1
        visible_pleura_count = countVisiblePleuraLines(logic)
        total_pleura_nodes = len(logic.pleuraLines)

        print("Frame 1 final check:")
        print("  - Total pleura nodes: {}".format(total_pleura_nodes))
        print("  - Visible pleura lines: {}".format(visible_pleura_count))

        # Show details of each line
        for i, line in enumerate(logic.pleuraLines):
            visible = line.GetDisplayNode().GetVisibility()
            rater = line.GetAttribute("rater")
            print("  - Line {}: visible={}, rater='{}'".format(i, visible, rater))

        # The bug test: frame 1 should have exactly 1 new visible pleura line
        new_visible_pleura_count = visible_pleura_count - frame1_visible_pleura_count
        if new_visible_pleura_count != 1:
            print("\nBUG DETECTED: Should have exactly 1 new visible pleura line at frame 1, "
                  "but found {}".format(new_visible_pleura_count))
            print("This indicates the updateCurrentFrame bug is present!")
            return False

        print("SUCCESS: Frame 1 check passed - exactly 1 new visible pleura line")

        # Step 7: Modify the points of the line at frame 1 and test persistence
        print("\n--- Step 7: Modifying points of frame 1 line ---")

        # Get the frame 1 line (should be the second line we added)
        frame1_line = logic.pleuraLines[0]

        # Record original positions
        original_point1 = [100, 100, 0]
        original_point2 = [150, 1500, 0]
        frame1_line.GetNthControlPointPosition(0, original_point1)
        frame1_line.GetNthControlPointPosition(1, original_point2)
        print("Original point 1: {}".format(original_point1))
        print("Original point 2: {}".format(original_point2))

        # Modify the points
        new_point1 = [300, 200, 0]
        new_point2 = [400, 250, 0]
        frame1_line.SetNthControlPointPosition(0, new_point1)
        frame1_line.SetNthControlPointPosition(1, new_point2)
        print("Modified point 1: {}".format(new_point1))
        print("Modified point 2: {}".format(new_point2))

        # Update current frame to save the changes
        logic.syncMarkupsToAnnotations()
        logic.refreshDisplay(updateOverlay=True, updateGui=True)
        print("SUCCESS: Updated current frame after modifying points")

        # Verify the points are modified
        modified_point1 = [0, 0, 0]
        modified_point2 = [0, 0, 0]
        frame1_line.GetNthControlPointPosition(0, modified_point1)
        frame1_line.GetNthControlPointPosition(1, modified_point2)
        print("Verified modified point 1: {}".format(modified_point1))
        print("Verified modified point 2: {}".format(modified_point2))

        # Check that points were actually modified
        if (abs(modified_point1[0] - new_point1[0]) > 0.1 or
            abs(modified_point1[1] - new_point1[1]) > 0.1 or
            abs(modified_point2[0] - new_point2[0]) > 0.1 or
            abs(modified_point2[1] - new_point2[1]) > 0.1):
            print("ERROR: Points were not properly modified")
            return False

        print("SUCCESS: Points successfully modified at frame 1")

        # Step 8: Navigate to frame 0 and back to frame 1 to test persistence
        print("\n--- Step 8: Testing point persistence across frame navigation ---")

        # Go to frame 0
        logic.sequenceBrowserNode.SetSelectedItemNumber(0)
        current_frame = logic.sequenceBrowserNode.GetSelectedItemNumber()
        print("SUCCESS: Navigated to frame {}".format(current_frame))

        # Trigger update
        logic.syncAnnotationsToMarkups()
        logic.refreshDisplay(updateOverlay=True, updateGui=True)
        print("SUCCESS: Updated line markups at frame 0")

        # Go back to frame 1
        logic.sequenceBrowserNode.SetSelectedItemNumber(1)
        current_frame = logic.sequenceBrowserNode.GetSelectedItemNumber()
        print("SUCCESS: Navigated back to frame {}".format(current_frame))

        # Trigger update
        logic.syncAnnotationsToMarkups()
        logic.refreshDisplay(updateOverlay=True, updateGui=True)
        print("SUCCESS: Updated line markups at frame 1")

        # Check that the modified points are still there
        final_point1 = [0, 0, 0]
        final_point2 = [0, 0, 0]
        frame1_line.GetNthControlPointPosition(0, final_point1)
        frame1_line.GetNthControlPointPosition(1, final_point2)
        print("Final point 1: {}".format(final_point1))
        print("Final point 2: {}".format(final_point2))

        # Verify points persisted correctly
        if (abs(final_point1[0] - new_point1[0]) > 0.1 or
            abs(final_point1[1] - new_point1[1]) > 0.1 or
            abs(final_point2[0] - new_point2[0]) > 0.1 or
            abs(final_point2[1] - new_point2[1]) > 0.1):
            print("\nBUG DETECTED: Modified points did not persist after frame navigation!")
            print("Expected point 1: {}, got: {}".format(new_point1, final_point1))
            print("Expected point 2: {}, got: {}".format(new_point2, final_point2))
            print("This indicates the updateCurrentFrame bug affects point persistence!")
            return False

        print("SUCCESS: Modified points persisted correctly after frame navigation")

        print("\nSUCCESS: updateCurrentFrame bug test PASSED - markup nodes and point modifications are properly managed!")
        return True

    except Exception as e:
        print("ERROR: Test failed with exception: {}".format(e))
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        try:
            if 'slicer' in locals():
                slicer.mrmlScene.Clear(0)
                print("SUCCESS: Scene cleared")
        except:
            pass


def countVisiblePleuraLines(logic):
    """Count the number of visible pleura lines."""
    count = 0
    for line in logic.pleuraLines:
        if line.GetDisplayNode().GetVisibility():
            count += 1
    return count


def main():
    """Main entry point."""
    print("Starting updateCurrentFrame bug test...")
    success = test_updateCurrentFrame_bug()

    if success:
        print("\nALL TESTS PASSED!")
        return 0
    else:
        print("\nTEST FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())