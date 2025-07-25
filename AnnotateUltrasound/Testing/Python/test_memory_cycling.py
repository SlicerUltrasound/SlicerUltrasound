#!/usr/bin/env python3
"""
Memory cycling test for AnnotateUltrasound module.
This test cycles through frames while drawing lines to monitor memory usage.
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

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    try:
        slicer.util.pip_install("psutil")
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False
        print("Warning: psutil not available. Memory monitoring will be disabled.")

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


class MemoryCyclingTest(ScriptedLoadableModuleTest):
    """
    Memory cycling test that cycles through frames while drawing lines.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear(0)

        # Initialize test parameters
        self.current_frame = 0
        self.total_frames = 0
        self.draw_cycle_count = 0
        self.play_cycle_count = 0
        self.max_cycles = 5  # Default number of cycles to run
        self.test_data_path = testDataPath
        self.dicom_file = "3038953328_70622118.dcm"
        self.annotation_file = "3038953328_70622118.json"

        # Override max_cycles if command line argument is provided
        if hasattr(self, 'args') and hasattr(self.args, 'max_cycles'):
            self.max_cycles = self.args.max_cycles

        # Memory tracking
        self.memory_log = []
        self.start_time = time.time()

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
        if self.widget:
            del self.widget
        slicer.mrmlScene.Clear(0)

    def log_memory_usage(self, action=""):
        """Log current memory usage."""
        timestamp = time.time() - self.start_time

        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                if not action.startswith("playback"):
                    self.memory_log.append({
                        'timestamp': timestamp,
                        'memory_mb': memory_mb,
                        'action': action,
                        'frame': self.current_frame,
                        'cycle': self.draw_cycle_count
                    })
                    print(f"Memory: {memory_mb:.1f}MB | Frame: {self.current_frame} | Cycle: {self.draw_cycle_count} | Action: {action}")
                else:
                    self.memory_log.append({
                        'timestamp': timestamp,
                        'memory_mb': memory_mb,
                        'action': action,
                        'cycle': self.play_cycle_count
                    })
                    print(f"Memory: {memory_mb:.1f}MB | Cycle: {self.play_cycle_count} | Action: {action}")
            except Exception as e:
                print(f"Could not log memory usage: {e}")
        else:
            # Log without memory info
            if not action.startswith("playback"):
                self.memory_log.append({
                    'timestamp': timestamp,
                    'memory_mb': 0,  # Placeholder
                    'action': action,
                    'frame': self.current_frame,
                    'cycle': self.draw_cycle_count
                })
                print(f"Frame: {self.current_frame} | Cycle: {self.draw_cycle_count} | Action: {action}")
            else:
                self.memory_log.append({
                    'timestamp': timestamp,
                    'memory_mb': 0,  # Placeholder
                    'action': action,
                    'cycle': self.play_cycle_count
                })
                print(f"Cycle: {self.play_cycle_count} | Action: {action}")

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
                self.total_frames = self.logic.sequenceBrowserNode.GetNumberOfItems()
                print(f"Loaded {self.total_frames} frames")

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

                self.log_memory_usage("data_loaded")

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

    def draw_test_lines(self):
        """Draw test lines on the current frame."""
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

            # Calculate offset based on current frame to move lines each frame
            # Move lines by a few pixels each frame, cycling back after 5 frames
            offset_x = (self.current_frame % 5) * 2.0  # Move 2 units in X every frame
            offset_y = (self.current_frame % 5) * 1.5  # Move 1.5 units in Y every frame

            # Convert LPS coordinates from JSON to RAS coordinates
            # LPS to RAS: L->R (flip X), P->A (flip Y), S->S (keep Z)

            # Use coordinates similar to the JSON annotations, converted to RAS
            # Original LPS coordinates from JSON: X: 80-150, Y: 35-85, Z: 0

            # Pleura line 1: longer line similar to annotation pattern (LPS: [85, 41, 0] to [97, 43, 0])
            pleura1_start = [-85.0 + offset_x, -41.0 + offset_y, 0.0]  # LPS to RAS: flip X and Y
            pleura1_end = [-97.0 + offset_x, -43.0 + offset_y, 0.0]

            # Pleura line 2: another longer line (LPS: [128, 44, 0] to [143, 38, 0])
            pleura2_start = [-128.0 + offset_x, -44.0 + offset_y, 0.0]  # LPS to RAS: flip X and Y
            pleura2_end = [-143.0 + offset_x, -38.0 + offset_y, 0.0]

            # B-line 1: shorter line similar to annotation pattern (LPS: [80, 82, 0] to [85, 83, 0])
            bline1_start = [-80.0 + offset_x, -82.0 + offset_y, 0.0]  # LPS to RAS: flip X and Y
            bline1_end = [-85.0 + offset_x, -83.0 + offset_y, 0.0]

            # B-line 2: another shorter line (LPS: [144, 85, 0] to [149, 84, 0])
            bline2_start = [-144.0 + offset_x, -85.0 + offset_y, 0.0]  # LPS to RAS: flip X and Y
            bline2_end = [-149.0 + offset_x, -84.0 + offset_y, 0.0]

            # Use the current rater from the logic
            parameterNode = self.logic.getParameterNode()
            rater = parameterNode.rater if parameterNode and parameterNode.rater else "test_rater"

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

            self.log_memory_usage("lines_drawn")
            return True

        except Exception as e:
            print(f"Error drawing test lines: {e}")
            return False

    def navigate_to_frame(self, frame_index):
        """Navigate to a specific frame."""
        try:
            if self.logic.sequenceBrowserNode:
                self.logic.sequenceBrowserNode.SetSelectedItemNumber(frame_index)
                self.current_frame = frame_index
                self.log_memory_usage("frame_navigated")
                return True
            return False
        except Exception as e:
            print(f"Error navigating to frame {frame_index}: {e}")
            return False

    def cycle_frame(self):
        """Cycle to the next frame and draw lines."""
        try:
            # Draw lines on current frame
            if not self.draw_test_lines():
                print("Failed to draw test lines")
                return

            # If not on first frame, go to previous frame, then back to current frame, then to next frame
            if self.current_frame > 0:
                # Go to previous frame
                prev_frame = self.current_frame - 1
                if not self.navigate_to_frame(prev_frame):
                    print("Failed to navigate to previous frame")
                    return

                # Refresh and wait a moment
                slicer.app.processEvents()
                time.sleep(0.5)

                # Go back to the frame we just drew on
                current_frame = self.current_frame + 1  # This is the frame we drew on
                if not self.navigate_to_frame(current_frame):
                    print("Failed to navigate back to current frame")
                    return

                # Refresh and wait a moment
                slicer.app.processEvents()
                time.sleep(0.5)

                # Go to next frame
                next_frame = (current_frame + 1) % self.total_frames
                if not self.navigate_to_frame(next_frame):
                    print("Failed to navigate to next frame")
                    return
            else:
                # If on first frame, just go to next frame
                next_frame = (self.current_frame + 1) % self.total_frames
                if not self.navigate_to_frame(next_frame):
                    print("Failed to navigate to next frame")
                    return

            # Increment cycle count
            self.draw_cycle_count += 1

            # Force garbage collection
            gc.collect()

            # Log progress
            print(f"Completed cycle {self.draw_cycle_count}/{self.max_cycles}")

        except Exception as e:
            print(f"Error in cycle_frame: {e}")

    def play_sequence(self):
        """Play the sequence from start to end using Slicer's playback."""
        try:
            print("Starting sequence playback using Slicer's player...")

            if not self.logic.sequenceBrowserNode:
                print("No sequence browser node available")
                return False

            # Set playback speed to 2.0 fps
            self.logic.sequenceBrowserNode.SetPlaybackRateFps(2.0)

            # Start playback
            self.logic.sequenceBrowserNode.SetPlaybackActive(True)
            self.log_memory_usage("playback_started")

            # Wait for playback to complete by monitoring the sequence browser
            print(f"Playing {self.total_frames} frames...")

            # Wait until playback is no longer active
            while self.logic.sequenceBrowserNode.GetPlaybackActive():
                time.sleep(0.1)  # Small check interval
                slicer.app.processEvents()  # Keep UI responsive

            # Stop playback (in case it's still running)
            self.logic.sequenceBrowserNode.SetPlaybackActive(False)
            self.log_memory_usage("playback_completed")
            self.play_cycle_count += 1
            print(f"Completed playback of {self.total_frames} frames")
            return True

        except Exception as e:
            print(f"Error during sequence playback: {e}")
            return False

    def runTest(self):
        """Run the memory cycling test."""
        print("Starting memory cycling test...")
        print("This test will:")
        print("1. Load the AnnotateUltrasound module")
        print("2. Load test DICOM data")
        print("3. Cycle through frames while drawing lines")
        print("4. Play the sequence from start to end multiple times")
        print("5. Monitor memory usage throughout the process")
        print("6. Save memory logs to a JSON file")

        # Load test data
        if not self.load_test_data():
            print("Failed to load test data")
            return

        # Run drawing cycles
        print(f"\n=== PHASE 1: Drawing Cycles ({self.max_cycles} cycles) ===")
        for cycle in range(self.max_cycles):
            self.cycle_frame()
            time.sleep(1)  # Small delay between cycles

        # Run playback cycles
        print(f"\n=== PHASE 2: Playback Cycles ({self.max_cycles} cycles) ===")
        for cycle in range(self.max_cycles):
            print(f"Playback cycle {cycle + 1}/{self.max_cycles}")
            if not self.play_sequence():
                print("Failed to complete playback cycle")
                break
            time.sleep(0.5)  # Small delay between playback cycles

        # Save memory log
        log_file = os.path.join(tempfile.gettempdir(), f"memory_cycling_test_{int(time.time())}.json")
        with open(log_file, 'w') as f:
            json.dump(self.memory_log, f, indent=2)

        print(f"\nMemory cycling test completed. Log saved to: {log_file}")
        print(f"Total drawing cycles completed: {self.draw_cycle_count}")
        print(f"Total playback cycles completed: {self.play_cycle_count}")

        # Print summary
        if PSUTIL_AVAILABLE and self.memory_log:
            initial_memory = self.memory_log[0]['memory_mb']
            final_memory = self.memory_log[-1]['memory_mb']
            memory_change = final_memory - initial_memory
            print(f"Memory change: {memory_change:+.1f}MB ({initial_memory:.1f}MB -> {final_memory:.1f}MB)")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Memory cycling test for AnnotateUltrasound')
    parser.add_argument('--max-cycles', type=int, default=5,
                       help='Maximum number of cycles to run (default: 5)')
    return parser.parse_args()

def runMemoryCyclingTest():
    """Run the memory cycling test."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        test = MemoryCyclingTest()
        # Store args in the test instance so setUp can access it
        test.args = args
        test.setUp()  # Explicitly call setUp
        test.runTest()
        test.tearDown()  # Explicitly call tearDown
        return True
    except Exception as e:
        print(f"Error running memory cycling test: {e}")
        return False


if __name__ == '__main__':
    runMemoryCyclingTest()
    slicer.util.exit(0)
