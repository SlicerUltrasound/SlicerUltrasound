#!/usr/bin/env python3
"""
GUI test for AnnotateUltrasound module using Slicer's GUI test harness.
This test simulates user interactions and verifies UI behavior.
"""

import sys
import os
import slicer
import vtk
import time


class AnnotateUltrasoundWidgetTest:
    """
    GUI test for AnnotateUltrasound module.
    This test simulates user interactions and verifies UI behavior.
    """

    def __init__(self):
        self.widget = None
        self.logic = None

    def setUp(self):
        """Set up the test environment."""
        slicer.mrmlScene.Clear(0)

        # Check if the module was installed
        print("Checking if AnnotateUltrasound module is available...")
        try:
            import AnnotateUltrasound
            print(f"✓ AnnotateUltrasound module imported successfully from: {AnnotateUltrasound.__file__}")
        except ImportError as e:
            print(f"❌ AnnotateUltrasound module not found: {e}")
            print("Available modules in slicer.modules:")
            for module_name in dir(slicer.modules):
                if not module_name.startswith('_'):
                    print(f"  - {module_name}")
            raise RuntimeError("AnnotateUltrasound module not installed or not accessible")

        print("Available modules in slicer.modules:")
        for module_name in dir(slicer.modules):
            if not module_name.startswith('_'):
                print(f"  - {module_name}")

        print("Trying to select AnnotateUltrasound module...")
        slicer.util.selectModule('AnnotateUltrasound')
        slicer.util.delayDisplay("Selected AnnotateUltrasound", 1000)

        # Check if module is now available
        print("After selectModule, available modules:")
        for module_name in dir(slicer.modules):
            if not module_name.startswith('_'):
                print(f"  - {module_name}")

        # Try to get the widget through Slicer's module system
        try:
            self.widget = AnnotateUltrasound.getAnnotateUltrasoundWidget()
            print("✓ Got widget via getAnnotateUltrasoundWidget()")
        except Exception as e:
            print(f"Failed to get widget via getAnnotateUltrasoundWidget(): {e}")

            # Fallback: Try to create widget directly
            try:
                print("Trying to create widget directly...")
                from AnnotateUltrasound import AnnotateUltrasoundWidget
                self.widget = AnnotateUltrasoundWidget()
                print("✓ Created widget directly")
            except Exception as e2:
                print(f"Failed to create widget directly: {e2}")

                # Last resort: Try to access via slicer.modules
                try:
                    print("Trying to access via slicer.modules...")
                    if hasattr(slicer.modules, 'annotateultrasound'):
                        moduleWidget = slicer.modules.annotateultrasound.widgetRepresentation()
                        if moduleWidget:
                            self.widget = moduleWidget.self()
                            print("✓ Got widget via slicer.modules")
                        else:
                            print("No widget representation found in slicer.modules")
                    else:
                        print("annotateultrasound not found in slicer.modules")
                except Exception as e3:
                    print(f"Failed to access via slicer.modules: {e3}")

        if not self.widget:
            raise RuntimeError("Failed to get AnnotateUltrasound widget instance")

        self.logic = self.widget.logic
        time.sleep(1)
        self.loadTestInput()

    def tearDown(self):
        """Clean up after tests."""
        if self.widget:
            del self.widget
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run all GUI tests."""
        print("=== Running AnnotateUltrasound WidgetTests ===")

        results = []

        self.setUp()

        try:
            # Run each test and collect results
            test_methods = [method for method in dir(self) if method.startswith('test_')]

            for test_method in test_methods:
                print(f"\n--- Running {test_method} ---")
                try:
                    getattr(self, test_method)()
                    results.append((test_method, True, None))
                    print(f"✅ {test_method} passed")
                except Exception as e:
                    results.append((test_method, False, str(e)))
                    print(f"❌ {test_method} failed: {e}")
                    import traceback
                    traceback.print_exc()

            # Print summary
            print(f"\n=== Test Summary ===")
            print(f"Tests run: {len(results)}")
            passed = sum(1 for _, success, _ in results if success)
            failed = len(results) - passed
            print(f"Passed: {passed}")
            print(f"Failed: {failed}")

            if failed > 0:
                print(f"\n=== Failed Tests ===")
                for test_name, success, error in results:
                    if not success:
                        print(f"FAIL: {test_name} - {error}")

            if failed == 0:
                print("✅ All GUI tests passed!")
            else:
                print(f"❌ {failed} test(s) failed")

        except Exception as e:
            print(f"❌ Test setup failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self.tearDown()
            print("Tests complete.")

        return failed == 0

    def test_minimal_rater_persistence(self):
        """Minimal test: set rater name, press Enter, click addPleuraButton, print state before and after."""
        print("Testing minimal rater persistence...")

        # Ensure parameter node is set
        if self.widget._parameterNode is None and self.logic:
            self.widget.setParameterNode(self.logic.getParameterNode())

        self.widget.ui.raterName.setText("minimal_rater")
        print(f"After setText rater name: '{self.widget.ui.raterName.text}'")

        self.widget.ui.raterName.returnPressed.emit()
        print(f"After returnPressed parameter node rater: '{self.widget._parameterNode.rater if self.widget._parameterNode else 'None'}'")
        time.sleep(0.5)
        print(f"Before addPleuraButton click: UI text='{self.widget.ui.raterName.text}', parameter node rater='{self.widget._parameterNode.rater if self.widget._parameterNode else 'None'}'")
        self.widget.ui.addPleuraButton.click()
        time.sleep(0.1)
        print(f"After addPleuraButton click: UI text='{self.widget.ui.raterName.text}', parameter node rater='{self.widget._parameterNode.rater if self.widget._parameterNode else 'None'}'")
        print("✅ Minimal rater persistence test complete")

    def test_widget_creation(self):
        """Test that the widget can be created and initialized."""
        print("Testing widget creation...")

        # Verify widget was created
        self.assertIsNotNone(self.widget)
        self.assertIsNotNone(self.logic)

        # Debug: Print UI attributes to see what's available
        print("UI object:", self.widget.ui)
        print("UI type:", type(self.widget.ui))
        print("UI attributes:", dir(self.widget.ui))

        # Verify UI elements exist
        self.assertIsNotNone(self.widget.ui)
        self.assertIsNotNone(self.widget.ui.addPleuraButton)
        self.assertIsNotNone(self.widget.ui.addBlineButton)
        self.assertIsNotNone(self.widget.ui.raterColorTable)

        print("✅ Widget creation test passed")

    def test_button_interactions(self):
        """Test button clicks and state changes."""
        print("Testing button interactions...")

        # Debug: Check initial state
        print(f"Initial rater name: '{self.widget.ui.raterName.text}'")
        print(f"Initial parameter node rater: '{self.widget._parameterNode.rater}'")

        # Set a rater name before annotation actions
        self.widget.ui.raterName.setText("test_rater")
        print(f"After setText rater name: '{self.widget.ui.raterName.text}'")

        # Simulate pressing Enter to trigger returnPressed signal
        self.widget.ui.raterName.returnPressed.emit()
        print(f"After returnPressed parameter node rater: '{self.widget._parameterNode.rater}'")

        # Wait for logic to update
        time.sleep(0.5)  # Increased delay

        # Test Add Pleura button
        self.widget.onRaterNameChanged()  # Ensure rater is synced before clicking
        print(f"Before addPleuraButton click: UI text='{self.widget.ui.raterName.text}', parameter node rater='{self.widget._parameterNode.rater}'")
        initial_state = self.widget.ui.addPleuraButton.isChecked()
        self.widget.ui.addPleuraButton.click()
        time.sleep(0.1)  # Allow UI to update

        # Button should change state
        new_state = self.widget.ui.addPleuraButton.isChecked()
        self.assertNotEqual(initial_state, new_state)
        self.widget._parameterNode.lineBeingPlaced = None

        # Test Add B-line button
        self.widget.onRaterNameChanged()  # Ensure rater is synced before clicking
        print(f"Before addBlineButton click: UI text='{self.widget.ui.raterName.text}', parameter node rater='{self.widget._parameterNode.rater}'")
        initial_state = self.widget.ui.addBlineButton.isChecked()
        self.widget.ui.addBlineButton.click()
        time.sleep(0.1)

        new_state = self.widget.ui.addBlineButton.isChecked()
        self.assertNotEqual(initial_state, new_state)
        self.widget._parameterNode.lineBeingPlaced = None

        print("✅ Button interactions test passed")

    def test_rater_table_interactions(self):
        """Test rater table interactions."""
        print("Testing rater table interactions...")

        # Verify table exists
        table = self.widget.ui.raterColorTable
        self.assertIsNotNone(table)

        # Test adding a rater (if there's an add rater button)
        if hasattr(self.widget.ui, 'addRaterButton'):
            initial_row_count = table.rowCount()
            self.widget.ui.addRaterButton.click()
            time.sleep(0.1)

            new_row_count = table.rowCount()
            self.assertGreater(new_row_count, initial_row_count)

        print("✅ Rater table interactions test passed")

    def test_keyboard_shortcuts(self):
        """Test keyboard shortcuts work properly and have the intended effect."""
        print("Testing keyboard shortcuts...")

        widget = self.widget
        print("✅ Using module-provided AnnotateUltrasound widget")

        # Assert parameter node is not None after Read Input
        assert widget._parameterNode is not None, "Parameter node should not be None after Read Input"

        # Check if the module processed the input
        if widget._parameterNode is not None:
            print(f"Parameter node dfLoaded: {widget._parameterNode.dfLoaded}")
            print(f"Logic dicomDf is None: {self.logic.dicomDf is None}")
            if self.logic.dicomDf is not None:
                print(f"Logic dicomDf shape: {self.logic.dicomDf.shape}")
        else:
            error_msg = "❌ Parameter node is still None after Read Input click"
            print(error_msg)
            raise AssertionError(error_msg)

        # Now continue with annotation shortcut and drawing simulation

        # Ensure module is active
        slicer.util.selectModule('AnnotateUltrasound')
        slicer.util.delayDisplay("AnnotateUltrasound UI loaded", 1000)

        # Check if shortcuts are enabled
        print(f"Shortcut W enabled: {widget.shortcutW.isEnabled()}")
        print(f"Shortcut S enabled: {widget.shortcutS.isEnabled()}")
        print(f"Shortcut Space enabled: {widget.shortcutSpace.isEnabled()}")

        # Check rater name - required for onAddLine to work
        print(f"Rater name before test: '{widget._parameterNode.rater}'")
        print(f"UI rater name text: '{widget.ui.raterName.text}'")
        if not widget._parameterNode.rater:
            print("Setting rater name for testing...")
            widget.ui.raterName.setText("test_rater")
            widget.onRaterNameChanged()
            print(f"Rater name after setting: '{widget._parameterNode.rater}'")

        # Ensure main window has focus
        mainWindow = slicer.util.mainWindow()
        mainWindow.activateWindow()
        mainWindow.raise_()

        # Add a delay to ensure UI is ready
        slicer.util.delayDisplay("About to test shortcuts", 1000)

        # Helper function to send key events to main window
        def sendKeyToMainWindow(key, modifiers=qt.Qt.NoModifier):
            mw = slicer.util.mainWindow()
            eventPress = qt.QKeyEvent(qt.QEvent.KeyPress, key, modifiers)
            eventRelease = qt.QKeyEvent(qt.QEvent.KeyRelease, key, modifiers)
            qt.QApplication.postEvent(mw, eventPress)
            qt.QApplication.postEvent(mw, eventRelease)
            # Process events to ensure they're handled
            qt.QApplication.processEvents()

        # Helper function to simulate mouse clicks in the 3D view
        def simulateLineDrawing():
            """Simulate drawing a line by clicking at two points in the 3D view."""
            # Get the 3D view widget
            viewWidget = slicer.app.layoutManager().threeDWidget(0).threeDView()
            print(f"viewWidget: {viewWidget}, type: {type(viewWidget)}")

            # Get view dimensions
            print(f"viewWidget.size: {viewWidget.size}, type: {type(viewWidget.size)}")
            viewSize = viewWidget.size
            print(f"viewSize: {viewSize}, type: {type(viewSize)}")
            centerX = viewSize.width() // 2
            centerY = viewSize.height() // 2

            # First click - start point
            print("Simulating first click (start point)...")
            pressEvent1 = qt.QMouseEvent(qt.QEvent.MouseButtonPress,
                                       qt.QPointF(centerX, centerY),
                                       qt.Qt.LeftButton, qt.Qt.LeftButton, qt.Qt.NoModifier)
            releaseEvent1 = qt.QMouseEvent(qt.QEvent.MouseButtonRelease,
                                         qt.QPointF(centerX, centerY),
                                         qt.Qt.LeftButton, qt.Qt.LeftButton, qt.Qt.NoModifier)
            qt.QApplication.postEvent(viewWidget, pressEvent1)
            qt.QApplication.postEvent(viewWidget, releaseEvent1)

            slicer.util.delayDisplay("First click sent", 500)

            # Second click - end point
            print("Simulating second click (end point)...")
            pressEvent2 = qt.QMouseEvent(qt.QEvent.MouseButtonPress,
                                       qt.QPointF(centerX + 100, centerY + 100),
                                       qt.Qt.LeftButton, qt.Qt.LeftButton, qt.Qt.NoModifier)
            releaseEvent2 = qt.QMouseEvent(qt.QEvent.MouseButtonRelease,
                                         qt.QPointF(centerX + 100, centerY + 100),
                                         qt.Qt.LeftButton, qt.Qt.LeftButton, qt.Qt.NoModifier)
            qt.QApplication.postEvent(viewWidget, pressEvent2)
            qt.QApplication.postEvent(viewWidget, releaseEvent2)

            slicer.util.delayDisplay("Second click sent", 500)

        # Test W key (toggle pleura annotation mode)
        widget.ui.addPleuraButton.setFocus()
        pleura_button_before = widget.ui.addPleuraButton.isChecked()
        print(f"Pleura button before W: {pleura_button_before}")
        sendKeyToMainWindow(qt.Qt.Key_W)
        slicer.util.delayDisplay("W key sent", 500)
        pleura_button_after = widget.ui.addPleuraButton.isChecked()
        print(f"Pleura button after W: {pleura_button_after}")
        if pleura_button_after != pleura_button_before:
            print("✅ W key toggled pleura annotation mode")
        else:
            error_msg = "❌ W key did not toggle pleura annotation mode"
            print(error_msg)
            raise AssertionError(error_msg)
        self.widget._parameterNode.lineBeingPlaced = None

        # Test S key (toggle B-line annotation mode)
        widget.ui.addBlineButton.setFocus()
        bline_button_before = widget.ui.addBlineButton.isChecked()
        print(f"B-line button before S: {bline_button_before}")
        sendKeyToMainWindow(qt.Qt.Key_S)
        slicer.util.delayDisplay("S key sent", 500)
        bline_button_after = widget.ui.addBlineButton.isChecked()
        print(f"B-line button after S: {bline_button_after}")
        if bline_button_after != bline_button_before:
            print("✅ S key toggled B-line annotation mode")
        else:
            error_msg = "❌ S key did not toggle B-line annotation mode"
            print(error_msg)
            raise AssertionError(error_msg)
        self.widget._parameterNode.lineBeingPlaced = None

        # Test O key (toggle overlay)
        widget.ui.overlayVisibilityButton.setFocus()
        overlay_before = widget.ui.overlayVisibilityButton.isChecked()
        print(f"Overlay visibility before Space: {overlay_before}")
        sendKeyToMainWindow(qt.Qt.Key_Space)
        slicer.util.delayDisplay("Space key sent", 500)
        overlay_after = widget.ui.overlayVisibilityButton.isChecked()
        print(f"Overlay visibility after Space: {overlay_after}")
        if overlay_after != overlay_before:
            print("✅ Space key toggled overlay visibility")
        else:
            error_msg = "❌ Space key did not toggle overlay visibility"
            print(error_msg)
            raise AssertionError(error_msg)

        # Test complete workflow: W key + mouse interaction to draw pleura line
        print("\nTesting complete pleura line drawing workflow...")
        pleura_count_before = len(self.logic.pleuraLines)
        print(f"Pleura lines before drawing: {pleura_count_before}")

        # Activate pleura mode with W key
        sendKeyToMainWindow(qt.Qt.Key_W)
        slicer.util.delayDisplay("W key sent for drawing", 500)

        # Simulate drawing a line
        simulateLineDrawing()
        slicer.util.delayDisplay("Mouse interaction completed", 1000)

        pleura_count_after = len(self.logic.pleuraLines)
        print(f"Pleura lines after drawing: {pleura_count_after}")
        if pleura_count_after > pleura_count_before:
            print("✅ Complete pleura line drawing workflow worked")
        else:
            error_msg = "❌ Complete pleura line drawing workflow failed"
            print(error_msg)
            raise AssertionError(error_msg)

        print("Keyboard shortcut test completed")

    def test_line_creation_workflow(self):
        """Test the complete line creation workflow."""
        print("Testing line creation workflow...")

        # Set up test data
        test_raters = {"rater1", "rater2"}
        self.logic.setSelectedRaters(test_raters)

        # Verify raters are set
        selected = self.logic.getSelectedRaters()
        self.assertEqual(selected, test_raters)

        # Test creating a pleura line
        coordinates = [[0, 0, 0], [1, 1, 1]]
        line = self.logic.createMarkupLine("Pleura", "test_rater", coordinates)
        self.assertIsNotNone(line)

        # Test creating a B-line
        line = self.logic.createMarkupLine("B-line", "test_rater", coordinates)
        self.assertIsNotNone(line)

        print("✅ Line creation workflow test passed")

    # Helper assertion methods
    def assertIsNotNone(self, obj):
        if obj is None:
            raise AssertionError(f"Expected not None, but got None")

    def assertIsNone(self, obj):
        if obj is not None:
            raise AssertionError(f"Expected None, but got {obj}")

    def assertEqual(self, a, b):
        if a != b:
            raise AssertionError(f"Expected {a}, but got {b}")

    def assertNotEqual(self, a, b):
        if a == b:
            raise AssertionError(f"Expected not {a}, but got {b}")

    def assertGreater(self, a, b):
        if not a > b:
            raise AssertionError(f"Expected {a} > {b}")

    def assertGreaterEqual(self, a, b):
        if not a >= b:
            raise AssertionError(f"Expected {a} >= {b}")

    def loadTestInput(self):
        """Set test input directory and trigger Read Input."""
        test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        self.widget.ui.inputDirectoryButton.directory = test_data_dir
        print(f"Set input directory to: {test_data_dir}")
        slicer.util.delayDisplay("Input directory set", 500)
        self.widget.onReadInputButton()
        slicer.util.delayDisplay("Read Input triggered", 2000)

def runGUITest():
    """Run the GUI test."""
    test = AnnotateUltrasoundWidgetTest()
    success = test.runTest()

    # Print final result
    if success:
        print("✅ All GUI tests completed successfully!")
    else:
        print("❌ Some GUI tests failed!")

    # Exit Slicer after test completion
    print("Tests complete. Exiting Slicer...")
    slicer.util.quit()

    return success

if __name__ == '__main__':
    # Run the test directly
    runGUITest()
else:
    # Defer test execution with QTimer.singleShot to ensure the Slicer GUI is fully initialized
    qt.QTimer.singleShot(0, runGUITest)
