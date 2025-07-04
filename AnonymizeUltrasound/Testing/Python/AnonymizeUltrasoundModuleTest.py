#!/usr/bin/env python3
"""
Slicer test for AnonymizeUltrasound module.
This test runs within Slicer's Python environment.
"""

import sys
import os
import slicer
import vtk

# Add the module path to sys.path
modulePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, modulePath)

# Import the module
from AnonymizeUltrasound import AnonymizeUltrasoundLogic, AnonymizeUltrasoundWidget

# Import Slicer test base class
try:
    from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest
except ImportError:
    import unittest
    ScriptedLoadableModuleTest = unittest.TestCase


class AnonymizeUltrasoundModuleTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_AnonymizeUltrasoundLogic()
        self.test_parameter_node()

    def test_AnonymizeUltrasoundLogic(self):
        """ Test the logic class.
        """
        logic = AnonymizeUltrasoundLogic()

        # Test initialization
        self.assertIsNotNone(logic)

        # Test basic properties exist
        self.assertTrue(hasattr(logic, 'getParameterNode'))
        self.assertTrue(hasattr(logic, 'process'))

        print("✅ AnonymizeUltrasoundLogic initialization test passed")

    def test_parameter_node(self):
        """ Test parameter node creation.
        """
        logic = AnonymizeUltrasoundLogic()
        param_node = logic.getParameterNode()

        self.assertIsNotNone(param_node)

        # Test that parameter node has expected properties
        self.assertTrue(hasattr(param_node, 'inputVolume'))
        self.assertTrue(hasattr(param_node, 'outputVolume'))

        print("✅ Parameter node test passed")


def runTest():
    """
    Run the tests.
    """
    test = AnonymizeUltrasoundModuleTest()
    test.runTest()

if __name__ == '__main__':
    runTest()