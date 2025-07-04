#!/usr/bin/env python3
"""
Debug script to see what nodes Slicer creates when importing our synthetic DICOM.
"""

import sys
import os
import slicer
import vtk
import time

# Add the module path to sys.path
modulePath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(modulePath, 'AnnotateUltrasound'))

# Import the module
from AnnotateUltrasound import AnnotateUltrasoundLogic

def debug_dicom_import():
    """Debug what happens when importing our synthetic DICOM."""

    print("=== Debugging DICOM Import ===")

    # Clear scene
    slicer.mrmlScene.Clear(0)

    # Create logic
    logic = AnnotateUltrasoundLogic()

    # Set up test data
    test_data_dir = os.path.join(modulePath, 'test_data')
    logic.setRater("andrew")

    # Load DICOM directory
    print("Loading DICOM directory...")
    num_files, num_annotations = logic.updateInputDf("andrew", test_data_dir)
    print(f"Found {num_files} DICOM files")

    if logic.dicomDf is None or len(logic.dicomDf) == 0:
        print("❌ No DICOM files found!")
        return

    # Try to load the first sequence
    print("\nLoading first sequence...")
    current_index = logic.loadNextSequence()
    print(f"Load result: {current_index}")

    # Check what nodes were created
    print("\n=== Nodes in Scene ===")
    nodes = slicer.mrmlScene.GetNodes()
    for i in range(nodes.GetNumberOfItems()):
        node = nodes.GetItemAsObject(i)
        if node:
            print(f"Node {i}: {node.GetClassName()} - {node.GetName()}")

            # Check if it's a sequence browser
            if hasattr(node, 'IsA') and node.IsA("vtkMRMLSequenceBrowserNode"):
                print(f"  ✅ Found Sequence Browser Node: {node.GetName()}")
                print(f"  Master Sequence Node: {node.GetMasterSequenceNode()}")

            # Check if it's a volume node
            if hasattr(node, 'IsA') and node.IsA("vtkMRMLScalarVolumeNode"):
                print(f"  📊 Found Scalar Volume Node: {node.GetName()}")

            # Check if it's a sequence node
            if hasattr(node, 'IsA') and node.IsA("vtkMRMLSequenceNode"):
                print(f"  🔄 Found Sequence Node: {node.GetName()}")
                print(f"  Number of data nodes: {node.GetNumberOfDataNodes()}")

    # Check if we have a sequence browser
    sequence_browsers = slicer.mrmlScene.GetNodesByClass("vtkMRMLSequenceBrowserNode")
    print(f"\nNumber of sequence browser nodes: {sequence_browsers.GetNumberOfItems()}")

    if sequence_browsers.GetNumberOfItems() > 0:
        browser = sequence_browsers.GetItemAsObject(0)
        print(f"✅ Success! Found sequence browser: {browser.GetName()}")
        master_seq = browser.GetMasterSequenceNode()
        if master_seq:
            print(f"Master sequence: {master_seq.GetName()}")
            print(f"Number of frames: {master_seq.GetNumberOfDataNodes()}")
    else:
        print("❌ No sequence browser nodes found!")

        # Check what volume nodes we have
        volume_nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
        print(f"Number of scalar volume nodes: {volume_nodes.GetNumberOfItems()}")

        for i in range(volume_nodes.GetNumberOfItems()):
            vol = volume_nodes.GetItemAsObject(i)
            print(f"Volume {i}: {vol.GetName()}")

if __name__ == "__main__":
    debug_dicom_import()

    # Add a small delay to ensure all output is printed
    time.sleep(1)

    # Quit Slicer after debug completion
    print("Debug completed. Quitting Slicer...")
    slicer.util.quit()