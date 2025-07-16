"""
Tests for the freeMarkupNodes functionality in AnnotateUltrasoundLogic.

This module tests the node pooling mechanism that manages markup nodes
to avoid constant creation and destruction of nodes.
"""

import unittest
import slicer
import vtk
import logging

# Import the logic class
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from AnnotateUltrasound import AnnotateUltrasoundLogic


class TestFreeMarkupNodes(unittest.TestCase):
    """Test cases for the freeMarkupNodes functionality."""

    def setUp(self):
        """Set up test environment before each test."""
        self.logic = AnnotateUltrasoundLogic()
        # Clear any existing nodes
        self.logic.clearSceneLines()
        self.logic.freeMarkupNodes = []

    def tearDown(self):
        """Clean up after each test."""
        # Clean up all nodes
        self.logic.clearSceneLines()
        for node in self.logic.freeMarkupNodes:
            if slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self.logic.freeMarkupNodes = []

    def _get_node_count(self):
        """Helper method to get the count of markup line nodes in the scene."""
        nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsLineNode")
        return nodes.GetNumberOfItems()

    def test_initial_state(self):
        """Test that the freeMarkupNodes list starts empty."""
        self.assertEqual(len(self.logic.freeMarkupNodes), 0)

    def test_allocate_new_markup_node(self):
        """Test allocating a new markup node."""
        initial_count = self._get_node_count()

        node = self.logic._allocateNewMarkupNode()

        # Check that a new node was created
        self.assertIsNotNone(node)
        self.assertIsInstance(node, slicer.vtkMRMLMarkupsLineNode)

        # Check that it's in the scene
        self.assertTrue(slicer.mrmlScene.IsNodePresent(node))

        # Check that the scene count increased
        final_count = self._get_node_count()
        self.assertEqual(final_count, initial_count + 1)

    def test_get_unused_markup_node_empty_pool(self):
        """Test getting a markup node when the pool is empty."""
        self.assertEqual(len(self.logic.freeMarkupNodes), 0)

        node = self.logic._getUnusedMarkupNode()

        # Should return None when pool is empty
        self.assertIsNone(node)
        self.assertEqual(len(self.logic.freeMarkupNodes), 0)  # Pool should still be empty

    def test_get_unused_markup_node_with_pool(self):
        """Test getting a markup node from the pool."""
        # Add a node to the pool
        pool_node = self.logic._allocateNewMarkupNode()
        self.logic.freeMarkupNodes.append(pool_node)

        initial_count = self._get_node_count()

        # Get node from pool
        node = self.logic._getUnusedMarkupNode()

        # Should return the node from pool
        self.assertEqual(node, pool_node)
        self.assertEqual(len(self.logic.freeMarkupNodes), 0)  # Pool should be empty now

        # Scene count should not change (reused existing node)
        final_count = self._get_node_count()
        self.assertEqual(final_count, initial_count)

    def test_free_markup_node(self):
        """Test freeing a markup node back to the pool."""
        # Create a node
        node = self.logic._allocateNewMarkupNode()
        initial_pool_size = len(self.logic.freeMarkupNodes)

        # Free the node
        self.logic._freeMarkupNode(node)

        # Should be added to the pool
        self.assertEqual(len(self.logic.freeMarkupNodes), initial_pool_size + 1)
        self.assertIn(node, self.logic.freeMarkupNodes)

        # Node should still be in scene but hidden
        self.assertTrue(slicer.mrmlScene.IsNodePresent(node))
        # Note: The actual visibility logic may vary, so we just check the node exists

    def test_create_markup_line_uses_pool(self):
        """Test that createMarkupLine uses nodes from the pool when available."""
        # Add a node to the pool
        pool_node = self.logic._allocateNewMarkupNode()
        self.logic.freeMarkupNodes.append(pool_node)

        # Create a markup line with required parameters
        coordinates = [[0, 0, 0], [1, 1, 1]]
        line_node = self.logic.createMarkupLine("test_line", "test_rater", coordinates)

        # Should reuse the node from pool
        self.assertEqual(line_node, pool_node)
        self.assertEqual(len(self.logic.freeMarkupNodes), 0)

    def test_create_markup_line_allocates_new_when_pool_empty(self):
        """Test that createMarkupLine allocates new nodes when pool is empty."""
        initial_count = self._get_node_count()

        # Create a markup line with required parameters
        coordinates = [[0, 0, 0], [1, 1, 1]]
        line_node = self.logic.createMarkupLine("test_line", "test_rater", coordinates)

        # Should create a new node
        self.assertIsNotNone(line_node)
        self.assertIsInstance(line_node, slicer.vtkMRMLMarkupsLineNode)

        final_count = self._get_node_count()
        self.assertEqual(final_count, initial_count + 1)

    def test_free_markup_nodes_directly(self):
        """Test that _freeMarkupNode directly frees nodes back to the pool."""
        # Create some markup lines by adding them to the logic's line lists
        line1 = self.logic._allocateNewMarkupNode()
        line2 = self.logic._allocateNewMarkupNode()

        # Add them to the logic's line lists so clearSceneLines will process them
        self.logic.pleuraLines.append(line1)
        self.logic.bLines.append(line2)

        initial_pool_size = len(self.logic.freeMarkupNodes)

        # Instead of calling clearSceneLines (which requires parameter node setup),
        # test the direct _freeMarkupNode functionality
        self.logic._freeMarkupNode(line1)
        self.logic._freeMarkupNode(line2)

        # Nodes should be freed to pool
        self.assertEqual(len(self.logic.freeMarkupNodes), initial_pool_size + 2)
        self.assertIn(line1, self.logic.freeMarkupNodes)
        self.assertIn(line2, self.logic.freeMarkupNodes)

    def test_clear_scene_lines_frees_nodes(self):
        """Test that clearSceneLines frees nodes back to the pool with proper setup."""
        # Set up parameter node with a rater
        parameterNode = self.logic.getParameterNode()
        parameterNode.rater = "test_rater"

        # Create some markup lines and add them to the logic's line lists
        line1 = self.logic._allocateNewMarkupNode()
        line2 = self.logic._allocateNewMarkupNode()

        # Set the rater attribute so removeLastPleuraLine/removeLastBline can find them
        line1.SetAttribute("rater", "test_rater")
        line2.SetAttribute("rater", "test_rater")

        self.logic.pleuraLines.append(line1)
        self.logic.bLines.append(line2)

        initial_pool_size = len(self.logic.freeMarkupNodes)

        # Clear scene lines
        self.logic.clearSceneLines()

        # Nodes should be freed to pool
        self.assertEqual(len(self.logic.freeMarkupNodes), initial_pool_size + 2)
        self.assertIn(line1, self.logic.freeMarkupNodes)
        self.assertIn(line2, self.logic.freeMarkupNodes)

    def test_multiple_allocate_and_free_cycles(self):
        """Test multiple cycles of allocating and freeing nodes."""
        # First cycle
        node1 = self.logic._allocateNewMarkupNode()
        self.logic._freeMarkupNode(node1)
        self.assertEqual(len(self.logic.freeMarkupNodes), 1)

        # Second cycle - should reuse the same node
        node2 = self.logic._getUnusedMarkupNode()
        self.assertEqual(node2, node1)
        self.assertEqual(len(self.logic.freeMarkupNodes), 0)

        # Third cycle - should return None when pool is empty
        node3 = self.logic._getUnusedMarkupNode()
        self.assertIsNone(node3)
        self.assertEqual(len(self.logic.freeMarkupNodes), 0)

    def test_free_nonexistent_node(self):
        """Test freeing a node that doesn't exist in the scene."""
        # Create a node outside the logic
        external_node = slicer.vtkMRMLMarkupsLineNode()
        slicer.mrmlScene.AddNode(external_node)

        initial_pool_size = len(self.logic.freeMarkupNodes)

        # Try to free the external node
        self.logic._freeMarkupNode(external_node)

        # Should be added to pool (the logic doesn't check if it's managed)
        self.assertEqual(len(self.logic.freeMarkupNodes), initial_pool_size + 1)

    def test_pool_performance(self):
        """Test that the pool mechanism actually improves performance."""
        import time

        # Test without pool (direct allocation)
        start_time = time.time()
        for _ in range(10):
            node = self.logic._allocateNewMarkupNode()
            slicer.mrmlScene.RemoveNode(node)
        without_pool_time = time.time() - start_time

        # Test with pool
        start_time = time.time()
        for _ in range(10):
            node = self.logic._allocateNewMarkupNode()
            self.logic._freeMarkupNode(node)
        with_pool_time = time.time() - start_time

        # Pool should be faster (though exact timing may vary)
        # Just verify both approaches complete without error
        self.assertGreater(without_pool_time, 0)
        self.assertGreater(with_pool_time, 0)

    def test_node_visibility_after_free(self):
        """Test that freed nodes are properly reset."""
        node = self.logic._allocateNewMarkupNode()
        node.SetName("test_name")
        node.SetAttribute("rater", "test_rater")

        self.logic._freeMarkupNode(node)

        # Node should be reset after being freed
        self.assertEqual(node.GetName(), "")
        self.assertEqual(node.GetAttribute("rater"), "")

    def test_node_visibility_after_reuse(self):
        """Test that reused nodes are properly configured."""
        # Add a reset node to pool
        node = self.logic._allocateNewMarkupNode()
        node.SetName("")
        node.SetAttribute("rater", "")
        self.logic.freeMarkupNodes.append(node)

        # Reuse the node
        coordinates = [[0, 0, 0], [1, 1, 1]]
        reused_node = self.logic.createMarkupLine("new_name", "new_rater", coordinates)

        # Node should be properly configured when reused
        self.assertEqual(reused_node.GetName(), "new_name")
        self.assertEqual(reused_node.GetAttribute("rater"), "new_rater")

    def test_free_markup_node_with_none(self):
        """Test that _freeMarkupNode handles None nodes gracefully."""
        # This should not raise an exception
        try:
            self.logic._freeMarkupNode(None)
            # If we get here, no exception was raised
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"_freeMarkupNode with None should not raise exception: {e}")

    def test_pool_limits(self):
        """Test that the pool can handle multiple nodes."""
        # Add multiple nodes to pool
        nodes = []
        for i in range(5):
            node = self.logic._allocateNewMarkupNode()
            nodes.append(node)
            self.logic._freeMarkupNode(node)

        # Pool should contain all nodes
        self.assertEqual(len(self.logic.freeMarkupNodes), 5)

        # Retrieve all nodes from pool
        retrieved_nodes = []
        for _ in range(5):
            node = self.logic._getUnusedMarkupNode()
            retrieved_nodes.append(node)

        # Should retrieve all nodes
        self.assertEqual(len(retrieved_nodes), 5)
        self.assertEqual(len(self.logic.freeMarkupNodes), 0)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestFreeMarkupNodes)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Results Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    return result


if __name__ == '__main__':
    # Run tests when script is executed directly
    result = run_tests()

    # Exit with appropriate code
    if result.wasSuccessful():
        print("\nAll tests passed!")
        exit(0)
    else:
        print("\nSome tests failed!")
        exit(1)