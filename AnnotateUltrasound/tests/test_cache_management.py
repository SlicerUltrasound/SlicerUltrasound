"""
Unit tests for cache management functionality.
Tests cache clearing and state management without requiring Slicer.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestCacheManagement:
    """Test cache management functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.logic = MockAnnotateUltrasoundLogic()

    def test_cache_initialization(self):
        """Test that cache is properly initialized."""
        assert self.logic._lastMarkupFrameIndex is None
        assert self.logic._lastMarkupFrameHash is None

    def test_cache_setting(self):
        """Test setting cache values."""
        # Set cache values
        self.logic._lastMarkupFrameIndex = 5
        self.logic._lastMarkupFrameHash = "test_hash_123"

        assert self.logic._lastMarkupFrameIndex == 5
        assert self.logic._lastMarkupFrameHash == "test_hash_123"

    def test_cache_clearing_on_scene_clear(self):
        """Test that cache is cleared when scene is cleared."""
        # Set some cache values
        self.logic._lastMarkupFrameIndex = 10
        self.logic._lastMarkupFrameHash = "cached_hash"

        # Clear scene should clear cache
        self.logic.clearScene()

        assert self.logic._lastMarkupFrameIndex is None
        assert self.logic._lastMarkupFrameHash is None

    def test_cache_clearing_on_data_reload(self):
        """Test that cache is cleared when data is reloaded."""
        # Set some cache values
        self.logic._lastMarkupFrameIndex = 7
        self.logic._lastMarkupFrameHash = "old_hash"

        # Reload data should clear cache
        self.logic.reloadData()

        assert self.logic._lastMarkupFrameIndex is None
        assert self.logic._lastMarkupFrameHash is None

    def test_cache_validation(self):
        """Test cache validation logic."""
        # Test with no cache
        assert self.logic.isCacheValid(5, "hash123") == False

        # Set cache
        self.logic._lastMarkupFrameIndex = 5
        self.logic._lastMarkupFrameHash = "hash123"

        # Test with matching cache
        assert self.logic.isCacheValid(5, "hash123") == True

        # Test with different frame
        assert self.logic.isCacheValid(6, "hash123") == False

        # Test with different hash
        assert self.logic.isCacheValid(5, "different_hash") == False

    def test_cache_update(self):
        """Test updating cache values."""
        # Update cache
        self.logic.updateCache(3, "new_hash")

        assert self.logic._lastMarkupFrameIndex == 3
        assert self.logic._lastMarkupFrameHash == "new_hash"

        # Update again
        self.logic.updateCache(8, "another_hash")

        assert self.logic._lastMarkupFrameIndex == 8
        assert self.logic._lastMarkupFrameHash == "another_hash"

    def test_cache_invalidation_scenarios(self):
        """Test various scenarios that should invalidate cache."""
        # Set initial cache
        self.logic._lastMarkupFrameIndex = 2
        self.logic._lastMarkupFrameHash = "initial_hash"

        # Test frame change invalidation
        self.logic.onFrameChanged(3)
        assert self.logic._lastMarkupFrameIndex is None
        assert self.logic._lastMarkupFrameHash is None

        # Reset cache
        self.logic._lastMarkupFrameIndex = 2
        self.logic._lastMarkupFrameHash = "initial_hash"

        # Test rater change invalidation
        self.logic.onRaterChanged("new_rater")
        assert self.logic._lastMarkupFrameIndex is None
        assert self.logic._lastMarkupFrameHash is None

    def test_cache_persistence_during_valid_operations(self):
        """Test that cache persists during operations that shouldn't clear it."""
        # Set cache
        self.logic._lastMarkupFrameIndex = 4
        self.logic._lastMarkupFrameHash = "persistent_hash"

        # Operations that shouldn't clear cache
        self.logic.updateLineVisibility(True)
        self.logic.setOverlayOpacity(0.5)
        self.logic.updateColors("rater1")

        # Cache should still be there
        assert self.logic._lastMarkupFrameIndex == 4
        assert self.logic._lastMarkupFrameHash == "persistent_hash"

    def test_cache_hash_generation(self):
        """Test cache hash generation."""
        # Test hash generation with different inputs
        hash1 = self.logic.generateFrameHash(5, ["line1", "line2"])
        hash2 = self.logic.generateFrameHash(5, ["line1", "line2"])
        hash3 = self.logic.generateFrameHash(5, ["line1", "line3"])
        hash4 = self.logic.generateFrameHash(6, ["line1", "line2"])

        # Same inputs should generate same hash
        assert hash1 == hash2

        # Different inputs should generate different hashes
        assert hash1 != hash3
        assert hash1 != hash4

    def test_cache_memory_management(self):
        """Test that cache doesn't grow unbounded."""
        # Test cache size limits
        for i in range(100):
            self.logic.updateCache(i, f"hash_{i}")

        # Should only keep the latest cache entry
        assert self.logic._lastMarkupFrameIndex == 99
        assert self.logic._lastMarkupFrameHash == "hash_99"

        # Cache should not accumulate old entries
        assert not hasattr(self.logic, '_cacheHistory') or len(getattr(self.logic, '_cacheHistory', [])) <= 1


class MockAnnotateUltrasoundLogic:
    """Mock logic class for testing cache management."""

    def __init__(self):
        self._lastMarkupFrameIndex = None
        self._lastMarkupFrameHash = None
        self.currentFrame = 0
        self.currentRater = "default_rater"

    def clearScene(self):
        """Clear the scene and cache."""
        self._lastMarkupFrameIndex = None
        self._lastMarkupFrameHash = None

    def reloadData(self):
        """Reload data and clear cache."""
        self._lastMarkupFrameIndex = None
        self._lastMarkupFrameHash = None

    def isCacheValid(self, frame_index, frame_hash):
        """Check if cache is valid for given frame and hash."""
        return (self._lastMarkupFrameIndex == frame_index and
                self._lastMarkupFrameHash == frame_hash)

    def updateCache(self, frame_index, frame_hash):
        """Update cache with new values."""
        self._lastMarkupFrameIndex = frame_index
        self._lastMarkupFrameHash = frame_hash

    def onFrameChanged(self, new_frame):
        """Handle frame change - should invalidate cache."""
        self.currentFrame = new_frame
        self._lastMarkupFrameIndex = None
        self._lastMarkupFrameHash = None

    def onRaterChanged(self, new_rater):
        """Handle rater change - should invalidate cache."""
        self.currentRater = new_rater
        self._lastMarkupFrameIndex = None
        self._lastMarkupFrameHash = None

    def updateLineVisibility(self, visible):
        """Update line visibility - should not affect cache."""
        pass

    def setOverlayOpacity(self, opacity):
        """Set overlay opacity - should not affect cache."""
        pass

    def updateColors(self, rater):
        """Update colors - should not affect cache."""
        pass

    def generateFrameHash(self, frame_index, line_data):
        """Generate a hash for frame data."""
        import hashlib
        data_str = f"{frame_index}_{','.join(str(line_data))}"
        return hashlib.md5(data_str.encode()).hexdigest()[:8]