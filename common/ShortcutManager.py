#!/usr/bin/env python3

import qt
import slicer
from typing import Dict, Callable, List

class ShortcutManager:
    """Manage keyboard shortcuts for ultrasound modules"""

    def __init__(self):
        self.shortcuts: Dict[str, qt.QShortcut] = {}

    def setup_shortcuts(self, shortcut_configs: Dict[str, Callable],
                       main_window=None) -> None:
        """Setup keyboard shortcuts with callbacks"""
        if main_window is None:
            main_window = slicer.util.mainWindow()

        for key, callback in shortcut_configs.items():
            if callback:
                shortcut = qt.QShortcut(main_window)
                shortcut.setKey(qt.QKeySequence(key))
                shortcut.connect('activated()', callback)
                self.shortcuts[key] = shortcut

    def disconnect_all(self) -> None:
        """Disconnect all shortcuts"""
        for shortcut in self.shortcuts.values():
            try:
                shortcut.activated.disconnect()
            except RuntimeError:
                pass  # Already disconnected
        self.shortcuts.clear()

    def is_shortcut_active(self, key: str) -> bool:
        """Check if shortcut is active"""
        return key in self.shortcuts
