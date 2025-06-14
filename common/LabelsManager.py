#!/usr/bin/env python3

import os
import qt
import slicer
from typing import List, Dict, Optional, Callable
from collections import defaultdict
import csv
import logging

class LabelsManager:
    """Manage annotation labels from CSV files"""

    def __init__(self, labels_scroll_area_contents):
        self.labels_scroll_area_contents = labels_scroll_area_contents
        self.categories = defaultdict(list)

    def load_labels_from_file(self, file_path: str) -> bool:
        """Load labels from CSV file"""
        try:
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                self.categories.clear()
                for row in reader:
                    if len(row) >= 2:
                        category, label = map(str.strip, row[:2])
                        self.categories[category].append(label)
            return True
        except (FileNotFoundError, PermissionError) as e:
            logging.error(f"Cannot read labels file: {file_path}, error: {e}")
            return False

    def populate_ui(self, checkbox_callback: Optional[Callable] = None):
        """Populate UI with loaded labels"""
        # Remove existing labels
        layout = self.labels_scroll_area_contents.layout()
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()

        # Add new labels
        for category, labels in self.categories.items():
            group_box = qt.QGroupBox(category, self.labels_scroll_area_contents)
            category_layout = qt.QVBoxLayout(group_box)

            for label in labels:
                checkbox = qt.QCheckBox(label, group_box)
                if checkbox_callback:
                    checkbox.toggled.connect(checkbox_callback)
                category_layout.addWidget(checkbox)

            group_box.setLayout(category_layout)
            layout.addWidget(group_box)

        # Add stretch to push everything to top
        layout.addStretch(1)

    def get_selected_labels(self) -> List[str]:
        """Get list of selected labels"""
        selected = []
        layout = self.labels_scroll_area_contents.layout()

        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if isinstance(widget, qt.QGroupBox):
                group_title = widget.title
                group_layout = widget.layout()

                for j in range(group_layout.count()):
                    checkbox = group_layout.itemAt(j).widget()
                    if isinstance(checkbox, qt.QCheckBox) and checkbox.isChecked():
                        selected.append(f"{group_title}/{checkbox.text}")

        return selected

    def clear_selections(self):
        """Clear all checkbox selections"""
        layout = self.labels_scroll_area_contents.layout()

        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if isinstance(widget, qt.QGroupBox):
                group_layout = widget.layout()
                for j in range(group_layout.count()):
                    checkbox = group_layout.itemAt(j).widget()
                    if isinstance(checkbox, qt.QCheckBox):
                        checkbox.blockSignals(True)
                        checkbox.setChecked(False)
                        checkbox.blockSignals(False)
