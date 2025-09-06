from abc import ABC, abstractmethod
from typing import Optional

class ProgressReporter(ABC):
    """Abstract interface for progress reporting"""

    @abstractmethod
    def start(self, total: int, description: str = "Processing"):
        """Start progress tracking"""
        pass

    @abstractmethod
    def update(self, current: int, message: Optional[str] = None):
        """Update progress"""
        pass

    @abstractmethod
    def finish(self):
        """Finish progress tracking"""
        pass

class TqdmProgressReporter(ProgressReporter):
    """Progress reporter using tqdm (for CLI)"""

    def __init__(self):
        self.pbar = None

    def start(self, total: int, description: str = "Processing"):
        from tqdm import tqdm
        self.pbar = tqdm(total=total, desc=description)

    def update(self, current: int, message: Optional[str] = None):
        if self.pbar:
            self.pbar.set_postfix_str(message or "")
            self.pbar.update(1)

    def finish(self):
        if self.pbar:
            self.pbar.close()

class SlicerProgressReporter(ProgressReporter):
    """Progress reporter using Slicer's QProgressDialog"""

    def __init__(self, parent_widget):
        self.progress_dialog = None
        self.parent = parent_widget

    def start(self, total: int, description: str = "Processing"):
        import qt
        import slicer
        self.progress_dialog = qt.QProgressDialog(description, "Cancel", 0, total, slicer.util.mainWindow())
        self.progress_dialog.setWindowModality(qt.Qt.WindowModal)
        self.progress_dialog.show()

    def update(self, current: int, message: Optional[str] = None):
        if self.progress_dialog:
            self.progress_dialog.setValue(current)
            if message:
                self.progress_dialog.setLabelText(message)
            # Process events to keep UI responsive
            import slicer
            slicer.app.processEvents()

    def finish(self):
        if self.progress_dialog:
            self.progress_dialog.close()
