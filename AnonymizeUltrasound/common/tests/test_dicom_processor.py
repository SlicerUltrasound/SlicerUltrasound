"""Tests for DicomProcessor — focused on sidecar correctness invariants.

The full processor exercises model inference and masking; those paths are
covered by integration/E2E. Here we test the small, deterministic helpers
that emit sidecar files, where source-vs-anonymized UID confusion can leak
PHI.

`dicom_processor` imports torch and matplotlib at module load time for the
inference / PDF-overview paths. The sidecar helper under test does not
touch either; we stub them via sys.modules so the test can run in a slim
venv without torch installed.
"""
import json
import os
import sys
import tempfile
import shutil
import types

import pytest


class _PermissiveStubModule(types.ModuleType):
    """Module that returns a dummy object for every attribute access.

    dicom_processor (and its imports) reference symbols from torch / cv2 /
    monai / matplotlib at module load time (type annotations, constants).
    We don't execute any of those code paths in the sidecar tests, so a
    permissive stub keeps imports cheap without faking real APIs.
    """

    def __getattr__(self, attr):  # noqa: D401
        return type(attr, (), {})


def _stub_module(name: str) -> types.ModuleType:
    """Install a permissive stub under `name` so `import name` succeeds."""
    if name in sys.modules:
        return sys.modules[name]
    mod = _PermissiveStubModule(name)
    sys.modules[name] = mod
    return mod


for _name in (
    "torch",
    "cv2",
    "requests",
    "monai",
    "monai.metrics",
    "monai.metrics.meandice",
    "monai.metrics.meaniou",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.patches",
    "matplotlib.backends",
    "matplotlib.backends.backend_pdf",
):
    _stub_module(_name)

# evaluation.py uses non-relative `from common.masking import create_mask`,
# which only resolves when AnonymizeUltrasound/ is on sys.path.
_anon_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _anon_dir not in sys.path:
    sys.path.insert(0, _anon_dir)

from ..dicom_processor import DicomProcessor, ProcessingConfig  # noqa: E402
from ..dicom_file_manager import DicomFileManager  # noqa: E402


class _DataframeRow:
    """Lightweight stand-in for a pandas dataframe row.

    `_save_sequence_info` uses both `row.DICOMDataset.SOPInstanceUID` (the
    bug) and `row.AnonSOPInstanceUID` (the fix). A simple attribute-bag is
    sufficient — no pandas required.
    """

    def __init__(self, *, anon_sop_uid: str, source_sop_uid: str):
        self.AnonSOPInstanceUID = anon_sop_uid

        class _DS:
            pass

        ds = _DS()
        ds.SOPInstanceUID = source_sop_uid
        self.DICOMDataset = ds


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def processor():
    """Minimal processor wired up enough to call _save_sequence_info.

    No model is loaded; the helper under test does not touch the model.
    """
    config = ProcessingConfig(
        model_path="",
        device="cpu",
        no_mask_generation=False,
        phi_only_mode=False,
    )
    return DicomProcessor(config, DicomFileManager())


class TestSaveSequenceInfoNoSopUidLeak:
    """F2 — _save_sequence_info must write the anonymized SOPInstanceUID.

    Currently it writes `row.DICOMDataset.SOPInstanceUID` (the source UID),
    creating a sidecar leak next to the anonymized .dcm.
    """

    SOURCE_SOP_UID = "1.2.840.SOURCE.SOP.LEAK"
    ANON_SOP_UID = "2.25.ANON.SOP.SAFE"

    def test_sequence_info_uses_anon_sop_uid_not_source(self, processor, temp_dir):
        """The JSON sidecar's SOPInstanceUID must equal row.AnonSOPInstanceUID,
        NOT row.DICOMDataset.SOPInstanceUID."""
        row = _DataframeRow(
            anon_sop_uid=self.ANON_SOP_UID,
            source_sop_uid=self.SOURCE_SOP_UID,
        )
        final_output_path = os.path.join(temp_dir, "out.dcm")

        processor._save_sequence_info(final_output_path, row, mask_config=None)

        json_path = final_output_path.replace(".dcm", ".json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            payload = json.load(f)

        assert payload["SOPInstanceUID"] == self.ANON_SOP_UID
        assert payload["SOPInstanceUID"] != self.SOURCE_SOP_UID

    def test_sequence_info_no_source_uid_in_payload(self, processor, temp_dir):
        """Belt-and-suspenders: the source UID string must not appear ANYWHERE
        in the serialized JSON."""
        row = _DataframeRow(
            anon_sop_uid=self.ANON_SOP_UID,
            source_sop_uid=self.SOURCE_SOP_UID,
        )
        final_output_path = os.path.join(temp_dir, "out.dcm")

        processor._save_sequence_info(final_output_path, row, mask_config=None)

        json_path = final_output_path.replace(".dcm", ".json")
        with open(json_path) as f:
            serialized = f.read()

        assert self.SOURCE_SOP_UID not in serialized
        assert self.ANON_SOP_UID in serialized

    def test_sequence_info_falls_back_to_none_when_anon_sop_uid_missing(
        self, processor, temp_dir
    ):
        """If AnonSOPInstanceUID is empty (degenerate input), the sidecar must
        write 'None' — never the source UID."""
        row = _DataframeRow(
            anon_sop_uid="",
            source_sop_uid=self.SOURCE_SOP_UID,
        )
        final_output_path = os.path.join(temp_dir, "out.dcm")

        processor._save_sequence_info(final_output_path, row, mask_config=None)

        json_path = final_output_path.replace(".dcm", ".json")
        with open(json_path) as f:
            payload = json.load(f)

        assert payload["SOPInstanceUID"] == "None"
        with open(json_path) as f:
            serialized = f.read()
        assert self.SOURCE_SOP_UID not in serialized


if __name__ == "__main__":
    pytest.main([__file__])
