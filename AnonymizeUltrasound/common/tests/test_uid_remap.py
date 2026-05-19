"""Tests for the pure remap_uid helper used by the DICOM de-id pipeline."""

import hashlib
import re

import pytest

from ..uid_remap import remap_uid


SAMPLE_SOURCE_UIDS = [
    "1.2.840.113619.2.55.3.604688432.781.1591781234.467",
    "1.2.840.113619.2.55.3.604688432.781.1591781234.468",
    "1.2.840.10008.5.1.4.1.1.6.1",
    "1.2.276.0.7230010.3.1.4.0.4242.20240101120000.1",
]


def _expected_remap(orig: str) -> str:
    digest = hashlib.sha256(orig.encode("ascii")).digest()[:15]
    return f"2.25.{int.from_bytes(digest, 'big')}"


@pytest.mark.parametrize("src", SAMPLE_SOURCE_UIDS)
def test_remap_uid_matches_specification(src):
    """remap_uid must use sha256, truncate to 15 bytes, big-endian int, '2.25.' arc."""
    assert remap_uid(src) == _expected_remap(src)


def test_remap_uid_is_deterministic():
    src = SAMPLE_SOURCE_UIDS[0]
    assert remap_uid(src) == remap_uid(src)


@pytest.mark.parametrize("src", SAMPLE_SOURCE_UIDS)
def test_remap_uid_starts_with_2_25_arc(src):
    assert remap_uid(src).startswith("2.25.")


@pytest.mark.parametrize("src", SAMPLE_SOURCE_UIDS)
def test_remap_uid_format_is_2_25_dot_digits(src):
    assert re.fullmatch(r"2\.25\.\d+", remap_uid(src)) is not None


@pytest.mark.parametrize("src", SAMPLE_SOURCE_UIDS)
def test_remap_uid_length_within_dicom_ui_limit(src):
    """DICOM UI VR limit is 64 characters."""
    assert len(remap_uid(src)) <= 64


def test_remap_uid_distinct_inputs_produce_distinct_outputs():
    outputs = {remap_uid(u) for u in SAMPLE_SOURCE_UIDS}
    assert len(outputs) == len(SAMPLE_SOURCE_UIDS)


def test_remap_uid_handles_empty_string():
    """Empty source should hash without error and stay in the 2.25 arc."""
    result = remap_uid("")
    assert result == _expected_remap("")
    assert result.startswith("2.25.")


def test_remap_uid_handles_pydicom_generated_uid_format():
    """remap_uid must accept pydicom-style UIDs (the generate_uid fallback path)."""
    pydicom_style = "1.2.826.0.1.3680043.8.498.123456789012345678901234567890"
    assert remap_uid(pydicom_style) == _expected_remap(pydicom_style)
