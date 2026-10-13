"""Tests for common/export.py: prefix rules, directory guards and the staged export pair."""
import json
import os
from unittest.mock import patch

import numpy as np
import pydicom
import pytest

from .. import export
from ..dicom_file_manager import DicomFileManager
from ..export import (
    ExportError,
    compose_patient_name,
    export_instance,
    is_export_complete,
    output_directory_has_dicom,
    resolve_output_path,
    resolve_session_directories,
    validate_prefix,
)
from ..key_store import KeyStore
from .fixtures import make_dataset, write_dataset

HEX32 = 'c' * 32


# ---------------------------------------------------------------------------
# Prefix and patient name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('text, expected', [
    ('', ''),
    ('   ', ''),
    ('BWH', 'BWH'),
    (' BWH ', 'BWH'),
    ('proj-01_x', 'proj-01_x'),
    ('a' * 16, 'a' * 16),
])
def test_validate_prefix_accepts_project_labels(text, expected):
    assert validate_prefix(text) == expected


@pytest.mark.parametrize('text', [
    'Jane_Doe_12345_xyz',
    'a' * 17,
    'a b',
    '/data/x',
    'a.b',
    'a^b',
    'ümlaut',
    'a\\b',
])
def test_validate_prefix_rejects_anything_else(text):
    with pytest.raises(ExportError):
        validate_prefix(text)


def test_compose_patient_name_is_the_pseudonym_with_an_optional_prefix():
    assert compose_patient_name('', HEX32) == HEX32
    assert compose_patient_name('PROJ', HEX32) == 'PROJ_' + HEX32


# ---------------------------------------------------------------------------
# Directory guards
# ---------------------------------------------------------------------------

def _three_dirs(tmp_path):
    dirs = [tmp_path / name for name in ('input', 'output', 'keys')]
    for d in dirs:
        d.mkdir()
    return [str(d) for d in dirs]


def test_resolve_session_directories_returns_real_paths(tmp_path):
    input_dir, output_dir, keys_dir = _three_dirs(tmp_path)
    resolved = resolve_session_directories(input_dir, output_dir, keys_dir)
    assert resolved == tuple(os.path.realpath(d) for d in (input_dir, output_dir, keys_dir))


@pytest.mark.parametrize('nested', ['keys_in_output', 'output_in_input', 'input_in_output'])
def test_resolve_session_directories_refuses_nested_directories(tmp_path, nested):
    input_dir, output_dir, keys_dir = _three_dirs(tmp_path)
    if nested == 'keys_in_output':
        keys_dir = os.path.join(output_dir, 'keys')
    elif nested == 'output_in_input':
        output_dir = os.path.join(input_dir, 'out')
    else:
        input_dir = os.path.join(output_dir, 'in')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(keys_dir, exist_ok=True)

    with pytest.raises(ExportError):
        resolve_session_directories(input_dir, output_dir, keys_dir)


@pytest.mark.parametrize('same', ['input_output', 'output_keys', 'input_keys'])
def test_resolve_session_directories_refuses_equal_directories(tmp_path, same):
    input_dir, output_dir, keys_dir = _three_dirs(tmp_path)
    if same == 'input_output':
        output_dir = input_dir
    elif same == 'output_keys':
        keys_dir = output_dir
    else:
        keys_dir = input_dir
    with pytest.raises(ExportError):
        resolve_session_directories(input_dir, output_dir, keys_dir)


def test_resolve_session_directories_refuses_missing_directories(tmp_path):
    input_dir, output_dir, keys_dir = _three_dirs(tmp_path)
    with pytest.raises(ExportError):
        resolve_session_directories(input_dir, output_dir, os.path.join(keys_dir, 'missing'))
    with pytest.raises(ExportError):
        resolve_session_directories('', output_dir, keys_dir)


def test_resolve_session_directories_sees_through_a_symlinked_keys_directory(tmp_path):
    input_dir, output_dir, keys_dir = _three_dirs(tmp_path)
    real_keys = os.path.join(output_dir, 'keys')
    os.makedirs(real_keys)
    link = str(tmp_path / 'keys_link')
    os.symlink(real_keys, link)
    with pytest.raises(ExportError):
        resolve_session_directories(input_dir, output_dir, link)


def test_output_directory_has_dicom(tmp_path):
    assert output_directory_has_dicom(str(tmp_path)) is False
    (tmp_path / 'note.json').write_text('{}')
    assert output_directory_has_dicom(str(tmp_path)) is False
    nested = tmp_path / 'a' / 'b'
    nested.mkdir(parents=True)
    (nested / 'x.DCM').write_bytes(b'')
    assert output_directory_has_dicom(str(tmp_path)) is True


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

def test_resolve_output_path_joins_under_the_root(tmp_path):
    root = str(tmp_path)
    assert resolve_output_path(root, 'a/b/x.dcm') == os.path.join(os.path.realpath(root), 'a', 'b', 'x.dcm')
    assert resolve_output_path(root, 'a/b/x.dcm', preserve_directory_structure=False) == \
        os.path.join(os.path.realpath(root), 'x.dcm')


@pytest.mark.parametrize('bad', ['../x.dcm', 'a/../../x.dcm', '/abs/x.dcm', 'a//x.dcm', './x.dcm', '', 'a\\x.dcm'])
def test_resolve_output_path_rejects_paths_that_are_not_plain_relative(tmp_path, bad):
    with pytest.raises(ExportError):
        resolve_output_path(str(tmp_path), bad)


def test_resolve_output_path_rejects_a_symlink_escaping_the_root(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    os.symlink(str(outside), str(root / 'link'))
    with pytest.raises(ExportError):
        resolve_output_path(str(root), 'link/x.dcm')


def test_is_export_complete_requires_both_files(tmp_path):
    row = {'OutputPath': 'sub/x.dcm'}
    assert is_export_complete(str(tmp_path), row) is False
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'x.dcm').write_bytes(b'')
    assert is_export_complete(str(tmp_path), row) is False
    (tmp_path / 'sub' / 'x.json').write_text('{}')
    assert is_export_complete(str(tmp_path), row) is True
    assert is_export_complete(str(tmp_path), row, preserve_directory_structure=False) is False


