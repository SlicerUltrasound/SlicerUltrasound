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
    resolve_output_path,
    resolve_session_directories,
    unlisted_output_dicoms,
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


def test_unlisted_output_dicoms_lists_every_dicom_file_an_empty_store_does_not_name(tmp_path):
    root = str(tmp_path)
    assert unlisted_output_dicoms(root, KeyStore()) == []
    (tmp_path / 'note.json').write_text('{}')
    assert unlisted_output_dicoms(root, KeyStore()) == []
    nested = tmp_path / 'a' / 'b'
    nested.mkdir(parents=True)
    (nested / 'x.DCM').write_bytes(b'')
    assert unlisted_output_dicoms(root, KeyStore()) == [os.path.join('a', 'b', 'x.DCM')]


def test_unlisted_output_dicoms_accepts_a_file_named_by_a_row_wherever_it_sits(scanned_manager, output_root):
    listed = scanned_manager.dicom_df.iloc[0]['AnonFilename']
    os.makedirs(os.path.join(output_root, 'sub'))
    with open(os.path.join(output_root, 'sub', listed), 'wb') as handle:
        handle.write(b'')
    assert unlisted_output_dicoms(output_root, scanned_manager.key_store) == []
    with open(os.path.join(output_root, 'other.dcm'), 'wb') as handle:
        handle.write(b'')
    assert unlisted_output_dicoms(output_root, scanned_manager.key_store) == ['other.dcm']


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


# ---------------------------------------------------------------------------
# The staged export pair
# ---------------------------------------------------------------------------

MASK_PARAMS = {'mask_type': 'rectangle', 'rectangle_left': 1, 'rectangle_right': 5,
               'rectangle_top': 1, 'rectangle_bottom': 3}


@pytest.fixture
def scanned_manager(tmp_path):
    input_dir = str(tmp_path / 'input')
    write_dataset(make_dataset(), input_dir, 'source.dcm')
    manager = DicomFileManager()
    assert manager.scan_directory(input_dir, False, key_store=KeyStore()) == 1
    manager.current_index = 0
    return manager


@pytest.fixture
def output_root(tmp_path):
    root = tmp_path / 'output'
    root.mkdir()
    return str(root)


@pytest.fixture
def image():
    return np.zeros((2, 4, 6, 1), dtype=np.uint8)


def _no_temp_files(root):
    return not any(name.count('.tmp-') for _, _, files in os.walk(root) for name in files)


def test_export_instance_writes_the_pair_and_leaves_no_temp_files(scanned_manager, output_root, image):
    row = scanned_manager.dicom_df.iloc[0]

    result = export_instance(scanned_manager, row, image, output_root=output_root, prefix='PROJ',
                             mask_params=MASK_PARAMS, labels=['Label A'])

    assert os.path.isfile(result.dicom_path) and os.path.isfile(result.json_path)
    assert os.path.basename(result.dicom_path) == row['AnonFilename']
    assert result.json_path == result.dicom_path[:-4] + '.json'
    assert _no_temp_files(output_root)
    assert sorted(os.listdir(output_root)) == sorted([row['AnonFilename'], row['AnonFilename'][:-4] + '.json'])

    ds = pydicom.dcmread(result.dicom_path)
    assert ds.SOPInstanceUID == row['AnonSOPInstanceUID']
    assert ds.PatientID == row['AnonPatientID']
    assert ds.PatientName == 'PROJ_' + row['AnonPatientID']
    assert ds.PatientIdentityRemoved == 'YES'

    with open(result.json_path, encoding='utf-8') as handle:
        sidecar = json.load(handle)
    assert sidecar['SOPInstanceUID'] == row['AnonSOPInstanceUID']
    assert sidecar['AnnotationLabels'] == ['Label A']
    assert sidecar['GrayscaleConversion'] is False
    assert isinstance(sidecar['git_sha'], str) and isinstance(sidecar['git_dirty'], bool)
    for key, value in MASK_PARAMS.items():
        assert sidecar[key] == value
    assert is_export_complete(output_root, row)


def test_export_instance_without_prefix_uses_the_pseudonym_as_name(scanned_manager, output_root, image):
    row = scanned_manager.dicom_df.iloc[0]
    result = export_instance(scanned_manager, row, image, output_root=output_root)
    ds = pydicom.dcmread(result.dicom_path, stop_before_pixels=True)
    assert ds.PatientName == row['AnonPatientID']
    with open(result.json_path, encoding='utf-8') as handle:
        assert 'AnnotationLabels' not in json.load(handle)


def test_export_instance_failure_after_dicom_staging_leaves_prior_pair_intact(scanned_manager, output_root, image):
    row = scanned_manager.dicom_df.iloc[0]
    dicom_final = os.path.join(output_root, row['AnonFilename'])
    json_final = dicom_final[:-4] + '.json'
    with open(dicom_final, 'wb') as handle:
        handle.write(b'prior-dicom')
    with open(json_final, 'w', encoding='utf-8') as handle:
        handle.write('prior-json')

    with patch.object(export.json, 'dump', side_effect=RuntimeError('boom')):
        with pytest.raises(ExportError):
            export_instance(scanned_manager, row, image, output_root=output_root)

    with open(dicom_final, 'rb') as handle:
        assert handle.read() == b'prior-dicom'
    with open(json_final, encoding='utf-8') as handle:
        assert handle.read() == 'prior-json'
    assert _no_temp_files(output_root)


def test_export_instance_json_rename_failure_leaves_no_complete_pair(scanned_manager, output_root, image):
    row = scanned_manager.dicom_df.iloc[0]
    dicom_final = os.path.join(output_root, row['AnonFilename'])
    json_final = dicom_final[:-4] + '.json'
    with open(dicom_final, 'wb') as handle:
        handle.write(b'prior-dicom')
    with open(json_final, 'w', encoding='utf-8') as handle:
        handle.write('prior-json')
    real_replace = os.replace

    def replace_unless_json(src, dst):
        if dst.endswith('.json'):
            raise OSError('sidecar is locked')
        return real_replace(src, dst)

    with patch.object(export.os, 'replace', side_effect=replace_unless_json):
        with pytest.raises(ExportError):
            export_instance(scanned_manager, row, image, output_root=output_root)

    assert not os.path.exists(dicom_final)
    assert is_export_complete(output_root, row) is False
    assert _no_temp_files(output_root)


def test_export_instance_rejects_a_staged_file_that_does_not_match_the_row(scanned_manager, output_root, image):
    row = scanned_manager.dicom_df.iloc[0].copy()
    row['AnonSOPInstanceUID'] = '2.25.1'

    with pytest.raises(ExportError):
        export_instance(scanned_manager, row, image, output_root=output_root)

    assert os.listdir(output_root) == []


def test_export_instance_raises_when_the_manager_cannot_save(scanned_manager, output_root):
    row = scanned_manager.dicom_df.iloc[0]
    with pytest.raises(ExportError):
        export_instance(scanned_manager, row, None, output_root=output_root)
    assert os.listdir(output_root) == []


def test_re_export_replaces_both_files_and_keeps_the_sop_uid(scanned_manager, output_root, image):
    row = scanned_manager.dicom_df.iloc[0]
    first = export_instance(scanned_manager, row, image, output_root=output_root, labels=['One'])
    second = export_instance(scanned_manager, row, image, output_root=output_root, labels=['Two'])

    assert first.dicom_path == second.dicom_path
    assert pydicom.dcmread(second.dicom_path, stop_before_pixels=True).SOPInstanceUID == row['AnonSOPInstanceUID']
    with open(second.json_path, encoding='utf-8') as handle:
        assert json.load(handle)['AnnotationLabels'] == ['Two']
    assert len(os.listdir(output_root)) == 2


def test_unlisted_output_dicoms_after_a_real_export(scanned_manager, output_root, image):
    row = scanned_manager.dicom_df.iloc[0]
    result = export_instance(scanned_manager, row, image, output_root=output_root)

    assert unlisted_output_dicoms(output_root, scanned_manager.key_store) == []
    assert unlisted_output_dicoms(output_root, KeyStore()) == [os.path.basename(result.dicom_path)]


def test_export_instance_preserves_the_input_tree_when_asked(tmp_path, image):
    input_dir = str(tmp_path / 'input')
    write_dataset(make_dataset(), os.path.join(input_dir, 'siteA', 'visit1'), 'source.dcm')
    manager = DicomFileManager()
    manager.scan_directory(input_dir, False, key_store=KeyStore())
    manager.current_index = 0
    row = manager.dicom_df.iloc[0]
    output_root = str(tmp_path / 'output')
    os.makedirs(output_root)

    result = export_instance(manager, row, image, output_root=output_root, preserve_directory_structure=True)

    assert result.dicom_path == os.path.join(os.path.realpath(output_root), 'siteA', 'visit1', row['AnonFilename'])
    assert is_export_complete(output_root, row, preserve_directory_structure=True)
