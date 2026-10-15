"""Tests for common/export.py: prefix rules, directory guards and the staged export pair."""
import json
import os
from unittest.mock import patch

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian

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
from .fixtures import SOURCE_UID_ROOT, US_MULTIFRAME_STORAGE, make_dataset, write_dataset

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
    assert 'git_sha' not in sidecar and 'git_dirty' not in sidecar
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


@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('transfer_syntax', [ExplicitVRLittleEndian, ImplicitVRLittleEndian])
def test_export_instance_preserves_complete_ultrasound_regions(
        tmp_path, output_root, image, transfer_syntax):
    region = Dataset()
    region.RegionSpatialFormat = 1
    region.PhysicalDeltaX = 0.1
    region.PhysicalDeltaY = 0.15
    region.RWaveTimeVector = [0.0, 100.0]  # Standard field omitted by the former allowlist.
    region.PatientName = '患者^領域'
    region.private_block(0x0011, 'REGION_PRIVATE', create=True).add_new(0x01, 'LO', '領域の識別子')
    reference = Dataset()
    reference.ReferencedSOPClassUID = US_MULTIFRAME_STORAGE
    reference.ReferencedSOPInstanceUID = SOURCE_UID_ROOT + '.77'
    reference.PatientName = '患者^参照'
    region.ReferencedImageSequence = Sequence([reference])

    second = Dataset()
    second.SpecificCharacterSet = 'ISO_IR 100'
    second.PatientName = 'Région^Deux'
    second.RegionSpatialFormat = 2
    source = make_dataset(SpecificCharacterSet='ISO_IR 192',
                          regions=[region, second, Dataset()])
    source.is_implicit_VR = transfer_syntax == ImplicitVRLittleEndian
    source.file_meta.TransferSyntaxUID = transfer_syntax
    input_dir = str(tmp_path / 'input')
    source_path = write_dataset(source, input_dir, 'source.dcm')
    parsed_source = pydicom.dcmread(source_path)
    original = parsed_source.SequenceOfUltrasoundRegions

    # Verify the fixture's own encoding before using it as the preservation oracle.
    assert original[0].PatientName == '患者^領域'
    assert original[0].ReferencedImageSequence[0].PatientName == '患者^参照'
    assert original[1].PatientName == 'Région^Deux'
    private_value = original[0].private_block(0x0011, 'REGION_PRIVATE')[0x01].value
    if isinstance(private_value, bytes):
        # Unknown private VRs are parsed as bytes from implicit-VR input.
        private_value = private_value.decode('utf-8').rstrip(' \0')
    assert private_value == '領域の識別子'

    manager = DicomFileManager()
    assert manager.scan_directory(input_dir, False, key_store=KeyStore()) == 1
    manager.current_index = 0
    result = export_instance(manager, manager.dicom_df.iloc[0], image, output_root=output_root)
    saved = pydicom.dcmread(result.dicom_path)

    # Compare parsed contents, including implicit-VR private bytes, not serialization.
    assert saved.SequenceOfUltrasoundRegions == original
    assert saved.SpecificCharacterSet == 'ISO_IR 192'
    assert saved.SequenceOfUltrasoundRegions[1].SpecificCharacterSet == 'ISO_IR 100'
    assert 'SpecificCharacterSet' not in saved.SequenceOfUltrasoundRegions[0]
    assert 'SpecificCharacterSet' not in saved.SequenceOfUltrasoundRegions[0].ReferencedImageSequence[0]
    assert saved.PixelSpacing == [1.0, 1.5]


@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('regions', ['missing', 'empty', 'present'])
@pytest.mark.parametrize('character_set', [None, '', 'ISO_IR 192'])
def test_export_instance_preserves_character_set_with_or_without_regions(
        tmp_path, output_root, image, regions, character_set):
    manufacturer = '超音波装置' if character_set == 'ISO_IR 192' else 'Ultrasound device'
    source = make_dataset(SpecificCharacterSet=character_set, Manufacturer=manufacturer)
    if regions == 'missing':
        del source.SequenceOfUltrasoundRegions
    elif regions == 'empty':
        source.SequenceOfUltrasoundRegions = Sequence([])
    input_dir = str(tmp_path / 'input')
    source_path = write_dataset(source, input_dir, 'source.dcm')
    parsed_source = pydicom.dcmread(source_path)
    assert parsed_source.Manufacturer == manufacturer
    if character_set is not None:
        assert parsed_source.SpecificCharacterSet == character_set

    manager = DicomFileManager()
    assert manager.scan_directory(input_dir, False, key_store=KeyStore()) == 1
    manager.current_index = 0
    result = export_instance(manager, manager.dicom_df.iloc[0], image, output_root=output_root)
    saved = pydicom.dcmread(result.dicom_path)

    assert saved.Manufacturer == manufacturer
    assert ('SequenceOfUltrasoundRegions' in saved) == (regions == 'present')
    if character_set is None:
        assert 'SpecificCharacterSet' not in saved
    else:
        assert saved.SpecificCharacterSet == character_set


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


@pytest.mark.parametrize('identity_removed', [None, 'NO'])
def test_export_instance_requires_identity_removed_yes(scanned_manager, output_root, image,
                                                       identity_removed):
    def set_identity_removed(ds):
        if identity_removed is not None:
            ds.PatientIdentityRemoved = identity_removed

    row = scanned_manager.dicom_df.iloc[0]
    with patch.object(scanned_manager, '_set_deidentification_attributes', set_identity_removed):
        with pytest.raises(ExportError, match='staged DICOM file does not match'):
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
