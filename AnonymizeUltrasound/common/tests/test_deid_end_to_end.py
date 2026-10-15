"""End-to-end proof of the interactive export path on a synthetic collection.

Builds a small input tree with planted identifiers, runs Import (lock, load,
scan, save keys.csv) and Export through the modules the Slicer widget calls,
and reads the results back. Only the preserved root ultrasound-region subtree
is exempt from identifier-removal checks; keys.csv intentionally holds mappings.
"""
import csv
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from ..dicom_file_manager import DicomFileManager
from ..export import export_instance, is_export_complete
from ..key_store import (
    KEYS_CSV_COLUMNS, KEYS_FILENAME, LOCK_FILENAME, KeyStore, KeysLock, uid_to_filename,
)
from .fixtures import (
    SECONDARY_CAPTURE_STORAGE,
    SOURCE_UID_ROOT,
    US_MULTIFRAME_STORAGE,
    make_dataset,
    set_unchecked,
    write_dataset,
)

PLANTED_TEXT = {
    'IssuerOfPatientID': 'HOSP_A',
    'PatientBirthDate': '19800515',
    'AccessionNumber': 'ACC778899',
    'StudyDescription': 'Patient Jane Doe referred by Dr Smith',
    'SeriesDescription': 'Dr Smith lung scan',
    'StationName': 'US_ROOM_3B',
    'InstitutionName': 'General Hospital',
    'OperatorsName': 'Operator^Olga',
    'ReferringPhysicianName': 'Smith^John',
    'ImageComments': 'Jane Doe follow-up comment',
}
PRIVATE_CREATOR = 'PLANTED_PRIVATE'
PRIVATE_VALUE = 'planted-private-value'
REGION_PRIVATE_VALUE = 'planted-region-private'
NESTED_UID = SOURCE_UID_ROOT + '.777'

PATIENTS = {
    'P1': {'PatientID': 'MRN0012345', 'PatientName': 'Doe^Jane'},
    'P2': {'PatientID': 'MRN0022222', 'PatientName': 'Roe^Richard'},
    'P3': {'PatientID': 'MRN0033333', 'PatientName': 'Poe^Edgar'},
}

# (file name, patient, study, series, instance, dates, extra attributes)
FILES = [
    ('p1/s1/a.dcm', 'P1', '.11', '.21', '.31', '20240229', {}),
    ('p1/s1/b.dcm', 'P1', '.11', '.21', '.32', '20240229', {}),
    ('p1/s2/c.dcm', 'P1', '.12', '.22', '.33', '20240315', {}),
    ('p2/s3/d.dcm', 'P2', '.13', '.23', '.34', '20230601', {}),
    ('p2/s4/e.dcm', 'P2', '.14', '.24', '.35', '20230901', {}),
    ('p3/s5/f.dcm', 'P3', '.15', '.25', '.36', None, {}),
    ('p3/s5/g.dcm', 'P3', '.15', '.26', '.37', '20231399', {}),
    ('p1/s1/h.dcm', 'P1', '.11', '.29', '.39', '20240229', {'SOPClassUID': SECONDARY_CAPTURE_STORAGE}),
]
MASK_PARAMS = {'mask_type': 'rectangle', 'rectangle_left': 1, 'rectangle_right': 5,
               'rectangle_top': 1, 'rectangle_bottom': 3}


def _planted_region():
    region = Dataset()
    region.RegionSpatialFormat = 1
    region.RegionDataType = 1
    region.RegionFlags = 0
    region.PhysicalUnitsXDirection = 3
    region.PhysicalUnitsYDirection = 3
    region.PhysicalDeltaX = 0.1
    region.PhysicalDeltaY = 0.15
    region.private_block(0x0011, PRIVATE_CREATOR, create=True).add_new(0x01, 'LO', REGION_PRIVATE_VALUE)
    nested = Dataset()
    nested.ReferencedSOPClassUID = US_MULTIFRAME_STORAGE
    nested.ReferencedSOPInstanceUID = NESTED_UID
    region.ReferencedImageSequence = Sequence([nested])
    return region


def _source(patient, study, series, instance, date, extra):
    attributes = dict(PLANTED_TEXT)
    attributes.update(PATIENTS[patient])
    attributes.update({
        'StudyInstanceUID': SOURCE_UID_ROOT + study,
        'SeriesInstanceUID': SOURCE_UID_ROOT + series,
        'SOPInstanceUID': SOURCE_UID_ROOT + instance,
        'FrameOfReferenceUID': SOURCE_UID_ROOT + study + '.9',
        'SeriesNumber': 3,
        'StudyDate': date,
        'SeriesDate': date,
        'ContentDate': date,
        'StudyTime': '101010' if date else None,
        'SeriesTime': '101011' if date else None,
        'ContentTime': '101012' if date else None,
    })
    attributes.update(extra)
    unparseable = date is not None and date != '' and len(date) == 8 and int(date[4:6]) > 12
    if unparseable:
        for keyword in ('StudyDate', 'SeriesDate', 'ContentDate'):
            attributes[keyword] = None
    ds = make_dataset(regions=[_planted_region()], **attributes)
    if unparseable:
        for keyword in ('StudyDate', 'SeriesDate', 'ContentDate'):
            set_unchecked(ds, keyword, 'DA', date)
    ds.private_block(0x0011, PRIVATE_CREATOR, create=True).add_new(0x01, 'LO', PRIVATE_VALUE)
    return ds


def _planted_strings():
    values = set(PLANTED_TEXT.values())
    values |= {PRIVATE_CREATOR, PRIVATE_VALUE, REGION_PRIVATE_VALUE, NESTED_UID}
    for patient in PATIENTS.values():
        values |= set(patient.values())
    for _, _, study, series, instance, _, _ in FILES:
        values |= {SOURCE_UID_ROOT + study, SOURCE_UID_ROOT + series, SOURCE_UID_ROOT + instance,
                   SOURCE_UID_ROOT + study + '.9'}
    return values


def _elements_outside_root_regions(ds):
    for element in ds:
        if element.keyword == 'SequenceOfUltrasoundRegions':
            continue
        yield element
        if element.VR == 'SQ':
            for item in element.value:
                yield from item.iterall()


def _element_values(ds):
    for element in ds.file_meta:
        yield str(element.value)
    for element in _elements_outside_root_regions(ds):
        if element.VR == 'SQ' or element.keyword == 'PixelData':
            continue
        yield str(element.value)


def test_identifier_scan_exempts_only_the_root_region_subtree():
    ds = make_dataset(regions=[_planted_region()])
    ds.file_meta.SourceApplicationEntityTitle = 'SOURCE_AE'
    values = list(_element_values(ds))
    assert 'SOURCE_AE' in values
    assert REGION_PRIVATE_VALUE not in values
    assert NESTED_UID not in values

    nested = Dataset()
    nested.SequenceOfUltrasoundRegions = Sequence([_planted_region()])
    ds.ReferencedImageSequence = Sequence([nested])
    values = list(_element_values(ds))
    assert REGION_PRIVATE_VALUE in values
    assert NESTED_UID in values


def _shifted(date, days):
    return (datetime.strptime(date, '%Y%m%d') + timedelta(days=days)).strftime('%Y%m%d')


@pytest.fixture
def collection(tmp_path):
    input_dir = str(tmp_path / 'input')
    output_dir = str(tmp_path / 'output')
    keys_dir = str(tmp_path / 'keys')
    for d in (input_dir, output_dir, keys_dir):
        os.makedirs(d)
    for name, patient, study, series, instance, date, extra in FILES:
        write_dataset(_source(patient, study, series, instance, date, extra),
                      os.path.join(input_dir, os.path.dirname(name)), os.path.basename(name))
    return input_dir, output_dir, keys_dir


def _import(input_dir, keys_dir):
    lock = KeysLock()
    lock.acquire(keys_dir)
    keys_path = os.path.join(keys_dir, KEYS_FILENAME)
    store = KeyStore.load(keys_path)
    manager = DicomFileManager()
    manager.scan_directory(input_dir, False, key_store=store)
    manager.save_keys_csv(keys_path)
    return lock, manager


def test_end_to_end_export_of_a_synthetic_collection(collection):
    input_dir, output_dir, keys_dir = collection
    lock, manager = _import(input_dir, keys_dir)
    df = manager.dicom_df

    assert len(df) == 7
    assert manager.skipped_unsupported_sop_class == 1
    assert manager.issuers_of_patient_id == {'HOSP_A'}
    assert sorted(os.listdir(keys_dir)) == [KEYS_FILENAME, LOCK_FILENAME]
    with open(os.path.join(keys_dir, KEYS_FILENAME), newline='', encoding='utf-8') as handle:
        mappings = {row['InstanceUID']: row for row in csv.DictReader(handle)}
    assert set(mappings) == set(df['InstanceUID'])

    image = np.zeros((2, 4, 6, 1), dtype=np.uint8)
    for index in range(len(df)):
        manager.current_index = index
        export_instance(manager, df.iloc[index], image, output_root=output_dir, prefix='PROJ',
                        mask_params=MASK_PARAMS, labels=['BLines'], preserve_directory_structure=False)

    planted = _planted_strings()
    by_source = {}
    for index in range(len(df)):
        row = df.iloc[index]
        dicom_path = os.path.join(output_dir, row['AnonFilename'])
        ds = pydicom.dcmread(dicom_path)
        by_source[row['InstanceUID']] = ds
        mapping = mappings[row['InstanceUID']]
        assert mapping['InputPath'] == os.path.relpath(row['InputPath'], input_dir).replace(os.sep, '/')
        for column in KEYS_CSV_COLUMNS:
            if column != 'InputPath':
                assert mapping[column] == str(row[column]), column

        for value in _element_values(ds):
            for text in planted:
                assert text not in value, f'{text!r} found in exported element'
        assert os.path.basename(dicom_path) == uid_to_filename(ds.SOPInstanceUID)
        assert ds.PatientID == row['AnonPatientID']
        assert ds.PatientName == 'PROJ_' + row['AnonPatientID']
        assert ds.StudyInstanceUID == mapping['AnonStudyUID']
        assert ds.SeriesInstanceUID == mapping['AnonSeriesUID']
        assert ds.SOPInstanceUID == mapping['AnonSOPInstanceUID']
        assert ds.FrameOfReferenceUID == mapping['AnonFrameOfReferenceUID']
        assert ds.PatientIdentityRemoved == 'YES'
        assert ds.LongitudinalTemporalInformationModified == 'MODIFIED'
        assert 'DeidentificationMethod' not in ds
        assert len(ds.DeidentificationMethodCodeSequence) == 1
        code = ds.DeidentificationMethodCodeSequence[0]
        assert (code.CodingSchemeDesignator, code.CodeValue, code.CodeMeaning) == (
            '99SLICERUS', 'ANON_US', 'SlicerUltrasound_AnonymizeUltrasound')
        assert ds.StudyID == ''
        for keyword in ('StudyDescription', 'SeriesDescription', 'StationName', 'InstitutionName',
                        'OperatorsName', 'ImageComments', 'IssuerOfPatientID', 'PatientBirthDate'):
            assert keyword not in ds or ds[keyword].value == '', keyword
        assert not any(element.tag.is_private for element in _elements_outside_root_regions(ds))
        assert not any(element.tag.is_private for element in ds.file_meta)

        source = pydicom.dcmread(row['InputPath'], stop_before_pixels=True)
        assert ds.SequenceOfUltrasoundRegions == source.SequenceOfUltrasoundRegions
        assert len(ds.SequenceOfUltrasoundRegions) == 1
        item = ds.SequenceOfUltrasoundRegions[0]
        assert float(item.PhysicalDeltaX) == 0.1
        assert item.RegionSpatialFormat == 1
        assert item.ReferencedImageSequence[0].ReferencedSOPInstanceUID == NESTED_UID
        assert item.private_block(0x0011, PRIVATE_CREATOR)[0x01].value == REGION_PRIVATE_VALUE

        with open(dicom_path[:-4] + '.json', encoding='utf-8') as handle:
            sidecar_text = handle.read()
        sidecar = json.loads(sidecar_text)
        assert sidecar['SOPInstanceUID'] == ds.SOPInstanceUID
        assert sidecar['AnnotationLabels'] == ['BLines']
        for text in planted:
            assert text not in sidecar_text

    for root, _, names in os.walk(output_dir):
        for name in names:
            relative_path = os.path.relpath(os.path.join(root, name), output_dir)
            for text in planted:
                assert text not in relative_path

    shifts = df.groupby('PatientUID')['DateShiftDays'].nunique()
    assert (shifts == 1).all()
    assert df[df['PatientUID'] == 'MRN0012345']['AnonStudyUID'].nunique() == 2
    assert df['AnonPatientID'].nunique() == 3

    for _, patient, study, _, instance, date, extra in FILES:
        if extra:
            continue
        ds = by_source[SOURCE_UID_ROOT + instance]
        shift = int(df[df['InstanceUID'] == SOURCE_UID_ROOT + instance]['DateShiftDays'].iloc[0])
        if date == '20231399' or date is None:
            assert ds.StudyDate == '' and ds.ContentDate == '' and 'SeriesDate' not in ds
        else:
            assert ds.StudyDate == _shifted(date, shift)
            assert ds.SeriesDate == _shifted(date, shift)
            assert ds.ContentDate == _shifted(date, shift)
            assert ds.StudyTime == '101010' and ds.SeriesTime == '101011' and ds.ContentTime == '101012'
            assert ds.StudyDate > date

    sibling_a = by_source[SOURCE_UID_ROOT + '.31']
    sibling_b = by_source[SOURCE_UID_ROOT + '.32']
    assert sibling_a.SeriesInstanceUID == sibling_b.SeriesInstanceUID
    assert sibling_a.StudyInstanceUID == sibling_b.StudyInstanceUID
    assert sibling_a.FrameOfReferenceUID == sibling_b.FrameOfReferenceUID
    assert int(sibling_a.SeriesNumber) == 3 and int(sibling_b.SeriesNumber) == 3
    assert sibling_a.SOPInstanceUID != sibling_b.SOPInstanceUID
    assert sibling_a.StudyInstanceUID != by_source[SOURCE_UID_ROOT + '.33'].StudyInstanceUID

    lock.release()

    # A second session over the same input and keys directory reuses every value.
    lock2, manager2 = _import(input_dir, keys_dir)
    key_columns = ['InstanceUID', 'AnonPatientID', 'AnonStudyUID', 'AnonSeriesUID',
                   'AnonSOPInstanceUID', 'AnonFrameOfReferenceUID', 'DateShiftDays', 'AnonFilename']
    first = df[key_columns].sort_values('InstanceUID').reset_index(drop=True)
    second = manager2.dicom_df[key_columns].sort_values('InstanceUID').reset_index(drop=True)
    assert first.equals(second)
    assert len(KeyStore.load(os.path.join(keys_dir, KEYS_FILENAME)).rows()) == 7
    assert manager2.update_progress_from_output(output_dir, False) is None

    broken = manager2.dicom_df.iloc[3]
    os.remove(os.path.join(output_dir, broken['AnonFilename'][:-4] + '.json'))
    assert is_export_complete(output_dir, broken, preserve_directory_structure=False) is False
    assert manager2.update_progress_from_output(output_dir, False) == 3
    lock2.release()
