"""Tests for common/key_store.py: the persisted keys.csv mapping and its lock."""
import csv
import os
import stat
import sys
import uuid
from unittest.mock import patch

import pytest

from ..key_store import (
    DATE_SHIFT_MAX_DAYS,
    DATE_SHIFT_MIN_DAYS,
    KEYS_CSV_COLUMNS,
    LOCK_FILENAME,
    KeyStore,
    KeysFileError,
    KeysLock,
    filename_to_uid,
    new_date_shift_days,
    new_patient_id,
    new_uid,
    uid_to_filename,
)

HEX32_A = 'a' * 32
HEX32_B = 'b' * 32

LEGACY_HEADER = [
    'InputPath', 'OutputPath', 'AnonFilename', 'PatientUID', 'StudyUID', 'SeriesUID',
    'InstanceUID', 'AnonStudyUID', 'AnonSeriesUID', 'AnonSOPInstanceUID',
    'FrameOfReferenceUID', 'AnonFrameOfReferenceUID', 'PhysicalDeltaX', 'PhysicalDeltaY',
    'ContentDate', 'ContentTime', 'Patch', 'TransducerModel', 'SeriesNumber',
]


def _row(**overrides):
    anon_sop = overrides.get('AnonSOPInstanceUID', '2.25.13')
    filename = uid_to_filename(anon_sop)
    row = {
        'InputPath': 'site/a.dcm',
        'OutputPath': 'site/' + filename,
        'AnonFilename': filename,
        'PatientUID': 'P1',
        'AnonPatientID': HEX32_A,
        'StudyUID': '1.2.3.1',
        'AnonStudyUID': '2.25.11',
        'SeriesUID': '1.2.3.2',
        'AnonSeriesUID': '2.25.12',
        'InstanceUID': '1.2.3.3',
        'AnonSOPInstanceUID': anon_sop,
        'FrameOfReferenceUID': '1.2.3.4',
        'AnonFrameOfReferenceUID': '2.25.14',
        'DateShiftDays': '10',
    }
    row.update(overrides)
    return row


def _second_row(**overrides):
    """A second instance of the same patient, study and series as ``_row()``."""
    base = {'InputPath': 'site/b.dcm', 'InstanceUID': '1.2.3.5', 'AnonSOPInstanceUID': '2.25.15'}
    base.update(overrides)
    return _row(**base)


def _write_csv(path, rows, header=KEYS_CSV_COLUMNS):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(column, '') for column in header])


def _read_bytes(path):
    with open(path, 'rb') as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# Value generators
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('value', [
    0,
    2 ** 128 - 1,
    0x0123456789abcdef0123456789abcdef,
    0xfedcba9876543210fedcba9876543210,
    0x80000000000000000000000000000000,
    0x00000000000000000000000000000001,
    0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f,
])
def test_uid_and_filename_are_a_bijection(value):
    with patch('uuid.uuid4', return_value=uuid.UUID(int=value)):
        uid = new_uid()
    assert uid == f'2.25.{value}'
    filename = uid_to_filename(uid)
    assert filename == f'{uuid.UUID(int=value).hex}.dcm'
    assert filename_to_uid(filename) == uid


@pytest.mark.parametrize('bad_uid', ['1.2.3', '2.25.', '2.25.abc', '2.25.-1', f'2.25.{2 ** 128}'])
def test_uid_to_filename_rejects_values_outside_the_uuid_arc(bad_uid):
    with pytest.raises(ValueError):
        uid_to_filename(bad_uid)


@pytest.mark.parametrize('bad_name', ['x.dcm', 'A' * 32 + '.dcm', 'a' * 31 + '.dcm', 'a' * 32 + '.json'])
def test_filename_to_uid_rejects_non_uuid_names(bad_name):
    with pytest.raises(ValueError):
        filename_to_uid(bad_name)


def test_new_patient_id_is_32_lowercase_hex_characters():
    with patch('uuid.uuid4', return_value=uuid.UUID(int=0xabc)):
        assert new_patient_id() == uuid.UUID(int=0xabc).hex
    assert len(new_patient_id()) == 32


@pytest.mark.parametrize('randbelow_value, expected', [(0, 1), (1094, 1095)])
def test_new_date_shift_days_covers_exactly_one_to_1095(randbelow_value, expected):
    with patch('secrets.randbelow', return_value=randbelow_value) as randbelow:
        assert new_date_shift_days() == expected
    randbelow.assert_called_once_with(DATE_SHIFT_MAX_DAYS)
    assert DATE_SHIFT_MIN_DAYS == 1


def test_new_date_shift_days_is_never_zero_or_negative():
    draws = {new_date_shift_days() for _ in range(300)}
    assert min(draws) >= 1
    assert max(draws) <= 1095


# ---------------------------------------------------------------------------
# Mint on miss, reuse on hit
# ---------------------------------------------------------------------------

def test_patient_mints_once_and_returns_the_same_pair_afterwards():
    store = KeyStore()
    first = store.patient('P1')
    second = store.patient('P1')
    other = store.patient('P2')

    assert first == second
    assert first[0] != other[0]
    assert len(first[0]) == 32 and int(first[0], 16) >= 0
    assert 1 <= first[1] <= 1095
    assert isinstance(first[1], int)


def test_study_series_instance_and_frame_mint_once_and_reuse():
    store = KeyStore()
    assert store.study('S1') == store.study('S1')
    assert store.study('S1') != store.study('S2')
    assert store.series('Se1') == store.series('Se1')
    assert store.instance('I1') == store.instance('I1')
    anon_sop, filename = store.instance('I1')
    assert filename == uid_to_filename(anon_sop)
    assert store.frame_of_reference('S1', 'F1', 'Se1') == store.frame_of_reference('S1', 'F1', 'Se2')
    assert store.frame_of_reference('S1', 'F1', 'Se1') != store.frame_of_reference('S2', 'F1', 'Se1')


def test_frame_of_reference_without_source_value_is_per_series():
    store = KeyStore()
    assert store.frame_of_reference('S1', '', 'Se1') != store.frame_of_reference('S1', '', 'Se2')
    assert store.frame_of_reference('S1', '', 'Se1') == store.frame_of_reference('S1', '', 'Se1')


# ---------------------------------------------------------------------------
# record / save / load
# ---------------------------------------------------------------------------

def test_record_save_load_round_trip_keeps_rows_and_patient_pair(tmp_path):
    path = str(tmp_path / 'keys.csv')
    store = KeyStore()
    store.record(_row())
    store.record(_second_row())
    store.save(path)

    with open(path, newline='', encoding='utf-8') as handle:
        header = handle.readline().rstrip('\n')
    assert header == ','.join(KEYS_CSV_COLUMNS)

    loaded = KeyStore.load(path)
    assert loaded.rows() == [_row(), _second_row()]
    assert loaded.patient('P1') == (HEX32_A, 10)
    assert loaded.study('1.2.3.1') == '2.25.11'
    assert loaded.series('1.2.3.2') == '2.25.12'
    assert loaded.instance('1.2.3.3') == ('2.25.13', uid_to_filename('2.25.13'))
    assert loaded.frame_of_reference('1.2.3.1', '1.2.3.4', '1.2.3.2') == '2.25.14'


def test_save_writes_the_union_of_prior_and_recorded_rows(tmp_path):
    path = str(tmp_path / 'keys.csv')
    _write_csv(path, [_row()])

    store = KeyStore.load(path)
    store.record(_second_row())
    store.save(path)

    assert [r['InstanceUID'] for r in KeyStore.load(path).rows()] == ['1.2.3.3', '1.2.3.5']


def test_record_upserts_paths_of_a_known_instance():
    store = KeyStore()
    store.record(_row())
    store.record(_row(InputPath='moved/a.dcm', OutputPath='moved/' + uid_to_filename('2.25.13')))

    rows = store.rows()
    assert len(rows) == 1
    assert rows[0]['InputPath'] == 'moved/a.dcm'


def test_record_rejects_a_known_instance_under_another_series():
    store = KeyStore()
    store.record(_row())
    with pytest.raises(KeysFileError):
        store.record(_row(SeriesUID='1.2.3.8', AnonSeriesUID='2.25.18'))


def test_patient_pair_minted_before_save_is_the_pair_read_after_load(tmp_path):
    path = str(tmp_path / 'keys.csv')
    store = KeyStore()
    anon_patient_id, shift = store.patient('P9')
    row = _row(PatientUID='P9', AnonPatientID=anon_patient_id, DateShiftDays=str(shift))
    store.record(row)
    store.save(path)

    assert KeyStore.load(path).patient('P9') == (anon_patient_id, shift)


def test_load_missing_file_returns_an_empty_store(tmp_path):
    store = KeyStore.load(str(tmp_path / 'keys.csv'))
    assert store.rows() == []


def test_save_creates_the_temp_file_with_mode_0600_before_writing(tmp_path):
    path = str(tmp_path / 'keys.csv')
    store = KeyStore()
    store.record(_row())
    calls = []
    real_open = os.open

    def spy(file, flags, mode=0o777, *args, **kwargs):
        calls.append((str(file), flags, mode))
        return real_open(file, flags, mode, *args, **kwargs)

    with patch('os.open', side_effect=spy):
        store.save(path)

    temp_calls = [c for c in calls if c[0].startswith(path) and c[0] != path]
    assert len(temp_calls) == 1
    _, flags, mode = temp_calls[0]
    assert flags & os.O_CREAT and flags & os.O_EXCL
    assert mode == 0o600
    if sys.platform != 'win32':
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert sorted(os.listdir(str(tmp_path))) == ['keys.csv']


def test_interrupted_save_leaves_the_prior_file_intact(tmp_path):
    path = str(tmp_path / 'keys.csv')
    _write_csv(path, [_row()])
    before = _read_bytes(path)

    store = KeyStore.load(path)
    store.record(_second_row())
    with patch('os.replace', side_effect=OSError('disk full')):
        with pytest.raises(OSError):
            store.save(path)

    assert _read_bytes(path) == before
    assert sorted(os.listdir(str(tmp_path))) == ['keys.csv']


# ---------------------------------------------------------------------------
# load rejects anything that is not a well-formed, one-to-one map
# ---------------------------------------------------------------------------

MALFORMED_FILES = [
    ('legacy_header', LEGACY_HEADER, [_row()]),
    ('missing_last_column', KEYS_CSV_COLUMNS[:-1], [_row()]),
    ('extra_column', KEYS_CSV_COLUMNS + ['Extra'], [_row()]),
    ('empty_patient_uid', KEYS_CSV_COLUMNS, [_row(PatientUID='')]),
    ('empty_input_path', KEYS_CSV_COLUMNS, [_row(InputPath='')]),
    ('empty_anon_for', KEYS_CSV_COLUMNS, [_row(AnonFrameOfReferenceUID='')]),
    ('anon_patient_id_not_hex', KEYS_CSV_COLUMNS, [_row(AnonPatientID='XYZ')]),
    ('anon_patient_id_uppercase', KEYS_CSV_COLUMNS, [_row(AnonPatientID='A' * 32)]),
    ('anon_study_uid_wrong_arc', KEYS_CSV_COLUMNS, [_row(AnonStudyUID='1.2.3')]),
    ('anon_series_uid_leading_zero', KEYS_CSV_COLUMNS, [_row(AnonSeriesUID='2.25.0123')]),
    ('anon_for_uid_not_numeric', KEYS_CSV_COLUMNS, [_row(AnonFrameOfReferenceUID='2.25.abc')]),
    ('shift_zero', KEYS_CSV_COLUMNS, [_row(DateShiftDays='0')]),
    ('shift_1096', KEYS_CSV_COLUMNS, [_row(DateShiftDays='1096')]),
    ('shift_negative', KEYS_CSV_COLUMNS, [_row(DateShiftDays='-1')]),
    ('shift_not_an_int', KEYS_CSV_COLUMNS, [_row(DateShiftDays='abc')]),
    ('filename_does_not_match_sop_uid', KEYS_CSV_COLUMNS,
     [_row(AnonFilename='x.dcm', OutputPath='site/x.dcm')]),
    ('output_path_absolute', KEYS_CSV_COLUMNS, [_row(OutputPath='/site/' + uid_to_filename('2.25.13'))]),
    ('output_path_parent_reference', KEYS_CSV_COLUMNS,
     [_row(OutputPath='../' + uid_to_filename('2.25.13'))]),
    ('output_path_basename_mismatch', KEYS_CSV_COLUMNS, [_row(OutputPath='site/other.dcm')]),
    ('output_path_not_normalized', KEYS_CSV_COLUMNS,
     [_row(OutputPath='site//' + uid_to_filename('2.25.13'))]),
]

INCONSISTENT_FILES = [
    ('duplicate_instance_uid', [_row(), _row(InputPath='site/b.dcm')]),
    ('one_patient_two_pseudonyms', [_row(), _second_row(AnonPatientID=HEX32_B)]),
    ('one_patient_two_shifts', [_row(), _second_row(DateShiftDays='11')]),
    ('two_patients_one_pseudonym',
     [_row(), _second_row(PatientUID='P2', StudyUID='1.2.3.7', AnonStudyUID='2.25.17',
                          SeriesUID='1.2.3.8', AnonSeriesUID='2.25.18',
                          FrameOfReferenceUID='1.2.3.9', AnonFrameOfReferenceUID='2.25.19')]),
    ('one_study_two_anon_studies', [_row(), _second_row(AnonStudyUID='2.25.21')]),
    ('two_studies_one_anon_study',
     [_row(), _second_row(StudyUID='1.2.3.7', SeriesUID='1.2.3.8', AnonSeriesUID='2.25.18',
                          AnonFrameOfReferenceUID='2.25.19')]),
    ('one_series_two_anon_series', [_row(), _second_row(AnonSeriesUID='2.25.22')]),
    ('two_series_one_anon_series', [_row(), _second_row(SeriesUID='1.2.3.8')]),
    ('two_instances_one_anon_sop', [_row(), _second_row(AnonSOPInstanceUID='2.25.13')]),
    ('series_under_two_studies',
     [_row(), _second_row(StudyUID='1.2.3.7', AnonStudyUID='2.25.17', AnonFrameOfReferenceUID='2.25.19')]),
    ('study_under_two_patients',
     [_row(), _second_row(PatientUID='P2', AnonPatientID=HEX32_B, DateShiftDays='5')]),
    ('one_frame_pair_two_anon_frames', [_row(), _second_row(AnonFrameOfReferenceUID='2.25.99')]),
    ('two_frame_pairs_one_anon_frame',
     [_row(), _second_row(StudyUID='1.2.3.7', AnonStudyUID='2.25.17', SeriesUID='1.2.3.8',
                          AnonSeriesUID='2.25.18')]),
]


@pytest.mark.parametrize('case_id, header, rows', MALFORMED_FILES, ids=[c[0] for c in MALFORMED_FILES])
def test_load_rejects_malformed_files_and_leaves_them_untouched(tmp_path, case_id, header, rows):
    path = str(tmp_path / 'keys.csv')
    _write_csv(path, rows, header=header)
    before = _read_bytes(path)

    with pytest.raises(KeysFileError):
        KeyStore.load(path)

    assert _read_bytes(path) == before


@pytest.mark.parametrize('case_id, rows', INCONSISTENT_FILES, ids=[c[0] for c in INCONSISTENT_FILES])
def test_load_rejects_files_whose_mapping_is_not_one_to_one(tmp_path, case_id, rows):
    path = str(tmp_path / 'keys.csv')
    _write_csv(path, rows)
    before = _read_bytes(path)

    with pytest.raises(KeysFileError) as excinfo:
        KeyStore.load(path)

    assert 'row 2' in str(excinfo.value)
    assert _read_bytes(path) == before


def test_load_accepts_two_instances_of_one_series(tmp_path):
    path = str(tmp_path / 'keys.csv')
    _write_csv(path, [_row(), _second_row()])
    assert len(KeyStore.load(path).rows()) == 2


def test_load_rejects_an_empty_file(tmp_path):
    path = str(tmp_path / 'keys.csv')
    open(path, 'wb').close()
    with pytest.raises(KeysFileError):
        KeyStore.load(path)


def test_legacy_file_error_message_tells_the_operator_what_to_do(tmp_path):
    path = str(tmp_path / 'keys.csv')
    _write_csv(path, [_row()], header=LEGACY_HEADER)
    with pytest.raises(KeysFileError) as excinfo:
        KeyStore.load(path)
    assert 'unsupported format' in str(excinfo.value)
    assert 'empty keys directory' in str(excinfo.value)


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------

def test_second_lock_acquire_fails_until_the_first_is_released(tmp_path):
    keys_dir = str(tmp_path)
    first = KeysLock()
    second = KeysLock()

    first.acquire(keys_dir)
    with pytest.raises(KeysFileError):
        second.acquire(keys_dir)

    first.release()
    second.acquire(keys_dir)
    second.release()

    lock_path = os.path.join(keys_dir, LOCK_FILENAME)
    assert os.path.exists(lock_path)
    assert os.path.getsize(lock_path) == 0
    assert sorted(os.listdir(keys_dir)) == [LOCK_FILENAME]


def test_release_without_acquire_is_a_no_op():
    KeysLock().release()
