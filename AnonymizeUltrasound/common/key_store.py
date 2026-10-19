"""Persisted mapping between source DICOM identifiers and their export values.

``keys.csv`` is the one file that relates an exported instance back to its
source. It lives in the keys directory next to a zero-byte ``keys.csv.lock``
and nothing else. Every value in it is either copied from the source
(``PatientUID``, ``StudyUID``, ``SeriesUID``, ``InstanceUID``,
``FrameOfReferenceUID``, ``InputPath``) or drawn from the operating system's
random source the first time the source value is seen (``AnonPatientID``, the
anonymized UIDs, ``DateShiftDays``). Once drawn, a value is only ever read
back, so every later Import against the same keys directory reproduces the
same pseudonyms and the same per-patient date offset.

The file is read with :meth:`KeyStore.load`, which rejects anything that is
not a well-formed, one-to-one map and leaves the file untouched, and written
with :meth:`KeyStore.save`, which replaces it atomically.
"""
import csv
import os
import posixpath
import re
import secrets
import sys
import uuid
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pydicom.uid

KEYS_FILENAME = 'keys.csv'
LOCK_FILENAME = 'keys.csv.lock'

KEYS_CSV_COLUMNS = [
    'InputPath',
    'OutputPath',
    'AnonFilename',
    'PatientUID',
    'AnonPatientID',
    'StudyUID',
    'AnonStudyUID',
    'SeriesUID',
    'AnonSeriesUID',
    'InstanceUID',
    'AnonSOPInstanceUID',
    'FrameOfReferenceUID',
    'AnonFrameOfReferenceUID',
    'DateShiftDays',
]
OPTIONAL_COLUMNS = {'FrameOfReferenceUID'}
ANON_UID_COLUMNS = (
    'AnonStudyUID', 'AnonSeriesUID', 'AnonSOPInstanceUID', 'AnonFrameOfReferenceUID',
)

DATE_SHIFT_MIN_DAYS = 1
DATE_SHIFT_MAX_DAYS = 1095

UNSUPPORTED_FORMAT_MESSAGE = (
    'keys.csv has an unsupported format; '
    'choose an empty keys directory and an empty output directory'
)

_HEX32 = re.compile(r'^[0-9a-f]{32}$')
_UUID_ARC_UID = re.compile(r'^2\.25\.(0|[1-9]\d{0,38})$')
_UUID_FILENAME = re.compile(r'^([0-9a-f]{32})\.dcm$')
_DRIVE_LETTER = re.compile(r'^[A-Za-z]:')
_SERIES_SENTINEL = ':SeriesUID:'


class KeysFileError(Exception):
    """keys.csv or its lock cannot be used. The file on disk is left as it was."""


# ---------------------------------------------------------------------------
# Value generators and conversions
# ---------------------------------------------------------------------------

def new_uid() -> str:
    """Return a fresh UID in the ``2.25`` arc (the integer form of a random UUID)."""
    return str(pydicom.uid.generate_uid(prefix=None))


def new_patient_id() -> str:
    """Return a fresh patient pseudonym: 32 lowercase hex characters."""
    return uuid.uuid4().hex


def new_date_shift_days() -> int:
    """Return a date offset in days, uniform over [1, 1095]."""
    return secrets.randbelow(DATE_SHIFT_MAX_DAYS) + DATE_SHIFT_MIN_DAYS


def uid_to_filename(uid: str) -> str:
    """Return the ``<uuid hex>.dcm`` file name of a ``2.25`` UID."""
    uid = str(uid)
    if not _UUID_ARC_UID.match(uid):
        raise ValueError('UID is not in the 2.25 arc')
    return f'{uuid.UUID(int=int(uid[5:])).hex}.dcm'


def filename_to_uid(filename: str) -> str:
    """Return the ``2.25`` UID encoded in a ``<uuid hex>.dcm`` file name."""
    match = _UUID_FILENAME.match(str(filename))
    if not match:
        raise ValueError('file name is not a UUID-named DICOM file')
    return f'2.25.{uuid.UUID(hex=match.group(1)).int}'


def is_plain_relative_path(path: str) -> bool:
    """True for a normalized, ``/``-separated relative path with no ``.`` or ``..`` parts."""
    if not path or '\\' in path or posixpath.isabs(path) or _DRIVE_LETTER.match(path):
        return False
    parts = path.split('/')
    if '' in parts or '.' in parts or '..' in parts:
        return False
    return posixpath.normpath(path) == path


def _is_valid_anon_uid(value: str) -> bool:
    """A ``2.25`` UID with a single integer component of up to 39 digits, no leading zero."""
    return len(value) <= 64 and bool(_UUID_ARC_UID.match(value))


def _frame_key(study_uid: str, source_for: str, series_uid: str) -> Tuple[str, str]:
    """Mapping key for a FrameOfReferenceUID: per study, per source value or per series."""
    if source_for:
        return (str(study_uid), str(source_for))
    return (str(study_uid), f'{_SERIES_SENTINEL}{series_uid}')


def _validate_row(row: Dict[str, object], where: str) -> Dict[str, str]:
    """Return ``row`` with every column as a string, or raise :class:`KeysFileError`."""
    missing = [column for column in KEYS_CSV_COLUMNS if column not in row]
    if missing:
        raise KeysFileError(f'{where}: missing column(s) {", ".join(missing)}')
    clean = {
        column: ('' if row[column] is None else str(row[column])) for column in KEYS_CSV_COLUMNS
    }

    for column in KEYS_CSV_COLUMNS:
        if column not in OPTIONAL_COLUMNS and clean[column] == '':
            raise KeysFileError(f'{where}: {column} is empty')
    if not _HEX32.match(clean['AnonPatientID']):
        raise KeysFileError(f'{where}: AnonPatientID is not 32 lowercase hex characters')
    for column in ANON_UID_COLUMNS:
        if not _is_valid_anon_uid(clean[column]):
            raise KeysFileError(f'{where}: {column} is not a valid UID in the 2.25 arc')

    try:
        shift = int(clean['DateShiftDays'])
    except ValueError:
        raise KeysFileError(f'{where}: DateShiftDays is not an integer') from None
    in_range = DATE_SHIFT_MIN_DAYS <= shift <= DATE_SHIFT_MAX_DAYS
    if str(shift) != clean['DateShiftDays'] or not in_range:
        raise KeysFileError(
            f'{where}: DateShiftDays must be an integer from '
            f'{DATE_SHIFT_MIN_DAYS} to {DATE_SHIFT_MAX_DAYS}'
        )

    try:
        expected_filename = uid_to_filename(clean['AnonSOPInstanceUID'])
    except ValueError:
        raise KeysFileError(f'{where}: AnonSOPInstanceUID is outside the UUID range') from None
    if clean['AnonFilename'] != expected_filename:
        raise KeysFileError(f'{where}: AnonFilename does not match AnonSOPInstanceUID')
    output_path = clean['OutputPath']
    if (not is_plain_relative_path(output_path)
            or posixpath.basename(output_path) != clean['AnonFilename']):
        raise KeysFileError(
            f'{where}: OutputPath must be a plain relative path ending in AnonFilename'
        )
    return clean


# ---------------------------------------------------------------------------
# KeyStore
# ---------------------------------------------------------------------------

class KeyStore:
    """In-memory view of ``keys.csv``: mint on first sight, read back ever after.

    ``patient``, ``study``, ``series``, ``instance`` and ``frame_of_reference``
    return the stored value for a source key or draw a new one; ``record``
    adds the completed row of a scanned instance; ``save`` writes the union of
    the loaded and recorded rows. Every source-to-anonymized relation is kept
    one-to-one, and a row that would break that is rejected without changing
    the store.
    """

    def __init__(self) -> None:
        self._rows: Dict[str, Dict[str, str]] = {}
        self._patients: Dict[str, Tuple[str, int]] = {}
        self._studies: Dict[str, str] = {}
        self._series: Dict[str, str] = {}
        self._instances: Dict[str, Tuple[str, str]] = {}
        self._frames: Dict[Tuple[str, str], str] = {}
        self._reverse: Dict[str, Dict[str, object]] = {
            'patient': {}, 'study': {}, 'series': {}, 'instance': {}, 'frame': {},
        }
        self._study_patient: Dict[str, str] = {}
        self._series_study: Dict[str, str] = {}
        self._instance_series: Dict[str, str] = {}

    @classmethod
    def load(cls, path: str) -> 'KeyStore':
        """Read ``path``; a missing file gives an empty store, a bad file raises."""
        store = cls()
        if not os.path.exists(path):
            return store
        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False,
                                encoding='utf-8')
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
            raise KeysFileError(UNSUPPORTED_FORMAT_MESSAGE) from error
        if list(frame.columns) != KEYS_CSV_COLUMNS:
            raise KeysFileError(UNSUPPORTED_FORMAT_MESSAGE)
        for number, values in enumerate(frame.itertuples(index=False, name=None), start=1):
            row = dict(zip(KEYS_CSV_COLUMNS, values))
            where = f'keys.csv row {number}'
            if row['InstanceUID'] in store._rows:
                raise KeysFileError(f'{where}: InstanceUID is listed twice')
            store._record(row, where)
        return store

    # -- lookups that mint on a miss ------------------------------------------

    def patient(self, source_patient_id: str) -> Tuple[str, int]:
        """Return ``(AnonPatientID, DateShiftDays)`` for a source PatientID."""
        key = str(source_patient_id)
        if key not in self._patients:
            value = (new_patient_id(), new_date_shift_days())
            self._patients[key] = value
            self._reverse['patient'][value[0]] = key
        return self._patients[key]

    def study(self, source_study_uid: str) -> str:
        return self._mint(self._studies, self._reverse['study'], str(source_study_uid))

    def series(self, source_series_uid: str) -> str:
        return self._mint(self._series, self._reverse['series'], str(source_series_uid))

    def instance(self, source_sop_uid: str) -> Tuple[str, str]:
        """Return ``(AnonSOPInstanceUID, AnonFilename)`` for a source SOP Instance UID."""
        key = str(source_sop_uid)
        if key not in self._instances:
            anon_uid = new_uid()
            self._instances[key] = (anon_uid, uid_to_filename(anon_uid))
            self._reverse['instance'][anon_uid] = key
        return self._instances[key]

    def frame_of_reference(self, study_uid: str, source_for: str, series_uid: str) -> str:
        """Return the anonymized FrameOfReferenceUID for (study, source FOR or series)."""
        key = _frame_key(study_uid, source_for, series_uid)
        return self._mint(self._frames, self._reverse['frame'], key)

    @staticmethod
    def _mint(mapping: dict, reverse: dict, key) -> str:
        if key not in mapping:
            value = new_uid()
            mapping[key] = value
            reverse[value] = key
        return mapping[key]

    # -- rows -------------------------------------------------------------------

    def record(self, row: Dict[str, object]) -> Dict[str, str]:
        """Add or update the row of one instance and return it as stored.

        A known instance keeps its ``OutputPath``: the export location does not
        move when the source file does. Raises when the row contradicts the store.
        """
        return self._record(row, str(row.get('InputPath') or 'scanned instance'))

    def rows(self) -> List[Dict[str, str]]:
        return [dict(row) for row in self._rows.values()]

    def _record(self, row: Dict[str, object], where: str) -> Dict[str, str]:
        clean = _validate_row(row, where)
        known = self._rows.get(clean['InstanceUID'])
        if known is not None:
            clean['OutputPath'] = known['OutputPath']
        shift = int(clean['DateShiftDays'])
        frame_key = _frame_key(clean['StudyUID'], clean['FrameOfReferenceUID'], clean['SeriesUID'])
        reverse = self._reverse
        bindings = [
            (self._patients, clean['PatientUID'], (clean['AnonPatientID'], shift), 'PatientUID'),
            (self._studies, clean['StudyUID'], clean['AnonStudyUID'], 'StudyUID'),
            (self._series, clean['SeriesUID'], clean['AnonSeriesUID'], 'SeriesUID'),
            (self._instances, clean['InstanceUID'],
             (clean['AnonSOPInstanceUID'], clean['AnonFilename']), 'InstanceUID'),
            (self._frames, frame_key, clean['AnonFrameOfReferenceUID'], 'FrameOfReferenceUID'),
            (reverse['patient'], clean['AnonPatientID'], clean['PatientUID'], 'AnonPatientID'),
            (reverse['study'], clean['AnonStudyUID'], clean['StudyUID'], 'AnonStudyUID'),
            (reverse['series'], clean['AnonSeriesUID'], clean['SeriesUID'], 'AnonSeriesUID'),
            (reverse['instance'], clean['AnonSOPInstanceUID'], clean['InstanceUID'],
             'AnonSOPInstanceUID'),
            (reverse['frame'], clean['AnonFrameOfReferenceUID'], frame_key,
             'AnonFrameOfReferenceUID'),
            (self._study_patient, clean['StudyUID'], clean['PatientUID'], 'StudyUID'),
            (self._series_study, clean['SeriesUID'], clean['StudyUID'], 'SeriesUID'),
            (self._instance_series, clean['InstanceUID'], clean['SeriesUID'], 'InstanceUID'),
        ]
        # Check every relation before changing any, so a rejected row leaves the store as it was.
        for mapping, key, value, column in bindings:
            if key in mapping and mapping[key] != value:
                raise KeysFileError(f'{where}: {column} conflicts with an earlier row')
        for mapping, key, value, _ in bindings:
            mapping[key] = value
        self._rows[clean['InstanceUID']] = clean
        return dict(clean)

    # -- persistence ----------------------------------------------------------

    def save(self, path: str) -> None:
        """Write every row to ``path`` through a 0600 temp file and an atomic rename."""
        temp_path = f'{path}.tmp-{os.getpid()}'
        descriptor = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=KEYS_CSV_COLUMNS, lineterminator='\n')
                writer.writeheader()
                for row in self._rows.values():
                    writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except BaseException:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
        _fsync_directory(os.path.dirname(os.path.abspath(path)))


def _fsync_directory(directory: str) -> None:
    """Flush the directory entry after a rename where the platform allows it."""
    if not hasattr(os, 'O_DIRECTORY'):
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------

if sys.platform == 'win32':  # pragma: no cover - exercised on Windows only
    import msvcrt

    def _lock_exclusive(handle) -> None:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock_exclusive(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class KeysLock:
    """Exclusive advisory lock on a keys directory, held through ``keys.csv.lock``.

    The lock file is a separate zero-byte file because the lock must survive
    the atomic replacement of ``keys.csv`` itself.
    """

    def __init__(self) -> None:
        self._handle = None
        self.path: Optional[str] = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self, keys_dir: str) -> None:
        """Take the lock, or raise :class:`KeysFileError` when another session holds it."""
        self.release()
        path = os.path.join(keys_dir, LOCK_FILENAME)
        handle = open(path, 'a+b')
        try:
            _lock_exclusive(handle)
        except OSError:
            handle.close()
            raise KeysFileError(
                'The keys directory is in use by another session (keys.csv.lock is held).'
            ) from None
        self._handle = handle
        self.path = path

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            _unlock(self._handle)
        finally:
            self._handle.close()
            self._handle = None
            self.path = None
