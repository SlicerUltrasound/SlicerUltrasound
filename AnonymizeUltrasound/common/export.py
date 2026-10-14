"""Export orchestration without Slicer: directory guards, output paths and the
staged DICOM + annotation JSON pair.

An export writes two files for one instance, ``<uuid>.dcm`` and
``<uuid>.json``. Both are staged next to their final names, the staged DICOM
is read back and checked against the keys.csv row, and only then are the two
renamed into place, DICOM first. An instance counts as exported when both
final files exist.
"""
import itertools
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import re

import pydicom

from .key_store import is_plain_relative_path
from .version_info import get_provenance

PREFIX_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,16}$')
DICOM_EXTENSIONS = {'.dcm', '.dicom'}
_DIRECTORY_NAMES = ('input', 'output', 'keys')


class ExportError(Exception):
    """An export or an Import-time check could not be completed; nothing final was changed."""


@dataclass(frozen=True)
class ExportResult:
    dicom_path: str
    json_path: str


# ---------------------------------------------------------------------------
# Operator input
# ---------------------------------------------------------------------------

def validate_prefix(text: Optional[str]) -> str:
    """Return the project label, or raise when it is not 1-16 of ``[A-Za-z0-9_-]``."""
    text = (text or '').strip()
    if text and not PREFIX_PATTERN.match(text):
        raise ExportError(
            'The project label may only use letters, digits, "_" and "-" (1 to 16 characters), '
            'or be left empty.'
        )
    return text


def compose_patient_name(prefix: str, anon_patient_id: str) -> str:
    return f'{prefix}_{anon_patient_id}' if prefix else anon_patient_id


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

def resolve_session_directories(input_dir: str, output_dir: str,
                                keys_dir: str) -> Tuple[str, str, str]:
    """Return the real paths of the three directories, or raise when they overlap."""
    resolved = []
    for name, directory in zip(_DIRECTORY_NAMES, (input_dir, output_dir, keys_dir)):
        if not directory or not os.path.isdir(directory):
            raise ExportError(f'The {name} directory does not exist.')
        resolved.append(os.path.realpath(directory))
    pairs = itertools.combinations(zip(_DIRECTORY_NAMES, resolved), 2)
    for (name_a, path_a), (name_b, path_b) in pairs:
        if path_a == path_b:
            raise ExportError(f'The {name_a} and {name_b} directories must be different.')
        if _is_within(path_a, path_b):
            raise ExportError(f'The {name_a} directory must not be inside the {name_b} directory.')
        if _is_within(path_b, path_a):
            raise ExportError(f'The {name_b} directory must not be inside the {name_a} directory.')
    return resolved[0], resolved[1], resolved[2]


def output_directory_has_dicom(output_dir: str) -> bool:
    """True when any file under ``output_dir`` has a DICOM extension."""
    for _, _, files in os.walk(output_dir):
        if any(os.path.splitext(name)[1].lower() in DICOM_EXTENSIONS for name in files):
            return True
    return False


def _is_within(child: str, parent: str) -> bool:
    try:
        return child != parent and os.path.commonpath([child, parent]) == parent
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

def resolve_output_path(output_root: str, output_path: str,
                        preserve_directory_structure: bool = True) -> str:
    """Return the final ``.dcm`` path of a keys.csv ``OutputPath`` under ``output_root``.

    ``output_path`` must be a plain ``/``-separated relative path. With
    ``preserve_directory_structure`` off only its file name is used. The result
    must resolve (symlinks included) to a location under the output root.
    """
    output_path = str(output_path)
    if not is_plain_relative_path(output_path) or not output_path.endswith('.dcm'):
        raise ExportError('OutputPath in keys.csv must be a plain relative path to a .dcm file.')
    root = os.path.realpath(output_root)
    parts = output_path.split('/')
    if not preserve_directory_structure:
        parts = parts[-1:]
    candidate = os.path.join(root, *parts)
    resolved = os.path.realpath(candidate)
    if not _is_within(resolved, root):
        raise ExportError('OutputPath in keys.csv resolves outside the output directory.')
    return candidate


def _json_path(dicom_path: str) -> str:
    return dicom_path[:-len('.dcm')] + '.json'


def is_export_complete(output_root: str, row, preserve_directory_structure: bool = True) -> bool:
    """True when both the DICOM and the annotation JSON of ``row`` exist."""
    dicom_path = resolve_output_path(output_root, row['OutputPath'], preserve_directory_structure)
    return os.path.isfile(dicom_path) and os.path.isfile(_json_path(dicom_path))


# ---------------------------------------------------------------------------
# The export pair
# ---------------------------------------------------------------------------

def export_instance(manager, row, image_array, *, output_root: str, prefix: str = '',
                    mask_params: Optional[Dict] = None, labels: Optional[Iterable[str]] = None,
                    preserve_directory_structure: bool = True) -> ExportResult:
    """Write the DICOM and annotation JSON of the manager's current instance.

    ``row`` is the dataframe row of that instance. On any failure the staged
    files are removed and the error is raised as :class:`ExportError`. Up to
    the final renames the previously exported pair, if any, is left as it was.
    When the JSON rename fails after the DICOM rename succeeded, the new DICOM
    is removed again so the instance does not count as exported: the previous
    DICOM is gone by then and the previous JSON may remain.
    """
    prefix = validate_prefix(prefix)
    dicom_final = resolve_output_path(output_root, row['OutputPath'], preserve_directory_structure)
    json_final = _json_path(dicom_final)
    suffix = f'.tmp-{os.getpid()}'
    dicom_temp = dicom_final + suffix
    json_temp = json_final + suffix
    try:
        patient_name = compose_patient_name(prefix, row['AnonPatientID'])
        anonymized = manager.save_anonymized_dicom(image_array, dicom_temp, patient_name)
        if anonymized is None:
            raise ExportError('The DICOM file could not be written; see the log for the reason.')
        _verify_staged_dicom(dicom_temp, row)
        _write_sidecar(json_temp, row['AnonSOPInstanceUID'], mask_params, labels)
        os.replace(dicom_temp, dicom_final)
        try:
            os.replace(json_temp, json_final)
        except Exception:
            _discard(dicom_final)  # the new DICOM must not pair with the previous JSON
            raise
    except ExportError:
        _discard(dicom_temp, json_temp)
        raise
    except Exception as error:
        _discard(dicom_temp, json_temp)
        raise ExportError(f'Export failed: {error}') from error
    return ExportResult(dicom_final, json_final)


def _verify_staged_dicom(path: str, row) -> None:
    staged = pydicom.dcmread(path, stop_before_pixels=True)
    checks = (
        (str(staged.get('SOPInstanceUID', '')), str(row['AnonSOPInstanceUID'])),
        (str(staged.get('PatientID', '')), str(row['AnonPatientID'])),
        (str(staged.get('PatientIdentityRemoved', '')), 'YES'),
    )
    if any(actual != expected for actual, expected in checks):
        raise ExportError(
            'The staged DICOM file does not match its keys.csv row; nothing was exported.'
        )


def _write_sidecar(path: str, anon_sop_uid: str, mask_params: Optional[Dict],
                   labels: Optional[Iterable[str]]) -> None:
    provenance = get_provenance()
    info = {
        'SOPInstanceUID': anon_sop_uid,
        'GrayscaleConversion': False,
        'git_sha': provenance['git_sha'],
        'git_dirty': provenance['git_dirty'],
    }
    info.update(mask_params or {})
    if labels is not None:
        info['AnnotationLabels'] = list(labels)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(info, handle)


def _discard(*paths: str) -> None:
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass
