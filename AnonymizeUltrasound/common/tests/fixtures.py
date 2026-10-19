"""Synthetic ultrasound DICOM datasets shared by the AnonymizeUltrasound tests."""
import os

import pydicom.config
from pydicom.dataelem import DataElement
from pydicom.datadict import tag_for_keyword
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian

US_IMAGE_STORAGE = '1.2.840.10008.5.1.4.1.1.6.1'
US_MULTIFRAME_STORAGE = '1.2.840.10008.5.1.4.1.1.3.1'
US_IMAGE_STORAGE_RETIRED = '1.2.840.10008.5.1.4.1.1.6'
US_MULTIFRAME_STORAGE_RETIRED = '1.2.840.10008.5.1.4.1.1.3'
SECONDARY_CAPTURE_STORAGE = '1.2.840.10008.5.1.4.1.1.7'

SOURCE_UID_ROOT = '1.2.826.0.1.3680043.9999'

DEFAULT_ATTRIBUTES = {
    'PatientID': 'MRN0012345',
    'PatientName': 'Doe^Jane',
    'StudyInstanceUID': SOURCE_UID_ROOT + '.1',
    'SeriesInstanceUID': SOURCE_UID_ROOT + '.2',
    'SOPInstanceUID': SOURCE_UID_ROOT + '.3',
    'SOPClassUID': US_MULTIFRAME_STORAGE,
    'Modality': 'US',
    'NumberOfFrames': 2,
    'Rows': 4,
    'Columns': 6,
    'BitsAllocated': 8,
    'BitsStored': 8,
    'HighBit': 7,
    'PixelRepresentation': 0,
    'SamplesPerPixel': 1,
    'PhotometricInterpretation': 'MONOCHROME2',
    'ContentDate': '20240101',
    'ContentTime': '120000',
    'TransducerData': 'SC6-1s,02597',
}


def make_dataset(**overrides) -> FileDataset:
    """Return an in-memory ultrasound FileDataset.

    Keyword arguments override or add top-level attributes; a value of ``None``
    leaves the attribute out. ``PhysicalDeltaX`` and ``PhysicalDeltaY`` set the
    calibration of the single default region item; ``regions`` replaces the
    region items with the given list (``[]`` for an empty sequence).
    """
    attributes = {**DEFAULT_ATTRIBUTES, **overrides}
    delta_x = attributes.pop('PhysicalDeltaX', 0.1)
    delta_y = attributes.pop('PhysicalDeltaY', 0.15)
    regions = attributes.pop('regions', None)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = attributes['SOPClassUID']
    meta.MediaStorageSOPInstanceUID = attributes['SOPInstanceUID']
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = '1.2.826.0.1.3680043.8.498.1'

    ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0' * 128,
                     is_implicit_VR=False, is_little_endian=True)
    for keyword, value in attributes.items():
        if value is not None:
            setattr(ds, keyword, value)

    if regions is None:
        region = Dataset()
        region.PhysicalDeltaX = delta_x
        region.PhysicalDeltaY = delta_y
        regions = [region]
    ds.SequenceOfUltrasoundRegions = Sequence(regions)

    frames = int(attributes.get('NumberOfFrames') or 1)
    ds.PixelData = b'\0' * (ds.Rows * ds.Columns * frames)
    return ds


def set_unchecked(ds: Dataset, keyword: str, vr: str, value) -> None:
    """Set an attribute without pydicom's value validation (for deliberately bad source values)."""
    ds.add(DataElement(tag_for_keyword(keyword), vr, value, validation_mode=pydicom.config.IGNORE))


def write_dataset(ds: FileDataset, directory: str, filename: str) -> str:
    """Save ``ds`` under ``directory`` (created if needed) and return its path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    ds.save_as(path)
    return path
