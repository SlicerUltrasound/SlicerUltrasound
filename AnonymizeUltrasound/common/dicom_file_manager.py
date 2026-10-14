import copy
import datetime
import io
import logging
import os
import posixpath
from typing import List, Optional, Set

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydicom.dataset import FileMetaDataset

from .export import DICOM_EXTENSIONS, is_export_complete
from .key_store import KEYS_CSV_COLUMNS, KeyStore, KeysFileError
from .version_info import apply_provenance_tag, get_provenance


class DicomFileManager:
    """
    Shared DICOM file management functionality for ultrasound modules.

    This class provides common functionality for managing DICOM files across different
    ultrasound processing modules. It handles:

    - Scanning directories for DICOM files
    - Parsing DICOM metadata and creating structured dataframes
    - Managing file loading and temporary directories
    - Looking up or minting the export values of every instance in a KeyStore
    - Progress tracking over exported pairs

    The class maintains a pandas DataFrame (dicom_df) containing metadata for all
    discovered DICOM files, including file paths, source identifiers, their export
    values from keys.csv, physical spacing data and the source dataset.

    Attributes:
        dicom_df (pd.DataFrame): DataFrame containing DICOM file metadata
        next_index (int): Index of next file to process
        current_index (int): Index of currently loaded file
        key_store (KeyStore): Mapping used by the last scan
        issuers_of_patient_id (set): Distinct non-empty IssuerOfPatientID values of the last scan
        skipped_unsupported_sop_class (int): Files the last scan skipped for their SOP class
    """

    # Allowed DICOM file extensions (case-insensitive); the set export.py also scans for.
    DICOM_EXTENSIONS = DICOM_EXTENSIONS

    # SOP classes an export can be built from. Anything else is skipped at scan time.
    SUPPORTED_SOP_CLASSES = {
        '1.2.840.10008.5.1.4.1.1.6.1',  # Ultrasound Image Storage
        '1.2.840.10008.5.1.4.1.1.3.1',  # Ultrasound Multi-frame Image Storage
        '1.2.840.10008.5.1.4.1.1.6',    # Ultrasound Image Storage (Retired)
        '1.2.840.10008.5.1.4.1.1.3',    # Ultrasound Multi-frame Image Storage (Retired)
    }

    # Tags copied from source only if present (allowlist semantics).
    DICOM_TAGS_TO_COPY = [
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PatientAge",
        "PatientSex",
        "PixelRepresentation",
        "SeriesNumber",
        "Manufacturer",
    ]

    # Tags always written to the de-id dataset: source value if present, else "".
    # Downstream consumers rely on these tags being present on every anonymized DICOM.
    DICOM_TAGS_PRESERVE_OR_BLANK = [
        "TransducerData",
        "TransducerType",
        "ManufacturerModelName",
    ]

    # Attributes of a SequenceOfUltrasoundRegions item that are carried over
    # (PS3.3 Table C.8-17). Each output item is rebuilt from these alone.
    REGION_ALLOWLIST = (
        'RegionSpatialFormat', 'RegionDataType', 'RegionFlags',
        'RegionLocationMinX0', 'RegionLocationMinY0', 'RegionLocationMaxX1', 'RegionLocationMaxY1',
        'ReferencePixelX0', 'ReferencePixelY0',
        'PhysicalUnitsXDirection', 'PhysicalUnitsYDirection',
        'ReferencePixelPhysicalValueX', 'ReferencePixelPhysicalValueY',
        'PhysicalDeltaX', 'PhysicalDeltaY',
        'TransducerFrequency', 'TransducerType', 'PulseRepetitionFrequency',
        'DopplerCorrectionAngle', 'SteeringAngle',
        'DopplerSampleVolumeXPositionRetired', 'DopplerSampleVolumeXPosition',
        'DopplerSampleVolumeYPositionRetired', 'DopplerSampleVolumeYPosition',
        'TMLinePositionX0Retired', 'TMLinePositionX0',
        'TMLinePositionY0Retired', 'TMLinePositionY0',
        'TMLinePositionX1Retired', 'TMLinePositionX1',
        'TMLinePositionY1Retired', 'TMLinePositionY1',
        'PixelComponentOrganization', 'PixelComponentMask',
        'PixelComponentRangeStart', 'PixelComponentRangeStop',
        'PixelComponentPhysicalUnits', 'PixelComponentDataType',
        'NumberOfTableBreakPoints', 'TableOfXBreakPoints', 'TableOfYBreakPoints',
        'NumberOfTableEntries', 'TableOfPixelValues', 'TableOfParameterValues',
    )

    # Expected columns in the DICOM files dataframe: the keys.csv columns first,
    # then scan-time detail that is not persisted.
    DICOM_DATAFRAME_COLUMNS = KEYS_CSV_COLUMNS + [
        'PhysicalDeltaX',
        'PhysicalDeltaY',
        'ContentDate',
        'ContentTime',
        'Patch',
        'TransducerModel',
        'DICOMDataset',
    ]

    # Identity of this writer in the file meta information of every export.
    IMPLEMENTATION_CLASS_UID = '2.25.249430386050762985123801272949428663139'

    # DeidentificationMethod (0012,0063) values after the tool identification.
    # Each value is limited to 64 characters by its VR.
    DEIDENTIFICATION_METHOD = (
        'Dataset rebuilt from an allowlist of attributes',
        'Random UUID pseudonyms; mapping kept in keys.csv',
        'Dates shifted by a per-patient offset; times kept',
        'Device model, patient sex and capped age retained',
        'Pixels outside the operator mask zeroed; no OCR',
        'Restricted research use; separate review before sharing',
    )

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Forget the last scan: dataframe, indices, key store and scan counters."""
        self.dicom_df = None
        self.input_folder = None
        self.next_index = 0
        self.current_index = 0
        self.key_store: Optional[KeyStore] = None
        self.issuers_of_patient_id: Set[str] = set()
        self.skipped_unsupported_sop_class = 0
        # SOPInstanceUID -> (file path, source identifiers) of every listed instance.
        self._seen_instances = {}

    def _first_transducer_segment(self, raw) -> str:
        """Return the first segment of a TransducerData-like value with case preserved.

        Handles plain strings (comma- or backslash-delimited) and pydicom MultiValue
        objects (auto-split by VR LO separator). Returns an empty string for None,
        empty, or whitespace-only input.
        """
        if raw is not None and not isinstance(raw, str) and hasattr(raw, '__getitem__'):
            raw = raw[0] if len(raw) > 0 else ''

        source = str(raw).strip() if raw is not None else ''
        if not source:
            return ''

        return source.split('\\')[0].split(',')[0].strip()

    def get_transducer_model(self, transducer_data, manufacturer: str = '',
                             manufacturer_model_name: str = '') -> str:
        """
        Parse the transducer model identifier from DICOM metadata.

        For Butterfly Network manufacturers, the device identifier is stored in
        ManufacturerModelName (0008,1090) rather than TransducerData (0018,5010).
        For other vendors, TransducerData may be comma-delimited ('SC6-1s,02597')
        or backslash-delimited ('S4-1U\\UNUSED\\UNUSED'); pydicom may return the
        latter as a MultiValue because backslash is the native VR LO separator.

        Returns the lowercased model identifier, or 'unknown' if unavailable.
        """
        if 'butterfly' in str(manufacturer or '').lower():
            raw = manufacturer_model_name
        else:
            raw = transducer_data

        segment = self._first_transducer_segment(raw)
        return segment.lower() if segment else 'unknown'

    def scan_directory(self, input_folder: str, skip_single_frame: bool = False,
                       key_store: Optional[KeyStore] = None) -> int:
        """
        Scan directory for DICOM files and create dataframe

        Args:
            input_folder: Directory to scan
            skip_single_frame: Skip single frame DICOM files (AnonymizeUltrasound)
            key_store: Mapping to read export values from and record new ones in.
                A fresh, empty store is used when none is given.

        Returns:
            Number of DICOM files found

        Raises:
            KeysFileError: a scanned instance contradicts a mapping already in the store,
                or two files share a SOPInstanceUID but disagree on their source identifiers.
        """
        self.reset()
        self.key_store = key_store if key_store is not None else KeyStore()
        self.input_folder = input_folder
        dicom_data = []

        for root, dirs, files in os.walk(input_folder):
            # Sort to ensure consistent processing order
            dirs.sort()
            files.sort()
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)
                if ext.lower() not in self.DICOM_EXTENSIONS:
                    logging.info(f"Skipping non-DICOM file: {file_path}")
                    continue

                dicom_info = self._extract_dicom_info(file_path, input_folder, skip_single_frame)
                if dicom_info:
                    dicom_data.append(dicom_info)

        self._create_dataframe(dicom_data)

        return len(self.dicom_df) if self.dicom_df is not None else 0

    def get_number_of_instances(self) -> int:
        """Get number of instances in dataframe"""
        return len(self.dicom_df) if self.dicom_df is not None else 0

    def save_keys_csv(self, path: str) -> None:
        """Write the key store of the last scan to ``path`` (keys.csv)."""
        if self.key_store is None:
            raise KeysFileError('No scan to save: run scan_directory first')
        self.key_store.save(path)

    def _extract_dicom_info(self, file_path: str, input_folder: str,
                            skip_single_frame: bool) -> Optional[dict]:
        """Extract DICOM information from file

        Reads and parses a DICOM file, checks that it is a supported ultrasound
        image, and looks up (or mints) its export values in the key store.
        Everything read from the file is guarded: a file that cannot be read or
        parsed is logged and skipped. Conflicts with the store are not.

        Args:
            file_path: Path to the DICOM file to process
            input_folder: Path to the input folder
            skip_single_frame: If True, skip files with less than 2 frames

        Returns:
            dict: Dictionary containing extracted DICOM metadata
            None: If file cannot be read or parsed, is not a supported ultrasound
                image, doesn't meet frame requirements, or repeats an instance
                already listed

        Raises:
            KeysFileError: the instance contradicts a mapping already in the store,
                or repeats a listed SOPInstanceUID with different source identifiers.
        """
        try:
            dicom_ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            parsed = self._parse_source_dataset(dicom_ds, file_path, skip_single_frame)
        except Exception as e:
            logging.error(f"Skipping unreadable DICOM file {file_path}: {e}")
            return None
        if parsed is None:
            return None
        source_ids, issuer, fields = parsed

        if issuer:
            self.issuers_of_patient_id.add(issuer)

        listed = self._seen_instances.get(source_ids['InstanceUID'])
        if listed is not None:
            listed_path, listed_ids = listed
            if listed_ids != source_ids:
                raise KeysFileError(
                    f"{file_path}: SOPInstanceUID is already listed from {listed_path} "
                    "with different source identifiers")
            logging.info(f"Skipping {file_path}: its SOPInstanceUID is already listed "
                         f"from {listed_path}")
            return None

        # keys.csv paths are relative to the input folder and use '/' on every platform.
        relative_input = os.path.relpath(file_path, input_folder).replace(os.sep, '/')
        keys_row = self._record_mapping(source_ids, relative_input)
        self._seen_instances[source_ids['InstanceUID']] = (file_path, source_ids)

        return {
            **keys_row,
            **fields,
            'InputPath': file_path,
            'DateShiftDays': int(keys_row['DateShiftDays']),
        }

    def _parse_source_dataset(self, dicom_ds, file_path: str,
                              skip_single_frame: bool) -> Optional[tuple]:
        """Check that ``dicom_ds`` is a supported ultrasound image and read what the scan needs.

        Returns ``(source_ids, issuer, fields)``: the source identifiers that
        :meth:`_record_mapping` maps, the stripped IssuerOfPatientID ('' when
        absent) and the dataframe columns that come from the file alone.
        Returns ``None`` when the file is to be skipped; the reason is logged.
        """
        # Skip non-ultrasound modalities
        if dicom_ds.get("Modality", "") != "US":
            logging.info(f"Skipping non-ultrasound file: {file_path}")
            return None

        sop_class_uid = str(dicom_ds.get('SOPClassUID', '') or '')
        if sop_class_uid not in self.SUPPORTED_SOP_CLASSES:
            logging.info("Skipping file whose SOPClassUID is not an ultrasound image storage "
                         f"class: {file_path}")
            self.skipped_unsupported_sop_class += 1
            return None

        # Skip single frame if requested (AnonymizeUltrasound)
        if skip_single_frame and ('NumberOfFrames' not in dicom_ds or dicom_ds.NumberOfFrames < 2):
            logging.info(f"Skipping single frame file: {file_path}")
            return None

        # Extract required fields
        patient_uid = getattr(dicom_ds, 'PatientID', None)
        study_uid = getattr(dicom_ds, 'StudyInstanceUID', None)
        series_uid = getattr(dicom_ds, 'SeriesInstanceUID', None)
        instance_uid = getattr(dicom_ds, 'SOPInstanceUID', None)

        if not all([patient_uid, study_uid, series_uid, instance_uid]):
            logging.info(f"Missing required DICOM fields in file: {file_path}")
            return None

        source_ids = {
            'PatientUID': str(patient_uid),
            'StudyUID': str(study_uid),
            'SeriesUID': str(series_uid),
            'InstanceUID': str(instance_uid),
            'FrameOfReferenceUID': str(getattr(dicom_ds, 'FrameOfReferenceUID', '') or ''),
        }
        issuer = str(dicom_ds.get('IssuerOfPatientID', '') or '').strip()

        physical_delta_x, physical_delta_y = self._extract_spacing_info(dicom_ds)
        transducer_model = self.get_transducer_model(
            dicom_ds.get('TransducerData', ''),
            manufacturer=dicom_ds.get('Manufacturer', ''),
            manufacturer_model_name=dicom_ds.get('ManufacturerModelName', ''),
        )
        fields = {
            'PhysicalDeltaX': physical_delta_x,
            'PhysicalDeltaY': physical_delta_y,
            'ContentDate': str(getattr(dicom_ds, 'ContentDate', '') or ''),
            'ContentTime': str(getattr(dicom_ds, 'ContentTime', '') or ''),
            'Patch': physical_delta_x is None or physical_delta_y is None,
            'TransducerModel': transducer_model,
            'DICOMDataset': dicom_ds,
        }
        return source_ids, issuer, fields

    def _record_mapping(self, source_ids: dict, relative_input: str) -> dict:
        """Look up or mint the export values of one instance and record its keys.csv row.

        Args:
            source_ids: PatientUID, StudyUID, SeriesUID, InstanceUID and
                FrameOfReferenceUID ('' when absent) of the source instance
            relative_input: input path relative to the input folder, '/'-separated

        Returns:
            The keys.csv row as stored (every value a string). A known instance
            keeps the OutputPath of its earlier export.

        Raises:
            KeysFileError: the instance contradicts a mapping already in the store.
        """
        store = self.key_store
        anon_patient_id, date_shift_days = store.patient(source_ids['PatientUID'])
        anon_sop_uid, anon_filename = store.instance(source_ids['InstanceUID'])
        keys_row = {
            'InputPath': relative_input,
            'OutputPath': posixpath.join(posixpath.dirname(relative_input), anon_filename),
            'AnonFilename': anon_filename,
            'PatientUID': source_ids['PatientUID'],
            'AnonPatientID': anon_patient_id,
            'StudyUID': source_ids['StudyUID'],
            'AnonStudyUID': store.study(source_ids['StudyUID']),
            'SeriesUID': source_ids['SeriesUID'],
            'AnonSeriesUID': store.series(source_ids['SeriesUID']),
            'InstanceUID': source_ids['InstanceUID'],
            'AnonSOPInstanceUID': anon_sop_uid,
            'FrameOfReferenceUID': source_ids['FrameOfReferenceUID'],
            'AnonFrameOfReferenceUID': store.frame_of_reference(
                source_ids['StudyUID'], source_ids['FrameOfReferenceUID'], source_ids['SeriesUID']),
            'DateShiftDays': str(date_shift_days),
        }
        return store.record(keys_row)

    def _extract_spacing_info(self, dicom_ds):
        """Extract physical spacing information from DICOM dataset"""
        physical_delta_x = None
        physical_delta_y = None

        if hasattr(dicom_ds, 'SequenceOfUltrasoundRegions') and dicom_ds.SequenceOfUltrasoundRegions:
            region = dicom_ds.SequenceOfUltrasoundRegions[0]
            if hasattr(region, 'PhysicalDeltaX'):
                physical_delta_x = float(region.PhysicalDeltaX)
            if hasattr(region, 'PhysicalDeltaY'):
                physical_delta_y = float(region.PhysicalDeltaY)

        return physical_delta_x, physical_delta_y

    def _create_dataframe(self, dicom_data: List[dict]) -> None:
        """Create pandas DataFrame from DICOM data"""
        if not dicom_data:
            self.dicom_df = pd.DataFrame()
            return

        # Create DataFrame with proper column order
        self.dicom_df = pd.DataFrame(dicom_data, columns=self.DICOM_DATAFRAME_COLUMNS)

        # Sort and reset index
        self.dicom_df = (self.dicom_df
                         .sort_values(['InputPath', 'ContentDate', 'ContentTime'])
                         .reset_index(drop=True))

        # Fill missing spacing information using forward/backward fill
        spacing_cols = ['PhysicalDeltaX', 'PhysicalDeltaY']
        self.dicom_df[spacing_cols] = (self.dicom_df
                                       .groupby('StudyUID')[spacing_cols]
                                       .transform(lambda x: x.ffill().bfill()))

        self.next_index = 0

    def update_progress_from_output(self, output_directory: str, preserve_directory_structure: bool) -> Optional[int]:
        """Update progress based on existing output pairs

        An instance counts as processed when both its DICOM and its annotation
        JSON exist in the output directory. next_index is moved to the first
        instance that is not complete, so processing resumes there.

        Args:
            output_directory: Directory path where anonymized DICOM files are saved
            preserve_directory_structure: If True, the output filepath will be the same as the relative path.
        Returns:
            int: Number of files already processed (0 if none processed)
            None: If all files have been processed or no dataframe exists
        """
        if self.dicom_df is None:
            return None

        for index, (_, row) in enumerate(self.dicom_df.iterrows()):
            if not is_export_complete(output_directory, row, preserve_directory_structure):
                self.next_index = index
                return index

        return None

    def increment_dicom_index(self, output_directory: Optional[str] = None,
                              continue_progress: bool = False, preserve_directory_structure: bool = True) -> bool:
        """
        Increment the DICOM index to the next file to be processed.

        This method advances the internal index counter and optionally skips files whose
        export pair already exists in the output directory when continuing from a previous
        processing session.

        Args:
            output_directory (Optional[str]): The output directory path where processed files
                are stored. Required when continue_progress is True.
            continue_progress (bool): If True, skip files that already exist in the output
                directory to continue from where processing left off. Defaults to False.
            preserve_directory_structure (bool): Whether to preserve the original directory
                structure when generating output file paths. Defaults to True.

        Returns:
            bool: True if there are more files to process (index is within bounds),
                    False if all files have been processed or if dicom_df is None.
        """
        if self.dicom_df is None:
            return False

        # Increment the index to the next file to be processed.
        self.next_index += 1

        # If continue_progress is True, skip files whose export pair already exists.
        if continue_progress and output_directory:
            while self.next_index < len(self.dicom_df):
                row = self.dicom_df.iloc[self.next_index]
                if not is_export_complete(output_directory, row, preserve_directory_structure):
                    break
                self.next_index += 1

        return self.next_index < len(self.dicom_df)

    def save_anonymized_dicom(self, image_array: np.ndarray, output_path: str,
                              new_patient_name: str = '') -> Optional[pydicom.Dataset]:
        """
        Save image array as anonymized DICOM file and return the in-memory anon Dataset.

        PatientID, the UIDs and the date offset come from the current dataframe
        record, i.e. from keys.csv. The caller supplies only the display name.

        Args:
            image_array: Numpy array containing image data (frames, height, width, channels)
            output_path: Full path where DICOM file should be saved
            new_patient_name: PatientName to write; the AnonPatientID when empty

        Returns:
            The anonymized pydicom.Dataset on success, or None when the call
            fails its pre-conditions (no dataframe, invalid current_index,
            None image array).
        """
        if self.dicom_df is None:
            logging.error("No DICOM dataframe available")
            return None

        if self.current_index >= len(self.dicom_df):
            logging.error("No current DICOM record available")
            return None

        if image_array is None:
            logging.error("Image array is None")
            return None

        current_record = self.dicom_df.iloc[self.current_index]
        source_dataset = current_record.DICOMDataset

        # Create new anonymized dataset
        anonymized_ds = self._create_base_dicom_dataset(image_array, current_record)

        # Copy essential metadata
        self._copy_source_metadata(anonymized_ds, source_dataset, current_record)

        # Handle anonymization
        self._apply_anonymization(anonymized_ds, source_dataset, new_patient_name,
                                  current_record['AnonPatientID'], int(current_record['DateShiftDays']))

        # Set required conformance attributes
        self._set_conformance_attributes(anonymized_ds, source_dataset)

        # Describe what was done to the dataset
        self._set_deidentification_attributes(anonymized_ds)

        # Compress and set pixel data
        self._set_compressed_pixel_data(anonymized_ds, image_array)

        # Embed git provenance as a private tag so the on-disk .dcm and the
        # returned Dataset carry the same build identity.
        apply_provenance_tag(anonymized_ds)

        # Create and save file
        self._create_and_save_dicom_file(anonymized_ds, output_path)

        return anonymized_ds

    def _create_base_dicom_dataset(self, image_array: np.ndarray, current_record: dict) -> pydicom.Dataset:
        """Create base DICOM dataset with image dimensions and basic attributes."""
        ds = pydicom.Dataset()

        # Set image dimensions
        if len(image_array.shape) == 4:  # Multi-frame format
            frames, height, width, channels = image_array.shape
            ds.NumberOfFrames = frames
            ds.SamplesPerPixel = channels
        elif len(image_array.shape) == 3:  # Multi-frame grayscale
            frames, height, width = image_array.shape
            ds.NumberOfFrames = frames
            ds.SamplesPerPixel = 1
        elif len(image_array.shape) == 2:  # Single frame grayscale
            height, width = image_array.shape
            ds.SamplesPerPixel = 1

        ds.Rows = height
        ds.Columns = width
        ds.Modality = 'US'

        # Set photometric interpretation based on the number of channels
        if ds.SamplesPerPixel == 1:
            ds.PhotometricInterpretation = "MONOCHROME2"
        elif ds.SamplesPerPixel == 3:
            ds.PhotometricInterpretation = "YBR_FULL_422" # For JPEG compressed images

        self._copy_spacing_info(ds, current_record)

        return ds

    def _copy_spacing_info(self, ds: pydicom.Dataset, current_record: dict) -> None:
        """
        Carry the ultrasound region calibration over and set the conventional
        PixelSpacing tag for DICOM readers that don't support ultrasound regions.
        """
        source_dataset = current_record.DICOMDataset

        # Rebuild each region item from its standard calibration attributes.
        regions = getattr(source_dataset, 'SequenceOfUltrasoundRegions', None)
        if regions is not None and len(regions) > 0:
            ds.SequenceOfUltrasoundRegions = pydicom.Sequence(
                [self._rebuild_region_item(item) for item in regions])

        # Copy spacing to conventional PixelSpacing tag
        delta_x = current_record['PhysicalDeltaX']
        delta_y = current_record['PhysicalDeltaY']
        if delta_x is not None and delta_y is not None:
            delta_x_mm = float(delta_x) * 10
            delta_y_mm = float(delta_y) * 10
            ds.PixelSpacing = [f"{delta_x_mm:.14f}", f"{delta_y_mm:.14f}"]

    def _rebuild_region_item(self, item: pydicom.Dataset) -> pydicom.Dataset:
        """Return a new region item holding only the REGION_ALLOWLIST attributes of ``item``."""
        rebuilt = pydicom.Dataset()
        for keyword in self.REGION_ALLOWLIST:
            if keyword in item:
                rebuilt.add(copy.deepcopy(item[keyword]))
        return rebuilt

    def _copy_source_metadata(self, ds: pydicom.Dataset, source_ds: pydicom.Dataset, current_record) -> None:
        """Copy metadata from source dataset.

        TransducerData is reduced to its leading model segment (e.g. "SC6-1s")
        so vendor serial numbers are not carried over via 0018,5010. TransducerType
        and ManufacturerModelName are preserved verbatim — downstream tooling and
        Butterfly Network device routing depend on those exact values.
        SeriesNumber is copied from the source so the instances of one series
        agree on it, and written empty when the source has none.
        """
        for tag in self.DICOM_TAGS_TO_COPY:
            if hasattr(source_ds, tag):
                setattr(ds, tag, getattr(source_ds, tag))
        if 'SeriesNumber' not in ds:
            ds.SeriesNumber = ''

        for tag in self.DICOM_TAGS_PRESERVE_OR_BLANK:
            value = getattr(source_ds, tag, '') or ''
            if tag == 'TransducerData':
                value = self._first_transducer_segment(value)
            setattr(ds, tag, value)

        # Handle UIDs
        self._copy_and_generate_uids(ds, source_ds, current_record)

    def _copy_and_generate_uids(self, ds: pydicom.Dataset, source_ds: pydicom.Dataset, current_record) -> None:
        """Copy the SOP class and take every instance UID from the keys.csv row.

        SOPClassUID identifies the registered DICOM object type (e.g. Ultrasound
        Multi-frame Image Storage) and is copied verbatim; scan_directory only
        lists supported classes, so a missing one is a programming error.
        Study, Series, SOP Instance and Frame of Reference UIDs are the values
        the key store assigned to this instance at scan time.
        """
        sop_class_uid = getattr(source_ds, 'SOPClassUID', None)
        if not sop_class_uid:
            raise ValueError('Source dataset has no SOPClassUID')
        ds.SOPClassUID = sop_class_uid

        ds.SOPInstanceUID = current_record['AnonSOPInstanceUID']
        ds.SeriesInstanceUID = current_record['AnonSeriesUID']
        ds.StudyInstanceUID = current_record['AnonStudyUID']
        ds.FrameOfReferenceUID = current_record['AnonFrameOfReferenceUID']

    def _apply_anonymization(self, ds: pydicom.Dataset, source_ds: pydicom.Dataset,
                             new_patient_name: str, new_patient_id: str, date_shift_days: int) -> None:
        """Set the patient module and shift the dates.

        Args:
            ds: dataset being built
            source_ds: source dataset (read for PatientAge derivation and dates)
            new_patient_name: PatientName to write; ``new_patient_id`` when empty
            new_patient_id: AnonPatientID from keys.csv (required)
            date_shift_days: DateShiftDays from keys.csv
        """
        if not new_patient_id:
            raise ValueError('new_patient_id is required')

        ds.PatientName = new_patient_name if new_patient_name else new_patient_id
        ds.PatientID = new_patient_id
        ds.PatientBirthDate = ""
        ds.ReferringPhysicianName = ""
        ds.AccessionNumber = ""
        ds.StudyID = ""

        # HIPAA Safe Harbor: aggregate ages >89 years to "090Y". When the
        # source has no PatientAge, derive it from StudyDate - PatientBirthDate
        # so the demographic survives de-id when only PatientBirthDate was set.
        if hasattr(ds, 'PatientAge') and ds.PatientAge:
            ds.PatientAge = self._cap_patient_age(ds.PatientAge)
        else:
            computed = self._compute_patient_age_from_birthdate(source_ds)
            if computed is not None:
                ds.PatientAge = computed

        self._apply_date_shifting(ds, source_ds, date_shift_days)

    def _cap_patient_age(self, age_str) -> str:
        """Cap PatientAge at '090Y' for ages >= 90 years (HIPAA Safe Harbor).

        DICOM AS format is 'nnnX' where nnn is three digits and X is one of
        D (days), W (weeks), M (months), or Y (years). Only year values are
        capped; D/W/M values and empty strings pass through unchanged.
        Malformed values are logged at ERROR and passed through so source
        data is never silently rewritten.
        """
        if not isinstance(age_str, str) or age_str == '':
            return age_str

        is_valid_as = (
            len(age_str) == 4
            and age_str[-1] in ('D', 'W', 'M', 'Y')
            and age_str[:3].isdigit()
        )
        if not is_valid_as:
            logging.error(
                f"Invalid PatientAge value {age_str!r}; "
                "expected DICOM AS format 'nnnX' with X in (D, W, M, Y), e.g. '045Y'"
            )
            return age_str

        if age_str[-1] != 'Y':
            return age_str

        years = int(age_str[:3])
        return '090Y' if years >= 90 else age_str

    def _compute_patient_age_from_birthdate(self, source_ds: pydicom.Dataset) -> Optional[str]:
        """Compute DICOM AS PatientAge from source StudyDate - PatientBirthDate.

        Used only when the source dataset has no PatientAge — trusts the source
        value when present. Returns ``None`` (and logs a warning) if the math
        can't be performed: missing/empty dates, malformed dates, or a birth
        date after the study date. Output units are picked by the magnitude of
        the delta to keep the value clinically meaningful:

        * ``< 31 days``  -> ``nnnD``
        * ``< 365 days`` -> ``nnnM`` using 30.4375 days/month
        * else           -> ``nnnY`` using 365.25 days/year, capped at ``090Y``
          per HIPAA Safe Harbor §164.514(b)(2)(i)(C).
        """
        study_date_str = getattr(source_ds, 'StudyDate', None)
        birth_date_str = getattr(source_ds, 'PatientBirthDate', None)
        if not study_date_str or not birth_date_str:
            logging.warning(
                "PatientAge not set: source has no usable PatientBirthDate+StudyDate"
            )
            return None

        try:
            study_date = datetime.datetime.strptime(str(study_date_str), "%Y%m%d")
            birth_date = datetime.datetime.strptime(str(birth_date_str), "%Y%m%d")
        except ValueError:
            logging.warning(
                f"PatientAge not set: could not parse PatientBirthDate={birth_date_str!r} "
                f"or StudyDate={study_date_str!r} as %Y%m%d"
            )
            return None

        if study_date < birth_date:
            logging.warning(
                "PatientAge not set: PatientBirthDate is after StudyDate "
                f"(birth={birth_date_str}, study={study_date_str})"
            )
            return None

        delta_days = (study_date - birth_date).days
        if delta_days < 31:
            return f"{delta_days:03d}D"
        if delta_days < 365:
            months = int(delta_days // 30.4375)
            return f"{months:03d}M"
        # Calendar-year math avoids the int(365 // 365.25) == 0 boundary bug
        # that would report a patient exactly one year old as "000Y".
        years = study_date.year - birth_date.year
        if (study_date.month, study_date.day) < (birth_date.month, birth_date.day):
            years -= 1
        return '090Y' if years >= 90 else f"{years:03d}Y"

    def _shift_date(self, date_value, offset: int) -> str:
        """Return a DA value moved forward by ``offset`` days, or '' when it cannot be parsed."""
        text = str(date_value) if date_value is not None else ''
        if not text:
            return ''
        if len(text) != 8 or not text.isdigit():
            logging.warning("Date value is not in YYYYMMDD form; writing an empty date")
            return ''
        try:
            parsed = datetime.datetime.strptime(text, "%Y%m%d")
        except ValueError:
            logging.warning("Date value is not a calendar date; writing an empty date")
            return ''
        return (parsed + datetime.timedelta(days=offset)).strftime("%Y%m%d")

    def _apply_date_shifting(self, ds: pydicom.Dataset, source_ds: pydicom.Dataset,
                             date_shift_days: int) -> None:
        """Move every date forward by the patient's offset; copy times unchanged.

        StudyDate, StudyTime, ContentDate and ContentTime are always written
        (empty when the source has no parseable value). SeriesDate and
        SeriesTime are written only when the source has a usable value.
        """
        if (isinstance(date_shift_days, bool) or not isinstance(date_shift_days, int)
                or not 1 <= date_shift_days <= 1095):
            raise ValueError('date_shift_days must be an integer from 1 to 1095')

        ds.StudyDate = self._shift_date(source_ds.get('StudyDate'), date_shift_days)
        ds.StudyTime = str(source_ds.get('StudyTime', '') or '')
        ds.ContentDate = self._shift_date(source_ds.get('ContentDate'), date_shift_days)
        ds.ContentTime = str(source_ds.get('ContentTime', '') or '')

        series_date = self._shift_date(source_ds.get('SeriesDate'), date_shift_days)
        if series_date:
            ds.SeriesDate = series_date
        series_time = str(source_ds.get('SeriesTime', '') or '')
        if series_time:
            ds.SeriesTime = series_time

    def _set_deidentification_attributes(self, ds: pydicom.Dataset) -> None:
        """Record what was done to the dataset (PS3.15 E.1.1 attributes, text only)."""
        ds.PatientIdentityRemoved = 'YES'
        ds.LongitudinalTemporalInformationModified = 'MODIFIED'
        ds.DeidentificationMethod = self._deidentification_method_values()

    def _deidentification_method_values(self) -> List[str]:
        tool = f'SlicerUltrasound AnonymizeUltrasound {self._git_sha_short()}'
        return [tool, *self.DEIDENTIFICATION_METHOD]

    @staticmethod
    def _git_sha_short() -> str:
        return str(get_provenance().get('git_sha', 'unknown'))[:7]

    def _set_conformance_attributes(self, ds: pydicom.Dataset, source_ds: pydicom.Dataset) -> None:
        """Set required DICOM conformance attributes."""
        # Conditional elements: provide empty defaults if unknown.
        if not hasattr(ds, 'Laterality'):
            ds.Laterality = ''
        if not hasattr(ds, 'InstanceNumber'):
            ds.InstanceNumber = 1
        if not hasattr(ds, 'PatientOrientation'):
            ds.PatientOrientation = ''
        if not hasattr(ds, "ImageType"):
            ds.ImageType = r"ORIGINAL\PRIMARY\IMAGE"

        # Multi-frame specific attributes
        if hasattr(ds, 'NumberOfFrames') and int(ds.NumberOfFrames) > 1:
            ds.FrameTime = getattr(source_ds, 'FrameTime', 0.1)
            if hasattr(source_ds, 'FrameIncrementPointer'):
                ds.FrameIncrementPointer = source_ds.FrameIncrementPointer
            else:
                ds.FrameIncrementPointer = pydicom.tag.Tag(0x0018, 0x1063)

        # For color images, set PlanarConfiguration (Type 1C)
        if hasattr(ds, 'SamplesPerPixel') and ds.SamplesPerPixel == 3:
            ds.PlanarConfiguration = getattr(source_ds, 'PlanarConfiguration', 0)

    def _set_compressed_pixel_data(self, ds: pydicom.Dataset, image_array: np.ndarray) -> None:
        """Compress image frames and set pixel data."""
        compressed_frames = []
        for frame in image_array:
            compressed_frame = self._compress_frame_to_jpeg(frame)
            compressed_frames.append(compressed_frame)

        ds.PixelData = pydicom.encaps.encapsulate(compressed_frames)
        ds['PixelData'].VR = 'OB'
        ds['PixelData'].is_undefined_length = True
        ds.LossyImageCompression = '01'
        ds.LossyImageCompressionMethod = 'ISO_10918_1'

    def _compress_frame_to_jpeg(self, frame: np.ndarray, quality: int = 95) -> bytes:
        """Compress a single frame using JPEG compression."""

        # If frame is 2D, add a channel dimension because PIL doesn't support 2D images
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, axis=-1)

        # Convert to PIL Image grayscale or RGB
        if frame.shape[2] == 1:
            image = Image.fromarray(frame[:, :, 0]).convert("L")
        else:
            image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")

        # Compress to JPEG
        with io.BytesIO() as output:
            image.save(output, format="JPEG", quality=quality)
            return output.getvalue()

    def _create_and_save_dicom_file(self, ds: pydicom.Dataset, output_filepath: str) -> None:
        """Create file metadata and save DICOM file."""
        # Create file meta information
        meta = FileMetaDataset()
        meta.FileMetaInformationGroupLength = 0
        meta.FileMetaInformationVersion = b'\x00\x01'
        meta.MediaStorageSOPClassUID = ds.SOPClassUID
        meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        meta.ImplementationClassUID = self.IMPLEMENTATION_CLASS_UID
        meta.ImplementationVersionName = f"SLICERUS_{self._git_sha_short()}"
        meta.TransferSyntaxUID = pydicom.uid.JPEGBaseline8Bit

        # Create file dataset
        file_ds = pydicom.dataset.FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        for elem in ds:
            file_ds.add(elem)

        # Set encoding attributes
        file_ds.is_implicit_VR = False
        file_ds.is_little_endian = True

        # Save file to output path. Create the directories if they don't exist.
        directory = os.path.dirname(output_filepath)
        os.makedirs(directory, exist_ok=True)
        file_ds.save_as(output_filepath)
        logging.info(f"DICOM generated successfully: {output_filepath}")
