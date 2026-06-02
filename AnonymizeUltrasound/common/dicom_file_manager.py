import os
import hashlib
import pydicom
import pandas as pd
import logging
from typing import Optional, List
import numpy as np
from PIL import Image
import io
import random
import datetime
import json
from pydicom.dataset import FileMetaDataset
from pydicom.encaps import generate_pixel_data_frame, decode_data_sequence
from .uid_remap import remap_uid
from .version_info import apply_provenance_tag


class DicomFileManager:
    """
    Shared DICOM file management functionality for ultrasound modules.

    This class provides common functionality for managing DICOM files across different
    ultrasound processing modules. It handles:

    - Scanning directories for DICOM files
    - Parsing DICOM metadata and creating structured dataframes
    - Managing file loading and temporary directories
    - Generating anonymized filenames and patient IDs
    - Progress tracking for batch processing

    The class maintains a pandas DataFrame (dicom_df) containing metadata for all
    discovered DICOM files, including file paths, patient information, instance UIDs,
    physical spacing data, and anonymized filenames for export.

    Attributes:
        dicom_df (pd.DataFrame): DataFrame containing DICOM file metadata
        next_index (int): Index of next file to process
        current_index (int): Index of currently loaded file
        _temp_directories (List[str]): List of temporary directories for cleanup
    """

    # Define allowed DICOM file extensions (case-insensitive)
    DICOM_EXTENSIONS = {'.dcm', '.dicom'}

    # Tags copied from source only if present (allowlist semantics).
    DICOM_TAGS_TO_COPY = [
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PatientAge",
        "PatientSex",
        "PixelRepresentation",
        "SeriesNumber",
        "StationName",
        "StudyDate",
        "StudyDescription",
        "StudyTime",
        "Manufacturer",
    ]

    # Tags always written to the de-id dataset: source value if present, else "".
    # Downstream consumers rely on these tags being present on every anonymized DICOM.
    DICOM_TAGS_PRESERVE_OR_BLANK = [
        "TransducerData",
        "TransducerType",
        "ManufacturerModelName",
    ]

    # Expected columns in the DICOM files dataframe
    DICOM_DATAFRAME_COLUMNS = [
        'InputPath',
        'OutputPath',
        'AnonFilename',
        'PatientUID',
        'StudyUID',
        'SeriesUID',
        'InstanceUID',
        'AnonStudyUID',
        'AnonSeriesUID',
        'AnonSOPInstanceUID',
        'FrameOfReferenceUID',
        'AnonFrameOfReferenceUID',
        'PhysicalDeltaX',
        'PhysicalDeltaY',
        'ContentDate',
        'ContentTime',
        'Patch',
        'TransducerModel',
        'DICOMDataset',
    ]

    PATIENT_ID_HASH_LENGTH = 10
    INSTANCE_ID_HASH_LENGTH = 8
    DEFAULT_CONTENT_DATE = '19000101'
    DEFAULT_CONTENT_TIME = ''

    def __init__(self):
        self.dicom_df = None
        self.input_folder = None
        self.next_index = 0
        self.current_index = 0
        # Maps (source StudyInstanceUID, source FrameOfReferenceUID) → run-local
        # anonymized FOR UID. Populated by _populate_anon_for_column or seeded
        # from a prior export's keys.csv via _seed_for_map_from_keys_csv.
        self._for_map: dict[tuple[str, str], str] = {}

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
                       hash_patient_id: bool = True,
                       seed_keys_csv: Optional[str] = None) -> int:
        """
        Scan directory for DICOM files and create dataframe

        Args:
            input_folder: Directory to scan
            skip_single_frame: Skip single frame DICOM files (AnonymizeUltrasound)
            hash_patient_id: If True, hash the patient ID
            seed_keys_csv: Optional path to a prior export's keys.csv. When
                provided, the (StudyUID, source FOR) → AnonFrameOfReferenceUID
                mapping from that file pre-populates self._for_map so a resumed
                export reuses the same anon FOR UIDs as the prior partial run.

        Returns:
            Number of DICOM files found
        """
        dicom_data = []

        # Reset _for_map before seeding so a long-lived DicomFileManager (e.g.
        # the Slicer logic singleton) doesn't carry mappings between unrelated
        # scans — that would re-use anonymized FOR UIDs across exports and
        # break run-local unlinkability. Seeding uses setdefault, so the
        # reset-then-seed order preserves the resume contract.
        self._for_map = {}
        self._seed_for_map_from_keys_csv(seed_keys_csv)

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

                dicom_info = self._extract_dicom_info(file_path, input_folder, skip_single_frame, hash_patient_id)
                if dicom_info:
                    dicom_data.append(dicom_info)

        self._create_dataframe(dicom_data)

        return len(self.dicom_df) if self.dicom_df is not None else 0

    def _seed_for_map_from_keys_csv(self, path: Optional[str]) -> None:
        """Pre-populate self._for_map from an existing export's keys.csv.

        Enables resume safety: skipped + newly written files in a resumed export
        share one (StudyUID, source FOR) → AnonFrameOfReferenceUID mapping.
        Silently no-ops if path is None/empty/missing or the keys.csv lacks the
        FrameOfReferenceUID / AnonFrameOfReferenceUID columns (older export
        pre-dating this change). Existing entries in self._for_map are not
        overwritten — first-seen wins (use setdefault).
        """
        if not path or not os.path.exists(path):
            return
        prior = pd.read_csv(path, dtype=str).fillna('')
        required = {'StudyUID', 'FrameOfReferenceUID', 'AnonFrameOfReferenceUID'}
        if not required.issubset(prior.columns):
            return
        for _, row in prior.iterrows():
            study = row['StudyUID']
            source_for = row['FrameOfReferenceUID']
            anon = row['AnonFrameOfReferenceUID']
            if not (study and anon):
                continue
            if source_for:
                key = (study, source_for)
            else:
                key = (study, f":SeriesUID:{row.get('SeriesUID', '')}")
            self._for_map.setdefault(key, anon)

    def get_number_of_instances(self) -> int:
        """Get number of instances in dataframe"""
        return len(self.dicom_df) if self.dicom_df is not None else 0

    def build_csv_dataframe(self, input_folder: Optional[str]) -> Optional[pd.DataFrame]:
        """Return a copy of dicom_df ready to serialize to keys.csv.

        Drops the DICOMDataset column (binary payload, not CSV-serializable) and
        rewrites InputPath from absolute filepath to relative-to-input_folder.
        Leaves OutputPath and all other columns untouched.

        Args:
            input_folder: Root to compute InputPath relative to. If falsy,
                InputPath is left unchanged (absolute).
        Returns:
            DataFrame ready for to_csv(), or None if dicom_df is None or empty.
        """
        if self.dicom_df is None or self.dicom_df.empty:
            return None
        df = self.dicom_df.drop(columns=['DICOMDataset'], inplace=False, errors='ignore')
        if input_folder:
            normed = os.path.normpath(input_folder)
            df['InputPath'] = df['InputPath'].apply(
                lambda p: os.path.relpath(p, normed) if isinstance(p, str) and p else p
            )
        return df

    def _extract_dicom_info(self, file_path: str, input_folder: str, skip_single_frame: bool, hash_patient_id: bool = True) -> Optional[dict]:
        """Extract DICOM information from file

        Reads a DICOM file and extracts relevant metadata for ultrasound processing.
        Validates that the file is an ultrasound modality and optionally skips
        single-frame files based on the skip_single_frame parameter.

        Args:
            file_path: Path to the DICOM file to process
            input_folder: Path to the input folder
            skip_single_frame: If True, skip files with less than 2 frames
            hash_patient_id: If True, hash the patient ID

        Returns:
            dict: Dictionary containing extracted DICOM metadata
            None: If file cannot be read, is not ultrasound, or doesn't meet frame requirements
        """
        try:
            dicom_ds = pydicom.dcmread(file_path, stop_before_pixels=True)

            # Skip non-ultrasound modalities
            if dicom_ds.get("Modality", "") != "US":
                logging.info(f"Skipping non-ultrasound file: {file_path}")
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

            physical_delta_x, physical_delta_y = self._extract_spacing_info(dicom_ds)
            anon_filename = self._generate_filename_from_dicom(dicom_ds, hash_patient_id)
            content_date = getattr(dicom_ds, 'ContentDate', '19000101')
            content_time = getattr(dicom_ds, 'ContentTime', '000000')
            to_patch = physical_delta_x is None or physical_delta_y is None
            transducer_model = self.get_transducer_model(
                dicom_ds.get('TransducerData', ''),
                manufacturer=dicom_ds.get('Manufacturer', ''),
                manufacturer_model_name=dicom_ds.get('ManufacturerModelName', ''),
            )

            # Calculate relative path from input folder. Replace the filename with the anonymized filename.
            output_path = os.path.relpath(file_path, input_folder).replace(os.path.basename(file_path), anon_filename)

            return {
                'InputPath': file_path,
                'OutputPath': output_path,
                'AnonFilename': anon_filename,
                'PatientUID': patient_uid,
                'StudyUID': study_uid,
                'SeriesUID': series_uid,
                'InstanceUID': instance_uid,
                'AnonStudyUID': remap_uid(str(study_uid)) if study_uid else '',
                'AnonSeriesUID': remap_uid(str(series_uid)) if series_uid else '',
                'AnonSOPInstanceUID': remap_uid(str(instance_uid)) if instance_uid else '',
                'FrameOfReferenceUID': getattr(dicom_ds, 'FrameOfReferenceUID', ''),
                'PhysicalDeltaX': physical_delta_x,
                'PhysicalDeltaY': physical_delta_y,
                'ContentDate': content_date,
                'ContentTime': content_time,
                'Patch': to_patch,
                'TransducerModel': transducer_model,
                'DICOMDataset': dicom_ds
            }

        except Exception as e:
            logging.error(f"Failed to read DICOM file {file_path}: {e}")
            return None

    def generate_output_filepath(
        self, output_directory: str, output_path: str, preserve_directory_structure: bool) -> str:
        """
        Generate output filepath from relative path and output directory.
        If preserve_directory_structure is True, the output filepath will be the same as the relative path.
        """
        if preserve_directory_structure:
            return os.path.join(output_directory, output_path)
        else:
            filename = os.path.basename(output_path)
            return os.path.join(output_directory, filename)

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

    def _generate_filename_from_dicom(self, dicom_ds, hash_patient_id: bool = True):
        """
        Generate an anonymized filename from a DICOM dataset.

        Creates a standardized filename format for DICOM files using hashed identifiers
        to protect patient privacy while maintaining uniqueness.

        Args:
            dicom_ds: DICOM dataset containing patient and instance information
            hash_patient_id (bool): Whether to hash the patient ID (default: True)
                                If True, creates a 10-digit hash of the patient ID
                                If False, uses the original patient ID

        Returns:
            str: Generated filename in format "XXXXXXXXXX_YYYYYYYY.dcm" where:
                 - X is a 10-digit identifier (hashed patient ID or original)
                 - Y is an 8-digit hash of the SOP Instance UID
                 Returns empty string if required DICOM fields are missing

        Note:
            The filename format ensures uniqueness while anonymizing patient data.
            Both patient and instance identifiers are zero-padded to fixed lengths.
        """
        patientUID = dicom_ds.PatientID
        instanceUID = dicom_ds.SOPInstanceUID

        if patientUID is None or patientUID == "":
            logging.error("PatientID not found in DICOM header dict")
            return ""

        if instanceUID is None or instanceUID == "":
            logging.error("SOPInstanceUID not found in DICOM header dict")
            return ""

        if hash_patient_id:
            hash_object = hashlib.sha256()
            hash_object.update(str(patientUID).encode())
            patientId = int(hash_object.hexdigest(), 16) % 10**self.PATIENT_ID_HASH_LENGTH
        else:
            patientId = patientUID

        hash_object_instance = hashlib.sha256()
        hash_object_instance.update(str(instanceUID).encode())
        instanceId = int(hash_object_instance.hexdigest(), 16) % 10**self.INSTANCE_ID_HASH_LENGTH

        # Add trailing zeros
        patientId = str(patientId).zfill(self.PATIENT_ID_HASH_LENGTH)
        instanceId = str(instanceId).zfill(self.INSTANCE_ID_HASH_LENGTH)

        return f"{patientId}_{instanceId}.dcm"

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

        # Add series numbers
        self.dicom_df['SeriesNumber'] = (self.dicom_df
                                         .groupby(['PatientUID', 'StudyUID'])
                                         .cumcount() + 1)

        # Fill missing spacing information using forward/backward fill
        spacing_cols = ['PhysicalDeltaX', 'PhysicalDeltaY']
        self.dicom_df[spacing_cols] = (self.dicom_df
                                       .groupby('StudyUID')[spacing_cols]
                                       .transform(lambda x: x.ffill().bfill()))

        # Populate AnonFrameOfReferenceUID per (StudyUID, source FOR) group using
        # the run-local _for_map (possibly seeded from a prior keys.csv).
        self._populate_anon_for_column()

        self.next_index = 0

    def _populate_anon_for_column(self) -> None:
        """Assign AnonFrameOfReferenceUID per (StudyUID, source FrameOfReferenceUID) group.

        Uses self._for_map as the source of truth. Reuses entries already in the
        map (populated by _seed_for_map_from_keys_csv on resume). Generates fresh
        pydicom.uid.generate_uid(prefix=None) values for new groups, yielding
        run-local 2.25-arc UIDs.

        For rows whose source dataset lacks a FrameOfReferenceUID, the mapping
        key falls back to (StudyUID, ":SeriesUID:" + SeriesUID) so coordinate-less
        series within one study are NOT falsely linked to one another. Real DICOM
        UIDs contain only digits and dots, so the ":SeriesUID:" infix cannot
        collide with a real-FOR-derived key.
        """
        if self.dicom_df is None or self.dicom_df.empty:
            return

        def assign(row):
            study = str(row['StudyUID'])
            source_for = str(row.get('FrameOfReferenceUID') or '')
            if source_for:
                key = (study, source_for)
            else:
                key = (study, f":SeriesUID:{row['SeriesUID']}")
            if key not in self._for_map:
                self._for_map[key] = pydicom.uid.generate_uid(prefix=None)
            return self._for_map[key]

        self.dicom_df['AnonFrameOfReferenceUID'] = self.dicom_df.apply(assign, axis=1)

    def update_progress_from_output(self, output_directory: str, preserve_directory_structure: bool) -> Optional[int]:
        """Update progress based on existing output files

        This method checks which anonymized DICOM files already exist in the output
        directory and updates the next_index to skip over files that have
        already been processed. This enables resuming processing from where it
        left off in case of interruption.

        Args:
            output_directory: Directory path where anonymized DICOM files are saved
            preserve_directory_structure: If True, the output filepath will be the same as the relative path.
        Returns:
            int: Number of files already processed (0 if none processed)
            None: If all files have been processed or no dataframe exists
        """
        if self.dicom_df is None:
            return None

        # Create full paths vectorized
        output_paths = self.dicom_df['OutputPath'].apply(
            lambda x: self.generate_output_filepath(output_directory, x, preserve_directory_structure)
        )

        # Check existence vectorized
        exists_mask = output_paths.apply(os.path.exists)

        # If all files exist, return None indicating all files have been processed
        if exists_mask.all():
            return None

        # Find first False (first non-existing file)
        first_missing = exists_mask.idxmin()
        num_done = exists_mask[:first_missing].sum()

        self.next_index = num_done
        return num_done

    def _get_file_for_instance_uid(self, instance_uid: str) -> Optional[str]:
        """Get file path for given instance UID"""
        if self.dicom_df is None:
            return None

        matching_rows = self.dicom_df[self.dicom_df['InstanceUID'] == instance_uid]
        if not matching_rows.empty:
            return matching_rows.iloc[0]['InputPath']

        return None

    def dicom_header_to_dict(self, ds, parent=None):
        """
        Convert DICOM dataset to dictionary format.

        Recursively processes DICOM dataset elements, handling sequence (SQ) elements
        by creating nested dictionaries for each sequence item. Excludes PixelData
        to avoid memory issues with large image data.

        Args:
            ds: DICOM dataset to convert
            parent: Parent dictionary to populate (used for recursion)

        Returns:
            dict: Dictionary representation of DICOM dataset with nested structure
                 for sequence elements
        """
        if parent is None:
            parent = {}
        for elem in ds:
            # Skip PixelData to avoid memory issues with large image data
            if elem.name == "Pixel Data":
                continue

            if elem.VR == "SQ":
                parent[elem.name] = []
                for item in elem:
                    child = {}
                    self.dicom_header_to_dict(item, child)
                    parent[elem.name].append(child)
            else:
                parent[elem.name] = elem.value
        return parent

    def increment_dicom_index(self, output_directory: Optional[str] = None,
                              continue_progress: bool = False, preserve_directory_structure: bool = True) -> bool:
        """
        Increment the DICOM index to the next file to be processed.

        This method advances the internal index counter and optionally skips files that already
        exist in the output directory when continuing from a previous processing session.

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

        Note:
            This method modifies the internal next_index counter. When continue_progress
            is True, it will skip over files that already exist in the output directory.
        """
        if self.dicom_df is None:
            return False

        # Increment the index to the next file to be processed.
        self.next_index += 1

        # If continue_progress is True, skip files that already exist in output.
        if continue_progress and output_directory:
            while self.next_index < len(self.dicom_df):
                row = self.dicom_df.iloc[self.next_index]
                output_path = self.generate_output_filepath(
                    output_directory, row['OutputPath'], preserve_directory_structure)

                if not os.path.exists(output_path):
                    break

                self.next_index += 1

        return self.next_index < len(self.dicom_df)

    def save_anonymized_dicom(self, image_array: np.ndarray, output_path: str,
                            new_patient_name: str = '', new_patient_id: str = '', labels: Optional[List[str]] = None) -> Optional[pydicom.Dataset]:
        """
        Save image array as anonymized DICOM file and return the in-memory anon Dataset.

        The return value lets callers feed the same anonymized Dataset into
        `save_anonymized_dicom_header`, so the JSON sidecar serializes from
        the same source of truth as the saved .dcm — preventing source UIDs
        and untrimmed TransducerData from leaking into the header file.

        Args:
            image_array: Numpy array containing image data (frames, height, width, channels)
            output_path: Full path where DICOM file should be saved
            new_patient_name: New patient name for anonymization
            new_patient_id: New patient ID for anonymization
            labels: List of labels to add to the DICOM file

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

        # Set annotation labels as the series description
        if labels:
            anonymized_ds.SeriesDescription = " ".join(labels)

        # Copy essential metadata
        self._copy_source_metadata(anonymized_ds, source_dataset, output_path)

        # Handle anonymization
        self._apply_anonymization(anonymized_ds, source_dataset, new_patient_name, new_patient_id)

        # Set required conformance attributes
        self._set_conformance_attributes(anonymized_ds, source_dataset)

        # Compress and set pixel data
        self._set_compressed_pixel_data(anonymized_ds, image_array)

        # Embed git provenance as a private tag so the on-disk .dcm, the
        # returned Dataset, and any downstream header sidecar all carry the
        # same build identity.
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
        Copy spacing to conventional PixelSpacing tag for DICOM readers that don't support ultrasound regions.
        """
        source_dataset = current_record.DICOMDataset

        # Copy SequenceOfUltrasoundRegions if available
        if hasattr(source_dataset, "SequenceOfUltrasoundRegions") and len(source_dataset.SequenceOfUltrasoundRegions) > 0:
            ds.SequenceOfUltrasoundRegions = source_dataset.SequenceOfUltrasoundRegions

        # Copy spacing to conventional PixelSpacing tag
        delta_x = current_record['PhysicalDeltaX']
        delta_y = current_record['PhysicalDeltaY']
        if delta_x is not None and delta_y is not None:
            delta_x_mm = float(delta_x) * 10
            delta_y_mm = float(delta_y) * 10
            ds.PixelSpacing = [f"{delta_x_mm:.14f}", f"{delta_y_mm:.14f}"]

    def _copy_source_metadata(self, ds: pydicom.Dataset, source_ds: pydicom.Dataset, output_path: str) -> None:
        """Copy metadata from source dataset.

        TransducerData is reduced to its leading model segment (e.g. "SC6-1s")
        so vendor serial numbers do not leak via 0018,5010. TransducerType and
        ManufacturerModelName are preserved verbatim — downstream tooling and
        Butterfly Network device routing depend on those exact values.
        """
        for tag in self.DICOM_TAGS_TO_COPY:
            if hasattr(source_ds, tag):
                setattr(ds, tag, getattr(source_ds, tag))

        for tag in self.DICOM_TAGS_PRESERVE_OR_BLANK:
            value = getattr(source_ds, tag, '') or ''
            if tag == 'TransducerData':
                value = self._first_transducer_segment(value)
            setattr(ds, tag, value)

        # Handle UIDs
        self._copy_and_generate_uids(ds, source_ds, output_path)

        # Get series number from dataframe
        ds.SeriesNumber = self._get_series_number_for_current_instance()

    def _copy_and_generate_uids(self, ds: pydicom.Dataset, source_ds: pydicom.Dataset, output_path: str) -> None:
        """Remap instance UIDs deterministically; assign a run-local per-study FOR UID.

        SOPClassUID identifies the registered DICOM object type (e.g. Ultrasound
        Multi-frame Image Storage) and must NOT be remapped. Study, Series, and
        SOP Instance UIDs are routed through remap_uid so the de-id output sits
        in the 2.25 OID arc and original UIDs do not leak.

        FrameOfReferenceUID is assigned per (source StudyInstanceUID, source
        FrameOfReferenceUID) group and is generated run-locally — two separate
        de-id runs over the same source data produce different output FORs, so
        recipients cannot cluster de-id'd studies across exports. Within a single
        run, two source series in one study that legitimately shared a FOR UID
        get the SAME output FOR UID so downstream multi-series registration /
        volumetric fusion keeps working. Coordinate-less source series within
        one study get distinct output FOR UIDs (no false linkage).

        Resume contract: when scan_directory is called with seed_keys_csv, the
        prior export's (StudyUID, source FOR) → AnonFrameOfReferenceUID rows
        seed self._for_map, so a resumed export reuses the prior anon FORs for
        already-written files while minting fresh ones for new pairs.

        Three-tier read path for the output FOR UID:
          1. Primary — self.dicom_df.iloc[current_index]['AnonFrameOfReferenceUID']
             (the precomputed column populated by _populate_anon_for_column).
          2. Fallback — self._for_map.get((StudyUID, source FOR or
             ":SeriesUID:<series>" sentinel)). Used by direct callers that
             bypass scan_directory.
          3. Last resort — mint a fresh pydicom.uid.generate_uid(prefix=None)
             and log an error (indicates a bug; should not happen in normal
             scan_directory flows).
        """
        if hasattr(source_ds, 'SOPClassUID') and source_ds.SOPClassUID:
            ds.SOPClassUID = source_ds.SOPClassUID
        else:
            logging.error(f"SOPClassUID not found. Generating new one for {output_path}")
            ds.SOPClassUID = pydicom.uid.generate_uid()

        for attr in ('SOPInstanceUID', 'SeriesInstanceUID', 'StudyInstanceUID'):
            src_val = getattr(source_ds, attr, None)
            if src_val:
                setattr(ds, attr, remap_uid(str(src_val)))
            else:
                logging.error(f"{attr} not found. Generating new one for {output_path}")
                setattr(ds, attr, remap_uid(pydicom.uid.generate_uid()))

        ds.FrameOfReferenceUID = self._resolve_anon_for_uid(source_ds, output_path)

    def _resolve_anon_for_uid(self, source_ds: pydicom.Dataset, output_path: str) -> str:
        """Resolve the anonymized FOR UID via the three-tier read path.

        See _copy_and_generate_uids docstring for the policy. Split out so the
        read logic is exercised in isolation by the plumbing unit tests.
        """
        # Tier 1: dataframe column.
        if (self.dicom_df is not None
                and 0 <= self.current_index < len(self.dicom_df)
                and 'AnonFrameOfReferenceUID' in self.dicom_df.columns):
            candidate = self.dicom_df.iloc[self.current_index]['AnonFrameOfReferenceUID']
            if candidate:
                return str(candidate)

        # Tier 2: in-memory map keyed by source (StudyUID, FOR).
        study_uid = getattr(source_ds, 'StudyInstanceUID', None)
        source_for = getattr(source_ds, 'FrameOfReferenceUID', '') or ''
        if study_uid:
            if source_for:
                key = (str(study_uid), source_for)
            else:
                key = (str(study_uid),
                       f":SeriesUID:{getattr(source_ds, 'SeriesInstanceUID', '')}")
            mapped = self._for_map.get(key)
            if mapped:
                return mapped

        # Tier 3: mint and log error — should not happen in normal flows.
        logging.error(
            f"AnonFrameOfReferenceUID not available; generating fallback for {output_path}"
        )
        return pydicom.uid.generate_uid(prefix=None)

    def _apply_anonymization(self, ds: pydicom.Dataset, source_ds: pydicom.Dataset,
                            new_patient_name: str = "", new_patient_id: str = "") -> None:
        """Apply anonymization including patient info and date shifting."""
        # Anonymize patient information
        ds.PatientName = new_patient_name if new_patient_name else ""
        ds.PatientID = new_patient_id if new_patient_id else ""
        ds.PatientBirthDate = ""
        ds.ReferringPhysicianName = ""
        ds.AccessionNumber = ""

        # HIPAA Safe Harbor: aggregate ages >89 years to "090Y". When the
        # source has no PatientAge, derive it from StudyDate - PatientBirthDate
        # so the demographic survives de-id when only PatientBirthDate was set.
        if hasattr(ds, 'PatientAge') and ds.PatientAge:
            ds.PatientAge = self._cap_patient_age(ds.PatientAge)
        else:
            computed = self._compute_patient_age_from_birthdate(source_ds)
            if computed is not None:
                ds.PatientAge = computed

        # Apply date shifting for anonymization
        self._apply_date_shifting(ds, source_ds)

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

    def _shift_date(self, date_str: str, offset: int) -> str:
        """Shift a single date by the given offset."""
        try:
            date_obj = datetime.datetime.strptime(date_str, "%Y%m%d") + datetime.timedelta(days=offset)
            return date_obj.strftime("%Y%m%d")
        except Exception as e:
            logging.warning(f"Failed to parse date: {date_str}. Using original date. Error: {e}")
            return date_str

    def _apply_date_shifting(self, ds: pydicom.Dataset, source_ds: pydicom.Dataset) -> None:
        """Apply consistent date shifting based on patient ID."""
        patient_id = source_ds.PatientID
        random.seed(patient_id)
        random_offset = random.randint(0, 30)

        # Get dates with defaults
        study_date = getattr(source_ds, 'StudyDate', self.DEFAULT_CONTENT_DATE)
        series_date = getattr(source_ds, 'SeriesDate', self.DEFAULT_CONTENT_DATE)
        content_date = getattr(source_ds, 'ContentDate', self.DEFAULT_CONTENT_DATE)

        # Shift dates
        ds.StudyDate = self._shift_date(study_date, random_offset)
        ds.SeriesDate = self._shift_date(series_date, random_offset)
        ds.ContentDate = self._shift_date(content_date, random_offset)

        # Copy times
        ds.StudyTime = getattr(source_ds, 'StudyTime', self.DEFAULT_CONTENT_TIME)
        ds.SeriesTime = getattr(source_ds, 'SeriesTime', self.DEFAULT_CONTENT_TIME)
        ds.ContentTime = getattr(source_ds, 'ContentTime', self.DEFAULT_CONTENT_TIME)

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
        meta.ImplementationClassUID = pydicom.uid.generate_uid(None)
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

    def _get_series_number_for_current_instance(self) -> str:
        """Get series number for current instance from dataframe."""
        if self.dicom_df is None:
            return '1'

        current_record = self.dicom_df.iloc[self.current_index]
        current_instance_uid = current_record.DICOMDataset.SOPInstanceUID

        matching_rows = self.dicom_df[self.dicom_df['InstanceUID'] == current_instance_uid]
        if not matching_rows.empty:
            return str(matching_rows.iloc[0]['SeriesNumber'])

        return '1'

    def generate_filename_from_dicom_dataset(self, ds: pydicom.Dataset, hash_patient_id: bool = True) -> tuple[str, str, str]:
        """
        Generate a filename from a DICOM header dictionary.
        Optionally, the name will be a hash of the PatientID and the SOP Instance UID.
        The name will consist of two parts:
        X_Y.dcm
        X is generated by hashing the original patient UID to a 10-digit number.
        Y is generated from the DICOM instance UID, but limited to 8 digits

        :param ds: DICOM dataset
        :param hash_patient_id: If True, hash the patient ID
        :returns: tuple (filename, patientId, instanceId)
        """
        patient_id = ds.PatientID
        instance_id = ds.SOPInstanceUID

        if patient_id is None or patient_id == "":
            logging.error("PatientID not found in DICOM header dict")
            return "", "", ""

        if instance_id is None or instance_id == "":
            logging.error("SOPInstanceUID not found in DICOM header dict")
            return "", "", ""

        if hash_patient_id:
            hash_object = hashlib.sha256()
            hash_object.update(str(patient_id).encode())
            patient_id = int(hash_object.hexdigest(), 16) % 10**10
        else:
            patient_id = patient_id

        hash_object_instance_id = hashlib.sha256()
        hash_object_instance_id.update(str(instance_id).encode())
        instance_id = int(hash_object_instance_id.hexdigest(), 16) % 10**8

        # Add trailing zeros
        patient_id = str(patient_id).zfill(self.PATIENT_ID_HASH_LENGTH)
        instance_id = str(instance_id).zfill(self.INSTANCE_ID_HASH_LENGTH)

        return f"{patient_id}_{instance_id}.dcm", patient_id, instance_id

    def save_anonymized_dicom_header(
        self,
        current_dicom_record,
        output_filename: str,
        headers_directory: Optional[str] = None,
        *,
        anonymized_dataset: pydicom.Dataset,
    ) -> Optional[str]:
        """
        Save anonymized DICOM header information as a JSON file.

        Serializes from `anonymized_dataset` — the in-memory Dataset returned
        by `save_anonymized_dicom` — so the JSON sidecar contains the same
        remapped UIDs, run-local FrameOfReferenceUID, and trimmed
        TransducerData as the saved .dcm. Serializing from the source dataset
        would leak the original UIDs and full TransducerData into the sidecar
        next to an anonymized .dcm.

        The patient-name override (using output_filename) and partial
        birth-date scrub (keep year, set month/day to 01-01) are preserved
        so the JSON's PatientName/PatientBirthDate columns stay consistent
        across runs and don't expose the new_patient_id even when the anon
        Dataset's PatientName is set to something else.

        Args:
            current_dicom_record: Current DICOM record from the dataframe. Used
                only for input validation (the dataset comes from
                `anonymized_dataset`); retained so the kwarg contract matches
                the long-standing call sites.
            output_filename: Base filename for the output (drives PatientName
                anonymization).
            headers_directory: Directory path where header JSON files will be
                saved. If None, no header file is created.
            anonymized_dataset: The in-memory pydicom Dataset returned by
                save_anonymized_dicom. Required; passing the source dataset
                would defeat the de-id invariant.

        Returns:
            str: Full path to the saved JSON header file.
            None: If headers_directory is None.
        """
        if current_dicom_record is None:
            raise ValueError("Current DICOM record is required")

        if output_filename is None or output_filename == "":
            raise ValueError("Output filename is required")

        if anonymized_dataset is None:
            raise ValueError("anonymized_dataset is required to avoid source UID leak")

        if headers_directory is None:
            return None

        if not os.path.exists(headers_directory):
            os.makedirs(headers_directory)

        dicom_header_filename = output_filename.replace(".dcm", "_DICOMHeader.json")
        dicom_header_filepath = os.path.join(headers_directory, dicom_header_filename)
        os.makedirs(os.path.dirname(dicom_header_filepath), exist_ok=True)

        with open(dicom_header_filepath, 'w') as outfile:
            anonymized_header = self.dicom_header_to_dict(anonymized_dataset)

            if "Patient's Name" in anonymized_header:
                anonymized_header["Patient's Name"] = output_filename.split(".")[0]

            # Partial birth-date scrub: source DS retains the original
            # PatientBirthDate, so derive the year-only value from there.
            source_birth = getattr(
                current_dicom_record.DICOMDataset, 'PatientBirthDate', ''
            ) or ''
            if "Patient's Birth Date" in anonymized_header and source_birth:
                anonymized_header["Patient's Birth Date"] = str(source_birth)[:4] + "0101"

            json.dump(anonymized_header, outfile, default=self._convert_to_json_compatible)

        return dicom_header_filepath

    def _convert_to_json_compatible(self, obj):
        """
        Convert DICOM-specific data types to JSON-serializable formats.

        This method handles the conversion of pydicom data types that are not
        natively JSON-serializable, ensuring that DICOM header information
        can be properly saved as JSON files.

        Args:
            obj: Object to convert to JSON-compatible format

        Returns:
            Converted object in JSON-compatible format:
            - MultiValue objects are converted to lists
            - PersonName objects are converted to strings
            - Bytes objects are decoded using latin-1 encoding

        Raises:
            TypeError: If the object type is not supported for JSON serialization

        Note:
            This method is used as the 'default' parameter in json.dump() calls
            to handle DICOM-specific data types during JSON serialization.
        """
        if isinstance(obj, pydicom.multival.MultiValue):
            return list(obj)
        if isinstance(obj, pydicom.valuerep.PersonName):
            return str(obj)
        if isinstance(obj, bytes):
            return obj.decode('latin-1')
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    def read_frames_from_dicom(self, dicom_file_path):
        """
        Reads frames from a dicom file and returns a numpy array in the format of [Frames, Height, Width, Channels]. PyTorch convention is NCHW for image batches.
        :param dicom_file_path: path to the dicom file
        :return: numpy array [N,H,W,C]
        """
        ds = pydicom.dcmread(dicom_file_path)
        width = ds.Columns
        height = ds.Rows
        channels = ds.SamplesPerPixel

        try:
            num_frames = ds.NumberOfFrames
        except AttributeError:
            num_frames = 1
            logging.warning(f"Warning: No NumberOfFrames found in {dicom_file_path}, trying to read with num_frames=1")

        output = np.zeros((num_frames, height, width, channels), dtype=np.uint8)

        try:
            logging.info("Trying `dicom.encaps.generate_pixel_data_frame`")
            pixel_data_frames = generate_pixel_data_frame(ds.PixelData)

            for i in range(num_frames):
                frame_item = next(pixel_data_frames)
                image = Image.open(io.BytesIO(frame_item))  # jpeg uncompressed
                frame = np.array(image)
                # If frame is grayscale, add a channel dimension
                if len(frame.shape) == 2:
                    frame = np.expand_dims(frame, axis=2)
                output[i, :, :, :] = frame
        except Exception as e:
            logging.warning("dicom.encaps.generate_pixel_data_frame approach failed: %s", e)
            try:
                logging.info("Fallback to decode_data_sequence approach")
                frame_data = decode_data_sequence(ds.PixelData)

                for i, frame_item in enumerate(frame_data):
                    if i >= num_frames:
                        break
                    image = Image.open(io.BytesIO(frame_item))  # jpeg uncompressed
                    frame = np.array(image)
                    # If frame is grayscale, add a channel dimension
                    if len(frame.shape) == 2:
                        frame = np.expand_dims(frame, axis=2)
                    output[i, :, :, :] = frame

            except Exception as e:
                logging.warning("decode_data_sequence approach failed: %s", e)
                try:
                    logging.info("Fallback to ds.pixel_array approach")
                    pixel_data_frames = ds.pixel_array # this seems to be more robust? but slower
                    # ensure that the shape is (num_frames, height, width, channels).
                    # it is sometimes (height, width, channels)
                    if len(pixel_data_frames.shape) == 3 and num_frames == 1:
                        pixel_data_frames = np.expand_dims(pixel_data_frames, axis=0)

                    for i in range(num_frames):
                        frame = pixel_data_frames[i, :, :]
                        if len(frame.shape) == 2:
                            frame = np.expand_dims(frame, axis=2)
                        output[i, :, :, :] = frame

                except Exception as e:
                    raise e
        return output
