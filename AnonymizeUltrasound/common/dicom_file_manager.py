import os
import hashlib
import shutil
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

try:
    import qt
    import slicer
    from DICOMLib import DICOMUtils
    HAS_SLICER = True
except Exception:
    # CLI mode does not have Slicer or qt
    HAS_SLICER = False
    qt = None
    slicer = None
    DICOMUtils = None

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
        next_dicom_index (int): Index of next file to process
        current_dicom_index (int): Index of currently loaded file
        _temp_directories (List[str]): List of temporary directories for cleanup
    """

    # Define allowed DICOM file extensions (case-insensitive)
    DICOM_EXTENSIONS = {'.dcm', '.dicom'}

    # DICOM tags to copy directly
    DICOM_TAGS_TO_COPY = [
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "ManufacturerModelName",
        "PatientAge",
        "PatientSex",
        "PixelRepresentation",
        "SeriesNumber",
        "StationName",
        "StudyDate",
        "StudyDescription",
        "StudyID",
        "StudyTime",
        "TransducerType",
        "Manufacturer"
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
        self.next_dicom_index = 0
        self.current_dicom_index = 0
        self._temp_directories = []

    def get_transducer_model(self, transducerType: str) -> str:
        """
        Parse the transducer type string and return the transducer model or 'unknown'.
        For example, if transducerType is 'SC6-1s,02597', it returns 'sc6-1s'.
        """
        if not transducerType or transducerType.strip() == '':
            return 'unknown'

        return transducerType.split(",")[0].lower()

    def scan_directory(self, input_folder: str, skip_single_frame: bool = False, hash_patient_id: bool = True) -> int:
        """
        Scan directory for DICOM files and create dataframe

        Args:
            input_folder: Directory to scan
            skip_single_frame: Skip single frame DICOM files (AnonymizeUltrasound)
            hash_patient_id: If True, hash the patient ID

        Returns:
            Number of DICOM files found
        """
        dicom_data = []
        total_files = sum([len(files) for _, _, files in os.walk(input_folder)])

        progress_dialog = self._create_progress_dialog("Parsing DICOM files...", total_files) if HAS_SLICER else None

        try:
            file_count = 0
            for root, dirs, files in os.walk(input_folder):
                # Sort to ensure consistent processing order
                dirs.sort()
                files.sort()
                for file in files:
                    if progress_dialog:
                        progress_dialog.setValue(file_count)
                    file_count += 1
                    if HAS_SLICER:
                        slicer.app.processEvents()

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

        finally:
            if progress_dialog:
                progress_dialog.close()

    def load_sequence(self, parameter_node, output_directory: Optional[str] = None,
                     continue_progress: bool = False, preserve_directory_structure: bool = True):
        """
        Load next DICOM sequence from the dataframe.

        This method loads the next DICOM file in the sequence, creates a temporary directory,
        copies the DICOM file there, and loads it using Slicer's DICOM utilities. It then
        finds the sequence browser node and updates the parameter node.

        Args:
            parameter_node: Parameter node to store the loaded sequence browser
            output_directory: Optional output directory to check for existing files
            continue_progress: If True, skip files that already exist in output directory
            preserve_directory_structure: If True, the output filepath will be the same as the relative path.
        Returns:
            tuple: (current_dicom_df_index, sequence_browser) where:
                - current_dicom_df_index: The index of the current DICOM file in the dataframe
                - sequence_browser: The loaded sequence browser node
                Returns (None, None) if no more sequences available or loading fails.
        """
        if self.dicom_df is None or self.next_dicom_index is None or self.next_dicom_index >= len(self.dicom_df):
            return None, None

        next_row = self.dicom_df.iloc[self.next_dicom_index]
        temp_dicom_dir = self._setup_temp_directory()

        # Copy DICOM file to temporary folder
        shutil.copy(next_row['InputPath'], temp_dicom_dir)

        # Load DICOM using Slicer's DICOM utilities
        loaded_node_ids = self._load_dicom_from_temp(temp_dicom_dir)
        logging.info(f"Loaded DICOM nodes: {loaded_node_ids}")

        sequence_browser = self._find_sequence_browser(loaded_node_ids)

        if sequence_browser:
            parameter_node.ultrasoundSequenceBrowser = sequence_browser
        else:
            logging.error(f"Failed to find sequence browser node in {loaded_node_ids}")
            return None, None

        # Increment index
        next_index_val = self._increment_dicom_index(output_directory, continue_progress, preserve_directory_structure)

        # Cleanup
        self._cleanup_temp_directory(temp_dicom_dir)

        # Update current DICOM dataframe index
        self.current_dicom_index = self.next_dicom_index - 1 if self.next_dicom_index is not None and self.next_dicom_index > 0 else 0

        if next_index_val or self.next_dicom_index is not None:
            return self.current_dicom_index, sequence_browser

        return None, None

    def get_number_of_instances(self) -> int:
        """Get number of instances in dataframe"""
        return len(self.dicom_df) if self.dicom_df is not None else 0

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
            transducer_model = self.get_transducer_model(dicom_ds.get('TransducerType', ''))

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

        self.next_dicom_index = 0

    def update_progress_from_output(self, output_directory: str, preserve_directory_structure: bool) -> Optional[int]:
        """Update progress based on existing output files

        This method checks which anonymized DICOM files already exist in the output
        directory and updates the next_dicom_index to skip over files that have
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

        self.next_dicom_index = num_done
        return num_done

    def _create_progress_dialog(self, message: str, maximum: int):
        """Create progress dialog"""
        dialog = qt.QProgressDialog(message, "Cancel", 0, maximum, slicer.util.mainWindow())
        dialog.setWindowModality(qt.Qt.WindowModal)
        dialog.show()
        return dialog

    def _setup_temp_directory(self) -> str:
        """Setup temporary directory for DICOM files"""
        temp_dir = os.path.join(slicer.app.temporaryPath, 'UltrasoundModules')
        os.makedirs(temp_dir, exist_ok=True)

        # Clean existing files with error handling
        try:
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except OSError as e:
            logging.warning(f"Failed to clean temp directory {temp_dir}: {e}")

        self._temp_directories.append(temp_dir)
        return temp_dir

    def _load_dicom_from_temp(self, temp_dir: str) -> List[str]:
        """Load DICOM files using Slicer's DICOM utilities

        This method creates a temporary DICOM database and loads DICOM files
        from the specified directory into Slicer. It returns a list of node IDs
        for the loaded DICOM files.

        Args:
            temp_dir: Path to the temporary directory containing DICOM files

        Returns:
            List[str]: List of node IDs for the loaded DICOM files
        """
        loaded_node_ids = []
        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(temp_dir, db)
            patient_uids = db.patients()
            for patient_uid in patient_uids:
                loaded_node_ids.extend(DICOMUtils.loadPatientByUID(patient_uid))
        return loaded_node_ids

    def _find_sequence_browser(self, loaded_node_ids: List[str]):
        """Find sequence browser node from loaded nodes"""
        for node_id in loaded_node_ids:
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node and node.IsA("vtkMRMLSequenceBrowserNode"):
                return node
        return None

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

    def _increment_dicom_index(self, output_directory: Optional[str] = None,
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
            This method modifies the internal next_dicom_index counter. When continue_progress
            is True, it will skip over files that already exist in the output directory.
        """
        if self.dicom_df is None:
            return False

        # Increment the index to the next file to be processed.
        self.next_dicom_index += 1

        # If continue_progress is True, skip files that already exist in output.
        if continue_progress and output_directory:
            while self.next_dicom_index < len(self.dicom_df):
                row = self.dicom_df.iloc[self.next_dicom_index]
                output_path = self.generate_output_filepath(
                    output_directory, row['OutputPath'], preserve_directory_structure)

                if not os.path.exists(output_path):
                    break

                self.next_dicom_index += 1

        return self.next_dicom_index < len(self.dicom_df)

    def _cleanup_temp_directory(self, temp_dir: str):
        """Cleanup temporary directory"""
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if temp_dir in self._temp_directories:
                self._temp_directories.remove(temp_dir)
        except Exception as e:
            logging.warning(f"Failed to cleanup temporary directory {temp_dir}: {e}")

    def save_anonymized_dicom(self, image_array: np.ndarray, output_path: str,
                            new_patient_name: str = '', new_patient_id: str = '', labels: Optional[List[str]] = None) -> None:
        """
        Save image array as anonymized DICOM file.

        Args:
            image_array: Numpy array containing image data (frames, height, width, channels)
            output_path: Full path where DICOM file should be saved
            new_patient_name: New patient name for anonymization
            new_patient_id: New patient ID for anonymization
            labels: List of labels to add to the DICOM file
        """
        if self.dicom_df is None:
            logging.error("No DICOM dataframe available")
            return

        if self.current_dicom_index >= len(self.dicom_df):
            logging.error("No current DICOM record available")
            return

        if image_array is None:
            logging.error("Image array is None")
            return

        current_record = self.dicom_df.iloc[self.current_dicom_index]
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

        # Create and save file
        self._create_and_save_dicom_file(anonymized_ds, output_path)

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
        """Copy metadata from source dataset."""
        for tag in self.DICOM_TAGS_TO_COPY:
            if hasattr(source_ds, tag):
                setattr(ds, tag, getattr(source_ds, tag))

        # Handle UIDs
        self._copy_and_generate_uids(ds, source_ds, output_path)

        # Get series number from dataframe
        ds.SeriesNumber = self._get_series_number_for_current_instance()

    def _copy_and_generate_uids(self, ds: pydicom.Dataset, source_ds: pydicom.Dataset, output_path: str) -> None:
        """Copy or generate required UIDs."""
        # Copy or generate SOPClassUID
        if hasattr(source_ds, 'SOPClassUID') and source_ds.SOPClassUID:
            ds.SOPClassUID = source_ds.SOPClassUID
        else:
            logging.error(f"SOPClassUID not found. Generating new one for {output_path}")
            ds.SOPClassUID = pydicom.uid.generate_uid()

        # Copy or generate SOPInstanceUID
        if hasattr(source_ds, 'SOPInstanceUID') and source_ds.SOPInstanceUID:
            ds.SOPInstanceUID = source_ds.SOPInstanceUID
        else:
            logging.error(f"SOPInstanceUID not found. Generating new one for {output_path}")
            ds.SOPInstanceUID = pydicom.uid.generate_uid()

        # Generate a unique SeriesInstanceUID. This is because ultrasound machines often reuse the same SeriesInstanceUID, which can cause issues in the viewer.
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()

        # Copy or generate StudyInstanceUID
        if hasattr(source_ds, 'StudyInstanceUID') and source_ds.StudyInstanceUID:
            ds.StudyInstanceUID = source_ds.StudyInstanceUID
        else:
            logging.error(f"StudyInstanceUID not found. Generating new one for {output_path}")
            ds.StudyInstanceUID = pydicom.uid.generate_uid()

    def _apply_anonymization(self, ds: pydicom.Dataset, source_ds: pydicom.Dataset,
                            new_patient_name: str = "", new_patient_id: str = "") -> None:
        """Apply anonymization including patient info and date shifting."""
        # Anonymize patient information
        ds.PatientName = new_patient_name if new_patient_name else ""
        ds.PatientID = new_patient_id if new_patient_id else ""
        ds.PatientBirthDate = ""
        ds.ReferringPhysicianName = ""
        ds.AccessionNumber = ""

        # Apply date shifting for anonymization
        self._apply_date_shifting(ds, source_ds)

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

        current_record = self.dicom_df.iloc[self.current_dicom_index]
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

    def save_anonymized_dicom_header(self, current_dicom_record, output_filename: str, headers_directory: Optional[str] = None) -> Optional[str]:
        """
        Save anonymized DICOM header information as a JSON file.

        This method extracts DICOM header information from the current record,
        applies partial anonymization to sensitive fields, and saves the result
        as a JSON file alongside the anonymized DICOM file.

        Args:
            current_dicom_record: Current DICOM record from the dataframe containing
                                the DICOM dataset and metadata
            headers_directory: Directory path where header JSON files will be saved.
                            If None, no header file is created
            output_filename: Base filename for the output (used for patient name anonymization)

        Returns:
            str: Full path to the saved JSON header file
            None: If headers_directory is None or saving fails

        Note:
            - Creates necessary output directories if they don't exist
            - Applies partial anonymization to patient name and birth date
            - Patient name is replaced with the output filename (without extension)
            - Birth date is truncated to year only with "0101" appended
            - Uses convertToJsonCompatible for handling DICOM-specific data types
        """
        if current_dicom_record is None:
            raise ValueError("Current DICOM record is required")

        if output_filename is None or output_filename == "":
            raise ValueError("Output filename is required")

        if headers_directory is None:
            return None

        if not os.path.exists(headers_directory):
            os.makedirs(headers_directory)

        dicom_header_filename = output_filename.replace(".dcm", "_DICOMHeader.json")
        dicom_header_filepath = os.path.join(headers_directory, dicom_header_filename)
        os.makedirs(os.path.dirname(dicom_header_filepath), exist_ok=True)

        with open(dicom_header_filepath, 'w') as outfile:
            if self.dicom_df is not None:
                anonymized_header = self.dicom_header_to_dict(current_dicom_record.DICOMDataset)

                # Anonymize patient name
                if "Patient's Name" in anonymized_header:
                    anonymized_header["Patient's Name"] = output_filename.split(".")[0]

                # Partially anonymize birth date
                if "Patient's Birth Date" in anonymized_header:
                    anonymized_header["Patient's Birth Date"] = anonymized_header["Patient's Birth Date"][:4] + "0101"

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
