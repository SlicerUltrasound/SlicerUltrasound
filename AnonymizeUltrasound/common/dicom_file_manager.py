import os
import hashlib
import shutil
import pydicom
import pandas as pd
import logging
from typing import Optional, List
import qt
import slicer
from DICOMLib import DICOMUtils

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

    PATIENT_ID_HASH_LENGTH = 10
    INSTANCE_ID_HASH_LENGTH = 8
    DEFAULT_CONTENT_DATE = '19000101'
    DEFAULT_CONTENT_TIME = '000000'

    def __init__(self):
        self.dicom_df = None
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

    def scan_directory(self, input_folder: str, skip_single_frame: bool = False) -> int:
        """
        Scan directory for DICOM files and create dataframe

        Args:
            input_folder: Directory to scan
            skip_single_frame: Skip single frame DICOM files (AnonymizeUltrasound)

        Returns:
            Number of DICOM files found
        """
        dicom_data = []
        total_files = sum([len(files) for _, _, files in os.walk(input_folder)])

        progress_dialog = self._create_progress_dialog("Parsing DICOM files...", total_files)

        try:
            file_count = 0
            for root, dirs, files in os.walk(input_folder):
                # Sort to ensure consistent processing order
                dirs.sort()
                files.sort()
                for file in files:
                    progress_dialog.setValue(file_count)
                    file_count += 1
                    slicer.app.processEvents()

                    file_path = os.path.join(root, file)
                    _, ext = os.path.splitext(file)
                    if ext.lower() not in self.DICOM_EXTENSIONS:
                        logging.info(f"Skipping non-DICOM file: {file_path}")
                        continue

                    dicom_info = self._extract_dicom_info(file_path, skip_single_frame)
                    if dicom_info:
                        dicom_data.append(dicom_info)

            self._create_dataframe(dicom_data)

            return len(self.dicom_df) if self.dicom_df is not None else 0

        finally:
            progress_dialog.close()

    def load_sequence(self, parameter_node, output_directory: Optional[str] = None,
                     continue_progress: bool = False):
        """
        Load next DICOM sequence from the dataframe.
        
        This method loads the next DICOM file in the sequence, creates a temporary directory,
        copies the DICOM file there, and loads it using Slicer's DICOM utilities. It then
        finds the sequence browser node and updates the parameter node.
        
        Args:
            parameter_node: Parameter node to store the loaded sequence browser
            output_directory: Optional output directory to check for existing files
            continue_progress: If True, skip files that already exist in output directory
            
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
        shutil.copy(next_row['Filepath'], temp_dicom_dir)

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
        next_index_val = self._increment_dicom_index(output_directory, continue_progress)

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

    def _extract_dicom_info(self, file_path: str, skip_single_frame: bool) -> Optional[dict]:
        """Extract DICOM information from file
        
        Reads a DICOM file and extracts relevant metadata for ultrasound processing.
        Validates that the file is an ultrasound modality and optionally skips
        single-frame files based on the skip_single_frame parameter.
        
        Args:
            file_path: Path to the DICOM file to process
            skip_single_frame: If True, skip files with less than 2 frames
            
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
            exp_filename = self._generate_filename_from_dicom(dicom_ds)
            content_date = getattr(dicom_ds, 'ContentDate', '19000101')
            content_time = getattr(dicom_ds, 'ContentTime', '000000')
            to_patch = physical_delta_x is None or physical_delta_y is None
            transducer_model = self.get_transducer_model(dicom_ds.get('TransducerType', ''))

            return {
                'Filepath': file_path,
                'AnonFilename': exp_filename,
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

    def _generate_filename_from_dicom(self, dicom_ds, hashPatientId: bool = True):
        """
        Generate an anonymized filename from a DICOM dataset.
        
        Creates a standardized filename format for DICOM files using hashed identifiers
        to protect patient privacy while maintaining uniqueness.
        
        Args:
            dicom_ds: DICOM dataset containing patient and instance information
            hashPatientId (bool): Whether to hash the patient ID (default: True)
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

        if hashPatientId:
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
        expected_columns = [
            'Filepath', 'AnonFilename', 'PatientUID', 'StudyUID', 'SeriesUID', 
            'InstanceUID', 'PhysicalDeltaX', 'PhysicalDeltaY', 'ContentDate', 
            'ContentTime', 'Patch', 'TransducerModel', 'DICOMDataset'
        ]
        
        self.dicom_df = pd.DataFrame(dicom_data, columns=expected_columns)
        
        # Sort and reset index
        self.dicom_df = (self.dicom_df
                         .sort_values(['Filepath', 'ContentDate', 'ContentTime'])
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

    def update_progress_from_output(self, output_directory: str) -> Optional[int]:
        """Update progress based on existing output files
        
        This method checks which anonymized DICOM files already exist in the output
        directory and updates the next_dicom_index to skip over files that have
        already been processed. This enables resuming processing from where it
        left off in case of interruption.
        
        Args:
            output_directory: Directory path where anonymized DICOM files are saved
            
        Returns:
            int: Number of files already processed (0 if none processed)
            None: If all files have been processed or no dataframe exists
        """
        if self.dicom_df is None:
            return None
        
        # Create full paths vectorized
        output_paths = self.dicom_df['AnonFilename'].apply(
            lambda x: os.path.join(output_directory, x)
        )
        
        # Check existence vectorized
        exists_mask = output_paths.apply(os.path.exists)
        
        # Find first False (first non-existing file)
        if exists_mask.all():
            return None  # All files processed
        
        first_missing = exists_mask.idxmin() if not exists_mask.all() else len(exists_mask)
        num_done = exists_mask[:first_missing].sum()
        
        self.next_dicom_index = num_done
        return num_done if num_done > 0 else 0

    def _create_progress_dialog(self, message: str, maximum: int) -> qt.QProgressDialog:
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
            return matching_rows.iloc[0]['Filepath']

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
                              continue_progress: bool = False) -> bool:
        """Increment DICOM index, optionally checking for existing output files"""
        if self.dicom_df is None:
            return False

        self.next_dicom_index += 1

        if continue_progress and output_directory:
            # Skip files that already exist in output
            while self.next_dicom_index < len(self.dicom_df):
                row = self.dicom_df.iloc[self.next_dicom_index]
                expected_filename = row['AnonFilename']
                output_path = os.path.join(output_directory, expected_filename)

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
