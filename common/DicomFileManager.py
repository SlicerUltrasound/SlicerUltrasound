import os
import shutil
import pydicom
import pandas as pd
import logging
from typing import Optional, Tuple, List, Dict, Any
import qt
import slicer
from DICOMLib import DICOMUtils
import vtk
import numpy as np

class DicomFileManager:
    """Shared DICOM file management functionality for ultrasound modules"""

    # Define allowed DICOM file extensions (case-insensitive)
    DICOM_EXTENSIONS = {'.dcm', '.dicom'}

    
    def __init__(self):
        self.dicom_df = None
        self.next_dicom_df_index = 0
        self.current_dicom_dataset = None
        self.current_dicom_header = None
        self._temp_directories = []
    
    def scan_directory(self, input_folder: str, skip_single_frame: bool = False, 
                      rater: Optional[str] = None) -> int:
        """
        Scan directory for DICOM files and create dataframe

        Args:
            input_folder: Directory to scan
            skip_single_frame: Skip single frame DICOM files (AnonymizeUltrasound)
            rater: Optional rater name for annotation file paths (AnnotateUltrasound)

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

                    dicom_info = self._extract_dicom_info(file_path, skip_single_frame, rater)
                    if dicom_info:
                        dicom_data.append(dicom_info)

            self._create_dataframe(dicom_data, rater is not None)

            return len(self.dicom_df) if self.dicom_df is not None else 0

        finally:
            progress_dialog.close()
    
    def load_sequence(self, parameter_node, output_directory: Optional[str] = None, 
                     continue_progress: bool = False):
        """Load next DICOM sequence"""
        self._reset_scene()
        
        if self.next_dicom_df_index is None or self.next_dicom_df_index >= len(self.dicom_df):
            return None
        
        next_row = self.dicom_df.iloc[self.next_dicom_df_index]
        temp_dicom_dir = self._setup_temp_directory()
        
        # Copy DICOM file to temporary folder
        shutil.copy(next_row['Filepath'], temp_dicom_dir)
        
        # Load DICOM using Slicer's DICOM utilities
        loaded_node_ids = self._load_dicom_from_temp(temp_dicom_dir)
        logging.info(f"Loaded DICOM nodes: {loaded_node_ids}")

        sequence_browser = self._find_sequence_browser(loaded_node_ids)
        
        if sequence_browser:
            self.current_dicom_header = self._extract_dicom_header(sequence_browser)
            sequence_browser.SetAttribute("DicomFile", next_row['Filepath'])
            
            # Set the sequence browser in the parameter node
            if hasattr(parameter_node, 'ultrasoundSequenceBrowser'):
                parameter_node.ultrasoundSequenceBrowser = sequence_browser
            
            # Get proxy node and set it as input volume if parameter node has that field
            master_sequence = sequence_browser.GetMasterSequenceNode()
            if master_sequence:
                proxy_node = sequence_browser.GetProxyNode(master_sequence)
                if proxy_node and hasattr(parameter_node, 'inputVolume'):
                    parameter_node.inputVolume = proxy_node
        
        # Increment index
        next_index = self._increment_dicom_index(output_directory, continue_progress)
        
        # Cleanup
        self._cleanup_temp_directory(temp_dicom_dir)
        
        return self.next_dicom_df_index - 1 if next_index else None
    
    def load_previous_sequence(self) -> Optional[int]:
        """Load previous DICOM sequence"""
        if self.dicom_df is None or self.next_dicom_df_index <= 1:
            return None
        
        self.next_dicom_df_index -= 2
        return self.load_sequence()
    
    def get_number_of_instances(self) -> int:
        """Get number of instances in dataframe"""
        return len(self.dicom_df) if self.dicom_df is not None else 0
    
    def _extract_dicom_info(self, file_path: str, skip_single_frame: bool, 
                           rater: Optional[str]) -> Optional[List]:
        """Extract DICOM information from file"""
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
            
            # Build result based on module type
            if rater is not None:
                # AnnotateUltrasound mode
                base_filename = os.path.splitext(file_path)[0]
                annotation_path = self._determine_annotation_path(base_filename, rater)
                return [file_path, annotation_path, patient_uid, study_uid, series_uid, instance_uid]
            else:
                # AnonymizeUltrasound mode
                physical_delta_x, physical_delta_y = self._extract_spacing_info(dicom_ds)
                exp_filename = self._generate_filename_from_dicom(dicom_ds)
                content_date = getattr(dicom_ds, 'ContentDate', '19000101')
                content_time = getattr(dicom_ds, 'ContentTime', '000000')
                to_patch = physical_delta_x is None or physical_delta_y is None
                
                return [file_path, exp_filename, patient_uid, study_uid, series_uid, 
                       instance_uid, physical_delta_x, physical_delta_y, 
                       content_date, content_time, to_patch]
            
        except Exception as e:
            logging.error(f"Failed to read DICOM file {file_path}: {e}")
            return None
    
    def _determine_annotation_path(self, base_filename: str, rater: str) -> str:
        """Determine annotation file path for AnnotateUltrasound"""
        candidates = [
            f"{base_filename}.{rater}.json",
            f"{base_filename}.json"
        ]
        
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        
        # Default to rater-specific file
        return f"{base_filename}.{rater}.json"
    
    def _extract_spacing_info(self, dicom_ds):
        """Extract spacing information from DICOM dataset"""
        physical_delta_x = None
        physical_delta_y = None
        
        if hasattr(dicom_ds, 'SequenceOfUltrasoundRegions') and dicom_ds.SequenceOfUltrasoundRegions:
            region = dicom_ds.SequenceOfUltrasoundRegions[0]
            if hasattr(region, 'PhysicalDeltaX'):
                physical_delta_x = float(region.PhysicalDeltaX)
            if hasattr(region, 'PhysicalDeltaY'):
                physical_delta_y = float(region.PhysicalDeltaY)
        
        return physical_delta_x, physical_delta_y
    
    def _generate_filename_from_dicom(self, dicom_ds):
        """Generate expected filename from DICOM data"""
        patient_id = getattr(dicom_ds, 'PatientID', 'Unknown')
        instance_uid = getattr(dicom_ds, 'SOPInstanceUID', 'Unknown')
        return f"{patient_id}_{instance_uid}.dcm"
    
    def _create_dataframe(self, dicom_data: List, is_annotate_mode: bool):
        """Create pandas DataFrame from DICOM data"""
        if is_annotate_mode:
            columns = ['Filepath', 'AnnotationsFilepath', 'PatientUID', 'StudyUID', 
                      'SeriesUID', 'InstanceUID']
        else:
            columns = ['Filepath', 'AnonFilename', 'PatientUID', 'StudyUID', 
                      'SeriesUID', 'InstanceUID', 'PhysicalDeltaX', 'PhysicalDeltaY', 
                      'ContentDate', 'ContentTime', 'Patch']
        
        self.dicom_df = pd.DataFrame(dicom_data, columns=columns)
        
        if not is_annotate_mode:
            # Sort and add series numbers for AnonymizeUltrasound
            self.dicom_df = self.dicom_df.sort_values(by=['Filepath', 'ContentDate', 'ContentTime'])
            self.dicom_df['SeriesNumber'] = self.dicom_df.groupby(['PatientUID', 'StudyUID']).cumcount() + 1
            
            # Fill missing spacing information
            self.dicom_df['PhysicalDeltaX'] = self.dicom_df.groupby('StudyUID')['PhysicalDeltaX'].transform(lambda x: x.ffill().bfill())
            self.dicom_df['PhysicalDeltaY'] = self.dicom_df.groupby('StudyUID')['PhysicalDeltaY'].transform(lambda x: x.ffill().bfill())
        
        self.next_dicom_df_index = 0
    
    def update_progress_from_output(self, input_directory: str, output_directory: str) -> Optional[int]:
        """Update progress based on existing output files"""
        if self.dicom_df is None:
            return None
        
        num_done = 0
        for index, row in self.dicom_df.iterrows():
            expected_filename = row['AnonFilename']
            output_path = os.path.join(output_directory, expected_filename)
            
            if os.path.exists(output_path):
                num_done += 1
            else:
                break
        
        if num_done == len(self.dicom_df):
            return None  # All files processed
        elif num_done > 0:
            self.next_dicom_df_index = num_done
            return num_done
        else:
            self.next_dicom_df_index = 0
            return 0
    
    def _create_progress_dialog(self, message: str, maximum: int) -> qt.QProgressDialog:
        """Create progress dialog"""
        dialog = qt.QProgressDialog(message, "Cancel", 0, maximum, slicer.util.mainWindow())
        dialog.setWindowModality(qt.Qt.WindowModal)
        dialog.show()
        return dialog
    
    def _reset_scene(self):
        """Reset the scene before loading new data"""
        slicer.mrmlScene.Clear(0)
        self.current_dicom_dataset = None
        self.current_dicom_header = None
    
    def _setup_temp_directory(self) -> str:
        """Setup temporary directory for DICOM files"""
        temp_dir = os.path.join(slicer.app.temporaryPath, 'UltrasoundModules')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Clean existing files
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        self._temp_directories.append(temp_dir)
        return temp_dir
    
    def _load_dicom_from_temp(self, temp_dir: str) -> List[str]:
        """Load DICOM files using Slicer's DICOM utilities"""
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
    
    def _extract_dicom_header(self, browser_node) -> Optional[dict]:
        """Extract DICOM header from browser node"""
        if not browser_node:
            return None
        
        master_sequence = browser_node.GetMasterSequenceNode()
        proxy_node = browser_node.GetProxyNode(master_sequence)
        instance_uid = proxy_node.GetAttribute("DICOM.instanceUIDs")
        
        if isinstance(instance_uid, list):
            instance_uid = instance_uid[0]
        
        filepath = self._get_file_for_instance_uid(instance_uid)
        if filepath:
            ds = pydicom.dcmread(filepath)
            self.current_dicom_dataset = ds
            return self._dicom_header_to_dict(ds)
        
        return None
    
    def _get_file_for_instance_uid(self, instance_uid: str) -> Optional[str]:
        """Get file path for given instance UID"""
        if self.dicom_df is None:
            return None
        
        matching_rows = self.dicom_df[self.dicom_df['InstanceUID'] == instance_uid]
        if not matching_rows.empty:
            return matching_rows.iloc[0]['Filepath']
        
        return None
    
    def _dicom_header_to_dict(self, ds, parent=None):
        """Convert DICOM dataset to dictionary"""
        if parent is None:
            parent = {}
        for elem in ds:
            if elem.VR == "SQ":
                parent[elem.name] = []
                for item in elem:
                    child = {}
                    self._dicom_header_to_dict(item, child)
                    parent[elem.name].append(child)
            else:
                parent[elem.name] = elem.value
        return parent
    
    def _increment_dicom_index(self, output_directory: Optional[str] = None, 
                              continue_progress: bool = False) -> bool:
        """Increment DICOM index, optionally checking for existing output files"""
        self.next_dicom_df_index += 1
        
        if continue_progress and output_directory:
            # Skip files that already exist in output
            while self.next_dicom_df_index < len(self.dicom_df):
                row = self.dicom_df.iloc[self.next_dicom_df_index]
                expected_filename = row['AnonFilename']
                output_path = os.path.join(output_directory, expected_filename)
                
                if not os.path.exists(output_path):
                    break
                    
                self.next_dicom_df_index += 1
        
        return self.next_dicom_df_index < len(self.dicom_df)
    
    def _cleanup_temp_directory(self, temp_dir: str):
        """Cleanup temporary directory"""
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if temp_dir in self._temp_directories:
                self._temp_directories.remove(temp_dir)
        except Exception as e:
            logging.warning(f"Failed to cleanup temporary directory {temp_dir}: {e}")
    
    def cleanup_all_temp_directories(self):
        """Cleanup all temporary directories"""
        for temp_dir in self._temp_directories.copy():
            self._cleanup_temp_directory(temp_dir)