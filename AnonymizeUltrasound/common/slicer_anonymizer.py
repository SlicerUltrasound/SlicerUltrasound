import os
import hashlib
import shutil
import pydicom
import pandas as pd
import logging
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import io
import random
import datetime
import json
from pydicom.dataset import FileMetaDataset
from create_masks import corner_points_to_fan_mask_config


class UltrasoundAnonymizer:
    """
    Command-line ultrasound DICOM anonymizer that replicates 3D Slicer extension functionality.
    
    This class provides all the core anonymization functionality from the AnonymizeUltrasound 
    Slicer extension, adapted for command-line use without Slicer dependencies.
    """
    
    # Constants from original implementation
    DICOM_EXTENSIONS = {'.dcm', '.dicom'}
    PATIENT_ID_HASH_LENGTH = 10
    INSTANCE_ID_HASH_LENGTH = 8
    DEFAULT_CONTENT_DATE = '19000101'
    DEFAULT_CONTENT_TIME = ''
    
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
    
    def __init__(self):
        self.dicom_df = None
        self.mask_parameters = {}
        self.current_dicom_index = 0
        self.predicted_corners = None
        self.original_dims = None
        
    def scan_directory(self, input_folder: str, skip_single_frame: bool = False, hash_patient_id: bool = True) -> int:
        """
        Scan directory for DICOM files and create dataframe.
        
        Args:
            input_folder: Directory to scan
            skip_single_frame: Skip single frame DICOM files
            hash_patient_id: Whether to hash the patient ID in filenames.
            
        Returns:
            Number of DICOM files found
        """
        dicom_data = []
        
        for root, dirs, files in os.walk(input_folder):
            dirs.sort()
            files.sort()
            
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)
                
                if ext.lower() not in self.DICOM_EXTENSIONS:
                    continue
                    
                dicom_info = self._extract_dicom_info(file_path, input_folder, skip_single_frame, hash_patient_id)
                if dicom_info:
                    dicom_data.append(dicom_info)
                    
        self._create_dataframe(dicom_data)
        return len(self.dicom_df) if self.dicom_df is not None else 0
        
    def _extract_dicom_info(self, file_path: str, input_folder: str, skip_single_frame: bool, hash_patient_id: bool = True) -> Optional[Dict]:
        """Extract DICOM information from file."""
        try:
            dicom_ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            
            # Skip non-ultrasound modalities
            if dicom_ds.get("Modality", "") != "US":
                return None
                
            # Skip single frame if requested
            if skip_single_frame and ('NumberOfFrames' not in dicom_ds or dicom_ds.NumberOfFrames < 2):
                return None
                
            # Extract required fields
            patient_uid = getattr(dicom_ds, 'PatientID', None)
            study_uid = getattr(dicom_ds, 'StudyInstanceUID', None)
            series_uid = getattr(dicom_ds, 'SeriesInstanceUID', None)
            instance_uid = getattr(dicom_ds, 'SOPInstanceUID', None)
            
            if not all([patient_uid, study_uid, series_uid, instance_uid]):
                return None
                
            physical_delta_x, physical_delta_y = self._extract_spacing_info(dicom_ds)
            anon_filename = self._generate_filename_from_dicom(dicom_ds, hash_patient_id=hash_patient_id)
            content_date = getattr(dicom_ds, 'ContentDate', '19000101')
            content_time = getattr(dicom_ds, 'ContentTime', '000000')
            to_patch = physical_delta_x is None or physical_delta_y is None
            
            # Calculate relative path from input folder
            output_path = os.path.relpath(file_path, input_folder).replace(
                os.path.basename(file_path), anon_filename)
                
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
                'DICOMDataset': dicom_ds
            }
            
        except Exception as e:
            logging.error(f"Failed to read DICOM file {file_path}: {e}")
            return None
            
    def _create_dataframe(self, dicom_data: List[Dict]) -> None:
        """Create pandas DataFrame from DICOM data."""
        if not dicom_data:
            self.dicom_df = pd.DataFrame()
            return
            
        columns = [
            'InputPath', 'OutputPath', 'AnonFilename', 'PatientUID', 'StudyUID',
            'SeriesUID', 'InstanceUID', 'PhysicalDeltaX', 'PhysicalDeltaY',
            'ContentDate', 'ContentTime', 'Patch', 'TransducerModel', 'DICOMDataset'
        ]
        
        self.dicom_df = pd.DataFrame(dicom_data, columns=columns)
        
        # Sort and reset index
        self.dicom_df = (self.dicom_df
                        .sort_values(['InputPath', 'ContentDate', 'ContentTime'])
                        .reset_index(drop=True))
                        
        # Add series numbers
        self.dicom_df['SeriesNumber'] = (self.dicom_df
                                        .groupby(['PatientUID', 'StudyUID'])
                                        .cumcount() + 1)
                                        
        # Fill missing spacing information
        spacing_cols = ['PhysicalDeltaX', 'PhysicalDeltaY']
        self.dicom_df[spacing_cols] = (self.dicom_df
                                      .groupby('StudyUID')[spacing_cols]
                                      .transform(lambda x: x.ffill().bfill()))
                                      
    def _extract_spacing_info(self, dicom_ds):
        """Extract physical spacing information from DICOM dataset."""
        physical_delta_x = None
        physical_delta_y = None
        
        if hasattr(dicom_ds, 'SequenceOfUltrasoundRegions') and dicom_ds.SequenceOfUltrasoundRegions:
            region = dicom_ds.SequenceOfUltrasoundRegions[0]
            if hasattr(region, 'PhysicalDeltaX'):
                physical_delta_x = float(region.PhysicalDeltaX)
            if hasattr(region, 'PhysicalDeltaY'):
                physical_delta_y = float(region.PhysicalDeltaY)
                
        return physical_delta_x, physical_delta_y

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

    def _generate_filename_from_dicom(self, dicom_ds, hash_patient_id: bool = True) -> str:
        """Generate an anonymized filename from a DICOM dataset."""
        patient_uid = dicom_ds.PatientID
        instance_uid = dicom_ds.SOPInstanceUID
        
        if patient_uid is None or patient_uid == "":
            logging.error("PatientID not found in DICOM header")
            return ""
            
        if instance_uid is None or instance_uid == "":
            logging.error("SOPInstanceUID not found in DICOM header")
            return ""
            
        if hash_patient_id:
            hash_object = hashlib.sha256()
            hash_object.update(str(patient_uid).encode())
            patient_id = int(hash_object.hexdigest(), 16) % 10**self.PATIENT_ID_HASH_LENGTH
        else:
            patient_id = patient_uid
            
        hash_object_instance = hashlib.sha256()
        hash_object_instance.update(str(instance_uid).encode())
        instance_id = int(hash_object_instance.hexdigest(), 16) % 10**self.INSTANCE_ID_HASH_LENGTH
        
        # Add trailing zeros
        patient_id = str(patient_id).zfill(self.PATIENT_ID_HASH_LENGTH)
        instance_id = str(instance_id).zfill(self.INSTANCE_ID_HASH_LENGTH)
        
        return f"{patient_id}_{instance_id}.dcm"
        

    def _find_extreme_corners(self, points: np.ndarray) -> np.ndarray:
        """Find extreme corners from point set."""
        points = np.array(points)
        
        # Find corners based on extrema
        top_left = list(points[np.argmin(points[:, 0] + points[:, 1])])
        top_right = list(points[np.argmax(points[:, 0] - points[:, 1])])
        bottom_left = list(points[np.argmin(points[:, 0] - points[:, 1])])
        bottom_right = list(points[np.argmax(points[:, 0] + points[:, 1])])
        
        corners = [tuple(top_left), tuple(top_right), tuple(bottom_left), tuple(bottom_right)]
        unique_corners = set(corners)
        num_unique_corners = len(unique_corners)
        
        # Handle triangular mask case
        epsilon = 2
        if num_unique_corners == 3:
            top_point = top_left if top_left[1] < top_right[1] else top_right
            top_left = list(top_point)
            top_right = list(top_point)
            top_left[0] -= epsilon
            top_right[0] += epsilon
            
        if num_unique_corners < 3:
            return None
            
        return np.array([top_left, top_right, bottom_left, bottom_right])
                
    def apply_mask_to_sequence(self, image_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply mask to all frames in image sequence.
        
        Args:
            image_array: Image array with shape (frames, height, width, channels)
            mask: Binary mask array
            
        Returns:
            Masked image array
        """
        masked_array = image_array.copy()
        
        # Apply mask to each frame and channel
        for frame_idx in range(masked_array.shape[0]):
            for channel_idx in range(masked_array.shape[3]):
                masked_array[frame_idx, :, :, channel_idx] = np.multiply(
                    masked_array[frame_idx, :, :, channel_idx], mask)
                    
        return masked_array
        
    def save_anonymized_dicom(self, image_array: np.ndarray, output_path: str,
                            new_patient_name: str = '', new_patient_id: str = '', labels: List[str] = None) -> None:
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
        
    def _copy_spacing_info(self, ds: pydicom.Dataset, source_ds: pydicom.Dataset) -> None:
        """Copy spacing information."""
        # Copy SequenceOfUltrasoundRegions if available
        if hasattr(source_ds, "SequenceOfUltrasoundRegions") and len(source_ds.SequenceOfUltrasoundRegions) > 0:
            ds.SequenceOfUltrasoundRegions = source_ds.SequenceOfUltrasoundRegions
            
        # Copy spacing to conventional PixelSpacing tag
        if hasattr(source_ds, 'SequenceOfUltrasoundRegions') and source_ds.SequenceOfUltrasoundRegions:
            region = source_ds.SequenceOfUltrasoundRegions[0]
            if hasattr(region, 'PhysicalDeltaX') and hasattr(region, 'PhysicalDeltaY'):
                delta_x_mm = float(region.PhysicalDeltaX) * 10
                delta_y_mm = float(region.PhysicalDeltaY) * 10
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
        # Copy or generate UIDs
        for uid_tag in ['SOPClassUID', 'SOPInstanceUID', 'StudyInstanceUID']:
            if hasattr(source_ds, uid_tag) and getattr(source_ds, uid_tag):
                setattr(ds, uid_tag, getattr(source_ds, uid_tag))
            else:
                logging.warning(f"{uid_tag} not found. Generating new one for {output_path}")
                setattr(ds, uid_tag, pydicom.uid.generate_uid())
                
        # Generate unique SeriesInstanceUID
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        
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
        
    def _shift_date(self, date_str: str, offset: int) -> str:
        """Shift a single date by the given offset."""
        try:
            date_obj = datetime.datetime.strptime(date_str, "%Y%m%d") + datetime.timedelta(days=offset)
            return date_obj.strftime("%Y%m%d")
        except Exception as e:
            logging.warning(f"Failed to parse date: {date_str}. Using original date. Error: {e}")
            return date_str
            
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
        # Handle 2D frames
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, axis=-1)
            
        # Convert to PIL Image
        if frame.shape[2] == 1:
            image = Image.fromarray(frame[:, :, 0]).convert("L")
        else:
            # if its rgb with first dim as channels, then we need to transpose it
            if frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0)) # (C, H, W) -> (H, W, C)
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
        # file_ds.is_implicit_VR = False
        # file_ds.is_little_endian = True

        # Save file to output path. Create the directories if they don't exist.
        directory = os.path.dirname(output_filepath)
        os.makedirs(directory, exist_ok=True)
        file_ds.save_as(output_filepath, implicit_vr=False, little_endian=True)
        logging.info(f"DICOM generated successfully: {output_filepath}")
        
    def save_anonymized_dicom_header(self, output_filename: str, headers_directory: Optional[str] = None) -> Optional[str]:
        """
        Save anonymized DICOM header information as a JSON file.

        This method extracts DICOM header information from the current record,
        applies partial anonymization to sensitive fields, and saves the result
        as a JSON file alongside the anonymized DICOM file.

        Args:
            output_filename: Base filename for the output (used for patient name anonymization)
            headers_directory: Directory path where header JSON files will be saved.
                            If None, no header file is created

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
        if self.dicom_df is None or self.current_dicom_index >= len(self.dicom_df):
            logging.error("No current DICOM record available")
            return None

        current_dicom_record = self.dicom_df.iloc[self.current_dicom_index]

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
            anonymized_header = self._dicom_header_to_dict(current_dicom_record.DICOMDataset)

            # Anonymize patient name
            if "Patient's Name" in anonymized_header:
                anonymized_header["Patient's Name"] = output_filename.split(".")[0]

            # Partially anonymize birth date
            if "Patient's Birth Date" in anonymized_header:
                anonymized_header["Patient's Birth Date"] = anonymized_header["Patient's Birth Date"][:4] + "0101"

            json.dump(anonymized_header, outfile, default=self._convert_to_json_compatible)

        return dicom_header_filepath
        
    def _dicom_header_to_dict(self, ds: pydicom.Dataset, parent: Dict = None) -> Dict:
        """Convert DICOM dataset to dictionary format."""
        if parent is None:
            parent = {}
            
        for elem in ds:
            # Skip PixelData to avoid memory issues
            if elem.name == "Pixel Data":
                continue
                
            if elem.VR == "SQ":
                parent[elem.name] = []
                for item in elem:
                    child = {}
                    self._dicom_header_to_dict(item, child)
                    parent[elem.name].append(child)
            else:
                parent[elem.name] = elem.value
                
        return parent
        
    def _convert_to_json_compatible(self, obj):
        """Convert DICOM-specific data types to JSON-serializable formats."""
        if isinstance(obj, pydicom.multival.MultiValue):
            return list(obj)
        if isinstance(obj, pydicom.valuerep.PersonName):
            return str(obj)
        if isinstance(obj, bytes):
            return obj.decode('latin-1')
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')
        
    def save_json(self, output_path: str, sop_instance_uid: str, labels: List[str] = None) -> str:
        """
        Save sequence information, mask configuration, and annotations to JSON file.
        
        Args:
            output_path: DICOM output file path
            sop_instance_uid: SOP Instance UID
            labels: Annotation labels
            
        Returns:
            Path to saved JSON file
        """
        sequence_info = {
            'SOPInstanceUID': sop_instance_uid,
            'GrayscaleConversion': False
        }
        
        # Add mask parameters
        for key, value in self.mask_parameters.items():
            sequence_info[key] = value
            
        # Add mask configuration from predicted corners
        if self.predicted_corners is not None and self.original_dims is not None:
            try:
                mask_config = corner_points_to_fan_mask_config(
                    self.predicted_corners, 
                    image_size=self.original_dims
                )
                sequence_info['MaskConfig'] = mask_config
            except Exception as e:
                logging.warning(f"Failed to generate mask config: {e}")
            
        # Add annotation labels
        if labels:
            sequence_info["AnnotationLabels"] = labels
            
        json_path = output_path.replace(".dcm", ".json")
        with open(json_path, 'w') as outfile:
            json.dump(sequence_info, outfile, indent=2)
            
        return json_path
        
    def generate_output_filepath(self, output_directory: str, output_path: str, 
                                preserve_directory_structure: bool) -> str:
        """Generate output filepath considering directory structure preservation."""
        if preserve_directory_structure:
            return os.path.join(output_directory, output_path)
        else:
            filename = os.path.basename(output_path)
            return os.path.join(output_directory, filename)
