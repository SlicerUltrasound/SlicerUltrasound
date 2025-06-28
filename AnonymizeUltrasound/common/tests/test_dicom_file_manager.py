import pytest
import os
import sys
import tempfile
import shutil
import pandas as pd
import pydicom
import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom import Sequence
from unittest.mock import Mock, patch
from pathlib import Path

# Mock slicer module for testing
sys.modules['slicer'] = Mock()
sys.modules['slicer.app'] = Mock()
sys.modules['slicer.util'] = Mock()
sys.modules['slicer.mrmlScene'] = Mock()
sys.modules['qt'] = Mock()
sys.modules['vtk'] = Mock()
sys.modules['DICOMLib'] = Mock()
sys.modules['DICOMLib.DICOMUtils'] = Mock()

# Import the module under test
from common.dicom_file_manager import DicomFileManager

class TestDicomFileManager:
    """Test suite for DicomFileManager class"""

    PATIENT_ID = "TEST123"
    PATIENT_NAME = "Test^Patient"
    STUDY_UID = "1.2.840.113619.2.55.3.604688432.781.1591781234.467"
    SERIES_UID = "1.2.840.113619.2.55.3.604688432.781.1591781234.468"
    SOP_INSTANCE_UID = "1.2.840.113619.2.55.3.604688432.781.1591781234.469"
    SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.6.1"
    MODALITY = "US"
    NUMBER_OF_FRAMES = 10
    CONTENT_DATE = "20240101"
    CONTENT_TIME = "120000"
    TRANSDUCER_TYPE = "SC6-1s,02597"
    TRANSDUCER_MODEL = "sc6-1s"
    ROWS = 4
    COLUMNS = 6
    BITS_ALLOCATED = 8
    BITS_STORED = 8
    HIGH_BIT = 7
    PIXEL_REPRESENTATION = 0
    SAMPLES_PER_PIXEL = 1
    PHOTOMETRIC_INTERPRETATION = "MONOCHROME2" # Grayscale
    SEQUENCE_OF_ULTRASOUND_REGIONS = []
    TRANSFER_SYNTAX_UID = '1.2.840.10008.1.2'  # Implicit VR Little Endian
    IMPLEMENTATION_CLASS_UID = '1.2.826.0.1.3680043.8.498.1'  # Example UID
    PHYSICAL_DELTA_X = 0.1
    PHYSICAL_DELTA_Y = 0.15

    def create_test_dicom_file(self, **kwargs) -> FileDataset:
        """Create a temporary DICOM file for testing

        This method creates a FileDataset object that simulates a DICOM ultrasound file
        with configurable attributes. It sets up the necessary file metadata and dataset
        attributes required for testing DICOM file operations.

        Example:
            dicom_file = create_test_dicom_file(
                PatientID="CUSTOM123",
                NumberOfFrames=5,
                PhysicalDeltaX=0.2,
                PhysicalDeltaY=0.3
            )
        """
        # Default DICOM attributes
        defaults = {
            'PatientID': self.PATIENT_ID,
            'PatientName': self.PATIENT_NAME,
            'StudyInstanceUID': self.STUDY_UID,
            'SeriesInstanceUID': self.SERIES_UID,
            'SOPInstanceUID': self.SOP_INSTANCE_UID,
            'SOPClassUID': self.SOP_CLASS_UID,
            'Modality': self.MODALITY,
            'NumberOfFrames': self.NUMBER_OF_FRAMES,
            'ContentDate': self.CONTENT_DATE,
            'ContentTime': self.CONTENT_TIME,
            'TransducerType': self.TRANSDUCER_TYPE,
            'Rows': self.ROWS,
            'Columns': self.COLUMNS,
            'BitsAllocated': self.BITS_ALLOCATED,
            'BitsStored': self.BITS_STORED,
            'HighBit': self.HIGH_BIT,
            'PixelRepresentation': self.PIXEL_REPRESENTATION,
            'SamplesPerPixel': self.SAMPLES_PER_PIXEL,
            'PhotometricInterpretation': self.PHOTOMETRIC_INTERPRETATION,
            'SequenceOfUltrasoundRegions': self.SEQUENCE_OF_ULTRASOUND_REGIONS,
        }

        # Override defaults with provided kwargs
        attributes = {**defaults, **kwargs}

        # Create file meta information
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = attributes['SOPClassUID']
        file_meta.MediaStorageSOPInstanceUID = attributes['SOPInstanceUID']
        file_meta.ImplementationClassUID = self.IMPLEMENTATION_CLASS_UID
        file_meta.TransferSyntaxUID = self.TRANSFER_SYNTAX_UID

        # Create main dataset
        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Set all attributes
        for key, value in attributes.items():
            setattr(ds, key, value)

        region = Dataset()
        region.PhysicalDeltaX = kwargs.get('PhysicalDeltaX', self.PHYSICAL_DELTA_X)
        region.PhysicalDeltaY = kwargs.get('PhysicalDeltaY', self.PHYSICAL_DELTA_Y)
        ds.SequenceOfUltrasoundRegions = [region]

        # Create minimal pixel data (just zeros)
        if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
            pixel_array_size = ds.Rows * ds.Columns * ds.NumberOfFrames
        else:
            pixel_array_size = ds.Rows * ds.Columns

        ds.PixelData = b'\x00' * pixel_array_size

        return ds

    def save_dicom_file(self, ds: FileDataset, temp_dir: str) -> str:
        """Save the DICOM dataset to a file"""
        # Generate filename and save
        filename = f"{ds.PatientID}_{ds.SOPInstanceUID}.dcm"
        filepath = os.path.join(temp_dir, filename)

        ds.save_as(filepath)
        return filepath

    @pytest.fixture
    def sample_dicom_filepath(self, temp_dir):
        """Create a sample DICOM file for testing"""
        ds = self.create_test_dicom_file()
        filepath = self.save_dicom_file(ds, temp_dir)
        yield filepath

    @pytest.fixture
    def single_frame_dicom_filepath(self, temp_dir):
        """Create a single-frame DICOM file for testing"""
        ds = self.create_test_dicom_file(NumberOfFrames=1)
        filepath = self.save_dicom_file(ds, temp_dir)
        yield filepath

    @pytest.fixture
    def non_ultrasound_dicom_filepath(self, temp_dir):
        """Create a non-ultrasound DICOM file for testing"""
        ds = self.create_test_dicom_file(Modality='CT')
        filepath = self.save_dicom_file(ds, temp_dir)
        yield filepath

    @pytest.fixture
    def manager(self):
        """Create a DicomFileManager instance for testing"""
        return DicomFileManager()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_image_array_multi_frame(self):
        """Create a multi-frame RGB image array for testing (frames, height, width, channels)"""
        return np.random.randint(0, 255, (5, 10, 15, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_image_array_single_frame(self):
        """Create a single-frame grayscale image array for testing (frames, height, width, channels)"""
        return np.random.randint(0, 255, (1, 10, 15, 1), dtype=np.uint8)

    @pytest.fixture
    def manager_with_data(self, manager, temp_dir):
        """Create a manager with sample DICOM dataframe"""
        ds = self.create_test_dicom_file()
        filepath = self.save_dicom_file(ds, temp_dir)

        # Create dataframe with the test file
        dicom_data = [{
            'Filepath': filepath,
            'AnonFilename': 'test_output.dcm',
            'PatientUID': self.PATIENT_ID,
            'StudyUID': self.STUDY_UID,
            'StudyDate': self.CONTENT_DATE,
            'SeriesUID': self.SERIES_UID,
            'SeriesDate': self.CONTENT_DATE,
            'InstanceUID': self.SOP_INSTANCE_UID,
            'PhysicalDeltaX': self.PHYSICAL_DELTA_X,
            'PhysicalDeltaY': self.PHYSICAL_DELTA_Y,
            'ContentDate': self.CONTENT_DATE,
            'ContentTime': self.CONTENT_TIME,
            'Patch': False,
            'TransducerModel': self.TRANSDUCER_MODEL,
            'DICOMDataset': ds
        }]

        manager._create_dataframe(dicom_data)
        manager.current_dicom_index = 0
        return manager

    def test_init(self, manager):
        """Test DicomFileManager initialization"""
        assert manager.dicom_df is None
        assert manager.next_dicom_index == 0
        assert manager._temp_directories == []

    def test_get_transducer_model_valid(self, manager):
        """Test transducer model extraction with valid input"""
        assert manager.get_transducer_model("SC6-1s,02597") == "sc6-1s"
        assert manager.get_transducer_model("L12-3,12345") == "l12-3"
        assert manager.get_transducer_model("C1-5") == "c1-5"

    def test_get_transducer_model_invalid(self, manager):
        """Test transducer model extraction with invalid input"""
        assert manager.get_transducer_model("") == "unknown"
        assert manager.get_transducer_model(None) == "unknown"
        assert manager.get_transducer_model("   ") == "unknown"

    def test_get_number_of_instances_empty(self, manager):
        """Test get_number_of_instances with empty dataframe"""
        assert manager.get_number_of_instances() == 0

    def test_get_number_of_instances_with_data(self, manager):
        """Test get_number_of_instances with data"""
        manager.dicom_df = pd.DataFrame({'test': [1, 2, 3]})
        assert manager.get_number_of_instances() == 3

    def test_extract_dicom_info(self, manager, sample_dicom_filepath):
        """Test DICOM info extraction"""
        result = manager._extract_dicom_info(sample_dicom_filepath, False)

        assert result is not None
        assert len(result) == 13
        assert result['Filepath'] == sample_dicom_filepath
        assert result['AnonFilename'] == "4657494024_53302064.dcm"
        assert result['PatientUID'] == self.PATIENT_ID
        assert result['StudyUID'] == self.STUDY_UID
        assert result['SeriesUID'] == self.SERIES_UID
        assert result['InstanceUID'] == self.SOP_INSTANCE_UID
        assert result['ContentDate'] == self.CONTENT_DATE
        assert result['ContentTime'] == self.CONTENT_TIME
        assert result['Patch'] == False
        assert result['TransducerModel'] == self.TRANSDUCER_MODEL
        assert result['PhysicalDeltaX'] == self.PHYSICAL_DELTA_X
        assert result['PhysicalDeltaY'] == self.PHYSICAL_DELTA_Y
        assert result['DICOMDataset'] is not None

    def test_extract_dicom_info_skip_single_frame(self, manager, single_frame_dicom_filepath):
        """Test skipping single frame files when requested"""
        result = manager._extract_dicom_info(single_frame_dicom_filepath, True)
        assert result is None

    def test_extract_dicom_info_non_ultrasound(self, manager, non_ultrasound_dicom_filepath):
        """Test skipping non-ultrasound modalities"""
        result = manager._extract_dicom_info(non_ultrasound_dicom_filepath, False)
        assert result is None

    def test_extract_dicom_info_missing_required_fields(self, manager, temp_dir):
        """Test handling of missing required DICOM fields"""
        ds = self.create_test_dicom_file(PatientID=None)
        filepath = self.save_dicom_file(ds, temp_dir)
        Path(filepath).touch()

        result = manager._extract_dicom_info(filepath, False)

        assert result is None

    def test_extract_spacing_info_with_regions(self, manager):
        """Test spacing extraction with ultrasound regions"""
        dataset = self.create_test_dicom_file()
        region = Dataset()
        region.PhysicalDeltaX = 0.1
        region.PhysicalDeltaY = 0.15
        dataset.SequenceOfUltrasoundRegions = [region]

        delta_x, delta_y = manager._extract_spacing_info(dataset)
        assert delta_x == 0.1
        assert delta_y == 0.15

    def test_extract_spacing_info_without_regions(self, manager):
        """Test spacing extraction without ultrasound regions"""
        dataset = self.create_test_dicom_file()
        dataset.SequenceOfUltrasoundRegions = []

        delta_x, delta_y = manager._extract_spacing_info(dataset)
        assert delta_x is None
        assert delta_y is None

    def test_generate_filename_from_dicom(self, manager):
        """Test filename generation from DICOM data"""
        dataset = self.create_test_dicom_file()
        filename = manager._generate_filename_from_dicom(dataset)
        assert filename == "4657494024_53302064.dcm"

    def test_create_dataframe(self, manager):
        """Test dataframe creation"""
        dicom_data = [
            {
                'Filepath': 'file1.dcm',
                'ContentDate': '20240101',
                'ContentTime': '120000',
                'PatientUID': 'patient123',
                'StudyUID': 'study456',
                'PhysicalDeltaX': 0.1,
                'PhysicalDeltaY': 0.15,
                'TransducerModel': 'sc6-1s'
            },
            {
                'Filepath': 'file2.dcm',
                'ContentDate': '20240101',
                'ContentTime': '120100',
                'PatientUID': 'patient123',
                'StudyUID': 'study456',
                'PhysicalDeltaX': None,
                'PhysicalDeltaY': None,
                'TransducerModel': 'l12-3'
            }
        ]
        manager._create_dataframe(dicom_data)

        assert manager.dicom_df is not None
        assert len(manager.dicom_df) == 2
        assert 'TransducerModel' in manager.dicom_df.columns
        assert 'SeriesNumber' in manager.dicom_df.columns
        assert manager.next_dicom_index == 0

    def test_update_progress_from_output_no_dataframe(self, manager):
        """Test progress update with no dataframe"""
        result = manager.update_progress_from_output("output")
        assert result is None

    def test_update_progress_from_output_all_processed(self, manager, temp_dir):
        """Test progress update when all files are processed"""
        # Create test dataframe
        manager.dicom_df = pd.DataFrame({
            'AnonFilename': ['file1.dcm', 'file2.dcm']
        })

        # Create output files
        for filename in ['file1.dcm', 'file2.dcm']:
            Path(os.path.join(temp_dir, filename)).touch()

        result = manager.update_progress_from_output(temp_dir)
        assert result is None  # All files processed

    def test_update_progress_from_output_partial_processed(self, manager, temp_dir):
        """Test progress update with partially processed files"""
        # Create test dataframe
        manager.dicom_df = pd.DataFrame({
            'AnonFilename': ['file1.dcm', 'file2.dcm', 'file3.dcm']
        })

        # Create only first file
        Path(os.path.join(temp_dir, 'file1.dcm')).touch()

        result = manager.update_progress_from_output(temp_dir)
        assert result == 1
        assert manager.next_dicom_index == 1

    def test_get_file_for_instance_uid_found(self, manager):
        """Test getting file path for instance UID when found"""
        manager.dicom_df = pd.DataFrame({
            'InstanceUID': ['UID1', 'UID2', 'UID3'],
            'Filepath': ['file1.dcm', 'file2.dcm', 'file3.dcm']
        })

        result = manager._get_file_for_instance_uid('UID2')
        assert result == 'file2.dcm'

    def test_get_file_for_instance_uid_not_found(self, manager):
        """Test getting file path for instance UID when not found"""
        manager.dicom_df = pd.DataFrame({
            'InstanceUID': ['UID1', 'UID2'],
            'Filepath': ['file1.dcm', 'file2.dcm']
        })

        result = manager._get_file_for_instance_uid('UID999')
        assert result is None

    def test_get_file_for_instance_uid_no_dataframe(self, manager):
        """Test getting file path with no dataframe"""
        result = manager._get_file_for_instance_uid('UID1')
        assert result is None

    def test_dicom_header_to_dict_simple(self, manager):
        """Test DICOM header to dict conversion with simple elements"""
        dataset = self.create_test_dicom_file()
        result = manager.dicom_header_to_dict(dataset)

        assert result == {
            "Patient ID": self.PATIENT_ID,
            "Patient's Name": self.PATIENT_NAME,
            "Study Instance UID": self.STUDY_UID,
            "Series Instance UID": self.SERIES_UID,
            "SOP Class UID": self.SOP_CLASS_UID,
            "SOP Instance UID": self.SOP_INSTANCE_UID,
            "Modality": self.MODALITY,
            "Number of Frames": str(self.NUMBER_OF_FRAMES),
            "Content Date": self.CONTENT_DATE,
            "Content Time": self.CONTENT_TIME,
            "Transducer Type": self.TRANSDUCER_TYPE,
            "Rows": self.ROWS,
            "Columns": self.COLUMNS,
            "Bits Allocated": self.BITS_ALLOCATED,
            "Bits Stored": self.BITS_STORED,
            "High Bit": self.HIGH_BIT,
            "Pixel Representation": self.PIXEL_REPRESENTATION,
            "Samples per Pixel": self.SAMPLES_PER_PIXEL,
            "Photometric Interpretation": self.PHOTOMETRIC_INTERPRETATION,
            "Sequence of Ultrasound Regions": [
                {
                    "Physical Delta X": self.PHYSICAL_DELTA_X,
                    "Physical Delta Y": self.PHYSICAL_DELTA_Y
                }
            ]
        }

    def test_increment_dicom_index_basic(self, manager):
        """Test basic DICOM index increment"""
        manager.dicom_df = pd.DataFrame({'test': [1, 2, 3]})
        manager.next_dicom_index = 0

        result = manager._increment_dicom_index()
        assert result is True
        assert manager.next_dicom_index == 1

    def test_increment_dicom_index_at_end(self, manager):
        """Test DICOM index increment at end of dataframe"""
        manager.dicom_df = pd.DataFrame({'test': [1, 2]})
        manager.next_dicom_index = 1

        result = manager._increment_dicom_index()
        assert result is False
        assert manager.next_dicom_index == 2

    def test_increment_dicom_index_with_continue_progress(self, manager, temp_dir):
        """Test DICOM index increment with continue progress"""
        # Create test dataframe with multiple files
        manager.dicom_df = pd.DataFrame({
            'AnonFilename': ['file1.dcm', 'file2.dcm', 'file3.dcm', 'file4.dcm']
        })
        manager.next_dicom_index = 0

        # Create some existing output files (file1.dcm and file2.dcm already exist)
        Path(os.path.join(temp_dir, 'file1.dcm')).touch()
        Path(os.path.join(temp_dir, 'file2.dcm')).touch()
        # file3.dcm and file4.dcm don't exist

        # Test increment with continue_progress=True
        result = manager._increment_dicom_index(temp_dir, continue_progress=True)

        # Should skip to file3.dcm (index 2) since file1.dcm and file2.dcm already exist
        assert result is True
        assert manager.next_dicom_index == 2

        # Test increment again - should go to file4.dcm (index 3)
        result = manager._increment_dicom_index(temp_dir, continue_progress=True)
        assert result is True
        assert manager.next_dicom_index == 3

        # Test increment again - should go beyond end (index 4)
        result = manager._increment_dicom_index(temp_dir, continue_progress=True)
        assert result is False
        assert manager.next_dicom_index == 4

    @patch('os.path.exists')
    @patch('shutil.rmtree')
    def test_cleanup_temp_directory(self, mock_rmtree, mock_exists, manager):
        """Test temporary directory cleanup"""
        temp_dir = "/tmp/test_dir"
        manager._temp_directories = [temp_dir]
        mock_exists.return_value = True

        manager._cleanup_temp_directory(temp_dir)

        mock_rmtree.assert_called_once_with(temp_dir)
        assert temp_dir not in manager._temp_directories

    @patch('os.path.exists')
    @patch('shutil.rmtree')
    def test_cleanup_temp_directory_not_exists(self, mock_rmtree, mock_exists, manager):
        """Test cleanup of non-existent directory"""
        temp_dir = "/tmp/nonexistent"
        manager._temp_directories = [temp_dir]
        mock_exists.return_value = False

        manager._cleanup_temp_directory(temp_dir)

        mock_rmtree.assert_not_called()
        assert temp_dir not in manager._temp_directories

    @patch('os.walk')
    @patch.object(DicomFileManager, '_create_progress_dialog')
    @patch.object(DicomFileManager, '_extract_dicom_info')
    @patch.object(DicomFileManager, '_create_dataframe')
    def test_scan_directory_integration(self, mock_create_df, mock_extract, mock_dialog, mock_walk, manager):
        """Test the full scan_directory integration"""
        # Mock os.walk to return test files
        mock_walk.return_value = [
            ("/test", [], ["file1.dcm", "file2.txt", "file3.dcm"])
        ]

        # Mock progress dialog
        mock_progress = Mock()
        mock_dialog.return_value = mock_progress

        # Mock extract_dicom_info to return valid data for .dcm files only
        def extract_side_effect(file_path, skip_single):
            if file_path.endswith('.dcm'):
                return ["mock", "data"]
            return None

        mock_extract.side_effect = extract_side_effect

        # Mock create_dataframe
        manager.dicom_df = pd.DataFrame({'test': [1, 2]})

        result = manager.scan_directory("/test", False)

        assert result == 2
        assert mock_extract.call_count == 2  # Only called for .dcm files
        mock_create_df.assert_called_once()
        mock_progress.close.assert_called_once()

    @patch('pydicom.dcmread')
    def test_extract_dicom_info_read_error(self, mock_dcmread, manager, temp_dir):
        """Test handling of DICOM read errors"""
        mock_dcmread.side_effect = Exception("Cannot read DICOM file")

        test_file = os.path.join(temp_dir, "test.dcm")
        Path(test_file).touch()

        result = manager._extract_dicom_info(test_file, False)
        assert result is None

    @patch('shutil.rmtree')
    def test_cleanup_temp_directory_error(self, mock_rmtree, manager):
        """Test error handling in temp directory cleanup"""
        mock_rmtree.side_effect = PermissionError("Cannot delete directory")
        temp_dir = "/tmp/test_dir"
        manager._temp_directories = [temp_dir]

        # Should not raise exception
        manager._cleanup_temp_directory(temp_dir)

        # Directory should still be removed from tracking list
        assert temp_dir not in manager._temp_directories

    def test_save_anonymized_dicom_no_dataframe(self, manager, sample_image_array_single_frame, temp_dir):
        """Test save_anonymized_dicom with no dataframe"""
        output_path = os.path.join(temp_dir, "output.dcm")

        # Should not raise exception, just log error
        manager.save_anonymized_dicom(sample_image_array_single_frame, output_path)

        # File should not be created
        assert not os.path.exists(output_path)

    def test_save_anonymized_dicom_invalid_index(self, manager, sample_image_array_multi_frame, temp_dir):
        """Test save_anonymized_dicom with invalid current_dicom_index"""
        manager.dicom_df = pd.DataFrame({'test': [1, 2, 3]})
        manager.current_dicom_index = 5  # Out of bounds

        output_path = os.path.join(temp_dir, "output.dcm")

        # Should not raise exception, just log error
        manager.save_anonymized_dicom(sample_image_array_multi_frame, output_path)

        # File should not be created
        assert not os.path.exists(output_path)

    def test_save_anonymized_dicom_success_multi_frame(self, manager_with_data, sample_image_array_multi_frame, temp_dir):
        """Test successful save_anonymized_dicom with multi-frame image array"""
        output_path = os.path.join(temp_dir, "output.dcm")

        manager_with_data.save_anonymized_dicom(sample_image_array_multi_frame, output_path)

        # File should be created
        assert os.path.exists(output_path)

        # Verify the DICOM file
        saved_ds = pydicom.dcmread(output_path)
        assert saved_ds.NumberOfFrames == 5
        assert saved_ds.Rows == 10
        assert saved_ds.Columns == 15
        assert saved_ds.SamplesPerPixel == 3

    def test_save_anonymized_dicom_success_single_frame(self, manager_with_data, sample_image_array_single_frame, temp_dir):
        """Test successful save_anonymized_dicom with single-frame image array"""
        output_path = os.path.join(temp_dir, "output.dcm")

        manager_with_data.save_anonymized_dicom(sample_image_array_single_frame, output_path)

        # File should be created
        assert os.path.exists(output_path)

        # Verify the DICOM file
        saved_ds = pydicom.dcmread(output_path)
        assert saved_ds.NumberOfFrames == 1  # Single frame
        assert saved_ds.Rows == 10
        assert saved_ds.Columns == 15
        assert saved_ds.SamplesPerPixel == 1

    def test_create_base_dicom_dataset_multi_frame(self, manager_with_data, sample_image_array_multi_frame):
        """Test _create_base_dicom_dataset with multi-frame array"""
        current_record = manager_with_data.dicom_df.iloc[0]

        ds = manager_with_data._create_base_dicom_dataset(sample_image_array_multi_frame, current_record)

        assert ds.NumberOfFrames == 5
        assert ds.Rows == 10
        assert ds.Columns == 15
        assert ds.SamplesPerPixel == 3
        assert ds.Modality == 'US'
        assert ds.PhotometricInterpretation == "YBR_FULL_422"

    def test_create_base_dicom_dataset_single_frame(self, manager_with_data, sample_image_array_single_frame):
        """Test _create_base_dicom_dataset with single-frame array"""
        current_record = manager_with_data.dicom_df.iloc[0]

        ds = manager_with_data._create_base_dicom_dataset(sample_image_array_single_frame, current_record)

        assert ds.NumberOfFrames == 1
        assert ds.Rows == 10
        assert ds.Columns == 15
        assert ds.SamplesPerPixel == 1
        assert ds.PhotometricInterpretation == "MONOCHROME2"

    def test_copy_spacing_info_with_regions(self, manager_with_data):
        """Test _copy_spacing_info with ultrasound regions"""
        ds = pydicom.Dataset()
        current_record = manager_with_data.dicom_df.iloc[0]

        manager_with_data._copy_spacing_info(ds, current_record)

        assert hasattr(ds, 'SequenceOfUltrasoundRegions')
        assert len(ds.SequenceOfUltrasoundRegions) > 0
        assert hasattr(ds, 'PixelSpacing')
        assert len(ds.PixelSpacing) == 2

    def test_copy_spacing_info_without_regions(self, manager_with_data):
        """Test _copy_spacing_info without ultrasound regions"""
        ds = pydicom.Dataset()
        current_record = manager_with_data.dicom_df.iloc[0]

        # Remove ultrasound regions from source
        current_record.DICOMDataset.SequenceOfUltrasoundRegions = []

        manager_with_data._copy_spacing_info(ds, current_record)

        assert hasattr(ds, 'PixelSpacing')  # Should still have pixel spacing

    def test_copy_source_metadata(self, manager_with_data, temp_dir):
        """Test _copy_source_metadata"""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        # Check copied attributes
        assert ds.BitsAllocated == self.BITS_ALLOCATED
        assert ds.BitsStored == self.BITS_STORED
        assert ds.SeriesDescription == "test.dcm"
        assert hasattr(ds, 'SOPClassUID')
        assert hasattr(ds, 'SOPInstanceUID')
        assert hasattr(ds, 'SeriesInstanceUID')

    def test_copy_and_generate_uids_with_source_uids(self, manager_with_data, temp_dir):
        """Test _copy_and_generate_uids with existing UIDs in source"""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_and_generate_uids(ds, source_ds, output_path)

        # Should copy existing UIDs
        assert ds.SOPClassUID == source_ds.SOPClassUID
        assert ds.SOPInstanceUID == source_ds.SOPInstanceUID
        assert ds.StudyInstanceUID == source_ds.StudyInstanceUID
        # SeriesInstanceUID should always be generated new
        assert ds.SeriesInstanceUID != source_ds.SeriesInstanceUID

    def test_copy_and_generate_uids_missing_source_uids(self, manager_with_data, temp_dir):
        """Test _copy_and_generate_uids with missing UIDs in source"""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()  # Empty dataset
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_and_generate_uids(ds, source_ds, output_path)

        # Should generate new UIDs
        assert hasattr(ds, 'SOPClassUID')
        assert hasattr(ds, 'SOPInstanceUID')
        assert hasattr(ds, 'StudyInstanceUID')
        assert hasattr(ds, 'SeriesInstanceUID')

    def test_apply_anonymization_with_new_patient_info(self, manager_with_data):
        """Test _apply_anonymization with new patient information"""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset

        manager_with_data._apply_anonymization(
            ds, source_ds,
            new_patient_name="Anonymous^Patient",
            new_patient_id="ANON123"
        )

        assert ds.PatientName == "Anonymous^Patient"
        assert ds.PatientID == "ANON123"
        assert hasattr(ds, 'StudyDate')  # Date shifting should be applied

    def test_apply_anonymization_without_new_patient_info(self, manager_with_data):
        """Test _apply_anonymization without new patient information"""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset

        manager_with_data._apply_anonymization(ds, source_ds)

        assert ds.PatientName == source_ds.PatientName
        assert ds.PatientID == source_ds.PatientID

    def test_apply_date_shifting(self, manager_with_data):
        """Test _apply_date_shifting"""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset

        manager_with_data._apply_date_shifting(ds, source_ds)

        assert hasattr(ds, 'StudyDate')
        assert hasattr(ds, 'SeriesDate')
        assert hasattr(ds, 'ContentDate')
        assert hasattr(ds, 'StudyTime')
        assert hasattr(ds, 'SeriesTime')
        assert hasattr(ds, 'ContentTime')

        # Dates should be shifted (different from original)
        # Note: Due to seeding, the shift should be consistent
        assert ds.ContentDate != source_ds.ContentDate

    def test_apply_date_shifting_invalid_dates(self, manager_with_data):
        """Test _apply_date_shifting with invalid date formats"""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "TEST123"
        source_ds.StudyDate = "invalid_date"
        source_ds.SeriesDate = "20240101"  # Valid
        source_ds.ContentDate = ""

        manager_with_data._apply_date_shifting(ds, source_ds)

        # Should handle invalid dates gracefully
        assert ds.StudyDate == source_ds.StudyDate  # Should keep original invalid date
        assert ds.SeriesDate != source_ds.SeriesDate  # Should shift valid date
        assert ds.ContentDate == source_ds.ContentDate  # Should use default

    def test_set_conformance_attributes_multiframe(self, manager_with_data):
        """Test _set_conformance_attributes with multi-frame dataset"""
        ds = pydicom.Dataset()
        ds.NumberOfFrames = 5
        ds.SamplesPerPixel = 1

        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset

        manager_with_data._set_conformance_attributes(ds, source_ds)

        # Check Type 2 elements
        assert hasattr(ds, 'PatientBirthDate')
        assert hasattr(ds, 'ReferringPhysicianName')
        assert hasattr(ds, 'AccessionNumber')
        assert hasattr(ds, 'Laterality')
        assert hasattr(ds, 'InstanceNumber')
        assert hasattr(ds, 'PatientOrientation')
        assert hasattr(ds, 'ImageType')

        # Check multi-frame specific attributes
        assert hasattr(ds, 'FrameTime')
        assert hasattr(ds, 'FrameIncrementPointer')

    def test_set_conformance_attributes_color_image(self, manager_with_data):
        """Test _set_conformance_attributes with color image"""
        ds = pydicom.Dataset()
        ds.SamplesPerPixel = 3

        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset

        manager_with_data._set_conformance_attributes(ds, source_ds)

        # Check color image specific attributes
        assert hasattr(ds, 'PlanarConfiguration')

    def test_set_compressed_pixel_data(self, manager_with_data, sample_image_array_multi_frame):
        """Test _set_compressed_pixel_data"""
        ds = pydicom.Dataset()

        manager_with_data._set_compressed_pixel_data(ds, sample_image_array_multi_frame)

        assert hasattr(ds, 'PixelData')
        assert ds.LossyImageCompression == '01'
        assert ds.LossyImageCompressionMethod == 'ISO_10918_1'
        assert ds['PixelData'].VR == 'OB'
        assert ds['PixelData'].is_undefined_length == True

    def test_compress_frame_to_jpeg_2d(self, manager_with_data):
        """Test _compress_frame_to_jpeg with 2D frame"""
        frame = np.random.randint(0, 255, (10, 15), dtype=np.uint8)

        compressed = manager_with_data._compress_frame_to_jpeg(frame)

        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

    def test_compress_frame_to_jpeg_3d_grayscale(self, manager_with_data):
        """Test _compress_frame_to_jpeg with 3D grayscale frame"""
        frame = np.random.randint(0, 255, (10, 15, 1), dtype=np.uint8)

        compressed = manager_with_data._compress_frame_to_jpeg(frame)

        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

    def test_compress_frame_to_jpeg_3d_color(self, manager_with_data):
        """Test _compress_frame_to_jpeg with 3D color frame"""
        frame = np.random.randint(0, 255, (10, 15, 3), dtype=np.uint8)

        compressed = manager_with_data._compress_frame_to_jpeg(frame, quality=85)

        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

    @patch('pydicom.dataset.FileDataset.save_as')
    def test_create_and_save_dicom_file(self, mock_save_as, manager_with_data, temp_dir):
        """Test _create_and_save_dicom_file"""
        ds = pydicom.Dataset()
        ds.SOPClassUID = self.SOP_CLASS_UID
        ds.SOPInstanceUID = self.SOP_INSTANCE_UID

        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._create_and_save_dicom_file(ds, output_path)

        # Verify save_as was called
        mock_save_as.assert_called_once_with(output_path)

    def test_get_series_number_for_current_instance_found(self, manager_with_data):
        """Test _get_series_number_for_current_instance when found"""
        result = manager_with_data._get_series_number_for_current_instance()

        assert result == "1"  # Should be the series number from dataframe

    def test_get_series_number_for_current_instance_no_dataframe(self, manager):
        """Test _get_series_number_for_current_instance with no dataframe"""
        result = manager._get_series_number_for_current_instance()

        assert result == "1"  # Should return default

    def test_get_series_number_for_current_instance_not_found(self, manager_with_data):
        """Test _get_series_number_for_current_instance when instance not found"""
        # Modify the dataframe to have different instance UID
        manager_with_data.dicom_df.loc[0, 'InstanceUID'] = 'different_uid'

        result = manager_with_data._get_series_number_for_current_instance()

        assert result == "1"  # Should return default

    @patch.object(DicomFileManager, '_create_base_dicom_dataset')
    @patch.object(DicomFileManager, '_copy_source_metadata')
    @patch.object(DicomFileManager, '_apply_anonymization')
    @patch.object(DicomFileManager, '_set_conformance_attributes')
    @patch.object(DicomFileManager, '_set_compressed_pixel_data')
    @patch.object(DicomFileManager, '_create_and_save_dicom_file')
    def test_save_anonymized_dicom_integration(self, mock_save_file, mock_set_pixel_data,
                                              mock_set_conformance, mock_apply_anon,
                                              mock_copy_metadata, mock_create_base,
                                              manager_with_data, sample_image_array_multi_frame, temp_dir):
        """Test the full save_anonymized_dicom integration"""
        mock_ds = Mock()
        mock_create_base.return_value = mock_ds

        output_path = os.path.join(temp_dir, "output.dcm")

        manager_with_data.save_anonymized_dicom(
            sample_image_array_multi_frame,
            output_path,
            new_patient_name="Test^Patient",
            new_patient_id="TEST123"
        )

        # Verify all methods were called in correct order
        mock_create_base.assert_called_once()
        mock_copy_metadata.assert_called_once()
        mock_apply_anon.assert_called_once_with(
            mock_ds,
            manager_with_data.dicom_df.iloc[0].DICOMDataset,
            "Test^Patient",
            "TEST123"
        )
        mock_set_conformance.assert_called_once()
        mock_set_pixel_data.assert_called_once()
        mock_save_file.assert_called_once_with(mock_ds, output_path)

if __name__ == "__main__":
    pytest.main([__file__])
