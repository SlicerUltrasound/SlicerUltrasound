from pydicom import Sequence
import pytest
import os
import tempfile
import shutil
import pandas as pd
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
import os
from unittest.mock import Mock, patch
from pathlib import Path

# Mock slicer module for testing
import sys
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


if __name__ == "__main__":
    pytest.main([__file__])
