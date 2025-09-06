import pytest
import os
import sys
import tempfile
import shutil
import pandas as pd
import pydicom
import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from unittest.mock import Mock, patch
from pathlib import Path
import json

# Import the module under test
from ..dicom_file_manager import DicomFileManager

class TestDicomFileManager:
    """Test suite for DicomFileManager class"""

    PATIENT_ID = "TEST123"
    PATIENT_NAME = "REMOVE^THIS^PATIENT^NAME"
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
    INPUT_FOLDER = '/'
    FILE_NAME = "test_output.dcm"

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

    def save_dicom_file(self, ds: FileDataset, temp_dir: str, filename: str) -> str:
        """Save the DICOM dataset to a file"""
        # Generate filename and save
        filepath = os.path.join(temp_dir, filename)

        ds.save_as(filepath)
        return filepath

    @pytest.fixture
    def sample_dicom_filepath(self, temp_dir, manager):
        """Create a sample DICOM file for testing"""
        ds = self.create_test_dicom_file()
        filename = manager._generate_filename_from_dicom(ds)
        filepath = self.save_dicom_file(ds, temp_dir, filename)
        yield filepath

    @pytest.fixture
    def single_frame_dicom_filepath(self, temp_dir, manager):
        """Create a single-frame DICOM file for testing"""
        ds = self.create_test_dicom_file(NumberOfFrames=1)
        filename = manager._generate_filename_from_dicom(ds)
        filepath = self.save_dicom_file(ds, temp_dir, filename)
        yield filepath

    @pytest.fixture
    def non_ultrasound_dicom_filepath(self, temp_dir, manager):
        """Create a non-ultrasound DICOM file for testing"""
        ds = self.create_test_dicom_file(Modality='CT')
        filename = manager._generate_filename_from_dicom(ds)
        filepath = self.save_dicom_file(ds, temp_dir, filename)
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
        filename = manager._generate_filename_from_dicom(ds)
        filepath = self.save_dicom_file(ds, temp_dir, filename)

        # Create dataframe with the test file
        dicom_data = [{
            'InputPath': filepath,
            'OutputPath': os.path.relpath(filepath, temp_dir),
            'AnonFilename': self.FILE_NAME,
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
        manager.current_index = 0
        return manager

    def test_init(self, manager):
        """Test DicomFileManager initialization"""
        assert manager.dicom_df is None
        assert manager.next_index == 0

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
        result = manager._extract_dicom_info(sample_dicom_filepath, self.INPUT_FOLDER, False)
        filename, _, _ = manager.generate_filename_from_dicom_dataset(result['DICOMDataset'])

        assert result is not None
        assert len(result) == 14
        assert result['InputPath'] == sample_dicom_filepath
        assert filename in result['OutputPath']
        assert result['AnonFilename'] == filename
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
        result = manager._extract_dicom_info(single_frame_dicom_filepath, self.INPUT_FOLDER, True)
        assert result is None

    def test_extract_dicom_info_non_ultrasound(self, manager, non_ultrasound_dicom_filepath):
        """Test skipping non-ultrasound modalities"""
        result = manager._extract_dicom_info(non_ultrasound_dicom_filepath, self.INPUT_FOLDER, False)
        assert result is None

    def test_extract_dicom_info_missing_required_fields(self, manager, temp_dir):
        """Test handling of missing required DICOM fields"""
        ds = self.create_test_dicom_file(Modality='')
        filename = manager._generate_filename_from_dicom(ds)
        filepath = self.save_dicom_file(ds, temp_dir, filename)
        Path(filepath).touch()

        result = manager._extract_dicom_info(filepath, self.INPUT_FOLDER, False)

        assert result is None

    def test_extract_dicom_info_includes_output_path(self, manager, temp_dir):
        # Create a subdirectory structure
        subdir = os.path.join(temp_dir, "patient1", "study1")
        os.makedirs(subdir)

        ds = self.create_test_dicom_file()
        filename = manager._generate_filename_from_dicom(ds)
        filepath = self.save_dicom_file(ds, subdir, filename)

        result = manager._extract_dicom_info(filepath, temp_dir, False)

        assert result is not None
        assert 'OutputPath' in result
        assert result['OutputPath'] == os.path.relpath(filepath, temp_dir)
        # Should also have FilePath for backward compatibility
        assert result['InputPath'] == filepath

    def test_extract_dicom_info_with_nested_structure(self, manager, temp_dir):
        nested_dir = os.path.join(temp_dir, "patient", "study", "series", "instance")
        os.makedirs(nested_dir)

        ds = self.create_test_dicom_file()
        filename = manager._generate_filename_from_dicom(ds)
        filepath = self.save_dicom_file(ds, nested_dir, filename)

        result = manager._extract_dicom_info(filepath, temp_dir, False)

        assert result is not None
        expected_relative = os.path.join("patient", "study", "series", "instance", os.path.basename(filepath))
        assert result['OutputPath'] == expected_relative

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

    def test_create_dataframe(self, manager, temp_dir):
        """Test dataframe creation"""
        dicom_data = [
            {
                'InputPath': 'file1.dcm',
                'OutputPath': os.path.relpath('file1.dcm', temp_dir),
                'ContentDate': '20240101',
                'ContentTime': '120000',
                'PatientUID': 'patient123',
                'StudyUID': 'study456',
                'PhysicalDeltaX': 0.1,
                'PhysicalDeltaY': 0.15,
                'TransducerModel': 'sc6-1s'
            },
            {
                'InputPath': 'file2.dcm',
                'OutputPath': os.path.relpath('file2.dcm', temp_dir),
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
        assert manager.next_index == 0

    def test_update_progress_from_output_no_dataframe(self, manager):
        """Test progress update with no dataframe"""
        result = manager.update_progress_from_output("output", True)
        assert result is None

    def test_update_progress_from_output_all_processed(self, manager, temp_dir):
        """Test progress update when all files are processed"""
        # Create test dataframe
        manager.dicom_df = pd.DataFrame({
            'AnonFilename': ['file1.dcm', 'file2.dcm'],
            'OutputPath': ['file1.dcm', 'file2.dcm']
        })

        # Create output files
        for filename in ['file1.dcm', 'file2.dcm']:
            Path(os.path.join(temp_dir, filename)).touch()

        result = manager.update_progress_from_output(temp_dir, True)
        assert result is None  # All files processed

    def test_update_progress_from_output_partial_processed(self, manager, temp_dir):
        """Test progress update with partially processed files"""
        # Create test dataframe
        manager.dicom_df = pd.DataFrame({
            'AnonFilename': ['file1.dcm', 'file2.dcm', 'file3.dcm'],
            'OutputPath': ['file1.dcm', 'file2.dcm', 'file3.dcm']
        })

        # Create only first file
        Path(os.path.join(temp_dir, 'file1.dcm')).touch()

        result = manager.update_progress_from_output(temp_dir, True)
        assert result == 1
        assert manager.next_index == 1

    def test_update_progress_from_output_with_preserve_structure(self, manager, temp_dir):
        # Create test dataframe with OutputPath
        manager.dicom_df = pd.DataFrame({
            'OutputPath': ['patient1/file1.dcm', 'patient2/file2.dcm', 'patient3/file3.dcm']
        })

        # Create nested directory structure and first file
        os.makedirs(os.path.join(temp_dir, 'patient1'))
        Path(os.path.join(temp_dir, 'patient1', 'file1.dcm')).touch()

        result = manager.update_progress_from_output(temp_dir, preserve_directory_structure=True)

        assert result == 1
        assert manager.next_index == 1

    def test_update_progress_from_output_with_flatten_structure(self, manager, temp_dir):
        # Create test dataframe with OutputPath
        manager.dicom_df = pd.DataFrame({
            'OutputPath': ['patient1/subdir/file1.dcm', 'patient2/subdir/file2.dcm']
        })

        # Create flattened file (should be found at root level)
        Path(os.path.join(temp_dir, 'file1.dcm')).touch()

        result = manager.update_progress_from_output(temp_dir, preserve_directory_structure=False)

        assert result == 1
        assert manager.next_index == 1

    def test_get_file_for_instance_uid_found(self, manager):
        """Test getting file path for instance UID when found"""
        manager.dicom_df = pd.DataFrame({
            'InstanceUID': ['UID1', 'UID2', 'UID3'],
            'InputPath': ['file1.dcm', 'file2.dcm', 'file3.dcm']
        })

        result = manager._get_file_for_instance_uid('UID2')
        assert result == 'file2.dcm'

    def test_get_file_for_instance_uid_not_found(self, manager):
        """Test getting file path for instance UID when not found"""
        manager.dicom_df = pd.DataFrame({
            'InstanceUID': ['UID1', 'UID2'],
            'InputPath': ['file1.dcm', 'file2.dcm']
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
        manager.next_index = 0

        result = manager.increment_dicom_index()
        assert result is True
        assert manager.next_index == 1

    def test_increment_dicom_index_at_end(self, manager):
        """Test DICOM index increment at end of dataframe"""
        manager.dicom_df = pd.DataFrame({'test': [1, 2]})
        manager.next_index = 1

        result = manager.increment_dicom_index()
        assert result is False
        assert manager.next_index == 2

    def test_increment_dicom_index_with_continue_progress(self, manager, temp_dir):
        """Test DICOM index increment with continue progress"""
        # Create test dataframe with multiple files
        manager.dicom_df = pd.DataFrame({
            'AnonFilename': ['file1.dcm', 'file2.dcm', 'file3.dcm', 'file4.dcm'],
            'OutputPath': ['file1.dcm', 'file2.dcm', 'file3.dcm', 'file4.dcm']
        })
        manager.next_index = 0

        # Create some existing output files (file1.dcm and file2.dcm already exist)
        Path(os.path.join(temp_dir, 'file1.dcm')).touch()
        Path(os.path.join(temp_dir, 'file2.dcm')).touch()
        # file3.dcm and file4.dcm don't exist

        # Test increment with continue_progress=True
        result = manager.increment_dicom_index(temp_dir, continue_progress=True)

        # Should skip to file3.dcm (index 2) since file1.dcm and file2.dcm already exist
        assert result is True
        assert manager.next_index == 2

        # Test increment again - should go to file4.dcm (index 3)
        result = manager.increment_dicom_index(temp_dir, continue_progress=True)
        assert result is True
        assert manager.next_index == 3

        # Test increment again - should go beyond end (index 4)
        result = manager.increment_dicom_index(temp_dir, continue_progress=True)
        assert result is False
        assert manager.next_index == 4

    def test_increment_dicom_index_with_preserve_structure(self, manager, temp_dir):
        # Create test dataframe
        manager.dicom_df = pd.DataFrame({
            'OutputPath': ['dir1/file1.dcm', 'dir2/file2.dcm', 'dir3/file3.dcm']
        })
        manager.next_index = 0

        # Create first and second files in nested input directory
        os.makedirs(os.path.join(temp_dir, 'dir1'))
        Path(os.path.join(temp_dir, 'dir1', 'file1.dcm')).touch()
        os.makedirs(os.path.join(temp_dir, 'dir2'))
        Path(os.path.join(temp_dir, 'dir2', 'file2.dcm')).touch()

        result = manager.increment_dicom_index(
            temp_dir,
            continue_progress=True,
            preserve_directory_structure=True
        )

        # Since the output directory is preserved, the next file to be processed is the
        # third file in the input directory since the first two files already exist in the output directory.
        assert manager.next_index == 2
        assert result is True

    def test_increment_dicom_index_without_preserve_structure(self, manager, temp_dir):
        # Create test dataframe
        manager.dicom_df = pd.DataFrame({
            'OutputPath': ['dir1/file1.dcm', 'dir2/file2.dcm', 'dir3/file3.dcm']
        })

        # Create first and second files in nested input directory
        os.makedirs(os.path.join(temp_dir, 'dir1'))
        Path(os.path.join(temp_dir, 'dir1', 'file1.dcm')).touch()
        os.makedirs(os.path.join(temp_dir, 'dir2'))
        Path(os.path.join(temp_dir, 'dir2', 'file2.dcm')).touch()

        result = manager.increment_dicom_index(
            temp_dir,
            continue_progress=True,
            preserve_directory_structure=False
        )

        # Since the output directory is not preserved, the next file to be processed is the
        # second file in the input directory since the first file already exists in the output directory.
        assert manager.next_index == 1
        assert result is True

    @patch('pydicom.dcmread')
    def test_extract_dicom_info_read_error(self, mock_dcmread, manager, temp_dir):
        """Test handling of DICOM read errors"""
        mock_dcmread.side_effect = Exception("Cannot read DICOM file")

        test_file = os.path.join(temp_dir, "test.dcm")
        Path(test_file).touch()

        result = manager._extract_dicom_info(test_file, self.INPUT_FOLDER, False)
        assert result is None

    def test_save_anonymized_dicom_no_dataframe(self, manager, sample_image_array_single_frame, temp_dir):
        """Test save_anonymized_dicom with no dataframe"""
        output_path = os.path.join(temp_dir, "output.dcm")

        # Should not raise exception, just log error
        manager.save_anonymized_dicom(sample_image_array_single_frame, output_path)

        # File should not be created
        assert not os.path.exists(output_path)

    def test_save_anonymized_dicom_invalid_index(self, manager, sample_image_array_multi_frame, temp_dir):
        """Test save_anonymized_dicom with invalid current_index"""
        manager.dicom_df = pd.DataFrame({'test': [1, 2, 3]})
        manager.current_index = 5  # Out of bounds

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

    def test_save_anonymized_dicom_with_labels(self, manager_with_data, sample_image_array_multi_frame, temp_dir):
        """Test successful save_anonymized_dicom with labels"""
        output_path = os.path.join(temp_dir, "output.dcm")
        labels = ["label1", "label2"]
        manager_with_data.save_anonymized_dicom(sample_image_array_multi_frame, output_path, labels=labels)

        # File should be created
        assert os.path.exists(output_path)

        # Verify the DICOM file
        saved_ds = pydicom.dcmread(output_path)
        assert saved_ds.SeriesDescription == "label1 label2"

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

        assert ds.PatientName != source_ds.PatientName # Patient name should be set to a random value
        assert ds.PatientID != source_ds.PatientID # Patient ID should be set to a random value

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

    def test_copy_source_metadata_excludes_patient_sensitive_data(self, manager_with_data, temp_dir):
        """Test that PatientAge and PatientSex are no longer copied"""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset

        # Add patient age and sex to source
        source_ds.PatientAge = "025Y"
        source_ds.PatientSex = "M"

        output_path = os.path.join(temp_dir, "test.dcm")
        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        # These should NOT be copied
        assert not hasattr(ds, 'PatientID')
        assert not hasattr(ds, 'PatientBirthDate')

        # But these should still be copied
        assert hasattr(ds, 'BitsAllocated')
        assert hasattr(ds, 'TransducerType')

    def test_apply_anonymization_generates_uids_when_none_provided(self, manager_with_data):
        """Test that _apply_anonymization generates UIDs when no patient info provided"""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset

        # Don't provide new patient info
        manager_with_data._apply_anonymization(ds, source_ds)

        # Should be blank strings
        assert ds.PatientName == ""
        assert ds.PatientID == ""
        assert ds.PatientBirthDate == ""
        assert ds.ReferringPhysicianName == ""
        assert ds.AccessionNumber == ""

    def test_apply_anonymization_uses_provided_patient_info(self, manager_with_data):
        """Test that _apply_anonymization uses provided patient info when given"""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset

        manager_with_data._apply_anonymization(
            ds, source_ds,
            new_patient_name="Anonymous^Patient",
            new_patient_id="ANON123"
        )

        assert ds.PatientName == "Anonymous^Patient"
        assert ds.PatientID == "ANON123"
        assert ds.PatientBirthDate == ""
        assert ds.ReferringPhysicianName == ""
        assert ds.AccessionNumber == ""

    def test_apply_anonymization_forces_empty_type2_elements(self, manager_with_data):
        """Test that Type 2 elements are forced to empty strings regardless of source"""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()

        # Add patient info to source
        source_ds.PatientName = "Test^Patient"
        source_ds.PatientID = "TEST123"
        source_ds.PatientBirthDate = "19900101"
        source_ds.ReferringPhysicianName = "Dr. Smith"
        source_ds.AccessionNumber = "ACC123"

        manager_with_data._apply_anonymization(ds, source_ds)

        # Should be empty strings, not copied from source
        assert ds.PatientBirthDate == ""
        assert ds.ReferringPhysicianName == ""
        assert ds.AccessionNumber == ""

    def test_set_conformance_attributes_conditional_elements_only_when_missing(self, manager_with_data):
        """Test that conditional elements are only set when missing from dataset"""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()

        # Pre-set some attributes in the dataset
        ds.Laterality = "L"
        ds.InstanceNumber = 5
        ds.SamplesPerPixel = 1

        manager_with_data._set_conformance_attributes(ds, source_ds)

        # Pre-existing values should be preserved
        assert ds.Laterality == "L"
        assert ds.InstanceNumber == 5

        # Missing attributes should get defaults
        assert ds.PatientOrientation == ''
        assert ds.ImageType == ['ORIGINAL', 'PRIMARY', 'IMAGE']

    def test_set_conformance_attributes_missing_elements_get_defaults(self, manager_with_data):
        """Test that missing conditional elements get default values"""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()

        manager_with_data._set_conformance_attributes(ds, source_ds)

        # All missing elements should get defaults
        assert ds.Laterality == ''
        assert ds.InstanceNumber == 1
        assert ds.PatientOrientation == ''
        assert ds.ImageType == ['ORIGINAL', 'PRIMARY', 'IMAGE']

    def test_generate_filename_from_dicom_dataset_with_hashing(self, manager):
        """Test generate_filename_from_dicom_dataset with patient ID hashing enabled"""
        ds = self.create_test_dicom_file()

        filename, patient_id, instance_id = manager.generate_filename_from_dicom_dataset(ds, hash_patient_id=True)

        # Should return a properly formatted filename
        assert filename.endswith('.dcm')
        assert '_' in filename

        # Patient ID should be hashed (10 digits)
        assert len(patient_id) == 10
        assert patient_id.isdigit()
        assert patient_id != self.PATIENT_ID  # Should be different from original

        # Instance ID should be hashed (8 digits)
        assert len(instance_id) == 8
        assert instance_id.isdigit()

    def test_generate_filename_from_dicom_dataset_without_hashing(self, manager):
        """Test generate_filename_from_dicom_dataset with patient ID hashing disabled"""
        ds = self.create_test_dicom_file()

        filename, patient_id, instance_id = manager.generate_filename_from_dicom_dataset(ds, hash_patient_id=False)

        # Should return a properly formatted filename
        assert filename.endswith('.dcm')
        assert '_' in filename

        # Patient ID should be original (not hashed)
        assert patient_id == "000" + self.PATIENT_ID # Total length should be 10 digits, so we add 3 zeros to the front

        # Instance ID should still be hashed (8 digits)
        assert len(instance_id) == 8
        assert instance_id.isdigit()

    def test_generate_filename_from_dicom_dataset_missing_patient_id(self, manager):
        """Test generate_filename_from_dicom_dataset with missing PatientID"""
        ds = self.create_test_dicom_file(PatientID=None)

        filename, patient_id, instance_id = manager.generate_filename_from_dicom_dataset(ds)

        # Should return empty strings when patient ID is missing
        assert filename == ""
        assert patient_id == ""
        assert instance_id == ""

    def test_generate_filename_from_dicom_dataset_empty_patient_id(self, manager):
        """Test generate_filename_from_dicom_dataset with empty PatientID"""
        ds = self.create_test_dicom_file(PatientID="")

        filename, patient_id, instance_id = manager.generate_filename_from_dicom_dataset(ds)

        # Should return empty strings when patient ID is empty
        assert filename == ""
        assert patient_id == ""
        assert instance_id == ""

    def test_generate_filename_from_dicom_dataset_missing_instance_uid(self, manager):
        """Test generate_filename_from_dicom_dataset with missing SOPInstanceUID"""
        ds = self.create_test_dicom_file(SOPInstanceUID=None)

        filename, patient_id, instance_id = manager.generate_filename_from_dicom_dataset(ds)

        # Should return empty strings when instance UID is missing
        assert filename == ""
        assert patient_id == ""
        assert instance_id == ""

    def test_generate_filename_from_dicom_dataset_empty_instance_uid(self, manager):
        """Test generate_filename_from_dicom_dataset with empty SOPInstanceUID"""
        ds = self.create_test_dicom_file(SOPInstanceUID="")

        filename, patient_id, instance_id = manager.generate_filename_from_dicom_dataset(ds)

        # Should return empty strings when instance UID is empty
        assert filename == ""
        assert patient_id == ""
        assert instance_id == ""

    def test_generate_filename_from_dicom_dataset_deterministic_hashing(self, manager):
        """Test that filename generation is deterministic for same inputs"""
        ds1 = self.create_test_dicom_file()
        ds2 = self.create_test_dicom_file()  # Same values

        filename1, patient_id1, instance_id1 = manager.generate_filename_from_dicom_dataset(ds1)
        filename2, patient_id2, instance_id2 = manager.generate_filename_from_dicom_dataset(ds2)

        # Same inputs should produce same outputs
        assert filename1 == filename2
        assert patient_id1 == patient_id2
        assert instance_id1 == instance_id2

    def test_generate_filename_from_dicom_dataset_different_inputs_different_outputs(self, manager):
        """Test that different inputs produce different outputs"""
        ds1 = self.create_test_dicom_file(PatientID="PATIENT1")
        ds2 = self.create_test_dicom_file(PatientID="PATIENT2")

        filename1, patient_id1, instance_id1 = manager.generate_filename_from_dicom_dataset(ds1)
        filename2, patient_id2, instance_id2 = manager.generate_filename_from_dicom_dataset(ds2)

        # Different inputs should produce different outputs
        assert filename1 != filename2
        assert patient_id1 != patient_id2
        # Instance IDs should be the same since SOPInstanceUID is the same
        assert instance_id1 == instance_id2

    def test_generate_filename_from_dicom_dataset_zero_padding(self, manager):
        """Test that patient and instance IDs are properly zero-padded"""
        # Create a dataset that would generate small hash values
        ds = self.create_test_dicom_file(PatientID="A", SOPInstanceUID="B")

        filename, patient_id, instance_id = manager.generate_filename_from_dicom_dataset(ds)

        # Should be zero-padded to correct lengths
        assert len(patient_id) == manager.PATIENT_ID_HASH_LENGTH
        assert len(instance_id) == manager.INSTANCE_ID_HASH_LENGTH
        assert patient_id.isdigit()
        assert instance_id.isdigit()

    def test_generate_filename_from_dicom_dataset_return_type(self, manager):
        """Test that generate_filename_from_dicom_dataset returns correct types"""
        ds = self.create_test_dicom_file()

        result = manager.generate_filename_from_dicom_dataset(ds)

        # Should return a tuple of 3 strings
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(item, str) for item in result)

    # Test for updated comment in _copy_and_generate_uids (verification that logic still works)
    def test_copy_and_generate_uids_always_generates_series_uid(self, manager_with_data, temp_dir):
        """Test that SeriesInstanceUID is always generated (never copied from source)"""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        output_path = os.path.join(temp_dir, "test.dcm")

        original_series_uid = source_ds.SeriesInstanceUID

        manager_with_data._copy_and_generate_uids(ds, source_ds, output_path)

        # SeriesInstanceUID should always be different from source
        assert ds.SeriesInstanceUID != original_series_uid
        assert hasattr(ds, 'SeriesInstanceUID')
        assert ds.SeriesInstanceUID != ""

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

    def test_generate_output_filepath_preserve_structure(self, manager):
        """Test generate_output_filepath with preserve_directory_structure=True"""
        output_directory = "/output"
        output_path = "patient1/study1/series1/file.dcm"

        result = manager.generate_output_filepath(output_directory, output_path, True)

        assert result == "/output/patient1/study1/series1/file.dcm"

    def test_generate_output_filepath_flatten_structure(self, manager):
        """Test generate_output_filepath with preserve_directory_structure=False"""
        output_directory = "/output"
        output_path = "patient1/study1/series1/file.dcm"

        result = manager.generate_output_filepath(output_directory, output_path, False)

        assert result == "/output/file.dcm"

    def test_generate_output_filepath_simple_filename(self, manager):
        """Test generate_output_filepath with simple filename"""
        output_directory = "/output"
        output_path = "file.dcm"

        result_preserve = manager.generate_output_filepath(output_directory, output_path, True)
        result_flatten = manager.generate_output_filepath(output_directory, output_path, False)

        assert result_preserve == "/output/file.dcm"
        assert result_flatten == "/output/file.dcm"

    def test_save_anonymized_dicom_creates_nested_directories(self, manager_with_data, sample_image_array_single_frame, temp_dir):
        nested_output_path = os.path.join(temp_dir, "patient", "study", "series", "output.dcm")

        manager_with_data.save_anonymized_dicom(sample_image_array_single_frame, nested_output_path)

        # Directory should be created
        assert os.path.exists(os.path.dirname(nested_output_path))
        # File should be created
        assert os.path.exists(nested_output_path)

    def test_create_and_save_dicom_file_creates_directories(self, manager_with_data, temp_dir):
        ds = pydicom.Dataset()
        ds.SOPClassUID = self.SOP_CLASS_UID
        ds.SOPInstanceUID = self.SOP_INSTANCE_UID

        nested_output_path = os.path.join(temp_dir, "deep", "nested", "path", "test.dcm")

        manager_with_data._create_and_save_dicom_file(ds, nested_output_path)

        # Directory should be created
        assert os.path.exists(os.path.dirname(nested_output_path))

    def test_save_anonymized_dicom_header_creates_json_file(self, manager_with_data, temp_dir):
        headers_directory = os.path.join(temp_dir, "headers")
        current_record = manager_with_data.dicom_df.iloc[0]
        output_filename = "test_output.dcm"

        result = manager_with_data.save_anonymized_dicom_header(
            current_record,
            output_filename,
            headers_directory
        )

        assert result is not None
        assert result.endswith("_DICOMHeader.json")
        assert os.path.exists(result)

    def test_save_anonymized_dicom_header_anonymizes_patient_name(self, manager_with_data, temp_dir):
        headers_directory = os.path.join(temp_dir, "headers")
        current_record = manager_with_data.dicom_df.iloc[0]
        output_filename = "anonymized_patient.dcm"

        # Add patient name to source dataset
        current_record.DICOMDataset.PatientName = "Original^Patient^Name"

        result_path = manager_with_data.save_anonymized_dicom_header(
            current_record,
            output_filename,
            headers_directory
        )

        # Read the JSON file and verify anonymization
        with open(result_path, 'r') as f:
            header_data = json.load(f)

        assert header_data["Patient's Name"] == "anonymized_patient"

    def test_save_anonymized_dicom_header_anonymizes_birth_date(self, manager_with_data, temp_dir):
        headers_directory = os.path.join(temp_dir, "headers")
        current_record = manager_with_data.dicom_df.iloc[0]
        output_filename = "test.dcm"

        # Add birth date to source dataset
        current_record.DICOMDataset.PatientBirthDate = "19901215"

        result_path = manager_with_data.save_anonymized_dicom_header(
            current_record,
            output_filename,
            headers_directory
        )

        # Read the JSON file and verify anonymization
        with open(result_path, 'r') as f:
            header_data = json.load(f)

        assert header_data["Patient's Birth Date"] == "19900101"

    def test_save_anonymized_dicom_header_returns_none_when_no_directory(self, manager_with_data):
        current_record = manager_with_data.dicom_df.iloc[0]

        result = manager_with_data.save_anonymized_dicom_header(
            current_record,
            "test.dcm",
            None
        )

        assert result is None

    def test_save_anonymized_dicom_header_raises_error_when_no_filename(self, manager_with_data, temp_dir):
        headers_directory = os.path.join(temp_dir, "headers")
        current_record = manager_with_data.dicom_df.iloc[0]

        with pytest.raises(ValueError, match="Output filename is required"):
            manager_with_data.save_anonymized_dicom_header(
                current_record,
                "",
                headers_directory
            )

        with pytest.raises(ValueError, match="Output filename is required"):
            manager_with_data.save_anonymized_dicom_header(
                current_record,
                "",
                headers_directory
            )

    def test_save_anonymized_dicom_header_raises_error_when_no_record(self, manager_with_data, temp_dir):
        headers_directory = os.path.join(temp_dir, "headers")
        current_record = None

        with pytest.raises(ValueError, match="Current DICOM record is required"):
            manager_with_data.save_anonymized_dicom_header(
                current_record,
                "test.dcm",
                headers_directory
            )

    def test_save_anonymized_dicom_header_flatten_directory_structure(self, manager_with_data, temp_dir):
        headers_directory = os.path.join(temp_dir, "headers")

        # Create a record with nested relative path
        current_record = manager_with_data.dicom_df.iloc[0].copy()

        result_path = manager_with_data.save_anonymized_dicom_header(
            current_record,
            "test.dcm",
            headers_directory
        )

        expected_path = os.path.join(headers_directory, "test_DICOMHeader.json")
        assert result_path == expected_path
        assert os.path.exists(result_path)

    def test_convert_to_json_compatible_multival(self, manager):
        multival = pydicom.multival.MultiValue(str, ['value1', 'value2', 'value3'])

        result = manager._convert_to_json_compatible(multival)

        assert result == ['value1', 'value2', 'value3']

    def test_convert_to_json_compatible_person_name(self, manager):
        person_name = pydicom.valuerep.PersonName("Last^First^Middle")

        result = manager._convert_to_json_compatible(person_name)

        assert result == "Last^First^Middle"

    def test_convert_to_json_compatible_bytes(self, manager):
        byte_data = b'test_data'

        result = manager._convert_to_json_compatible(byte_data)

        assert result == 'test_data'

    def test_convert_to_json_compatible_unsupported_type(self, manager):
        unsupported_obj = object()

        with pytest.raises(TypeError, match="Object of type object is not JSON serializable"):
            manager._convert_to_json_compatible(unsupported_obj)

    def test_save_anonymized_dicom_with_empty_patient_info_defaults(self, manager_with_data, sample_image_array_single_frame, temp_dir):
        output_path = os.path.join(temp_dir, "output.dcm")

        # Test with default empty strings
        manager_with_data.save_anonymized_dicom(sample_image_array_single_frame, output_path)

        # File should be created
        assert os.path.exists(output_path)

        # Verify the DICOM file has empty patient info
        saved_ds = pydicom.dcmread(output_path)
        assert saved_ds.PatientName == ""
        assert saved_ds.PatientID == ""

    def test_apply_anonymization_with_empty_string_defaults(self, manager_with_data):
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset

        # Test with default empty strings
        manager_with_data._apply_anonymization(ds, source_ds)

        assert ds.PatientName == ""
        assert ds.PatientID == ""

if __name__ == "__main__":
    pytest.main([__file__])
