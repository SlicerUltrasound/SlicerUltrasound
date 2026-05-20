import pytest
import os
import tempfile
import shutil
import logging
import pandas as pd
import pydicom
import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from unittest.mock import Mock, patch
from pathlib import Path
import json

# Import the module under test
from ..dicom_file_manager import DicomFileManager
from ..uid_remap import remap_uid

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
    TRANSDUCER_DATA = "SC6-1s,02597"
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
            'TransducerData': self.TRANSDUCER_DATA,
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

    SOURCE_FOR_UID = "1.2.840.113619.2.1.1234"
    ANON_FOR_UID = "2.25.123456789012345678901234567890"

    @pytest.fixture
    def manager_with_data(self, manager, temp_dir):
        """Create a manager with sample DICOM dataframe"""
        ds = self.create_test_dicom_file()
        ds.FrameOfReferenceUID = self.SOURCE_FOR_UID
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
            'FrameOfReferenceUID': self.SOURCE_FOR_UID,
            'AnonFrameOfReferenceUID': self.ANON_FOR_UID,
            'PhysicalDeltaX': self.PHYSICAL_DELTA_X,
            'PhysicalDeltaY': self.PHYSICAL_DELTA_Y,
            'ContentDate': self.CONTENT_DATE,
            'ContentTime': self.CONTENT_TIME,
            'Patch': False,
            'TransducerModel': self.TRANSDUCER_MODEL,
            'DICOMDataset': ds
        }]

        manager._create_dataframe(dicom_data)
        # Seed the in-memory map so direct _copy_and_generate_uids calls hit
        # either the dataframe-column path or the _for_map fallback.
        manager._for_map[(self.STUDY_UID, self.SOURCE_FOR_UID)] = self.ANON_FOR_UID
        manager.current_index = 0
        return manager

    def test_init(self, manager):
        """Test DicomFileManager initialization"""
        assert manager.dicom_df is None
        assert manager.next_index == 0

    def test_get_transducer_model_valid(self, manager):
        """Test transducer model extraction with valid comma-delimited input"""
        assert manager.get_transducer_model("SC6-1s,02597") == "sc6-1s"
        assert manager.get_transducer_model("L12-3,12345") == "l12-3"
        assert manager.get_transducer_model("C1-5") == "c1-5"
        # Explicit non-Butterfly manufacturer arg should not change behavior
        assert manager.get_transducer_model("SC6-1s,02597", manufacturer="Philips") == "sc6-1s"

    def test_get_transducer_model_invalid(self, manager):
        """Test transducer model extraction with invalid input"""
        assert manager.get_transducer_model("") == "unknown"
        assert manager.get_transducer_model(None) == "unknown"
        assert manager.get_transducer_model("   ") == "unknown"

    def test_get_transducer_model_backslash_format(self, manager):
        """Test backslash-delimited TransducerData (VR LO native separator)"""
        assert manager.get_transducer_model("S4-1U\\UNUSED\\UNUSED") == "s4-1u"
        assert manager.get_transducer_model("L12-3\\UNUSED") == "l12-3"

    def test_get_transducer_model_backslash_multivalue(self, manager):
        """Test pydicom MultiValue input (auto-split backslash-delimited LO)"""
        from pydicom.multival import MultiValue
        mv = MultiValue(str, ["S4-1U", "UNUSED", "UNUSED"])
        assert manager.get_transducer_model(mv) == "s4-1u"

    def test_get_transducer_model_butterfly_uses_model_name(self, manager):
        """Test Butterfly manufacturer routes through ManufacturerModelName"""
        assert manager.get_transducer_model(
            "", manufacturer="Butterfly Network Inc", manufacturer_model_name="IQ"
        ) == "iq"
        # Butterfly branch ignores TransducerData even when present
        assert manager.get_transducer_model(
            "IgnoreMe,123", manufacturer="Butterfly Network Inc", manufacturer_model_name="IQ3"
        ) == "iq3"

    def test_get_transducer_model_butterfly_case_insensitive(self, manager):
        """Test Butterfly detection is case-insensitive substring match"""
        assert manager.get_transducer_model(
            "", manufacturer="butterfly network", manufacturer_model_name="IQ"
        ) == "iq"
        assert manager.get_transducer_model(
            "", manufacturer="BUTTERFLY NETWORK INC", manufacturer_model_name="IQ"
        ) == "iq"
        assert manager.get_transducer_model(
            "", manufacturer="Butterfly Network Inc.", manufacturer_model_name="IQ"
        ) == "iq"

    def test_get_transducer_model_butterfly_missing_model_name(self, manager):
        """Test Butterfly without ManufacturerModelName returns 'unknown' (no fallback)"""
        assert manager.get_transducer_model(
            "SC6-1s,02597", manufacturer="Butterfly Network Inc", manufacturer_model_name=""
        ) == "unknown"
        assert manager.get_transducer_model(
            "SC6-1s,02597", manufacturer="Butterfly Network Inc", manufacturer_model_name=None
        ) == "unknown"

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
        assert len(result) == 18
        assert result['InputPath'] == sample_dicom_filepath
        assert filename in result['OutputPath']
        assert result['AnonFilename'] == filename
        assert result['PatientUID'] == self.PATIENT_ID
        assert result['StudyUID'] == self.STUDY_UID
        assert result['SeriesUID'] == self.SERIES_UID
        assert result['InstanceUID'] == self.SOP_INSTANCE_UID
        assert result['AnonStudyUID'] == remap_uid(self.STUDY_UID)
        assert result['AnonSeriesUID'] == remap_uid(self.SERIES_UID)
        assert result['AnonSOPInstanceUID'] == remap_uid(self.SOP_INSTANCE_UID)
        assert result['ContentDate'] == self.CONTENT_DATE
        assert result['ContentTime'] == self.CONTENT_TIME
        assert result['Patch'] is False
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

    def test_extract_dicom_info_butterfly(self, manager, temp_dir):
        """Test Butterfly manufacturer DICOM uses ManufacturerModelName for TransducerModel"""
        ds = self.create_test_dicom_file(
            Manufacturer="Butterfly Network Inc",
            ManufacturerModelName="IQ",
            TransducerData="",
        )
        filename = manager._generate_filename_from_dicom(ds)
        filepath = self.save_dicom_file(ds, temp_dir, filename)

        result = manager._extract_dicom_info(filepath, self.INPUT_FOLDER, False)

        assert result is not None
        assert result['TransducerModel'] == "iq"

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
            "Transducer Data": self.TRANSDUCER_DATA,
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
        """Source instance UIDs are remapped via remap_uid; SOPClassUID stays verbatim."""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_and_generate_uids(ds, source_ds, output_path)

        # SOPClassUID identifies the registered DICOM object type — NOT remapped.
        assert ds.SOPClassUID == source_ds.SOPClassUID
        # Study/Series/SOP Instance UIDs are deterministically remapped.
        assert ds.SOPInstanceUID == remap_uid(str(source_ds.SOPInstanceUID))
        assert ds.SeriesInstanceUID == remap_uid(str(source_ds.SeriesInstanceUID))
        assert ds.StudyInstanceUID == remap_uid(str(source_ds.StudyInstanceUID))
        # Outputs must live in the 2.25.* arc.
        assert ds.SOPInstanceUID.startswith("2.25.")
        assert ds.SeriesInstanceUID.startswith("2.25.")
        assert ds.StudyInstanceUID.startswith("2.25.")

    def test_copy_and_generate_uids_missing_source_uids(self, manager_with_data, temp_dir):
        """When source UIDs are missing, instance UIDs land in the 2.25 arc."""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()  # Empty dataset
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_and_generate_uids(ds, source_ds, output_path)

        # SOPClassUID falls back to pydicom-generated; not constrained to 2.25.
        assert hasattr(ds, 'SOPClassUID')
        # Study/Series/SOP Instance UIDs are routed through remap_uid even on fallback.
        assert ds.SOPInstanceUID.startswith("2.25.")
        assert ds.StudyInstanceUID.startswith("2.25.")
        assert ds.SeriesInstanceUID.startswith("2.25.")

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
        assert ds['PixelData'].is_undefined_length is True

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
        assert hasattr(ds, 'TransducerData')

    def test_copy_source_metadata_preserves_station_name(self, manager_with_data, temp_dir):
        """StationName (0008,1010) is preserved in the de-id DICOM per institutional workflow"""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        source_ds.StationName = "US_ROOM_3B"
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert ds.StationName == "US_ROOM_3B"

    def test_copy_source_metadata_preserves_study_description(self, manager_with_data, temp_dir):
        """StudyDescription is preserved in the de-id DICOM per institutional workflow"""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        source_ds.StudyDescription = "Patient Jane Doe referred by Dr Smith"
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert ds.StudyDescription == "Patient Jane Doe referred by Dr Smith"

    def test_apply_anonymization_caps_patient_age_over_89(self, manager_with_data):
        """PatientAge >= 90 years is capped at '090Y' per HIPAA Safe Harbor"""
        ds = pydicom.Dataset()
        ds.PatientAge = "095Y"
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        manager_with_data._apply_anonymization(ds, source_ds)
        assert ds.PatientAge == "090Y"

    def test_apply_anonymization_caps_patient_age_exactly_90(self, manager_with_data):
        """PatientAge of exactly 90 years is also capped (boundary)"""
        ds = pydicom.Dataset()
        ds.PatientAge = "090Y"
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        manager_with_data._apply_anonymization(ds, source_ds)
        assert ds.PatientAge == "090Y"

    def test_apply_anonymization_preserves_patient_age_under_90(self, manager_with_data):
        """PatientAge under 90 years passes through unchanged"""
        ds = pydicom.Dataset()
        ds.PatientAge = "045Y"
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        manager_with_data._apply_anonymization(ds, source_ds)
        assert ds.PatientAge == "045Y"

    def test_apply_anonymization_preserves_non_year_patient_age(self, manager_with_data):
        """Non-year PatientAge formats (months, weeks, days) are not affected by the 90Y cap"""
        ds = pydicom.Dataset()
        ds.PatientAge = "025M"
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        manager_with_data._apply_anonymization(ds, source_ds)
        assert ds.PatientAge == "025M"

    def test_cap_patient_age_logs_error_for_invalid_format(self, manager, caplog):
        """Malformed PatientAge values are logged at ERROR with value + expected format, then passed through"""
        with caplog.at_level(logging.ERROR):
            result = manager._cap_patient_age("25YR")

        assert result == "25YR"
        assert any("25YR" in r.message for r in caplog.records), "log should include the invalid value"
        assert any("nnnX" in r.message for r in caplog.records), "log should include the expected format"

    def test_cap_patient_age_logs_error_for_bad_unit(self, manager, caplog):
        """Values with an invalid unit letter are flagged as errors"""
        with caplog.at_level(logging.ERROR):
            result = manager._cap_patient_age("045X")

        assert result == "045X"
        assert any("045X" in r.message for r in caplog.records)

    def test_cap_patient_age_logs_error_for_non_digit_prefix(self, manager, caplog):
        """Values whose prefix isn't three digits are flagged as errors"""
        with caplog.at_level(logging.ERROR):
            result = manager._cap_patient_age("abcY")

        assert result == "abcY"
        assert any("abcY" in r.message for r in caplog.records)

    def test_cap_patient_age_silent_for_empty(self, manager, caplog):
        """Empty PatientAge is a valid 'no value' state and should not log an error"""
        with caplog.at_level(logging.ERROR):
            result = manager._cap_patient_age("")

        assert result == ""
        assert len(caplog.records) == 0

    def test_cap_patient_age_silent_for_valid_non_year(self, manager, caplog):
        """Valid non-year DICOM AS values should not log an error"""
        with caplog.at_level(logging.ERROR):
            for value in ("025M", "014D", "003W"):
                assert manager._cap_patient_age(value) == value

        assert len(caplog.records) == 0

    def test_copy_and_generate_uids_does_not_copy_frame_of_reference_verbatim(
        self, manager_with_data, temp_dir
    ):
        """Leakage guard: source FrameOfReferenceUID never appears in the output,
        neither verbatim nor as its remap_uid hash. Under the run-local policy
        the output FOR is a freshly minted 2.25 UID with no derivation from the
        source value, so both inequality checks must hold."""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        source_ds.FrameOfReferenceUID = "1.2.840.113619.2.1.99"
        # _for_map has the test fixture's source FOR; this test uses a different
        # one, so _copy_and_generate_uids must mint or look up by the new key.
        manager_with_data._for_map[
            (str(source_ds.StudyInstanceUID), "1.2.840.113619.2.1.99")
        ] = "2.25.999999999999999999999999999"
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_and_generate_uids(ds, source_ds, output_path)

        assert ds.FrameOfReferenceUID != "1.2.840.113619.2.1.99"
        assert ds.FrameOfReferenceUID != remap_uid("1.2.840.113619.2.1.99")

    def test_copy_and_generate_uids_frame_of_reference_set_even_when_source_lacks_it(
        self, manager_with_data, temp_dir
    ):
        """FrameOfReferenceUID is Type 1/1C required for US IODs.

        Even when source lacks one, the output must have one — and it must live
        in the 2.25 arc. The fallback minting path takes over in this branch.
        """
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        if hasattr(source_ds, 'FrameOfReferenceUID'):
            delattr(source_ds, 'FrameOfReferenceUID')
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_and_generate_uids(ds, source_ds, output_path)

        assert hasattr(ds, 'FrameOfReferenceUID')
        assert ds.FrameOfReferenceUID
        assert ds.FrameOfReferenceUID.startswith("2.25.")

    # ----------------------------------------------------------------------
    # New integration tests for the run-local per-study FOR UID policy.
    # ----------------------------------------------------------------------

    def _make_source_dicom(self, temp_dir, subdir, *, sop_instance_uid,
                           series_instance_uid, study_instance_uid,
                           frame_of_reference_uid=None, patient_id=None):
        """Build and save a small DICOM with controllable UIDs for FOR tests."""
        kwargs = {
            'SOPInstanceUID': sop_instance_uid,
            'SeriesInstanceUID': series_instance_uid,
            'StudyInstanceUID': study_instance_uid,
        }
        if patient_id is not None:
            kwargs['PatientID'] = patient_id
        ds = self.create_test_dicom_file(**kwargs)
        if frame_of_reference_uid is not None:
            ds.FrameOfReferenceUID = frame_of_reference_uid
        elif hasattr(ds, 'FrameOfReferenceUID'):
            delattr(ds, 'FrameOfReferenceUID')
        nested = os.path.join(temp_dir, subdir)
        os.makedirs(nested, exist_ok=True)
        # Use a deterministic distinct filename per instance so files don't clobber.
        filename = f"{sop_instance_uid.replace('.', '_')}.dcm"
        return self.save_dicom_file(ds, nested, filename)

    def test_anon_for_uid_shared_across_series_with_same_source_for_in_same_study(
        self, manager, temp_dir
    ):
        """Two series in one study sharing source FOR → shared AnonFrameOfReferenceUID."""
        study_uid = "1.2.840.113619.2.55.STUDY.A"
        shared_for = "1.2.840.113619.2.1.FOR1"
        self._make_source_dicom(
            temp_dir, "studyA/seriesA",
            sop_instance_uid="1.2.A.1", series_instance_uid="1.2.A.SER1",
            study_instance_uid=study_uid, frame_of_reference_uid=shared_for,
        )
        self._make_source_dicom(
            temp_dir, "studyA/seriesB",
            sop_instance_uid="1.2.A.2", series_instance_uid="1.2.A.SER2",
            study_instance_uid=study_uid, frame_of_reference_uid=shared_for,
        )

        manager.scan_directory(temp_dir)

        rows = manager.dicom_df[manager.dicom_df['StudyUID'] == study_uid]
        assert len(rows) == 2
        anon_values = rows['AnonFrameOfReferenceUID'].unique()
        assert len(anon_values) == 1
        assert anon_values[0].startswith("2.25.")

    def test_anon_for_uid_differs_for_different_source_for_in_same_study(
        self, manager, temp_dir
    ):
        """Same study, different source FORs → distinct AnonFrameOfReferenceUIDs."""
        study_uid = "1.2.840.113619.2.55.STUDY.A"
        self._make_source_dicom(
            temp_dir, "studyA/seriesA",
            sop_instance_uid="1.2.A.1", series_instance_uid="1.2.A.SER1",
            study_instance_uid=study_uid, frame_of_reference_uid="1.2.FOR.X",
        )
        self._make_source_dicom(
            temp_dir, "studyA/seriesC",
            sop_instance_uid="1.2.A.3", series_instance_uid="1.2.A.SER3",
            study_instance_uid=study_uid, frame_of_reference_uid="1.2.FOR.Y",
        )

        manager.scan_directory(temp_dir)

        rows = manager.dicom_df[manager.dicom_df['StudyUID'] == study_uid]
        anon_values = list(rows['AnonFrameOfReferenceUID'])
        assert len(anon_values) == 2
        assert anon_values[0] != anon_values[1]

    def test_anon_for_uid_differs_across_studies_for_same_source_for(
        self, manager, temp_dir
    ):
        """Coincidentally-shared source FOR across two studies → distinct anon FORs."""
        shared_for = "1.2.840.113619.2.1.SHARED"
        self._make_source_dicom(
            temp_dir, "study1/seriesA",
            sop_instance_uid="1.2.S1.1", series_instance_uid="1.2.S1.SER1",
            study_instance_uid="1.2.STUDY.ONE", frame_of_reference_uid=shared_for,
            patient_id="P1",
        )
        self._make_source_dicom(
            temp_dir, "study2/seriesD",
            sop_instance_uid="1.2.S2.1", series_instance_uid="1.2.S2.SER1",
            study_instance_uid="1.2.STUDY.TWO", frame_of_reference_uid=shared_for,
            patient_id="P2",
        )

        manager.scan_directory(temp_dir)

        anon_s1 = manager.dicom_df[
            manager.dicom_df['StudyUID'] == "1.2.STUDY.ONE"
        ]['AnonFrameOfReferenceUID'].iloc[0]
        anon_s2 = manager.dicom_df[
            manager.dicom_df['StudyUID'] == "1.2.STUDY.TWO"
        ]['AnonFrameOfReferenceUID'].iloc[0]
        assert anon_s1 != anon_s2

    def test_anon_for_uid_cardinality_matches_unique_source_for_per_study(
        self, manager, temp_dir
    ):
        """3 unique source FORs across 5 series in one study → exactly 3 unique anon FORs."""
        study_uid = "1.2.STUDY.MULTI"
        spec = [
            ("ser1", "1.2.FOR.A"),
            ("ser2", "1.2.FOR.A"),
            ("ser3", "1.2.FOR.B"),
            ("ser4", "1.2.FOR.C"),
            ("ser5", "1.2.FOR.B"),
        ]
        for i, (ser_label, for_uid) in enumerate(spec):
            self._make_source_dicom(
                temp_dir, f"study/{ser_label}",
                sop_instance_uid=f"1.2.INST.{i}",
                series_instance_uid=f"1.2.SER.{i}",
                study_instance_uid=study_uid,
                frame_of_reference_uid=for_uid,
            )

        manager.scan_directory(temp_dir)

        nunique = manager.dicom_df.groupby('StudyUID')[
            'AnonFrameOfReferenceUID'
        ].nunique()
        assert nunique[study_uid] == 3

    def test_anon_for_uid_shared_across_instances_in_same_series(
        self, manager, temp_dir
    ):
        """Two instances in one source series → shared AnonFrameOfReferenceUID."""
        study_uid = "1.2.STUDY.ABC"
        series_uid = "1.2.SER.SAME"
        for_uid = "1.2.FOR.SAME"
        self._make_source_dicom(
            temp_dir, "study/ser1",
            sop_instance_uid="1.2.INST.1",
            series_instance_uid=series_uid,
            study_instance_uid=study_uid,
            frame_of_reference_uid=for_uid,
        )
        self._make_source_dicom(
            temp_dir, "study/ser1",
            sop_instance_uid="1.2.INST.2",
            series_instance_uid=series_uid,
            study_instance_uid=study_uid,
            frame_of_reference_uid=for_uid,
        )

        manager.scan_directory(temp_dir)

        rows = manager.dicom_df[manager.dicom_df['SeriesUID'] == series_uid]
        assert len(rows) == 2
        assert rows.iloc[0]['AnonFrameOfReferenceUID'] == rows.iloc[1]['AnonFrameOfReferenceUID']

    def test_anon_for_uid_emitted_when_source_lacks_for_and_does_not_link_coordinateless_series(
        self, manager, temp_dir
    ):
        """Two series in one study with no source FOR → both get UIDs but they differ.

        Type 1/1C compliance: every output instance has a non-empty 2.25 UID.
        Also: a missing source FOR must NOT be silently coalesced into a shared
        anon UID across two coordinate-less series — that would be a false link.
        """
        study_uid = "1.2.STUDY.NO_FOR"
        self._make_source_dicom(
            temp_dir, "study/ser1",
            sop_instance_uid="1.2.INST.X",
            series_instance_uid="1.2.SER.X",
            study_instance_uid=study_uid,
            frame_of_reference_uid=None,
        )
        self._make_source_dicom(
            temp_dir, "study/ser2",
            sop_instance_uid="1.2.INST.Y",
            series_instance_uid="1.2.SER.Y",
            study_instance_uid=study_uid,
            frame_of_reference_uid=None,
        )

        manager.scan_directory(temp_dir)

        rows = manager.dicom_df[manager.dicom_df['StudyUID'] == study_uid]
        anon_values = list(rows['AnonFrameOfReferenceUID'])
        assert len(anon_values) == 2
        assert all(v and v.startswith("2.25.") for v in anon_values)
        assert anon_values[0] != anon_values[1]

    def test_anon_for_uid_does_not_leak_source_value(self, manager, temp_dir):
        """Integration leakage guard: output FOR ≠ source verbatim AND ≠ remap_uid(source)."""
        leak_for = "1.2.840.113619.2.1.99"
        self._make_source_dicom(
            temp_dir, "study/ser1",
            sop_instance_uid="1.2.INST.LEAK",
            series_instance_uid="1.2.SER.LEAK",
            study_instance_uid="1.2.STUDY.LEAK",
            frame_of_reference_uid=leak_for,
        )

        manager.scan_directory(temp_dir)

        anon_for = manager.dicom_df.iloc[0]['AnonFrameOfReferenceUID']
        assert anon_for != leak_for
        assert anon_for != remap_uid(leak_for)
        assert anon_for.startswith("2.25.")

    def test_anon_for_uid_is_run_local_not_deterministic(self, temp_dir):
        """Two separate DicomFileManager runs over identical input → different anon FORs."""
        self._make_source_dicom(
            temp_dir, "study/ser1",
            sop_instance_uid="1.2.INST.RL",
            series_instance_uid="1.2.SER.RL",
            study_instance_uid="1.2.STUDY.RL",
            frame_of_reference_uid="1.2.FOR.RL",
        )

        m1 = DicomFileManager()
        m1.scan_directory(temp_dir)
        anon_first = m1.dicom_df.iloc[0]['AnonFrameOfReferenceUID']

        m2 = DicomFileManager()
        m2.scan_directory(temp_dir)
        anon_second = m2.dicom_df.iloc[0]['AnonFrameOfReferenceUID']

        assert anon_first != anon_second
        assert anon_first.startswith("2.25.")
        assert anon_second.startswith("2.25.")

    def test_anon_for_uid_resume_reuses_prior_keys_csv_mapping(self, temp_dir):
        """Resume: scan with seed_keys_csv from a prior run → prior anon FOR is reused."""
        # First run.
        self._make_source_dicom(
            temp_dir, "study/ser1",
            sop_instance_uid="1.2.INST.RESUME",
            series_instance_uid="1.2.SER.RESUME",
            study_instance_uid="1.2.STUDY.RESUME",
            frame_of_reference_uid="1.2.FOR.RESUME",
        )
        m1 = DicomFileManager()
        m1.scan_directory(temp_dir)
        prior_keys_csv = os.path.join(temp_dir, "keys.csv")
        m1.build_csv_dataframe(temp_dir).to_csv(prior_keys_csv, index=False)
        prior_anon = m1.dicom_df.iloc[0]['AnonFrameOfReferenceUID']

        # Second run with the seed.
        m2 = DicomFileManager()
        m2.scan_directory(temp_dir, seed_keys_csv=prior_keys_csv)

        # Filter out the rescanned keys.csv itself (only .dcm rows match).
        new_anon = m2.dicom_df[
            m2.dicom_df['StudyUID'] == "1.2.STUDY.RESUME"
        ]['AnonFrameOfReferenceUID'].iloc[0]
        assert new_anon == prior_anon

    def test_anon_for_uid_resume_mints_new_for_unseen_pairs(self, temp_dir):
        """Resume seeds Study1; scanning Study1+Study2 → Study1 reuses, Study2 fresh."""
        study1_subdir = os.path.join(temp_dir, "input1")
        os.makedirs(study1_subdir)
        self._make_source_dicom(
            study1_subdir, "studyA",
            sop_instance_uid="1.2.S1.INST",
            series_instance_uid="1.2.S1.SER",
            study_instance_uid="1.2.STUDY.SEEDED",
            frame_of_reference_uid="1.2.FOR.SEEDED",
        )
        m1 = DicomFileManager()
        m1.scan_directory(study1_subdir)
        prior_keys_csv = os.path.join(temp_dir, "keys.csv")
        m1.build_csv_dataframe(study1_subdir).to_csv(prior_keys_csv, index=False)
        seeded_anon = m1.dicom_df.iloc[0]['AnonFrameOfReferenceUID']

        # Second run scans a directory containing BOTH studies.
        combined_subdir = os.path.join(temp_dir, "input2")
        os.makedirs(combined_subdir)
        # Recreate the seeded study so its files exist in this new input root.
        self._make_source_dicom(
            combined_subdir, "studyA",
            sop_instance_uid="1.2.S1.INST",
            series_instance_uid="1.2.S1.SER",
            study_instance_uid="1.2.STUDY.SEEDED",
            frame_of_reference_uid="1.2.FOR.SEEDED",
        )
        self._make_source_dicom(
            combined_subdir, "studyB",
            sop_instance_uid="1.2.S2.INST",
            series_instance_uid="1.2.S2.SER",
            study_instance_uid="1.2.STUDY.UNSEEN",
            frame_of_reference_uid="1.2.FOR.UNSEEN",
            patient_id="P2",
        )

        m2 = DicomFileManager()
        m2.scan_directory(combined_subdir, seed_keys_csv=prior_keys_csv)

        seeded_row_anon = m2.dicom_df[
            m2.dicom_df['StudyUID'] == "1.2.STUDY.SEEDED"
        ]['AnonFrameOfReferenceUID'].iloc[0]
        unseen_row_anon = m2.dicom_df[
            m2.dicom_df['StudyUID'] == "1.2.STUDY.UNSEEN"
        ]['AnonFrameOfReferenceUID'].iloc[0]

        assert seeded_row_anon == seeded_anon
        assert unseen_row_anon != seeded_anon
        assert unseen_row_anon.startswith("2.25.")

    def test_anon_for_uid_resume_degrades_gracefully_for_old_keys_csv(self, temp_dir):
        """A keys.csv missing FOR columns → no exception; all rows minted fresh."""
        self._make_source_dicom(
            temp_dir, "study/ser1",
            sop_instance_uid="1.2.INST.OLDCSV",
            series_instance_uid="1.2.SER.OLDCSV",
            study_instance_uid="1.2.STUDY.OLDCSV",
            frame_of_reference_uid="1.2.FOR.OLDCSV",
        )
        old_csv = os.path.join(temp_dir, "old_keys.csv")
        # Synthesize an older keys.csv without the new columns.
        pd.DataFrame([{
            'InputPath': 'x.dcm',
            'OutputPath': 'x.dcm',
            'StudyUID': '1.2.STUDY.OLDCSV',
            'SeriesUID': '1.2.SER.OLDCSV',
            'AnonStudyUID': '2.25.111',
        }]).to_csv(old_csv, index=False)

        m = DicomFileManager()
        m.scan_directory(temp_dir, seed_keys_csv=old_csv)

        rows = m.dicom_df[m.dicom_df['StudyUID'] == "1.2.STUDY.OLDCSV"]
        assert len(rows) == 1
        anon = rows.iloc[0]['AnonFrameOfReferenceUID']
        assert anon.startswith("2.25.")
        # No partial seeding from the old csv was applied for FOR.
        assert ("1.2.STUDY.OLDCSV", "1.2.FOR.OLDCSV") in m._for_map

    def test_anon_for_uid_columns_in_keys_csv(self, manager, temp_dir):
        """Both new columns persist into the keys.csv-ready dataframe."""
        self._make_source_dicom(
            temp_dir, "study/ser1",
            sop_instance_uid="1.2.INST.CSV",
            series_instance_uid="1.2.SER.CSV",
            study_instance_uid="1.2.STUDY.CSV",
            frame_of_reference_uid="1.2.FOR.CSV",
        )

        manager.scan_directory(temp_dir)
        csv_df = manager.build_csv_dataframe(temp_dir)

        assert csv_df is not None
        assert 'FrameOfReferenceUID' in csv_df.columns
        assert 'AnonFrameOfReferenceUID' in csv_df.columns
        assert 'DICOMDataset' not in csv_df.columns

    # ----------------------------------------------------------------------
    # Unit-level _copy_and_generate_uids three-tier read path tests.
    # ----------------------------------------------------------------------

    def test_copy_and_generate_uids_uses_anon_for_from_dataframe(
        self, manager_with_data, temp_dir
    ):
        """Primary read path: AnonFrameOfReferenceUID column wins."""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        output_path = os.path.join(temp_dir, "test.dcm")
        # Sentinel that does NOT match the fixture-seeded _for_map value,
        # so we know the column is what was read.
        manager_with_data.dicom_df.at[0, 'AnonFrameOfReferenceUID'] = "2.25.SENTINEL"

        manager_with_data._copy_and_generate_uids(ds, source_ds, output_path)

        assert ds.FrameOfReferenceUID == "2.25.SENTINEL"

    def test_copy_and_generate_uids_falls_back_to_for_map_when_df_column_missing(
        self, manager_with_data, temp_dir
    ):
        """Fallback path: when the dataframe column is absent, _for_map is consulted."""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        output_path = os.path.join(temp_dir, "test.dcm")
        # Drop the column so the primary path can't fire.
        manager_with_data.dicom_df = manager_with_data.dicom_df.drop(
            columns=['AnonFrameOfReferenceUID']
        )
        manager_with_data._for_map[
            (str(source_ds.StudyInstanceUID), str(source_ds.FrameOfReferenceUID))
        ] = "2.25.MAPSENTINEL"

        manager_with_data._copy_and_generate_uids(ds, source_ds, output_path)

        assert ds.FrameOfReferenceUID == "2.25.MAPSENTINEL"

    def test_copy_and_generate_uids_last_resort_mints_for_when_no_mapping_available(
        self, manager_with_data, temp_dir, caplog
    ):
        """Last-resort fallback: no dataframe column AND no _for_map entry → mint + log error."""
        ds = pydicom.Dataset()
        # Construct a source_ds whose (StudyUID, FOR) is not in the manager's
        # seeded map and that has a brand-new FOR not in the column.
        source_ds = pydicom.Dataset()
        source_ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.6.1"
        source_ds.SOPInstanceUID = "1.2.NOMAP.INST"
        source_ds.SeriesInstanceUID = "1.2.NOMAP.SER"
        source_ds.StudyInstanceUID = "1.2.NOMAP.STUDY"
        source_ds.FrameOfReferenceUID = "1.2.NOMAP.FOR"
        # Drop the column AND clear the map.
        manager_with_data.dicom_df = manager_with_data.dicom_df.drop(
            columns=['AnonFrameOfReferenceUID']
        )
        manager_with_data._for_map.clear()
        output_path = os.path.join(temp_dir, "test.dcm")

        with caplog.at_level(logging.ERROR):
            manager_with_data._copy_and_generate_uids(ds, source_ds, output_path)

        assert ds.FrameOfReferenceUID
        assert ds.FrameOfReferenceUID.startswith("2.25.")
        assert any("AnonFrameOfReferenceUID" in r.message for r in caplog.records)

    # ----------------------------------------------------------------------
    # End-to-end save → read round-trip.
    # ----------------------------------------------------------------------

    def test_save_anonymized_dicom_emits_shared_for_uid_for_co_located_series(
        self, manager, temp_dir, sample_image_array_single_frame
    ):
        """Two series sharing source FOR in one study → saved files share output FOR."""
        study_uid = "1.2.STUDY.E2E"
        shared_for = "1.2.FOR.E2E"
        self._make_source_dicom(
            temp_dir, "study/serA",
            sop_instance_uid="1.2.INST.E2E.A",
            series_instance_uid="1.2.SER.E2E.A",
            study_instance_uid=study_uid,
            frame_of_reference_uid=shared_for,
        )
        self._make_source_dicom(
            temp_dir, "study/serB",
            sop_instance_uid="1.2.INST.E2E.B",
            series_instance_uid="1.2.SER.E2E.B",
            study_instance_uid=study_uid,
            frame_of_reference_uid=shared_for,
        )

        manager.scan_directory(temp_dir)

        outputs = []
        for idx in range(len(manager.dicom_df)):
            manager.current_index = idx
            out_path = os.path.join(temp_dir, "out", f"out_{idx}.dcm")
            manager.save_anonymized_dicom(sample_image_array_single_frame, out_path)
            outputs.append(out_path)

        a = pydicom.dcmread(outputs[0])
        b = pydicom.dcmread(outputs[1])
        assert a.FrameOfReferenceUID == b.FrameOfReferenceUID
        assert a.FrameOfReferenceUID.startswith("2.25.")

    def test_apply_anonymization_computes_age_in_years_from_birthdate(self, manager_with_data):
        """When source has BirthDate+StudyDate but no PatientAge, age is computed in years."""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "TEST123"
        source_ds.PatientBirthDate = "19800101"
        source_ds.StudyDate = "20200101"

        manager_with_data._apply_anonymization(ds, source_ds)

        assert ds.PatientAge == "040Y"

    def test_apply_anonymization_computes_age_in_months_for_infant(self, manager_with_data):
        """Infant age (>=31 days, <365) is computed in months using 30.4375 days/month."""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "TEST123"
        source_ds.PatientBirthDate = "20191215"
        source_ds.StudyDate = "20200615"

        manager_with_data._apply_anonymization(ds, source_ds)

        assert ds.PatientAge == "006M"

    def test_apply_anonymization_computes_age_in_days_for_newborn(self, manager_with_data):
        """Newborn age (<31 days) is computed in days, zero-padded to three digits."""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "TEST123"
        source_ds.PatientBirthDate = "20200115"
        source_ds.StudyDate = "20200125"

        manager_with_data._apply_anonymization(ds, source_ds)

        assert ds.PatientAge == "010D"

    def test_apply_anonymization_computed_age_capped_at_90y(self, manager_with_data):
        """Computed age >= 90 years is capped at '090Y' per HIPAA Safe Harbor."""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "TEST123"
        source_ds.PatientBirthDate = "19200101"
        source_ds.StudyDate = "20200101"

        manager_with_data._apply_anonymization(ds, source_ds)

        assert ds.PatientAge == "090Y"

    def test_apply_anonymization_preserves_existing_patient_age_does_not_recompute(
        self, manager_with_data
    ):
        """Source PatientAge wins over a (different) BirthDate-derived value — no overwrite."""
        ds = pydicom.Dataset()
        ds.PatientAge = "045Y"
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "TEST123"
        source_ds.PatientBirthDate = "19500101"
        source_ds.StudyDate = "20200101"

        manager_with_data._apply_anonymization(ds, source_ds)

        assert ds.PatientAge == "045Y"

    def test_apply_anonymization_missing_birthdate_no_age_set_logs_warning(
        self, manager_with_data, caplog
    ):
        """With no PatientAge and no BirthDate, PatientAge is left absent and a warning is logged."""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "TEST123"
        source_ds.StudyDate = "20200101"

        with caplog.at_level(logging.WARNING):
            manager_with_data._apply_anonymization(ds, source_ds)

        assert not hasattr(ds, 'PatientAge')
        assert any("PatientAge" in r.message for r in caplog.records)

    def test_apply_anonymization_missing_studydate_no_age_set_logs_warning(
        self, manager_with_data, caplog
    ):
        """With no PatientAge and no StudyDate, PatientAge is left absent and a warning is logged."""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "TEST123"
        source_ds.PatientBirthDate = "19800101"

        with caplog.at_level(logging.WARNING):
            manager_with_data._apply_anonymization(ds, source_ds)

        assert not hasattr(ds, 'PatientAge')
        assert any("PatientAge" in r.message for r in caplog.records)

    def test_apply_anonymization_birthdate_after_studydate_logs_warning(
        self, manager_with_data, caplog
    ):
        """If BirthDate is after StudyDate (data error), no age is set and a warning is logged."""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "TEST123"
        source_ds.PatientBirthDate = "20200101"
        source_ds.StudyDate = "20190101"

        with caplog.at_level(logging.WARNING):
            manager_with_data._apply_anonymization(ds, source_ds)

        assert not hasattr(ds, 'PatientAge')
        assert any("PatientAge" in r.message or "BirthDate" in r.message
                   for r in caplog.records)

    def test_apply_anonymization_malformed_birthdate_logs_warning(
        self, manager_with_data, caplog
    ):
        """A non-parseable BirthDate yields no PatientAge and logs a warning (not an error)."""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "TEST123"
        source_ds.PatientBirthDate = "BADDATE"
        source_ds.StudyDate = "20200101"

        with caplog.at_level(logging.WARNING):
            manager_with_data._apply_anonymization(ds, source_ds)

        assert not hasattr(ds, 'PatientAge')
        assert any("PatientAge" in r.message or "BirthDate" in r.message
                   for r in caplog.records)

    def test_apply_anonymization_empty_birthdate_no_age_set_no_error(
        self, manager_with_data, caplog
    ):
        """Empty BirthDate is a valid 'no value' state — no age set, no ERROR-level log."""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "TEST123"
        source_ds.PatientBirthDate = ""
        source_ds.StudyDate = "20200101"

        with caplog.at_level(logging.WARNING):
            manager_with_data._apply_anonymization(ds, source_ds)

        assert not hasattr(ds, 'PatientAge')
        assert all(r.levelno < logging.ERROR for r in caplog.records)

    def test_apply_anonymization_clears_birthdate_when_age_computed(self, manager_with_data):
        """PatientBirthDate is still cleared to '' even when PatientAge is computed from it."""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "TEST123"
        source_ds.PatientBirthDate = "19800101"
        source_ds.StudyDate = "20200101"

        manager_with_data._apply_anonymization(ds, source_ds)

        assert ds.PatientBirthDate == ""
        assert ds.PatientAge == "040Y"

    def test_copy_source_metadata_preserves_transducer_type(self, manager_with_data, temp_dir):
        """Test TransducerType present in source is preserved in de-id output"""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        source_ds.TransducerType = "LINEAR"
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert ds.TransducerType == "LINEAR"

    def test_copy_source_metadata_blanks_missing_transducer_type(self, manager_with_data, temp_dir):
        """Test TransducerType is present as empty string when absent in source"""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        if hasattr(source_ds, 'TransducerType'):
            delattr(source_ds, 'TransducerType')
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert hasattr(ds, 'TransducerType')
        assert ds.TransducerType == ""

    def test_copy_source_metadata_blanks_missing_transducer_data(self, manager_with_data, temp_dir):
        """Test TransducerData is present as empty string when absent in source"""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        if hasattr(source_ds, 'TransducerData'):
            delattr(source_ds, 'TransducerData')
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert hasattr(ds, 'TransducerData')
        assert ds.TransducerData == ""

    def test_copy_source_metadata_blanks_missing_manufacturer_model_name(self, manager_with_data, temp_dir):
        """Test ManufacturerModelName is present as empty string when absent in source"""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        if hasattr(source_ds, 'ManufacturerModelName'):
            delattr(source_ds, 'ManufacturerModelName')
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert hasattr(ds, 'ManufacturerModelName')
        assert ds.ManufacturerModelName == ""

    def test_copy_source_metadata_preserves_empty_transducer_data(self, manager_with_data, temp_dir):
        """Test TransducerData empty string in source is preserved as empty in de-id"""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        source_ds.TransducerData = ""
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert hasattr(ds, 'TransducerData')
        assert ds.TransducerData == ""

    def test_copy_source_metadata_strips_transducer_data_serial(self, manager_with_data, temp_dir):
        """Comma-delimited TransducerData is trimmed to the leading model segment."""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        source_ds.TransducerData = "SC6-1s,JK9U41102597"
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert ds.TransducerData == "SC6-1s"

    def test_copy_source_metadata_strips_transducer_data_backslash(self, manager_with_data, temp_dir):
        """Backslash-delimited TransducerData (VR LO MultiValue) collapses to the first segment string."""
        from pydicom.multival import MultiValue
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        source_ds.TransducerData = MultiValue(str, ["S4-1U", "UNUSED", "UNUSED"])
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert ds.TransducerData == "S4-1U"

    def test_copy_source_metadata_preserves_transducer_data_case(self, manager_with_data, temp_dir):
        """A single-segment TransducerData (no delimiter) survives unchanged with original case."""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        source_ds.TransducerData = "C1-5"
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert ds.TransducerData == "C1-5"

    def test_copy_source_metadata_strips_transducer_data_sp5_cb3c(self, manager_with_data, temp_dir):
        """User example: 'SP5-1s,CB3C' -> 'SP5-1s'."""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        source_ds.TransducerData = "SP5-1s,CB3C"
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert ds.TransducerData == "SP5-1s"

    def test_copy_source_metadata_strips_transducer_data_sc6_jk9(self, manager_with_data, temp_dir):
        """User example: 'SC6-1s,JK9' -> 'SC6-1s'."""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        source_ds.TransducerData = "SC6-1s,JK9"
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert ds.TransducerData == "SC6-1s"

    def test_copy_source_metadata_strips_transducer_data_only_delimiters(self, manager_with_data, temp_dir):
        """Pathological all-delimiter input becomes empty."""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        source_ds.TransducerData = "\\\\"
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert ds.TransducerData == ""

    def test_copy_source_metadata_strips_transducer_data_with_whitespace(self, manager_with_data, temp_dir):
        """Internal whitespace around the leading segment is stripped."""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        source_ds.TransducerData = " SC6-1s ,02597"
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert ds.TransducerData == "SC6-1s"

    def test_copy_source_metadata_preserves_manufacturer(self, manager_with_data, temp_dir):
        """Manufacturer (0008,0070) round-trips verbatim — regression lock."""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        source_ds.Manufacturer = "GE Healthcare"
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert ds.Manufacturer == "GE Healthcare"

    def test_copy_source_metadata_preserves_manufacturer_model_name(self, manager_with_data, temp_dir):
        """ManufacturerModelName (0008,1090) round-trips verbatim — regression lock."""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        source_ds.ManufacturerModelName = "Vivid E95"
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert ds.ManufacturerModelName == "Vivid E95"

    def test_copy_source_metadata_preserves_ge_vivid_pair(self, manager_with_data, temp_dir):
        """Most common GE configuration: both vendor/model survive intact together."""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        source_ds.Manufacturer = "GE Healthcare"
        source_ds.ManufacturerModelName = "Vivid E95"
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_source_metadata(ds, source_ds, output_path)

        assert ds.Manufacturer == "GE Healthcare"
        assert ds.ManufacturerModelName == "Vivid E95"

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

    def test_copy_and_generate_uids_remaps_series_uid(self, manager_with_data, temp_dir):
        """SeriesInstanceUID is remapped via remap_uid when source provides one."""
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        output_path = os.path.join(temp_dir, "test.dcm")

        original_series_uid = source_ds.SeriesInstanceUID

        manager_with_data._copy_and_generate_uids(ds, source_ds, output_path)

        assert ds.SeriesInstanceUID == remap_uid(str(original_series_uid))
        assert ds.SeriesInstanceUID != original_series_uid

    def test_copy_and_generate_uids_generates_series_uid_when_missing(self, manager_with_data, temp_dir):
        """When SeriesInstanceUID is missing, the fallback still lands in the 2.25 arc."""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()  # empty — no SeriesInstanceUID
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._copy_and_generate_uids(ds, source_ds, output_path)

        assert ds.SeriesInstanceUID.startswith("2.25.")

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
            headers_directory,
            anonymized_dataset=current_record.DICOMDataset,
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
            headers_directory,
            anonymized_dataset=current_record.DICOMDataset,
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
            headers_directory,
            anonymized_dataset=current_record.DICOMDataset,
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
            None,
            anonymized_dataset=current_record.DICOMDataset,
        )

        assert result is None

    def test_save_anonymized_dicom_header_raises_error_when_no_filename(self, manager_with_data, temp_dir):
        headers_directory = os.path.join(temp_dir, "headers")
        current_record = manager_with_data.dicom_df.iloc[0]

        with pytest.raises(ValueError, match="Output filename is required"):
            manager_with_data.save_anonymized_dicom_header(
                current_record,
                "",
                headers_directory,
                anonymized_dataset=current_record.DICOMDataset,
            )

        with pytest.raises(ValueError, match="Output filename is required"):
            manager_with_data.save_anonymized_dicom_header(
                current_record,
                "",
                headers_directory,
                anonymized_dataset=current_record.DICOMDataset,
            )

    def test_save_anonymized_dicom_header_raises_error_when_no_record(self, manager_with_data, temp_dir):
        headers_directory = os.path.join(temp_dir, "headers")
        current_record = None
        real_record = manager_with_data.dicom_df.iloc[0]

        with pytest.raises(ValueError, match="Current DICOM record is required"):
            manager_with_data.save_anonymized_dicom_header(
                current_record,
                "test.dcm",
                headers_directory,
                anonymized_dataset=real_record.DICOMDataset,
            )

    def test_save_anonymized_dicom_header_flatten_directory_structure(self, manager_with_data, temp_dir):
        headers_directory = os.path.join(temp_dir, "headers")

        # Create a record with nested relative path
        current_record = manager_with_data.dicom_df.iloc[0].copy()

        result_path = manager_with_data.save_anonymized_dicom_header(
            current_record,
            "test.dcm",
            headers_directory,
            anonymized_dataset=current_record.DICOMDataset,
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


class TestBuildCsvDataframe:
    """Tests for DicomFileManager.build_csv_dataframe.

    The helper prepares dicom_df for keys.csv serialization: drops the
    DICOMDataset binary column and rewrites InputPath relative to the given
    input_folder. Internal dicom_df must NOT be mutated.
    """

    ROOT = os.path.normpath('/root')

    @pytest.fixture
    def manager(self):
        return DicomFileManager()

    def _make_row(self, input_path, output_path='anon.dcm'):
        return {
            'InputPath': input_path,
            'OutputPath': output_path,
            'AnonFilename': 'anon.dcm',
            'PatientUID': 'P1',
            'StudyUID': 'S1',
            'SeriesUID': 'Se1',
            'InstanceUID': 'I1',
            'AnonStudyUID': '2.25.111',
            'AnonSeriesUID': '2.25.222',
            'AnonSOPInstanceUID': '2.25.333',
            'PhysicalDeltaX': 0.1,
            'PhysicalDeltaY': 0.1,
            'ContentDate': '20240101',
            'ContentTime': '000000',
            'Patch': False,
            'TransducerModel': 'tm',
            'DICOMDataset': object(),
        }

    def _populate(self, manager, rows):
        manager.dicom_df = pd.DataFrame(rows, columns=DicomFileManager.DICOM_DATAFRAME_COLUMNS)

    def test_rewrites_input_path_to_relative(self, manager):
        abs_path = os.path.join(self.ROOT, 'a', 'b', 'IM001.dcm')
        self._populate(manager, [self._make_row(abs_path)])

        df = manager.build_csv_dataframe(self.ROOT)

        assert df.iloc[0]['InputPath'] == os.path.join('a', 'b', 'IM001.dcm')

    def test_preserves_nested_subdirs(self, manager):
        abs_path = os.path.join(self.ROOT, 'p1', 's1', 'sub', 'IM001.dcm')
        self._populate(manager, [self._make_row(abs_path)])

        df = manager.build_csv_dataframe(self.ROOT)

        assert df.iloc[0]['InputPath'] == os.path.join('p1', 's1', 'sub', 'IM001.dcm')

    def test_drops_dicomdataset_column(self, manager):
        abs_path = os.path.join(self.ROOT, 'a', 'IM001.dcm')
        self._populate(manager, [self._make_row(abs_path)])

        df = manager.build_csv_dataframe(self.ROOT)

        assert 'DICOMDataset' not in df.columns
        for col in ('InputPath', 'OutputPath', 'AnonFilename', 'PatientUID'):
            assert col in df.columns

    def test_preserves_output_path_unchanged(self, manager):
        abs_path = os.path.join(self.ROOT, 'a', 'IM001.dcm')
        self._populate(manager, [self._make_row(abs_path, output_path='a/anon_123.dcm')])

        df = manager.build_csv_dataframe(self.ROOT)

        assert df.iloc[0]['OutputPath'] == 'a/anon_123.dcm'
        assert df.iloc[0]['OutputPath'] == manager.dicom_df.iloc[0]['OutputPath']

    def test_returns_none_for_empty_df(self, manager):
        manager.dicom_df = pd.DataFrame()

        assert manager.build_csv_dataframe(self.ROOT) is None

    def test_returns_none_when_df_is_none(self, manager):
        manager.dicom_df = None

        assert manager.build_csv_dataframe(self.ROOT) is None

    def test_no_root_leaves_input_path_absolute(self, manager):
        abs_path = os.path.join(self.ROOT, 'a', 'IM001.dcm')
        self._populate(manager, [self._make_row(abs_path)])

        df_none = manager.build_csv_dataframe(None)
        df_empty = manager.build_csv_dataframe('')

        assert df_none.iloc[0]['InputPath'] == abs_path
        assert df_empty.iloc[0]['InputPath'] == abs_path

    def test_handles_trailing_slash_in_input_folder(self, manager):
        abs_path = os.path.join(self.ROOT, 'a', 'IM001.dcm')
        self._populate(manager, [self._make_row(abs_path)])

        df_slash = manager.build_csv_dataframe(self.ROOT + os.sep)
        df_no_slash = manager.build_csv_dataframe(self.ROOT)

        assert df_slash.iloc[0]['InputPath'] == df_no_slash.iloc[0]['InputPath']

    def test_anon_uid_columns_pass_through_unchanged(self, manager):
        """Anon* UID columns are persisted verbatim — not subject to the InputPath relative rewrite."""
        abs_path = os.path.join(self.ROOT, 'a', 'IM001.dcm')
        self._populate(manager, [self._make_row(abs_path)])

        df = manager.build_csv_dataframe(self.ROOT)

        assert 'AnonStudyUID' in df.columns
        assert 'AnonSeriesUID' in df.columns
        assert 'AnonSOPInstanceUID' in df.columns
        assert df.iloc[0]['AnonStudyUID'] == '2.25.111'
        assert df.iloc[0]['AnonSeriesUID'] == '2.25.222'
        assert df.iloc[0]['AnonSOPInstanceUID'] == '2.25.333'

    def test_dataframe_columns_match_constant_ordering(self, manager):
        """build_csv_dataframe preserves DICOM_DATAFRAME_COLUMNS ordering minus DICOMDataset."""
        abs_path = os.path.join(self.ROOT, 'a', 'IM001.dcm')
        self._populate(manager, [self._make_row(abs_path)])

        df = manager.build_csv_dataframe(self.ROOT)

        expected = [c for c in DicomFileManager.DICOM_DATAFRAME_COLUMNS if c != 'DICOMDataset']
        assert list(df.columns) == expected


class TestRedScaffolding:
    """RED scaffolding for the de-id review fixes.

    These tests fail on the current code and pass once the corresponding
    GREEN step lands. Grouped by reviewer finding so each block can be
    pointed at its production fix.
    """

    PATIENT_ID = "REDPATIENT"
    PATIENT_NAME = "Red^Scaffold^Test"
    STUDY_UID = "1.2.840.RED.STUDY"
    SERIES_UID = "1.2.840.RED.SERIES"
    SOP_INSTANCE_UID = "1.2.840.RED.SOP"
    SOURCE_FOR_UID = "1.2.840.RED.FOR"
    ANON_FOR_UID = "2.25.RED.ANONFOR"
    TRANSDUCER_DATA = "SC6-1s,SERIAL12345"
    TRANSDUCER_MODEL = "sc6-1s"
    FILE_NAME = "red_test.dcm"

    @pytest.fixture
    def manager(self):
        return DicomFileManager()

    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d)

    def _make_source_ds(self, *, sop_uid=None, series_uid=None,
                        study_uid=None, for_uid=None,
                        patient_id=None, patient_name=None,
                        transducer_data=None):
        """Build a minimal source DICOM dataset for the RED tests."""
        helper = TestDicomFileManager()
        ds = helper.create_test_dicom_file(
            PatientID=patient_id or self.PATIENT_ID,
            PatientName=patient_name or self.PATIENT_NAME,
            StudyInstanceUID=study_uid or self.STUDY_UID,
            SeriesInstanceUID=series_uid or self.SERIES_UID,
            SOPInstanceUID=sop_uid or self.SOP_INSTANCE_UID,
            TransducerData=transducer_data or self.TRANSDUCER_DATA,
        )
        ds.FrameOfReferenceUID = for_uid or self.SOURCE_FOR_UID
        return ds

    @pytest.fixture
    def manager_with_red_data(self, manager, temp_dir):
        """Populate manager.dicom_df with one row that has all Anon* columns set."""
        helper = TestDicomFileManager()
        ds = self._make_source_ds()
        filename = manager._generate_filename_from_dicom(ds)
        filepath = helper.save_dicom_file(ds, temp_dir, filename)

        dicom_data = [{
            'InputPath': filepath,
            'OutputPath': os.path.relpath(filepath, temp_dir),
            'AnonFilename': self.FILE_NAME,
            'PatientUID': self.PATIENT_ID,
            'StudyUID': self.STUDY_UID,
            'SeriesUID': self.SERIES_UID,
            'InstanceUID': self.SOP_INSTANCE_UID,
            'AnonStudyUID': remap_uid(self.STUDY_UID),
            'AnonSeriesUID': remap_uid(self.SERIES_UID),
            'AnonSOPInstanceUID': remap_uid(self.SOP_INSTANCE_UID),
            'FrameOfReferenceUID': self.SOURCE_FOR_UID,
            'AnonFrameOfReferenceUID': self.ANON_FOR_UID,
            'PhysicalDeltaX': 0.1,
            'PhysicalDeltaY': 0.15,
            'ContentDate': "20240101",
            'ContentTime': "120000",
            'Patch': False,
            'TransducerModel': self.TRANSDUCER_MODEL,
            'DICOMDataset': ds,
        }]

        manager._create_dataframe(dicom_data)
        manager._for_map[(self.STUDY_UID, self.SOURCE_FOR_UID)] = self.ANON_FOR_UID
        manager.current_index = 0
        return manager

    @pytest.fixture
    def sample_image_array(self):
        return np.random.randint(0, 255, (1, 10, 15, 1), dtype=np.uint8)

    # ------------------------------------------------------------------
    # F1 — _DICOMHeader.json must serialize from the anonymized dataset.
    # ------------------------------------------------------------------

    def test_save_anonymized_dicom_returns_anonymized_dataset(
        self, manager_with_red_data, temp_dir, sample_image_array
    ):
        """save_anonymized_dicom must return the in-memory anonymized DS so
        the header sidecar can serialize from the same source of truth as the
        saved .dcm. Currently returns None — RED."""
        output_path = os.path.join(temp_dir, "out.dcm")
        # _populate_anon_for_column mints the actual run-local FOR UID
        # during _create_dataframe; assert against whatever it minted, not a
        # fixture sentinel.
        expected_anon_for = manager_with_red_data.dicom_df.iloc[0]['AnonFrameOfReferenceUID']
        result = manager_with_red_data.save_anonymized_dicom(
            sample_image_array, output_path,
            new_patient_name="anon_pt", new_patient_id="ANON_PID"
        )

        assert isinstance(result, pydicom.Dataset)
        assert result.PatientID == "ANON_PID"
        assert result.SOPInstanceUID == remap_uid(self.SOP_INSTANCE_UID)
        assert result.StudyInstanceUID == remap_uid(self.STUDY_UID)
        assert result.SeriesInstanceUID == remap_uid(self.SERIES_UID)
        assert result.FrameOfReferenceUID == expected_anon_for
        assert result.FrameOfReferenceUID.startswith("2.25.")

    def test_save_anonymized_dicom_header_accepts_anonymized_dataset_kwarg(
        self, manager_with_red_data, temp_dir, sample_image_array
    ):
        """save_anonymized_dicom_header must accept anonymized_dataset kwarg.
        Currently no such param — RED with TypeError."""
        headers_dir = os.path.join(temp_dir, "headers")
        output_path = os.path.join(temp_dir, "out.dcm")
        anon_ds = manager_with_red_data.save_anonymized_dicom(
            sample_image_array, output_path,
            new_patient_name="anon_pt", new_patient_id="ANON_PID"
        )

        result_path = manager_with_red_data.save_anonymized_dicom_header(
            manager_with_red_data.dicom_df.iloc[0],
            "anon_pt.dcm",
            headers_dir,
            anonymized_dataset=anon_ds,
        )
        assert result_path is not None
        assert os.path.exists(result_path)

    def test_header_json_does_not_leak_source_uids(
        self, manager_with_red_data, temp_dir, sample_image_array
    ):
        """The header JSON sidecar must contain remapped UIDs and must NOT
        contain any original source UID values."""
        headers_dir = os.path.join(temp_dir, "headers")
        output_path = os.path.join(temp_dir, "out.dcm")
        expected_anon_for = manager_with_red_data.dicom_df.iloc[0]['AnonFrameOfReferenceUID']
        anon_ds = manager_with_red_data.save_anonymized_dicom(
            sample_image_array, output_path,
            new_patient_name="anon_pt", new_patient_id="ANON_PID"
        )
        result_path = manager_with_red_data.save_anonymized_dicom_header(
            manager_with_red_data.dicom_df.iloc[0],
            "anon_pt.dcm",
            headers_dir,
            anonymized_dataset=anon_ds,
        )

        with open(result_path) as f:
            serialized = f.read()

        assert self.STUDY_UID not in serialized
        assert self.SERIES_UID not in serialized
        assert self.SOP_INSTANCE_UID not in serialized
        assert self.SOURCE_FOR_UID not in serialized
        assert remap_uid(self.STUDY_UID) in serialized
        assert remap_uid(self.SERIES_UID) in serialized
        assert remap_uid(self.SOP_INSTANCE_UID) in serialized
        assert expected_anon_for in serialized

    def test_header_json_uses_trimmed_transducer_data(
        self, manager_with_red_data, temp_dir, sample_image_array
    ):
        """The header JSON sidecar must contain only the trimmed first
        segment of TransducerData. The source's vendor serial number must
        not leak."""
        headers_dir = os.path.join(temp_dir, "headers")
        output_path = os.path.join(temp_dir, "out.dcm")
        anon_ds = manager_with_red_data.save_anonymized_dicom(
            sample_image_array, output_path,
            new_patient_name="anon_pt", new_patient_id="ANON_PID"
        )
        result_path = manager_with_red_data.save_anonymized_dicom_header(
            manager_with_red_data.dicom_df.iloc[0],
            "anon_pt.dcm",
            headers_dir,
            anonymized_dataset=anon_ds,
        )

        with open(result_path) as f:
            serialized = f.read()

        assert "SERIAL12345" not in serialized
        assert "SC6-1s" in serialized

    # ------------------------------------------------------------------
    # F4 — _for_map must be cleared at start of each scan_directory call.
    # ------------------------------------------------------------------

    def test_scan_directory_clears_for_map_between_unrelated_scans(self, temp_dir):
        """Two scans of unrelated studies on one DicomFileManager must NOT
        share _for_map entries. Currently entries from scan #1 persist into
        scan #2 — RED."""
        helper = TestDicomFileManager()

        # Scan 1 input.
        scan1_dir = os.path.join(temp_dir, "scan1")
        os.makedirs(scan1_dir)
        ds1 = helper.create_test_dicom_file(
            StudyInstanceUID="1.2.RED.STUDY.A",
            SeriesInstanceUID="1.2.RED.SER.A",
            SOPInstanceUID="1.2.RED.SOP.A",
        )
        ds1.FrameOfReferenceUID = "1.2.RED.FOR.A"
        helper.save_dicom_file(ds1, scan1_dir, "a.dcm")

        # Scan 2 input — completely different study.
        scan2_dir = os.path.join(temp_dir, "scan2")
        os.makedirs(scan2_dir)
        ds2 = helper.create_test_dicom_file(
            StudyInstanceUID="1.2.RED.STUDY.B",
            SeriesInstanceUID="1.2.RED.SER.B",
            SOPInstanceUID="1.2.RED.SOP.B",
        )
        ds2.FrameOfReferenceUID = "1.2.RED.FOR.B"
        helper.save_dicom_file(ds2, scan2_dir, "b.dcm")

        m = DicomFileManager()
        m.scan_directory(scan1_dir)
        scan1_keys = set(m._for_map.keys())
        assert ("1.2.RED.STUDY.A", "1.2.RED.FOR.A") in scan1_keys

        # Re-scan a different directory on the same manager.
        m.scan_directory(scan2_dir)

        # After scan #2, _for_map must contain ONLY scan #2's entries.
        # Currently scan #1's entries persist — RED.
        assert ("1.2.RED.STUDY.A", "1.2.RED.FOR.A") not in m._for_map
        assert ("1.2.RED.STUDY.B", "1.2.RED.FOR.B") in m._for_map

    def test_scan_directory_remints_anon_for_when_rescanning_same_study(self, temp_dir):
        """Run-local unlinkability: scanning the same study twice on one manager
        without seed_keys_csv must yield a different anonymized FOR UID the
        second time. Currently the second scan reuses the first scan's
        _for_map entry — RED."""
        helper = TestDicomFileManager()
        ds = helper.create_test_dicom_file(
            StudyInstanceUID="1.2.RED.STUDY.RESCAN",
            SeriesInstanceUID="1.2.RED.SER.RESCAN",
            SOPInstanceUID="1.2.RED.SOP.RESCAN",
        )
        ds.FrameOfReferenceUID = "1.2.RED.FOR.RESCAN"
        helper.save_dicom_file(ds, temp_dir, "rescan.dcm")

        m = DicomFileManager()
        m.scan_directory(temp_dir)
        first_anon = m.dicom_df.iloc[0]['AnonFrameOfReferenceUID']

        # Re-scan the same directory on the same manager — no seed.
        m.scan_directory(temp_dir)
        second_anon = m.dicom_df.iloc[0]['AnonFrameOfReferenceUID']

        assert first_anon != second_anon
        assert first_anon.startswith("2.25.")
        assert second_anon.startswith("2.25.")

    def test_scan_directory_preserves_seeded_mapping_after_reset(self, temp_dir):
        """Reset-then-seed ordering must not break the resume contract:
        when a seed_keys_csv is provided, the seeded entries survive the
        reset and the second scan reuses them."""
        helper = TestDicomFileManager()
        ds = helper.create_test_dicom_file(
            StudyInstanceUID="1.2.RED.STUDY.RESUME",
            SeriesInstanceUID="1.2.RED.SER.RESUME",
            SOPInstanceUID="1.2.RED.SOP.RESUME",
        )
        ds.FrameOfReferenceUID = "1.2.RED.FOR.RESUME"
        helper.save_dicom_file(ds, temp_dir, "resume.dcm")

        m1 = DicomFileManager()
        m1.scan_directory(temp_dir)
        prior_anon = m1.dicom_df.iloc[0]['AnonFrameOfReferenceUID']
        prior_keys = os.path.join(temp_dir, "keys.csv")
        m1.build_csv_dataframe(temp_dir).to_csv(prior_keys, index=False)

        # Second scan on a DIFFERENT manager with the seed CSV.
        m2 = DicomFileManager()
        # Pre-pollute m2._for_map with stale entries to verify they get cleared
        # before seeding so they don't shadow seeded mappings.
        m2._for_map[("1.2.STALE.STUDY", "1.2.STALE.FOR")] = "2.25.STALE"
        m2.scan_directory(temp_dir, seed_keys_csv=prior_keys)

        rescanned_anon = m2.dicom_df.iloc[0]['AnonFrameOfReferenceUID']
        assert rescanned_anon == prior_anon
        # Stale entry must have been cleared by the reset-then-seed sequence.
        assert ("1.2.STALE.STUDY", "1.2.STALE.FOR") not in m2._for_map

    # ------------------------------------------------------------------
    # F5 — PatientAge calendar-year math at the exact one-year boundary.
    # ------------------------------------------------------------------

    def test_compute_patient_age_returns_001y_at_exact_one_year(self, manager_with_red_data):
        """A patient exactly 365 days old (no leap year between dates) is one
        calendar year old and must yield '001Y'. Current `int(365 // 365.25)`
        returns 0 → '000Y' — RED."""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "REDPATIENT"
        source_ds.PatientBirthDate = "20210601"  # 2021 non-leap
        source_ds.StudyDate = "20220601"          # exactly 365 days later

        manager_with_red_data._apply_anonymization(ds, source_ds)

        assert ds.PatientAge == "001Y"

    def test_compute_patient_age_handles_leap_spanning_one_year(self, manager_with_red_data):
        """A patient born 2020-02-15 and studied 2021-02-15 spans a leap day
        (delta_days = 366). Both old and new math return '001Y'; this guards
        against accidentally over-correcting."""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "REDPATIENT"
        source_ds.PatientBirthDate = "20200215"
        source_ds.StudyDate = "20210215"

        manager_with_red_data._apply_anonymization(ds, source_ds)

        assert ds.PatientAge == "001Y"

    def test_compute_patient_age_returns_months_when_birthday_not_yet(
        self, manager_with_red_data
    ):
        """Birthday hasn't arrived yet: 2020-06-01 → 2021-05-31 is 364 days.
        Falls into the months branch with current code (`< 365`) and must
        keep doing so after the fix."""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "REDPATIENT"
        source_ds.PatientBirthDate = "20200601"
        source_ds.StudyDate = "20210531"

        manager_with_red_data._apply_anonymization(ds, source_ds)

        # 364 / 30.4375 ≈ 11.96 → 11M (current behavior is correct here).
        assert ds.PatientAge == "011M"

    def test_compute_patient_age_at_two_years_exact(self, manager_with_red_data):
        """delta_days = 731 (one leap year between 2-year span) must yield
        '002Y'. Regression test for the years branch under the new math."""
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "REDPATIENT"
        source_ds.PatientBirthDate = "20190601"
        source_ds.StudyDate = "20210601"  # 2020 was a leap year → 731 days

        manager_with_red_data._apply_anonymization(ds, source_ds)

        assert ds.PatientAge == "002Y"


if __name__ == "__main__":
    pytest.main([__file__])
