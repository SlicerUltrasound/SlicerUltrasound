import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from ..dicom_file_manager import DicomFileManager
from ..key_store import KEYS_CSV_COLUMNS, KeyStore, KeysFileError, uid_to_filename
from .fixtures import (
    SECONDARY_CAPTURE_STORAGE,
    SOURCE_UID_ROOT,
    US_IMAGE_STORAGE,
    US_IMAGE_STORAGE_RETIRED,
    US_MULTIFRAME_STORAGE,
    US_MULTIFRAME_STORAGE_RETIRED,
    make_dataset,
    set_unchecked,
    write_dataset,
)

HEX32 = 'd' * 32
UID_2_25 = re.compile(r'^2\.25\.\d{1,39}$')


class TestDicomFileManager:
    """Test suite for DicomFileManager."""

    PATIENT_ID = 'MRN0012345'
    STUDY_UID = SOURCE_UID_ROOT + '.1'
    SERIES_UID = SOURCE_UID_ROOT + '.2'
    SOP_INSTANCE_UID = SOURCE_UID_ROOT + '.3'
    SOURCE_FOR_UID = SOURCE_UID_ROOT + '.4'
    SOP_CLASS_UID = US_MULTIFRAME_STORAGE
    TRANSDUCER_MODEL = 'sc6-1s'
    BITS_ALLOCATED = 8
    BITS_STORED = 8

    @pytest.fixture
    def manager(self):
        return DicomFileManager()

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_image_array_multi_frame(self):
        """Multi-frame RGB image array (frames, height, width, channels)."""
        return np.random.randint(0, 255, (5, 10, 15, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_image_array_single_frame(self):
        """Single-frame grayscale image array (frames, height, width, channels)."""
        return np.random.randint(0, 255, (1, 10, 15, 1), dtype=np.uint8)

    def _scan(self, manager, input_dir, files, key_store=None):
        """Write ``files`` ({relative name: dataset}) under ``input_dir`` and scan them."""
        for name, ds in files.items():
            write_dataset(ds, os.path.join(input_dir, os.path.dirname(name)), os.path.basename(name))
        count = manager.scan_directory(input_dir, False, key_store=key_store or KeyStore())
        manager.current_index = 0
        return count

    @pytest.fixture
    def manager_with_data(self, manager, temp_dir):
        """Manager whose dataframe holds one scanned instance (current_index 0)."""
        ds = make_dataset(FrameOfReferenceUID=self.SOURCE_FOR_UID, SeriesNumber=4,
                          StudyDate='20240101', StudyTime='101010',
                          SeriesDate='20240101', SeriesTime='101011')
        assert self._scan(manager, temp_dir, {'source.dcm': ds}) == 1
        return manager

    def _anonymize(self, manager, ds, source_ds, name='', patient_id=HEX32, shift=10):
        manager._apply_anonymization(ds, source_ds, name, patient_id, shift)

    # ------------------------------------------------------------------
    # Basics
    # ------------------------------------------------------------------

    def test_init(self, manager):
        assert manager.dicom_df is None
        assert manager.next_index == 0
        assert manager.current_index == 0
        assert manager.issuers_of_patient_id == set()
        assert manager.skipped_unsupported_sop_class == 0

    def test_reset_clears_scan_state(self, manager_with_data):
        manager_with_data.reset()
        assert manager_with_data.dicom_df is None
        assert manager_with_data.next_index == 0
        assert manager_with_data.current_index == 0
        assert manager_with_data.key_store is None
        assert manager_with_data.issuers_of_patient_id == set()

    def test_get_transducer_model_valid(self, manager):
        assert manager.get_transducer_model("SC6-1s,02597") == "sc6-1s"
        assert manager.get_transducer_model("L12-3,12345") == "l12-3"
        assert manager.get_transducer_model("C1-5") == "c1-5"
        assert manager.get_transducer_model("SC6-1s,02597", manufacturer="Philips") == "sc6-1s"

    def test_get_transducer_model_invalid(self, manager):
        assert manager.get_transducer_model("") == "unknown"
        assert manager.get_transducer_model(None) == "unknown"
        assert manager.get_transducer_model("   ") == "unknown"

    def test_get_transducer_model_backslash_format(self, manager):
        assert manager.get_transducer_model("S4-1U\\UNUSED\\UNUSED") == "s4-1u"
        assert manager.get_transducer_model("L12-3\\UNUSED") == "l12-3"

    def test_get_transducer_model_backslash_multivalue(self, manager):
        from pydicom.multival import MultiValue
        mv = MultiValue(str, ["S4-1U", "UNUSED", "UNUSED"])
        assert manager.get_transducer_model(mv) == "s4-1u"

    def test_get_transducer_model_butterfly_uses_model_name(self, manager):
        assert manager.get_transducer_model(
            "", manufacturer="Butterfly Network Inc", manufacturer_model_name="IQ"
        ) == "iq"
        assert manager.get_transducer_model(
            "IgnoreMe,123", manufacturer="Butterfly Network Inc", manufacturer_model_name="IQ3"
        ) == "iq3"

    def test_get_transducer_model_butterfly_case_insensitive(self, manager):
        for manufacturer in ("butterfly network", "BUTTERFLY NETWORK INC", "Butterfly Network Inc."):
            assert manager.get_transducer_model(
                "", manufacturer=manufacturer, manufacturer_model_name="IQ"
            ) == "iq"

    def test_get_transducer_model_butterfly_missing_model_name(self, manager):
        assert manager.get_transducer_model(
            "SC6-1s,02597", manufacturer="Butterfly Network Inc", manufacturer_model_name=""
        ) == "unknown"
        assert manager.get_transducer_model(
            "SC6-1s,02597", manufacturer="Butterfly Network Inc", manufacturer_model_name=None
        ) == "unknown"

    def test_get_number_of_instances_empty(self, manager):
        assert manager.get_number_of_instances() == 0

    def test_get_number_of_instances_with_data(self, manager):
        manager.dicom_df = pd.DataFrame({'test': [1, 2, 3]})
        assert manager.get_number_of_instances() == 3

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def test_scan_directory_populates_mapping_columns(self, manager_with_data, temp_dir):
        row = manager_with_data.dicom_df.iloc[0]

        assert list(manager_with_data.dicom_df.columns) == DicomFileManager.DICOM_DATAFRAME_COLUMNS
        assert row['InputPath'] == os.path.join(temp_dir, 'source.dcm')
        assert row['OutputPath'] == row['AnonFilename']
        assert row['AnonFilename'] == uid_to_filename(row['AnonSOPInstanceUID'])
        assert row['PatientUID'] == self.PATIENT_ID
        assert row['StudyUID'] == self.STUDY_UID
        assert row['SeriesUID'] == self.SERIES_UID
        assert row['InstanceUID'] == self.SOP_INSTANCE_UID
        assert row['FrameOfReferenceUID'] == self.SOURCE_FOR_UID
        assert re.fullmatch(r'[0-9a-f]{32}', row['AnonPatientID'])
        for column in ('AnonStudyUID', 'AnonSeriesUID', 'AnonSOPInstanceUID', 'AnonFrameOfReferenceUID'):
            assert UID_2_25.match(row[column]), column
            assert row[column] not in (self.STUDY_UID, self.SERIES_UID, self.SOP_INSTANCE_UID, self.SOURCE_FOR_UID)
        assert 1 <= int(row['DateShiftDays']) <= 1095
        assert row['PhysicalDeltaX'] == 0.1
        assert row['TransducerModel'] == self.TRANSDUCER_MODEL
        assert row['Patch'] is False or row['Patch'] == False  # noqa: E712 - numpy bool
        assert isinstance(row['DICOMDataset'], pydicom.Dataset)

    def test_scan_directory_records_relative_input_path_in_the_store(self, manager, temp_dir):
        store = KeyStore()
        self._scan(manager, temp_dir, {'siteA/visit1/source.dcm': make_dataset()}, key_store=store)

        stored = store.rows()[0]
        assert stored['InputPath'] == 'siteA/visit1/source.dcm'
        assert stored['OutputPath'] == 'siteA/visit1/' + stored['AnonFilename']
        assert manager.dicom_df.iloc[0]['InputPath'] == os.path.join(temp_dir, 'siteA', 'visit1', 'source.dcm')
        assert manager.dicom_df.iloc[0]['OutputPath'] == stored['OutputPath']

    def test_scan_directory_skips_single_frame_when_asked(self, manager, temp_dir):
        write_dataset(make_dataset(NumberOfFrames=1), temp_dir, 'single.dcm')
        assert manager.scan_directory(temp_dir, True, key_store=KeyStore()) == 0
        assert manager.scan_directory(temp_dir, False, key_store=KeyStore()) == 1

    def test_scan_directory_skips_non_ultrasound_modality(self, manager, temp_dir):
        assert self._scan(manager, temp_dir, {'ct.dcm': make_dataset(Modality='CT')}) == 0

    def test_scan_directory_skips_files_missing_required_fields(self, manager, temp_dir):
        assert self._scan(manager, temp_dir, {'nopid.dcm': make_dataset(PatientID=None)}) == 0

    def test_scan_directory_skips_non_dicom_extensions(self, manager, temp_dir):
        Path(os.path.join(temp_dir, 'notes.txt')).write_text('x')
        assert self._scan(manager, temp_dir, {'ok.dcm': make_dataset()}) == 1

    @pytest.mark.parametrize('sop_class', [
        US_IMAGE_STORAGE, US_MULTIFRAME_STORAGE, US_IMAGE_STORAGE_RETIRED, US_MULTIFRAME_STORAGE_RETIRED,
    ])
    def test_scan_directory_accepts_the_ultrasound_storage_classes(self, manager, temp_dir, sop_class):
        assert self._scan(manager, temp_dir, {'us.dcm': make_dataset(SOPClassUID=sop_class)}) == 1
        assert manager.skipped_unsupported_sop_class == 0

    def test_scan_directory_skips_unsupported_sop_classes_and_invents_none(self, manager, temp_dir, caplog):
        with caplog.at_level(logging.INFO):
            count = self._scan(manager, temp_dir, {
                'sc.dcm': make_dataset(SOPClassUID=SECONDARY_CAPTURE_STORAGE),
                'ok.dcm': make_dataset(SOPInstanceUID=SOURCE_UID_ROOT + '.8'),
            })

        assert count == 1
        assert manager.skipped_unsupported_sop_class == 1
        assert manager.dicom_df.iloc[0]['InstanceUID'] == SOURCE_UID_ROOT + '.8'
        assert any('SOPClassUID' in record.message for record in caplog.records)

    def test_scan_directory_collects_distinct_issuers(self, manager, temp_dir):
        self._scan(manager, temp_dir, {
            'a.dcm': make_dataset(IssuerOfPatientID='HOSP_A'),
            'b.dcm': make_dataset(IssuerOfPatientID='HOSP_B', SOPInstanceUID=SOURCE_UID_ROOT + '.8'),
            'c.dcm': make_dataset(SOPInstanceUID=SOURCE_UID_ROOT + '.9'),
        })
        assert manager.issuers_of_patient_id == {'HOSP_A', 'HOSP_B'}

    def test_scan_directory_reuses_store_values_on_a_second_scan(self, manager, temp_dir):
        store = KeyStore()
        files = {
            'a.dcm': make_dataset(),
            'b.dcm': make_dataset(SOPInstanceUID=SOURCE_UID_ROOT + '.8', PatientID='MRN2',
                                  StudyInstanceUID=SOURCE_UID_ROOT + '.18',
                                  SeriesInstanceUID=SOURCE_UID_ROOT + '.28'),
        }
        self._scan(manager, temp_dir, files, key_store=store)
        first = manager.dicom_df[KEYS_CSV_COLUMNS[2:]].copy()

        second_manager = DicomFileManager()
        second_manager.scan_directory(temp_dir, False, key_store=store)
        second = second_manager.dicom_df[KEYS_CSV_COLUMNS[2:]]

        assert first.equals(second)
        assert first['AnonPatientID'].nunique() == 2
        assert len(store.rows()) == 2

    def test_scan_directory_with_a_fresh_store_mints_new_values(self, manager, temp_dir):
        self._scan(manager, temp_dir, {'a.dcm': make_dataset()})
        first = manager.dicom_df.iloc[0]
        second_manager = DicomFileManager()
        second_manager.scan_directory(temp_dir, False, key_store=KeyStore())
        second = second_manager.dicom_df.iloc[0]

        for column in ('AnonPatientID', 'AnonStudyUID', 'AnonSeriesUID', 'AnonSOPInstanceUID',
                       'AnonFrameOfReferenceUID'):
            assert first[column] != second[column], column

    def test_scan_directory_shares_anon_frame_of_reference_within_a_study(self, manager, temp_dir):
        shared_for = SOURCE_UID_ROOT + '.44'
        self._scan(manager, temp_dir, {
            'a.dcm': make_dataset(FrameOfReferenceUID=shared_for),
            'b.dcm': make_dataset(FrameOfReferenceUID=shared_for, SeriesInstanceUID=SOURCE_UID_ROOT + '.22',
                                  SOPInstanceUID=SOURCE_UID_ROOT + '.8'),
            'c.dcm': make_dataset(FrameOfReferenceUID=shared_for, StudyInstanceUID=SOURCE_UID_ROOT + '.11',
                                  SeriesInstanceUID=SOURCE_UID_ROOT + '.23', SOPInstanceUID=SOURCE_UID_ROOT + '.9'),
        })
        df = manager.dicom_df.set_index('InstanceUID')
        same_study = {SOURCE_UID_ROOT + '.3', SOURCE_UID_ROOT + '.8'}
        anon_for_same_study = {df.loc[uid, 'AnonFrameOfReferenceUID'] for uid in same_study}
        assert len(anon_for_same_study) == 1
        assert df.loc[SOURCE_UID_ROOT + '.9', 'AnonFrameOfReferenceUID'] not in anon_for_same_study
        assert shared_for not in set(df['AnonFrameOfReferenceUID'])

    def test_scan_directory_keeps_coordinate_less_series_apart(self, manager, temp_dir):
        self._scan(manager, temp_dir, {
            'a.dcm': make_dataset(),
            'b.dcm': make_dataset(SeriesInstanceUID=SOURCE_UID_ROOT + '.22', SOPInstanceUID=SOURCE_UID_ROOT + '.8'),
            'c.dcm': make_dataset(SOPInstanceUID=SOURCE_UID_ROOT + '.9'),
        })
        df = manager.dicom_df.set_index('InstanceUID')
        assert (df['FrameOfReferenceUID'] == '').all()
        assert df.loc[SOURCE_UID_ROOT + '.3', 'AnonFrameOfReferenceUID'] == \
            df.loc[SOURCE_UID_ROOT + '.9', 'AnonFrameOfReferenceUID']
        assert df.loc[SOURCE_UID_ROOT + '.3', 'AnonFrameOfReferenceUID'] != \
            df.loc[SOURCE_UID_ROOT + '.8', 'AnonFrameOfReferenceUID']

    def test_scan_directory_raises_when_a_file_contradicts_the_store(self, manager, temp_dir):
        store = KeyStore()
        self._scan(manager, temp_dir, {'a.dcm': make_dataset()}, key_store=store)
        os.remove(os.path.join(temp_dir, 'a.dcm'))
        write_dataset(make_dataset(SeriesInstanceUID=SOURCE_UID_ROOT + '.22'), temp_dir, 'a.dcm')

        with pytest.raises(KeysFileError):
            DicomFileManager().scan_directory(temp_dir, False, key_store=store)

    def test_scan_directory_lists_a_duplicated_instance_once(self, manager, temp_dir, caplog):
        with caplog.at_level(logging.INFO):
            count = self._scan(manager, temp_dir, {'a.dcm': make_dataset(), 'copy/a.dcm': make_dataset()})
        assert count == 1
        assert any('already listed' in record.message for record in caplog.records)

    def test_scan_directory_skips_a_file_with_an_empty_physical_delta(self, manager, temp_dir, caplog):
        with caplog.at_level(logging.ERROR):
            count = self._scan(manager, temp_dir, {
                'bad.dcm': make_dataset(PhysicalDeltaX=None),
                'ok.dcm': make_dataset(SOPInstanceUID=SOURCE_UID_ROOT + '.8'),
            })
        assert count == 1
        assert [os.path.basename(p) for p in manager.dicom_df['InputPath']] == ['ok.dcm']
        assert any('bad.dcm' in record.message for record in caplog.records
                   if record.levelno == logging.ERROR)

    def test_scan_directory_lists_an_instance_whose_first_copy_was_unreadable(self, manager, temp_dir):
        count = self._scan(manager, temp_dir, {
            'a.dcm': make_dataset(PhysicalDeltaX=None),
            'copy/a.dcm': make_dataset(),
        })
        assert count == 1
        assert manager.dicom_df.iloc[0]['InputPath'] == os.path.join(temp_dir, 'copy', 'a.dcm')

    def test_save_keys_csv_writes_the_store(self, manager, temp_dir):
        store = KeyStore()
        self._scan(manager, temp_dir, {'siteA/a.dcm': make_dataset()}, key_store=store)
        keys_path = os.path.join(temp_dir, 'keys.csv')

        manager.save_keys_csv(keys_path)

        loaded = KeyStore.load(keys_path)
        assert loaded.rows() == store.rows()
        assert loaded.rows()[0]['InputPath'] == 'siteA/a.dcm'

    @patch('pydicom.dcmread')
    def test_extract_dicom_info_read_error(self, mock_dcmread, manager, temp_dir):
        mock_dcmread.side_effect = Exception("Cannot read DICOM file")
        test_file = os.path.join(temp_dir, "test.dcm")
        Path(test_file).touch()
        manager.key_store = KeyStore()

        assert manager._extract_dicom_info(test_file, temp_dir, False) is None

    def test_create_dataframe_sorts_and_fills_spacing(self, manager):
        dicom_data = [
            {'InputPath': 'b.dcm', 'ContentDate': '20240101', 'ContentTime': '120100',
             'StudyUID': 'S', 'PhysicalDeltaX': None, 'PhysicalDeltaY': None},
            {'InputPath': 'a.dcm', 'ContentDate': '20240101', 'ContentTime': '120000',
             'StudyUID': 'S', 'PhysicalDeltaX': 0.1, 'PhysicalDeltaY': 0.15},
        ]
        manager._create_dataframe(dicom_data)

        assert list(manager.dicom_df['InputPath']) == ['a.dcm', 'b.dcm']
        assert list(manager.dicom_df['PhysicalDeltaX']) == [0.1, 0.1]
        assert 'SeriesNumber' not in manager.dicom_df.columns
        assert manager.next_index == 0

    def test_create_dataframe_empty(self, manager):
        manager._create_dataframe([])
        assert manager.dicom_df is not None and manager.dicom_df.empty

    # ------------------------------------------------------------------
    # Progress: an instance is done only when its .dcm and .json both exist
    # ------------------------------------------------------------------

    @staticmethod
    def _touch_pair(directory, name):
        os.makedirs(directory, exist_ok=True)
        Path(os.path.join(directory, name)).touch()
        Path(os.path.join(directory, name[:-4] + '.json')).touch()

    def test_update_progress_from_output_no_dataframe(self, manager):
        assert manager.update_progress_from_output("output", True) is None

    def test_update_progress_from_output_all_processed(self, manager, temp_dir):
        manager.dicom_df = pd.DataFrame({'OutputPath': ['file1.dcm', 'file2.dcm']})
        for filename in ['file1.dcm', 'file2.dcm']:
            self._touch_pair(temp_dir, filename)
        assert manager.update_progress_from_output(temp_dir, True) is None

    def test_update_progress_from_output_partial_processed(self, manager, temp_dir):
        manager.dicom_df = pd.DataFrame({'OutputPath': ['file1.dcm', 'file2.dcm', 'file3.dcm']})
        self._touch_pair(temp_dir, 'file1.dcm')
        Path(os.path.join(temp_dir, 'file2.dcm')).touch()  # no .json: not done

        assert manager.update_progress_from_output(temp_dir, True) == 1
        assert manager.next_index == 1

    def test_update_progress_from_output_with_preserve_structure(self, manager, temp_dir):
        manager.dicom_df = pd.DataFrame({'OutputPath': ['patient1/file1.dcm', 'patient2/file2.dcm']})
        self._touch_pair(os.path.join(temp_dir, 'patient1'), 'file1.dcm')
        assert manager.update_progress_from_output(temp_dir, preserve_directory_structure=True) == 1

    def test_update_progress_from_output_with_flatten_structure(self, manager, temp_dir):
        manager.dicom_df = pd.DataFrame({'OutputPath': ['patient1/subdir/file1.dcm', 'patient2/subdir/file2.dcm']})
        self._touch_pair(temp_dir, 'file1.dcm')
        assert manager.update_progress_from_output(temp_dir, preserve_directory_structure=False) == 1

    def test_increment_dicom_index_basic(self, manager):
        manager.dicom_df = pd.DataFrame({'test': [1, 2, 3]})
        manager.next_index = 0
        assert manager.increment_dicom_index() is True
        assert manager.next_index == 1

    def test_increment_dicom_index_at_end(self, manager):
        manager.dicom_df = pd.DataFrame({'test': [1, 2]})
        manager.next_index = 1
        assert manager.increment_dicom_index() is False
        assert manager.next_index == 2

    def test_increment_dicom_index_with_continue_progress(self, manager, temp_dir):
        manager.dicom_df = pd.DataFrame({'OutputPath': ['file1.dcm', 'file2.dcm', 'file3.dcm', 'file4.dcm']})
        manager.next_index = 0
        self._touch_pair(temp_dir, 'file1.dcm')
        self._touch_pair(temp_dir, 'file2.dcm')

        assert manager.increment_dicom_index(temp_dir, continue_progress=True) is True
        assert manager.next_index == 2
        assert manager.increment_dicom_index(temp_dir, continue_progress=True) is True
        assert manager.next_index == 3
        assert manager.increment_dicom_index(temp_dir, continue_progress=True) is False
        assert manager.next_index == 4

    def test_increment_dicom_index_with_preserve_structure(self, manager, temp_dir):
        manager.dicom_df = pd.DataFrame({'OutputPath': ['dir1/file1.dcm', 'dir2/file2.dcm', 'dir3/file3.dcm']})
        manager.next_index = 0
        self._touch_pair(os.path.join(temp_dir, 'dir1'), 'file1.dcm')
        self._touch_pair(os.path.join(temp_dir, 'dir2'), 'file2.dcm')

        result = manager.increment_dicom_index(temp_dir, continue_progress=True, preserve_directory_structure=True)
        assert manager.next_index == 2
        assert result is True

    def test_increment_dicom_index_without_preserve_structure(self, manager, temp_dir):
        manager.dicom_df = pd.DataFrame({'OutputPath': ['dir1/file1.dcm', 'dir2/file2.dcm', 'dir3/file3.dcm']})
        self._touch_pair(os.path.join(temp_dir, 'dir1'), 'file1.dcm')
        self._touch_pair(os.path.join(temp_dir, 'dir2'), 'file2.dcm')

        result = manager.increment_dicom_index(temp_dir, continue_progress=True, preserve_directory_structure=False)
        assert manager.next_index == 1
        assert result is True

    # ------------------------------------------------------------------
    # save_anonymized_dicom
    # ------------------------------------------------------------------

    def test_save_anonymized_dicom_no_dataframe(self, manager, sample_image_array_single_frame, temp_dir):
        output_path = os.path.join(temp_dir, "output.dcm")
        assert manager.save_anonymized_dicom(sample_image_array_single_frame, output_path) is None
        assert not os.path.exists(output_path)

    def test_save_anonymized_dicom_invalid_index(self, manager, sample_image_array_multi_frame, temp_dir):
        manager.dicom_df = pd.DataFrame({'test': [1, 2, 3]})
        manager.current_index = 5
        output_path = os.path.join(temp_dir, "output.dcm")
        assert manager.save_anonymized_dicom(sample_image_array_multi_frame, output_path) is None
        assert not os.path.exists(output_path)

    def test_save_anonymized_dicom_none_image(self, manager_with_data, temp_dir):
        output_path = os.path.join(temp_dir, "output.dcm")
        assert manager_with_data.save_anonymized_dicom(None, output_path) is None
        assert not os.path.exists(output_path)

    def test_save_anonymized_dicom_success_multi_frame(self, manager_with_data, sample_image_array_multi_frame, temp_dir):
        output_path = os.path.join(temp_dir, "output.dcm")
        row = manager_with_data.dicom_df.iloc[0]

        returned = manager_with_data.save_anonymized_dicom(sample_image_array_multi_frame, output_path, 'PROJ_' + HEX32)

        saved_ds = pydicom.dcmread(output_path)
        assert isinstance(returned, pydicom.Dataset)
        assert saved_ds.NumberOfFrames == 5
        assert saved_ds.Rows == 10
        assert saved_ds.Columns == 15
        assert saved_ds.SamplesPerPixel == 3
        assert saved_ds.SOPInstanceUID == row['AnonSOPInstanceUID']
        assert saved_ds.SeriesInstanceUID == row['AnonSeriesUID']
        assert saved_ds.StudyInstanceUID == row['AnonStudyUID']
        assert saved_ds.FrameOfReferenceUID == row['AnonFrameOfReferenceUID']
        assert saved_ds.PatientID == row['AnonPatientID']
        assert saved_ds.PatientName == 'PROJ_' + HEX32
        assert saved_ds.PatientIdentityRemoved == 'YES'
        assert 'SeriesDescription' not in saved_ds
        assert saved_ds.file_meta.MediaStorageSOPInstanceUID == row['AnonSOPInstanceUID']

    def test_save_anonymized_dicom_success_single_frame(self, manager_with_data, sample_image_array_single_frame, temp_dir):
        output_path = os.path.join(temp_dir, "output.dcm")
        manager_with_data.save_anonymized_dicom(sample_image_array_single_frame, output_path)

        saved_ds = pydicom.dcmread(output_path)
        assert saved_ds.NumberOfFrames == 1
        assert saved_ds.Rows == 10
        assert saved_ds.Columns == 15
        assert saved_ds.SamplesPerPixel == 1
        assert saved_ds.PatientName == manager_with_data.dicom_df.iloc[0]['AnonPatientID']

    def test_save_anonymized_dicom_creates_nested_directories(self, manager_with_data, sample_image_array_single_frame, temp_dir):
        nested_output_path = os.path.join(temp_dir, "patient", "study", "series", "output.dcm")
        manager_with_data.save_anonymized_dicom(sample_image_array_single_frame, nested_output_path)
        assert os.path.exists(nested_output_path)

    @patch.object(DicomFileManager, '_create_base_dicom_dataset')
    @patch.object(DicomFileManager, '_copy_source_metadata')
    @patch.object(DicomFileManager, '_apply_anonymization')
    @patch.object(DicomFileManager, '_set_conformance_attributes')
    @patch.object(DicomFileManager, '_set_deidentification_attributes')
    @patch.object(DicomFileManager, '_set_compressed_pixel_data')
    @patch.object(DicomFileManager, '_create_and_save_dicom_file')
    def test_save_anonymized_dicom_integration(self, mock_save_file, mock_set_pixel_data, mock_set_deid,
                                              mock_set_conformance, mock_apply_anon, mock_copy_metadata,
                                              mock_create_base, manager_with_data,
                                              sample_image_array_multi_frame, temp_dir):
        mock_ds = Mock()
        mock_create_base.return_value = mock_ds
        output_path = os.path.join(temp_dir, "output.dcm")
        row = manager_with_data.dicom_df.iloc[0]

        manager_with_data.save_anonymized_dicom(sample_image_array_multi_frame, output_path, "Test^Patient")

        mock_create_base.assert_called_once()
        mock_copy_metadata.assert_called_once()
        mock_apply_anon.assert_called_once_with(
            mock_ds, row['DICOMDataset'], "Test^Patient", row['AnonPatientID'], int(row['DateShiftDays'])
        )
        mock_set_conformance.assert_called_once()
        mock_set_deid.assert_called_once_with(mock_ds)
        mock_set_pixel_data.assert_called_once()
        mock_save_file.assert_called_once_with(mock_ds, output_path)

    # ------------------------------------------------------------------
    # Base dataset and regions
    # ------------------------------------------------------------------

    def test_create_base_dicom_dataset_multi_frame(self, manager_with_data, sample_image_array_multi_frame):
        current_record = manager_with_data.dicom_df.iloc[0]
        ds = manager_with_data._create_base_dicom_dataset(sample_image_array_multi_frame, current_record)

        assert ds.NumberOfFrames == 5
        assert ds.Rows == 10
        assert ds.Columns == 15
        assert ds.SamplesPerPixel == 3
        assert ds.Modality == 'US'
        assert ds.PhotometricInterpretation == "YBR_FULL_422"

    def test_create_base_dicom_dataset_single_frame(self, manager_with_data, sample_image_array_single_frame):
        current_record = manager_with_data.dicom_df.iloc[0]
        ds = manager_with_data._create_base_dicom_dataset(sample_image_array_single_frame, current_record)

        assert ds.NumberOfFrames == 1
        assert ds.Rows == 10
        assert ds.Columns == 15
        assert ds.SamplesPerPixel == 1
        assert ds.PhotometricInterpretation == "MONOCHROME2"

    def test_copy_spacing_info_with_regions(self, manager_with_data):
        ds = pydicom.Dataset()
        manager_with_data._copy_spacing_info(ds, manager_with_data.dicom_df.iloc[0])

        assert len(ds.SequenceOfUltrasoundRegions) == 1
        assert ds.PixelSpacing == ['1.00000000000000', '1.50000000000000']

    def test_copy_spacing_info_without_regions(self, manager_with_data):
        ds = pydicom.Dataset()
        current_record = manager_with_data.dicom_df.iloc[0]
        current_record.DICOMDataset.SequenceOfUltrasoundRegions = []

        manager_with_data._copy_spacing_info(ds, current_record)

        assert 'SequenceOfUltrasoundRegions' not in ds
        assert hasattr(ds, 'PixelSpacing')

    def test_copy_spacing_info_rebuilds_region_items_from_the_standard_attributes(self, manager_with_data):
        current_record = manager_with_data.dicom_df.iloc[0]
        region = Dataset()
        region.RegionSpatialFormat = 1
        region.RegionDataType = 1
        region.RegionFlags = 0
        region.RegionLocationMinX0 = 10
        region.PhysicalUnitsXDirection = 3
        region.PhysicalDeltaX = 0.2
        region.PhysicalDeltaY = 0.3
        region.TransducerFrequency = 5000
        region.private_block(0x0011, 'PLANTED_PRIVATE', create=True).add_new(0x01, 'LO', 'planted')
        nested = Dataset()
        nested.ReferencedSOPInstanceUID = SOURCE_UID_ROOT + '.77'
        region.ReferencedImageSequence = Sequence([nested])
        current_record.DICOMDataset.SequenceOfUltrasoundRegions = Sequence([region, Dataset()])

        ds = pydicom.Dataset()
        manager_with_data._copy_spacing_info(ds, current_record)

        assert len(ds.SequenceOfUltrasoundRegions) == 2
        item = ds.SequenceOfUltrasoundRegions[0]
        assert item is not region
        for keyword in ('RegionSpatialFormat', 'RegionDataType', 'RegionFlags', 'RegionLocationMinX0',
                        'PhysicalUnitsXDirection', 'PhysicalDeltaX', 'PhysicalDeltaY', 'TransducerFrequency'):
            assert item[keyword].value == region[keyword].value, keyword
        assert 'ReferencedImageSequence' not in item
        assert not any(element.tag.is_private for element in item)
        assert len(ds.SequenceOfUltrasoundRegions[1]) == 0
        # The source item is left as it was.
        assert 'ReferencedImageSequence' in region
        assert any(element.tag.is_private for element in region)

    def test_region_allowlist_holds_only_region_calibration_keywords(self):
        for keyword in DicomFileManager.REGION_ALLOWLIST:
            tag = pydicom.datadict.tag_for_keyword(keyword)
            assert tag is not None, keyword
            assert (tag >> 16) == 0x0018 and 0x6012 <= (tag & 0xFFFF) <= 0x605A, keyword
        assert 'PhysicalDeltaX' in DicomFileManager.REGION_ALLOWLIST
        assert 'TableOfParameterValues' in DicomFileManager.REGION_ALLOWLIST

    # ------------------------------------------------------------------
    # Metadata copy
    # ------------------------------------------------------------------

    def test_copy_source_metadata(self, manager_with_data):
        ds = pydicom.Dataset()
        current_record = manager_with_data.dicom_df.iloc[0]

        manager_with_data._copy_source_metadata(ds, current_record.DICOMDataset, current_record)

        assert ds.BitsAllocated == self.BITS_ALLOCATED
        assert ds.BitsStored == self.BITS_STORED
        assert ds.SOPClassUID == self.SOP_CLASS_UID
        assert ds.SOPInstanceUID == current_record['AnonSOPInstanceUID']
        assert ds.SeriesInstanceUID == current_record['AnonSeriesUID']
        assert int(ds.SeriesNumber) == 4

    def test_copy_source_metadata_excludes_patient_identifiers(self, manager_with_data):
        ds = pydicom.Dataset()
        current_record = manager_with_data.dicom_df.iloc[0]
        source_ds = current_record.DICOMDataset
        source_ds.PatientAge = "025Y"
        source_ds.PatientSex = "M"

        manager_with_data._copy_source_metadata(ds, source_ds, current_record)

        assert 'PatientID' not in ds
        assert 'PatientName' not in ds
        assert 'PatientBirthDate' not in ds
        assert ds.PatientAge == "025Y"
        assert ds.PatientSex == "M"
        assert hasattr(ds, 'TransducerData')

    @pytest.mark.parametrize('keyword, value', [
        ('StationName', 'US_ROOM_3B'),
        ('StudyDescription', 'Patient Jane Doe referred by Dr Smith'),
        ('SeriesDescription', 'Dr Smith lung scan'),
        ('StudyDate', '20240101'),
        ('StudyTime', '101010'),
        ('InstitutionName', 'General Hospital'),
    ])
    def test_copy_source_metadata_does_not_copy_free_text_or_dates(self, manager_with_data, keyword, value):
        ds = pydicom.Dataset()
        current_record = manager_with_data.dicom_df.iloc[0]
        setattr(current_record.DICOMDataset, keyword, value)

        manager_with_data._copy_source_metadata(ds, current_record.DICOMDataset, current_record)

        assert keyword not in ds

    def test_copy_source_metadata_blanks_missing_series_number(self, manager_with_data):
        ds = pydicom.Dataset()
        current_record = manager_with_data.dicom_df.iloc[0]
        del current_record.DICOMDataset.SeriesNumber

        manager_with_data._copy_source_metadata(ds, current_record.DICOMDataset, current_record)

        assert 'SeriesNumber' in ds
        assert ds.SeriesNumber is None or ds.SeriesNumber == ''

    def _copy_with(self, manager, **source_values):
        ds = pydicom.Dataset()
        current_record = manager.dicom_df.iloc[0]
        source_ds = current_record.DICOMDataset
        for keyword, value in source_values.items():
            if value is None:
                if hasattr(source_ds, keyword):
                    delattr(source_ds, keyword)
            else:
                setattr(source_ds, keyword, value)
        manager._copy_source_metadata(ds, source_ds, current_record)
        return ds

    def test_copy_source_metadata_preserves_transducer_type(self, manager_with_data):
        assert self._copy_with(manager_with_data, TransducerType="LINEAR").TransducerType == "LINEAR"

    def test_copy_source_metadata_blanks_missing_transducer_type(self, manager_with_data):
        ds = self._copy_with(manager_with_data, TransducerType=None)
        assert ds.TransducerType == ""

    def test_copy_source_metadata_blanks_missing_transducer_data(self, manager_with_data):
        ds = self._copy_with(manager_with_data, TransducerData=None)
        assert ds.TransducerData == ""

    def test_copy_source_metadata_blanks_missing_manufacturer_model_name(self, manager_with_data):
        ds = self._copy_with(manager_with_data, ManufacturerModelName=None)
        assert ds.ManufacturerModelName == ""

    def test_copy_source_metadata_preserves_empty_transducer_data(self, manager_with_data):
        assert self._copy_with(manager_with_data, TransducerData="").TransducerData == ""

    @pytest.mark.parametrize('source, expected', [
        ("SC6-1s,JK9U41102597", "SC6-1s"),
        ("C1-5", "C1-5"),
        ("SP5-1s,CB3C", "SP5-1s"),
        ("SC6-1s,JK9", "SC6-1s"),
        ("\\\\", ""),
        (" SC6-1s ,02597", "SC6-1s"),
    ])
    def test_copy_source_metadata_reduces_transducer_data_to_the_model_segment(self, manager_with_data, source, expected):
        assert self._copy_with(manager_with_data, TransducerData=source).TransducerData == expected

    def test_copy_source_metadata_strips_transducer_data_backslash(self, manager_with_data):
        from pydicom.multival import MultiValue
        ds = self._copy_with(manager_with_data, TransducerData=MultiValue(str, ["S4-1U", "UNUSED", "UNUSED"]))
        assert ds.TransducerData == "S4-1U"

    def test_copy_source_metadata_preserves_manufacturer_and_model(self, manager_with_data):
        ds = self._copy_with(manager_with_data, Manufacturer="GE Healthcare", ManufacturerModelName="Vivid E95")
        assert ds.Manufacturer == "GE Healthcare"
        assert ds.ManufacturerModelName == "Vivid E95"

    # ------------------------------------------------------------------
    # UIDs
    # ------------------------------------------------------------------

    def test_copy_and_generate_uids_takes_values_from_the_record(self, manager_with_data):
        ds = pydicom.Dataset()
        current_record = manager_with_data.dicom_df.iloc[0]
        source_ds = current_record.DICOMDataset

        manager_with_data._copy_and_generate_uids(ds, source_ds, current_record)

        assert ds.SOPClassUID == source_ds.SOPClassUID
        assert ds.SOPInstanceUID == current_record['AnonSOPInstanceUID']
        assert ds.SeriesInstanceUID == current_record['AnonSeriesUID']
        assert ds.StudyInstanceUID == current_record['AnonStudyUID']
        assert ds.FrameOfReferenceUID == current_record['AnonFrameOfReferenceUID']
        for keyword in ('SOPInstanceUID', 'SeriesInstanceUID', 'StudyInstanceUID', 'FrameOfReferenceUID'):
            assert UID_2_25.match(ds[keyword].value), keyword
            assert ds[keyword].value != source_ds[keyword].value, keyword

    def test_copy_and_generate_uids_never_invents_a_sop_class(self, manager_with_data):
        ds = pydicom.Dataset()
        current_record = manager_with_data.dicom_df.iloc[0]
        source_ds = pydicom.Dataset()

        with pytest.raises(ValueError):
            manager_with_data._copy_and_generate_uids(ds, source_ds, current_record)
        assert 'SOPClassUID' not in ds

    # ------------------------------------------------------------------
    # Anonymization
    # ------------------------------------------------------------------

    def test_apply_anonymization_uses_provided_patient_info(self, manager_with_data):
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset

        self._anonymize(manager_with_data, ds, source_ds, name="PROJ_" + HEX32)

        assert ds.PatientName == "PROJ_" + HEX32
        assert ds.PatientID == HEX32
        assert ds.PatientBirthDate == ""
        assert ds.ReferringPhysicianName == ""
        assert ds.AccessionNumber == ""
        assert ds.StudyID == ""

    def test_apply_anonymization_without_name_uses_the_pseudonym(self, manager_with_data):
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset

        self._anonymize(manager_with_data, ds, source_ds, name="")

        assert ds.PatientName == HEX32
        assert ds.PatientID == HEX32

    def test_apply_anonymization_requires_a_patient_id(self, manager_with_data):
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset
        with pytest.raises(ValueError):
            manager_with_data._apply_anonymization(ds, source_ds, "", "", 10)

    def test_apply_anonymization_forces_empty_type2_elements(self, manager_with_data):
        ds = pydicom.Dataset()
        source_ds = pydicom.Dataset()
        source_ds.PatientName = "Test^Patient"
        source_ds.PatientID = "TEST123"
        source_ds.PatientBirthDate = "19900101"
        source_ds.ReferringPhysicianName = "Dr. Smith"
        source_ds.AccessionNumber = "ACC123"
        source_ds.StudyID = "STUDY-77"

        self._anonymize(manager_with_data, ds, source_ds)

        assert ds.PatientBirthDate == ""
        assert ds.ReferringPhysicianName == ""
        assert ds.AccessionNumber == ""
        assert ds.StudyID == ""
        assert ds.PatientName == HEX32

    # --- PatientAge --------------------------------------------------------

    @pytest.mark.parametrize('age, expected', [
        ("095Y", "090Y"),
        ("090Y", "090Y"),
        ("045Y", "045Y"),
        ("025M", "025M"),
    ])
    def test_apply_anonymization_caps_patient_age(self, manager_with_data, age, expected):
        ds = pydicom.Dataset()
        ds.PatientAge = age
        self._anonymize(manager_with_data, ds, manager_with_data.dicom_df.iloc[0].DICOMDataset)
        assert ds.PatientAge == expected

    def test_cap_patient_age_logs_error_for_invalid_format(self, manager, caplog):
        with caplog.at_level(logging.ERROR):
            result = manager._cap_patient_age("25YR")
        assert result == "25YR"
        assert any("25YR" in r.message for r in caplog.records)
        assert any("nnnX" in r.message for r in caplog.records)

    def test_cap_patient_age_logs_error_for_bad_unit(self, manager, caplog):
        with caplog.at_level(logging.ERROR):
            result = manager._cap_patient_age("045X")
        assert result == "045X"
        assert any("045X" in r.message for r in caplog.records)

    def test_cap_patient_age_logs_error_for_non_digit_prefix(self, manager, caplog):
        with caplog.at_level(logging.ERROR):
            result = manager._cap_patient_age("abcY")
        assert result == "abcY"
        assert any("abcY" in r.message for r in caplog.records)

    def test_cap_patient_age_silent_for_empty(self, manager, caplog):
        with caplog.at_level(logging.ERROR):
            assert manager._cap_patient_age("") == ""
        assert len(caplog.records) == 0

    def test_cap_patient_age_silent_for_valid_non_year(self, manager, caplog):
        with caplog.at_level(logging.ERROR):
            for value in ("025M", "014D", "003W"):
                assert manager._cap_patient_age(value) == value
        assert len(caplog.records) == 0

    def _age_source(self, birth=None, study=None):
        source_ds = pydicom.Dataset()
        source_ds.PatientID = "TEST123"
        if birth is not None:
            set_unchecked(source_ds, 'PatientBirthDate', 'DA', birth)
        if study is not None:
            source_ds.StudyDate = study
        return source_ds

    @pytest.mark.parametrize('birth, study, expected', [
        ("19800101", "20200101", "040Y"),
        ("20191215", "20200615", "006M"),
        ("20200115", "20200125", "010D"),
        ("19200101", "20200101", "090Y"),
        ("20210601", "20220601", "001Y"),
        ("20200215", "20210215", "001Y"),
        ("20200601", "20210531", "011M"),
        ("20190601", "20210601", "002Y"),
    ])
    def test_apply_anonymization_computes_age_from_birthdate(self, manager_with_data, birth, study, expected):
        ds = pydicom.Dataset()
        self._anonymize(manager_with_data, ds, self._age_source(birth, study))
        assert ds.PatientAge == expected
        assert ds.PatientBirthDate == ""

    def test_apply_anonymization_preserves_existing_patient_age_does_not_recompute(self, manager_with_data):
        ds = pydicom.Dataset()
        ds.PatientAge = "045Y"
        self._anonymize(manager_with_data, ds, self._age_source("19500101", "20200101"))
        assert ds.PatientAge == "045Y"

    @pytest.mark.parametrize('birth, study', [
        (None, "20200101"),
        ("19800101", None),
        ("20200101", "20190101"),
        ("BADDATE", "20200101"),
    ])
    def test_apply_anonymization_leaves_age_absent_and_warns(self, manager_with_data, caplog, birth, study):
        ds = pydicom.Dataset()
        with caplog.at_level(logging.WARNING):
            self._anonymize(manager_with_data, ds, self._age_source(birth, study))
        assert 'PatientAge' not in ds
        assert any("PatientAge" in r.message or "BirthDate" in r.message for r in caplog.records)

    def test_apply_anonymization_empty_birthdate_no_age_set_no_error(self, manager_with_data, caplog):
        ds = pydicom.Dataset()
        with caplog.at_level(logging.WARNING):
            self._anonymize(manager_with_data, ds, self._age_source("", "20200101"))
        assert 'PatientAge' not in ds
        assert all(r.levelno < logging.ERROR for r in caplog.records)

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------

    def _dated_source(self, **values):
        """Source dataset with date/time values set without pydicom's validation."""
        source_ds = pydicom.Dataset()
        for keyword, value in values.items():
            set_unchecked(source_ds, keyword, 'DA' if keyword.endswith('Date') else 'TM', value)
        return source_ds

    @pytest.mark.parametrize('shift', [1, 1095])
    def test_apply_date_shifting_moves_every_date_forward_by_the_offset(self, manager, shift):
        ds = pydicom.Dataset()
        source_ds = self._dated_source(StudyDate='20240115', StudyTime='101010', SeriesDate='20240116',
                                       SeriesTime='111111', ContentDate='20240117', ContentTime='121212')
        expected = lambda date: (datetime.strptime(date, '%Y%m%d') + timedelta(days=shift)).strftime('%Y%m%d')

        manager._apply_date_shifting(ds, source_ds, shift)

        assert ds.StudyDate == expected('20240115')
        assert ds.SeriesDate == expected('20240116')
        assert ds.ContentDate == expected('20240117')
        assert ds.StudyDate > '20240115'
        assert ds.StudyDate[:4] in ('2024', '2027')
        assert ds.StudyTime == '101010'
        assert ds.SeriesTime == '111111'
        assert ds.ContentTime == '121212'

    def test_apply_date_shifting_handles_a_leap_day_source(self, manager):
        ds = pydicom.Dataset()
        manager._apply_date_shifting(ds, self._dated_source(StudyDate='20240229'), 1)
        assert ds.StudyDate == '20240301'

    def test_apply_date_shifting_writes_empty_dates_when_the_source_has_none(self, manager):
        ds = pydicom.Dataset()
        manager._apply_date_shifting(ds, pydicom.Dataset(), 10)

        assert ds.StudyDate == '' and ds.StudyTime == ''
        assert ds.ContentDate == '' and ds.ContentTime == ''
        assert 'SeriesDate' not in ds and 'SeriesTime' not in ds

    def test_apply_date_shifting_writes_empty_dates_for_unparseable_values(self, manager):
        ds = pydicom.Dataset()
        source_ds = self._dated_source(StudyDate='20231399', SeriesDate='2024-01-01', ContentDate='invalid')

        manager._apply_date_shifting(ds, source_ds, 10)

        assert ds.StudyDate == ''
        assert ds.ContentDate == ''
        assert 'SeriesDate' not in ds

    def test_apply_date_shifting_keeps_series_time_only_when_present(self, manager):
        ds = pydicom.Dataset()
        manager._apply_date_shifting(ds, self._dated_source(SeriesDate='20240101', SeriesTime='090000'), 2)
        assert ds.SeriesDate == '20240103'
        assert ds.SeriesTime == '090000'

        ds = pydicom.Dataset()
        manager._apply_date_shifting(ds, self._dated_source(SeriesTime='090000'), 2)
        assert 'SeriesDate' not in ds
        assert ds.SeriesTime == '090000'

    @pytest.mark.parametrize('shift', [0, 1096, -1, 'abc', None, 10.0])
    def test_apply_date_shifting_rejects_offsets_outside_the_contract(self, manager, shift):
        with pytest.raises((ValueError, TypeError)):
            manager._apply_date_shifting(pydicom.Dataset(), self._dated_source(StudyDate='20240101'), shift)

    def test_apply_anonymization_uses_the_given_shift(self, manager_with_data):
        ds = pydicom.Dataset()
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset  # StudyDate 20240101

        self._anonymize(manager_with_data, ds, source_ds, shift=31)

        assert ds.StudyDate == '20240201'
        assert ds.StudyTime == '101010'

    # ------------------------------------------------------------------
    # De-identification attributes
    # ------------------------------------------------------------------

    def test_set_deidentification_attributes(self, manager):
        ds = pydicom.Dataset()
        manager._set_deidentification_attributes(ds)

        assert ds.PatientIdentityRemoved == 'YES'
        assert ds.LongitudinalTemporalInformationModified == 'MODIFIED'
        methods = list(ds.DeidentificationMethod)
        assert len(methods) >= 5
        assert all(0 < len(text) <= 64 for text in methods)
        assert methods[0].startswith('SlicerUltrasound AnonymizeUltrasound ')
        assert any('keys.csv' in text for text in methods)
        assert any('per-patient offset' in text for text in methods)
        assert any('mask' in text for text in methods)
        assert not any(re.search(r'\d', text) for text in methods[1:])
        assert 'DeidentificationMethodCodeSequence' not in ds
        assert 'BurnedInAnnotation' not in ds

    # ------------------------------------------------------------------
    # Conformance and pixel data
    # ------------------------------------------------------------------

    def test_set_conformance_attributes_multiframe(self, manager_with_data):
        ds = pydicom.Dataset()
        ds.NumberOfFrames = 5
        ds.SamplesPerPixel = 1
        source_ds = manager_with_data.dicom_df.iloc[0].DICOMDataset

        manager_with_data._set_conformance_attributes(ds, source_ds)

        assert hasattr(ds, 'Laterality')
        assert hasattr(ds, 'InstanceNumber')
        assert hasattr(ds, 'PatientOrientation')
        assert hasattr(ds, 'ImageType')
        assert hasattr(ds, 'FrameTime')
        assert hasattr(ds, 'FrameIncrementPointer')

    def test_set_conformance_attributes_color_image(self, manager_with_data):
        ds = pydicom.Dataset()
        ds.SamplesPerPixel = 3
        manager_with_data._set_conformance_attributes(ds, manager_with_data.dicom_df.iloc[0].DICOMDataset)
        assert hasattr(ds, 'PlanarConfiguration')

    def test_set_conformance_attributes_conditional_elements_only_when_missing(self, manager_with_data):
        ds = pydicom.Dataset()
        ds.Laterality = "L"
        ds.InstanceNumber = 5
        ds.SamplesPerPixel = 1

        manager_with_data._set_conformance_attributes(ds, pydicom.Dataset())

        assert ds.Laterality == "L"
        assert ds.InstanceNumber == 5
        assert ds.PatientOrientation == ''
        assert ds.ImageType == ['ORIGINAL', 'PRIMARY', 'IMAGE']

    def test_set_conformance_attributes_missing_elements_get_defaults(self, manager_with_data):
        ds = pydicom.Dataset()
        manager_with_data._set_conformance_attributes(ds, pydicom.Dataset())

        assert ds.Laterality == ''
        assert ds.InstanceNumber == 1
        assert ds.PatientOrientation == ''
        assert ds.ImageType == ['ORIGINAL', 'PRIMARY', 'IMAGE']

    def test_set_compressed_pixel_data(self, manager_with_data, sample_image_array_multi_frame):
        ds = pydicom.Dataset()
        manager_with_data._set_compressed_pixel_data(ds, sample_image_array_multi_frame)

        assert hasattr(ds, 'PixelData')
        assert ds.LossyImageCompression == '01'
        assert ds.LossyImageCompressionMethod == 'ISO_10918_1'
        assert ds['PixelData'].VR == 'OB'
        assert ds['PixelData'].is_undefined_length is True

    @pytest.mark.parametrize('shape, kwargs', [((10, 15), {}), ((10, 15, 1), {}), ((10, 15, 3), {'quality': 85})])
    def test_compress_frame_to_jpeg(self, manager_with_data, shape, kwargs):
        frame = np.random.randint(0, 255, shape, dtype=np.uint8)
        compressed = manager_with_data._compress_frame_to_jpeg(frame, **kwargs)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

    # ------------------------------------------------------------------
    # File writing
    # ------------------------------------------------------------------

    @patch('pydicom.dataset.FileDataset.save_as')
    def test_create_and_save_dicom_file(self, mock_save_as, manager_with_data, temp_dir):
        ds = pydicom.Dataset()
        ds.SOPClassUID = self.SOP_CLASS_UID
        ds.SOPInstanceUID = '2.25.5'
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._create_and_save_dicom_file(ds, output_path)

        mock_save_as.assert_called_once_with(output_path)

    def test_create_and_save_dicom_file_uses_a_fixed_implementation_identity(self, manager_with_data, temp_dir):
        ds = pydicom.Dataset()
        ds.SOPClassUID = self.SOP_CLASS_UID
        ds.SOPInstanceUID = '2.25.5'
        output_path = os.path.join(temp_dir, "test.dcm")

        manager_with_data._create_and_save_dicom_file(ds, output_path)
        meta = pydicom.dcmread(output_path, stop_before_pixels=True).file_meta

        assert UID_2_25.match(DicomFileManager.IMPLEMENTATION_CLASS_UID)
        assert meta.ImplementationClassUID == DicomFileManager.IMPLEMENTATION_CLASS_UID
        assert meta.ImplementationVersionName.startswith('SLICERUS_')
        assert len(meta.ImplementationVersionName) <= 16
        assert meta.MediaStorageSOPInstanceUID == '2.25.5'
        assert meta.TransferSyntaxUID == pydicom.uid.JPEGBaseline8Bit
        with open(output_path, 'rb') as handle:
            assert handle.read(128) == b'\0' * 128

    def test_create_and_save_dicom_file_creates_directories(self, manager_with_data, temp_dir):
        ds = pydicom.Dataset()
        ds.SOPClassUID = self.SOP_CLASS_UID
        ds.SOPInstanceUID = '2.25.5'
        nested_output_path = os.path.join(temp_dir, "deep", "nested", "path", "test.dcm")

        manager_with_data._create_and_save_dicom_file(ds, nested_output_path)

        assert os.path.exists(nested_output_path)


if __name__ == "__main__":
    pytest.main([__file__])
