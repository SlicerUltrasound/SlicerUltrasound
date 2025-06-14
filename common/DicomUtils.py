import os
import logging
import hashlib
import json
from typing import Optional, Dict, Any, List
import pydicom
from pydicom.dataset import Dataset

class DicomUtils:
    """Utility class for DICOM operations"""

    @staticmethod
    def generate_filename_from_dicom(ds: Dataset, hash_patient_id: bool = True) -> tuple:
        """Generate filename from DICOM data"""
        patient_uid = ds.PatientID
        instance_uid = ds.SOPInstanceUID

        if not patient_uid or not instance_uid:
            logging.error("PatientID or SOPInstanceUID not found in DICOM header")
            return "", "", ""

        if hash_patient_id:
            hash_object = hashlib.sha256()
            hash_object.update(str(patient_uid).encode())
            patient_id = int(hash_object.hexdigest(), 16) % 10**10
        else:
            patient_id = patient_uid

        hash_object_instance = hashlib.sha256()
        hash_object_instance.update(str(instance_uid).encode())
        instance_id = int(hash_object_instance.hexdigest(), 16) % 10**8

        patient_id = str(patient_id).zfill(10)
        instance_id = str(instance_id).zfill(8)

        return f"{patient_id}_{instance_id}.dcm", patient_id, instance_id

    @staticmethod
    def dicom_header_to_dict(ds: Dataset, parent: Optional[str] = None) -> Dict[str, Any]:
        """Convert DICOM dataset to dictionary"""
        header_dict = {}
        for elem in ds:
            if elem.keyword and elem.keyword != "PixelData":
                if parent:
                    key = f"{parent}.{elem.keyword}"
                else:
                    key = elem.keyword
                header_dict[key] = str(elem.value)
        return header_dict

    @staticmethod
    def convert_to_json_compatible(obj: Any) -> Any:
        """Convert DICOM objects to JSON compatible format"""
        if isinstance(obj, pydicom.multival.MultiValue):
            return list(obj)
        if isinstance(obj, pydicom.valuerep.PersonName):
            return str(obj)
        if isinstance(obj, bytes):
            return obj.decode('latin-1')
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    @staticmethod
    def save_dicom_header(ds: Dataset, output_path: str, anonymize: bool = True) -> None:
        """Save DICOM header to JSON file"""
        header_dict = DicomUtils.dicom_header_to_dict(ds)
        
        if anonymize:
            # Anonymize sensitive fields
            if "Patient's Name" in header_dict:
                header_dict["Patient's Name"] = os.path.splitext(os.path.basename(output_path))[0]
            if "Patient's Birth Date" in header_dict:
                header_dict["Patient's Birth Date"] = header_dict["Patient's Birth Date"][:4] + "0101"

        with open(output_path, 'w') as outfile:
            json.dump(header_dict, outfile, default=DicomUtils.convert_to_json_compatible)

    @staticmethod
    def extract_spacing_info(ds: Dataset) -> tuple:
        """Extract spacing information from DICOM dataset"""
        try:
            pixel_spacing = ds.PixelSpacing
            if pixel_spacing:
                return float(pixel_spacing[0]), float(pixel_spacing[1])
        except:
            pass
        return 1.0, 1.0  # Default spacing if not found 