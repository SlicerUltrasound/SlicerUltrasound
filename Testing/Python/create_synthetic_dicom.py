#!/usr/bin/env python3
"""
Create synthetic DICOM files for testing.
These files will match the prefixes of existing JSON annotation files.
"""

import os
import pydicom
import numpy as np
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
import datetime
from pydicom.sequence import Sequence as DicomSequence

def create_synthetic_dicom(base_filename, output_dir="test_data"):
    """
    Create a synthetic multi-frame DICOM file with ultrasound-like data.

    Args:
        base_filename: Base filename without extension (e.g., "0561119268_08109698")
        output_dir: Directory to save the DICOM file
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create file metadata
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.3.1'  # US Multi-frame Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()

    # Create dataset
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # File info
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # Patient info
    ds.PatientName = "Test^Patient"
    ds.PatientID = "TEST001"
    ds.PatientBirthDate = "19800101"

    # Study info
    ds.StudyInstanceUID = generate_uid()
    ds.StudyDate = datetime.date.today().strftime('%Y%m%d')
    ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
    ds.StudyDescription = "Synthetic Ultrasound Test"

    # Series info
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = "1"
    ds.SeriesDescription = "Synthetic US Series"
    ds.Modality = "US"

    # Image info
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.3.1'  # US Multi-frame Image Storage
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16

    # Image dimensions (small for testing)
    ds.Rows = 256
    ds.Columns = 256
    num_frames = 10
    ds.NumberOfFrames = str(num_frames)

    # Spacing and orientation
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 1.0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, 0]

    # SharedFunctionalGroupsSequence (required for multi-frame)
    shared_fg = Dataset()
    pixel_measures = Dataset()
    pixel_measures.PixelSpacing = ds.PixelSpacing
    pixel_measures.SliceThickness = ds.SliceThickness
    pixel_measures.SpacingBetweenSlices = 1.0
    shared_fg.PixelMeasuresSequence = DicomSequence([pixel_measures])
    plane_orientation = Dataset()
    plane_orientation.ImageOrientationPatient = ds.ImageOrientationPatient
    shared_fg.PlaneOrientationSequence = DicomSequence([plane_orientation])
    ds.SharedFunctionalGroupsSequence = DicomSequence([shared_fg])

    # PerFrameFunctionalGroupsSequence (one for each frame)
    per_frame_seq = []
    for i in range(num_frames):
        fg = Dataset()
        plane_position = Dataset()
        # Slightly shift the image position for each frame
        plane_position.ImagePositionPatient = [0, 0, float(i)]
        fg.PlanePositionSequence = DicomSequence([plane_position])
        per_frame_seq.append(fg)
    ds.PerFrameFunctionalGroupsSequence = DicomSequence(per_frame_seq)

    # Create synthetic multi-frame image data
    frames = []
    for i in range(num_frames):
        image_data = np.zeros((256, 256), dtype=np.uint16)
        np.random.seed(42 + i)
        noise = np.random.normal(1000, 200, (256, 256)).astype(np.uint16)
        y, x = np.ogrid[:256, :256]
        # Move the circle center for each frame
        center_y, center_x = 128, 128 + i*5
        radius = 50
        circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        image_data[circle] += 2000
        for j in range(50, 200, 30):
            image_data[j:j+2, :] += 1500
        image_data += noise
        image_data = np.clip(image_data, 0, 65535).astype(np.uint16)
        frames.append(image_data)
    all_frames = np.stack(frames, axis=0)
    ds.PixelData = all_frames.tobytes()

    # Save the file
    output_path = os.path.join(output_dir, f"{base_filename}.dcm")
    ds.save_as(output_path, write_like_original=False)

    print(f"Created synthetic multi-frame DICOM: {output_path}")
    return output_path

def main():
    """Create synthetic DICOM files for all our test cases."""

    # Base filenames from our existing JSON files
    base_filenames = [
        "0561119268_08109698",
        "0561119268_32202817"
    ]

    print("Creating synthetic DICOM files for testing...")

    for base_filename in base_filenames:
        create_synthetic_dicom(base_filename)

    print("\n✅ All synthetic DICOM files created!")
    print("These files match the prefixes of existing JSON annotation files.")
    print("You can now run the tests with: make test-dicom")

if __name__ == "__main__":
    main()