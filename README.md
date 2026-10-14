# SlicerUltrasound

Modules for ultrasound data processing, including anonymization, conversion between data formats and imaging modes, and annotations.

## Overview

This Slicer extension provides several modules for ultrasound data processing:

- **AnnotateUltrasound**: Interactive annotation of ultrasound images with pleura lines and B-lines
- **AnonymizeUltrasound**: Anonymization of ultrasound DICOM data
- **MmodeAnalysis**: M-mode analysis of ultrasound sequences
- **TimeSeriesAnnotation**: Time series annotation tools
- **SceneCleaner**: Scene cleanup utilities

## Installation

### Prerequisites

- 3D Slicer 4.11 or later
- Python 3.9+ (included with Slicer)

### Building from Source

1. Clone the repository:

   ```bash
   git clone https://github.com/SlicerUltrasound/SlicerUltrasound.git
   cd SlicerUltrasound
   ```

2. Build the extension:

   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Install in Slicer:
   - Copy the built extension to Slicer's extension directory
   - Or use Slicer's Extension Manager to install from the built package

## Testing

This project uses 3D Slicer's native testing infrastructure with CTest.

### Running Tests

#### Quick Start

```bash
# Build with testing enabled
make build-testing

# Run all tests
make test

# Run specific test patterns
make test-pattern PATTERN=AnnotateUltrasound
```

#### Manual CTest Commands

```bash
# Build with testing
mkdir build
cd build
cmake -DBUILD_TESTING=ON ..
make

# Run all tests
ctest -V

# Run tests with specific labels
ctest -L unit
ctest -L integration
ctest -L gui

# Run tests with coverage
ctest -V --output-on-failure
```

### Test Categories

- **Unit Tests** (`unit` label): Test individual module logic and functions
- **Integration Tests** (`integration` label): Test module interactions and workflows
- **GUI Tests** (`gui` label): Test user interface and interactions
- **Slicer Tests** (`slicer` label): Tests that run within Slicer's environment
- **Ultrasound Tests** (`ultrasound` label): Ultrasound-specific functionality tests

### Available Test Targets

| Target               | Description                               |
| -------------------- | ----------------------------------------- |
| `make test`          | Run all Slicer-native tests (recommended) |
| `make test-slicer`   | Run Slicer-native tests with CTest        |
| `make test-gui`      | Run GUI tests (requires display)          |
| `make test-pattern`  | Run tests matching a pattern              |
| `make test-cov`      | Run tests with coverage reporting         |
| `make build-testing` | Build with testing enabled                |

### Test Files

- `AnnotateUltrasound/Testing/Python/AnnotateUltrasoundLogicTest.py` - Unit tests for AnnotateUltrasound
- `AnnotateUltrasound/Testing/Python/AnnotateUltrasoundWidgetTest.py` - GUI tests for AnnotateUltrasound
- `AnonymizeUltrasound/Testing/Python/AnonymizeUltrasoundModuleTest.py` - Unit tests for AnonymizeUltrasound

### Test Configuration

The project includes:

- `CTestConfig.cmake` - CTest configuration
- `CMakeLists.txt` - Build and test configuration
- Module-specific test configurations in each module's `Testing/` directory

## Development

### Adding New Tests

1. Create test file in `ModuleName/Testing/Python/ModuleNameTest.py`
2. Inherit from `ScriptedLoadableModuleTest`
3. Add test methods with descriptive names
4. Update the module's `Testing/Python/CMakeLists.txt` to include the test
5. Add appropriate test labels and properties

### Test Best Practices

- Use descriptive test method names
- Test both success and failure cases
- Include edge case testing
- Use appropriate test labels for categorization
- Keep tests independent and isolated
- Clean up resources in `setUp()` and `tearDown()`

### Debugging Tests

```bash
# Run tests with verbose output
ctest -V

# Run specific test with debug output
cd build
./bin/AnnotateUltrasoundLogicTest

# Run GUI tests manually
/Applications/Slicer.app/Contents/MacOS/Slicer --python-script AnnotateUltrasound/Testing/Python/AnnotateUltrasoundWidgetTest.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

(MIT License)[https://github.com/SlicerUltrasound/SlicerUltrasound?tab=MIT-1-ov-file#readme]

## Support

For issues and questions:

- Create an issue on GitHub
- Check the documentation
- Review existing test cases for examples

## Anonymize Ultrasound

This module reads a folder of ultrasound cine loops in DICOM format and iterates through them. It allows masking the fan shaped or rectangular image area to erase non-ultrasound information from the images. It also allows associating labels to loops, so classification annotations can start during anonymization. The masked loops are exported as new DICOM files built from a fixed list of attributes; the mapping back to the source files is written to a separate `keys.csv`. The output is prepared for restricted research use; a separate review is required before sharing it.

### How to use Anonymize Ultrasound module

1. Select **Input folder** as the folder that contains all your DICOM files. They can be in subfolders.
1. Select **Output folder** as an empty folder, or the folder that holds the files exported earlier with the same keys directory.
1. Select the **Keys directory**: a folder that only the research coordinator can read. `keys.csv` is read from and written to this folder. It must not be inside the input or output folder.
1. Optionally type a **Project label** (1 to 16 letters, digits, `_` or `-`). It is put in front of the patient pseudonym in the exported PatientName. It is never filled in from a folder name.
1. Click **Import DICOM folder** and wait until the input folder is scanned. `keys.csv` is updated before the first file is loaded.
1. Click **Next** to load the next DICOM file in alphabetical order.
1. Click **Mask** and click on the corners of the ultrasound area. Corner points can be modified later by dragging.
1. Click **Export** to produce two files in the output folder: the de-identified DICOM `<uuid>.dcm` and the annotation file `<uuid>.json` with the mask parameters, the checked labels and the git revision of the module. An instance counts as exported only when both files exist.

### What the export contains

- The area outside the defined fan or rectangle region is erased on all frames. Text inside the region is not detected.
- The output dataset is built from an empty dataset. Only these attributes are carried over from the source: the image encoding (bits, samples, photometric interpretation), the ultrasound region calibration (standard attributes of each region item only), `PixelSpacing`, `Manufacturer`, `ManufacturerModelName`, `TransducerType`, the model segment of `TransducerData`, `PatientSex`, `PatientAge` (capped at `090Y`), `SeriesNumber` and the SOP class. Descriptions, station and institution names, operator and physician names, comments and private attributes are not copied.
- `PatientID` is a random 32-character pseudonym drawn the first time a source patient is seen and reused afterwards. `PatientName` is the pseudonym, with the project label in front when one is given.
- Study, Series, SOP Instance and Frame of Reference UIDs are new random UIDs in the `2.25` arc. One source series stays one exported series. The file name is the SOP Instance UID in UUID form, so file name and UID always match.
- Dates are moved forward by a whole number of days between 1 and 1095, drawn once per source patient and stored in `keys.csv`. Every study of a patient uses the same offset, so the intervals between studies are kept. Output dates therefore lie after the real dates and can lie in the future. Times are copied unchanged. A source date that is missing or not a valid `YYYYMMDD` value is written empty.
- Every exported file carries `PatientIdentityRemoved = YES` and `LongitudinalTemporalInformationModified = MODIFIED`.

### keys.csv

`keys.csv` contains the re-identification mapping and must be restricted to the research coordinator. The keys directory holds only `keys.csv` and `keys.csv.lock` (a zero-byte file that keeps two sessions from writing at the same time). The file has one row per source instance and these columns:

`InputPath, OutputPath, AnonFilename, PatientUID, AnonPatientID, StudyUID, AnonStudyUID, SeriesUID, AnonSeriesUID, InstanceUID, AnonSOPInstanceUID, FrameOfReferenceUID, AnonFrameOfReferenceUID, DateShiftDays`

- `InputPath` and `OutputPath` are relative to the input and output folders.
- `PatientUID`, `StudyUID`, `SeriesUID`, `InstanceUID` and `FrameOfReferenceUID` are the source values; the `Anon*` columns are the exported values; `DateShiftDays` is the patient's date offset.
- Import reads the file, checks that every mapping is complete and one-to-one, adds the newly scanned instances and writes the file back atomically. A file that does not pass the checks is left untouched and the Import is refused.
- Use one keys directory per patient-identifier authority. Import is refused when the files of one Import carry more than one `IssuerOfPatientID`; the check covers that Import only, not earlier Imports into the same keys directory.
- A row reserves the mapping of an instance; it does not prove that the instance was exported. The two files in the output folder do.
- Import is refused when the output folder holds DICOM files that `keys.csv` does not list.

Options under Settings collapsible button:

- **Preserve input directory structure**: keeps the subfolder structure of the input files in the output folder. File names are always replaced.
- **Use cached mask**: reuses the last mask defined for the same transducer model.
- **Automatically suggest mask**: lets the bundled model propose the corner points.
- **Skip single-frame input DICOM files**: use this for efficiency if you only need to export multi-frame DICOM files.
- **Continue progress**: skips input files whose DICOM and JSON pair already exists in the output folder.
- **Enable 3-Point Fan Mask**: defines a fan from one top point and two bottom points.
- **Annotation labels CSV file**: Select a CSV file that contains classification labels. These labels will populate the GUI with checkboxes. Checked labels will be saved in the exported annotation json files.

![2024-04-14_AnonymizeUltrasound](https://github.com/SlicerUltrasound/SlicerUltrasound/assets/2071850/52ff3ab6-94ea-41d5-88c5-596b66d6a659)

## M-mode Analysis

This module generates an M-mode image from a B-mode loop along a user-specified line (see [US types](https://en.wikipedia.org/wiki/Medical_ultrasound#Types)). The line can be arbitrary, not just ultrasound scan lines. Measurements on the M-mode image can be saved.

### How to use Mmode Analysis module

1. Select **Output folder** where exported files will be saved.
1. Click **Scanline** to add a line markup on the B-mode image.
1. Click **Generate M-mode image** to generate the M-mode ultrasound in the lower view.
1. Optionally click **Measurement line** to add a ruler on the M-mode image.
1. Click **Save M-mode image** to export both the current B-mode image and the M-mode image with their lines. And the measurement in a CSV file in the output folder.
1.

![2024-04-14_MmodeAnalysis](https://github.com/SlicerUltrasound/SlicerUltrasound/assets/2071850/16fd839c-0959-4e39-bdeb-789e945bae90)

## Time Series Annotation

This module facilitates the process of creating segmentations for time-series 2D images (ultrasound) and saving segmentations into an output sequence browser.

### How to use Time Series Annotation module

1. Prepare a Slicer scene with a recorded sequence browser with image frames and optionally tracking (transform) data, registered CT or MRI, ultrasound beam surface model, etc. Althernatively, use the 'Load sample data' button.
1. Make selection in the module's Inputs section.
1. Use the Segmentation section to create segmentations for frames.
1. Keyboard shortcuts facilitate the workflow: C saves current frame, S skips current frame without saving, D erases current segmentation, A shows/hides foreground volume.

![TimeSeriesAnnotation_2024-06-27.png](https://raw.githubusercontent.com/ungi/SlicerUltrasound/b4c3fdea3025d2891f849a9061a89ca8cbb30b99/Screenshots/TimeSeriesAnnotation_2024-06-27.png)

## Additional Documentation

Refer to the docs/ directory for more information about development, debugging, and testing.
