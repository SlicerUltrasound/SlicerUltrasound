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

- `AnnotateUltrasound/Testing/Python/AnnotateUltrasoundModuleTest.py` - Unit tests for AnnotateUltrasound
- `AnnotateUltrasound/Testing/Python/AnnotateUltrasoundGUITest.py` - GUI tests for AnnotateUltrasound
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
./bin/AnnotateUltrasoundModuleTest

# Run GUI tests manually
/Applications/Slicer.app/Contents/MacOS/Slicer --python-script AnnotateUltrasound/Testing/Python/AnnotateUltrasoundGUITest.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:

- Create an issue on GitHub
- Check the documentation
- Review existing test cases for examples

## Anonymize Ultrasound

This module reads a folder of ultrasound cine loops in DICOM format and iterates through them. It allows masking the fan shaped or rectangular image area to erase non-ultrasound information from the images. It also allows associating labels to loops, so classification annotations can start during anonymization. The masked loops are exported in DICOM format. The exported DICOM files only contain a specific set of DICOM tags that do not contain any personal information about the patient.

### How to use Anonymize Ultrasound module

1. Select **Input folder** as the folder that contains all your DICOM files. They can be in subfolders.
1. Select **Output folder** as an empty folder, or which contains previously exported files using this module.
1. Click **Read DICOM folder** and wait until the input and output folders are scanned to compile a list of files to be anonymized.
1. Click **Load next** to load the next DICOM file in alphabetical order.
1. Click **Define mask** and click on the four corners of the ultrasound area. Corner points can be modified later by dragging.
1. Click **Export** to produce three files in the output folder: the anonymized DICOM, an annotation file with the corner points, and a json file with all the original DICOM tags that went through light anonymization.

### Anonymization steps

- The area outside the defined fan or rectangle region is erased on all frames.
- A PatientUID is generated as a 10-digit hash of the input DICOM Patient ID, and a FileUID is generated as an 8-digit hash of the DICOM SOP Intance UID.
- The exported output file will be named **PatientUID_FileUID.dcm**
- When **Hash Patient ID** is selected, the patient name is replaced the PatientUID and the Series Description is replaced by PatientUID_FileUID.
- New SOP Series UID is generated for every exported file (cine loop) so all DICOM browsers can load them one-by-one. (Most DICOM browsers load all instances of a series at a time, but some ultrasound machines export all cine loops of an exam under a single series UID.)

Options under Settings collapsible button:

- **Skip single-frame input files**: use this for efficiency if you only need to export multi-frame DICOM files.
- **Skip input if output already exists**: prevents repeated loading of files whose output already exists.
- **Keep input folders and filenames in output folder**: keeps the subfolder structure and file names of input files.
- **Convert color to grayscale**: use this if you want to eliminate colors in the exported images.
- **JPEG compression in DICOM**: use this to apply compression in exported files without visual information loss.
- **Labels file**: Select a CSV file that contains classification labels. These labels will populate the GUI with checkboxes. Checked labels will be saved in the exported annotation json files.

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
