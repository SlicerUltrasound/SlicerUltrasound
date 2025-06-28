# Local Development and Testing

## Python Version
For simplicity, we'll use the python version that ships with the latest stable Slicer release
and will update it to track the version used in Slicer.

## Package Manager
Below is a optional guide to set up a local development environment using the UV package manager.
You can also use other tools like `pyenv` and `venv` if you prefer.

### Pin Python version for this project
Saves required python version to `.python-version` file

```sh
uv python pin 3.9.10
```

### Create new venv with specified Python

```sh
uv venv --python 3.9.10
```

### Activate and install dependencies

```sh
source .venv/bin/activate
uv pip install -r requirements-test.txt
```

## Testing
To run the tests:

1. Install test dependencies: `pip install -r requirements-test.txt`
2. Run tests: `pytest common/tests/test_dicom_file_manager.py -v`
3. Run with coverage: `pytest common/tests/test_dicom_file_manager.py --cov=common.DicomFileManager --cov-report=html`



