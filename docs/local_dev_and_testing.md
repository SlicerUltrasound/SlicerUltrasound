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



## Logging (AnonymizeUltrasound)

The AnonymizeUltrasound module supports optional file logging for easier debugging:

- Enable in the module UI under Settings → "Enable file logging".
- Choose a log level (DEBUG/INFO/WARNING/ERROR). INFO is the default.
- Choose a log directory. If unset, it defaults to your Documents folder; a module default logs directory is also used internally when needed.
- Log files are plain text with names like `AnonymizeUltrasound_YYYYMMDD_HHMMSS.log`.
- Logs are written as events occur. Rotation is size‑based (10MB, 3 backups).
- When disabled, no file handler is attached (no overhead).

Tip: set level to DEBUG while reproducing an issue, then attach the resulting `.log` file when reporting.

