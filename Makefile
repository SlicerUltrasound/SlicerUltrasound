.PHONY: test test-slicer test-gui install-deps clean find-slicer-python

# Default target
all: test

# Find Slicer's Python executable
find-slicer-python:
	@echo "Finding Slicer's Python executable..."
	@if command -v Slicer >/dev/null 2>&1; then \
		echo "Slicer found in PATH"; \
	elif [ -f "/Applications/Slicer.app/Contents/MacOS/Slicer" ]; then \
		echo "Slicer found at /Applications/Slicer.app/Contents/MacOS/Slicer"; \
	elif [ -f "/usr/local/bin/Slicer" ]; then \
		echo "Slicer found at /usr/local/bin/Slicer"; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/bin/PythonSlicer" ]; then \
		echo "Slicer found at $$SLICER_HOME/bin/PythonSlicer"; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/bin/Slicer" ]; then \
		echo "Slicer found at $$SLICER_HOME/bin/Slicer"; \
	else \
		echo "ERROR: Slicer not found. Please install Slicer or add it to your PATH."; \
		exit 1; \
	fi

# Run GUI tests locally (requires display and user interaction simulation)
test-gui: find-slicer-python
	@echo "Running GUI tests locally (requires display)..."
	@SLICER_EXE=""; \
	if [ -f "/Applications/Slicer.app/Contents/MacOS/Slicer" ]; then \
		SLICER_EXE="/Applications/Slicer.app/Contents/MacOS/Slicer"; \
	elif [ -f "/usr/local/bin/Slicer" ]; then \
		SLICER_EXE="/usr/local/bin/Slicer"; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/bin/Slicer" ]; then \
		SLICER_EXE="$$SLICER_HOME/bin/Slicer"; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/Slicer" ]; then \
		SLICER_EXE="$$SLICER_HOME/Slicer"; \
	else \
		echo "❌ Slicer not found. Please install Slicer first."; \
		exit 1; \
	fi; \
	echo "Using Slicer: $$SLICER_EXE"; \
	echo "Running AnnotateUltrasound GUI test..."; \
	"$$SLICER_EXE" --python-script AnnotateUltrasound/Testing/Python/AnnotateUltrasoundGUITest.py; \
	echo "Running DICOM loading test..."; \
	"$$SLICER_EXE" --python-script AnnotateUltrasound/Testing/Python/test_dicom_loading.py

# Run tests for CI (skips GUI tests)
test-ci: test-py-slicer test-slicer-modules
	@echo "CI tests completed (GUI tests skipped)"

# Run DICOM loading tests (requires display and real DICOM data)
test-dicom: find-slicer-python
	@echo "Running DICOM loading tests (requires display)..."
	@if [ -f "/Applications/Slicer.app/Contents/MacOS/Slicer" ]; then \
		/Applications/Slicer.app/Contents/MacOS/Slicer --python-script AnnotateUltrasound/Testing/Python/test_dicom_loading.py; \
	elif [ -f "/usr/local/bin/Slicer" ]; then \
		/usr/local/bin/Slicer --python-script AnnotateUltrasound/Testing/Python/test_dicom_loading.py; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/bin/Slicer" ]; then \
		$$SLICER_HOME/bin/Slicer --python-script AnnotateUltrasound/Testing/Python/test_dicom_loading.py; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/Slicer" ]; then \
		$$SLICER_HOME/Slicer --python-script AnnotateUltrasound/Testing/Python/test_dicom_loading.py; \
	else \
		echo "❌ Slicer not found. Please install Slicer first."; \
		echo "Checked paths:"; \
		echo "  /Applications/Slicer.app/Contents/MacOS/Slicer"; \
		echo "  /usr/local/bin/Slicer"; \
		echo "  $$SLICER_HOME/bin/Slicer"; \
		echo "  $$SLICER_HOME/Slicer"; \
		exit 1; \
	fi

# Build and install the module into Slicer
install-module: find-slicer-python
	@echo "Building and installing SlicerUltrasound module..."
	@echo "Current directory: $$(pwd)"
	@echo "Checking for install_module.py:"
	@ls -la Testing/Python/install_module.py || echo "install_module.py not found in Testing/Python directory"
	@if [ -f "/Applications/Slicer.app/Contents/bin/PythonSlicer" ]; then \
		/Applications/Slicer.app/Contents/bin/PythonSlicer $$(pwd)/Testing/Python/install_module.py; \
	elif [ -f "/usr/local/bin/Slicer" ]; then \
		/usr/local/bin/Slicer --python-script $$(pwd)/Testing/Python/install_module.py; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/bin/PythonSlicer" ]; then \
		$$SLICER_HOME/bin/PythonSlicer $$(pwd)/Testing/Python/install_module.py; \
	else \
		echo "❌ Slicer not found. Please install Slicer first."; \
		exit 1; \
	fi

# Run Slicer-dependent tests (requires Slicer modules to be loaded)
test-slicer-modules: find-slicer-python
	@echo "Running pytest-style tests in Slicer Python environment..."
	@if [ -f "/Applications/Slicer.app/Contents/bin/PythonSlicer" ]; then \
		/Applications/Slicer.app/Contents/bin/PythonSlicer Testing/Python/run_slicer_tests.py --install-deps; \
	elif [ -f "/usr/local/bin/Slicer" ]; then \
		/usr/local/bin/Slicer --python-script Testing/Python/run_slicer_tests.py --install-deps; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/bin/PythonSlicer" ]; then \
		$$SLICER_HOME/bin/PythonSlicer Testing/Python/run_slicer_tests.py --install-deps; \
	else \
		echo "❌ Slicer not found. Please install Slicer first."; \
		exit 1; \
	fi

# Run all pure Python/pytest tests in any submodule's tests/ directory using Slicer's Python
test-py-slicer:
	@echo "Running pure Python tests in submodule tests/ directories using Slicer's Python..."
	@if [ -f "/Applications/Slicer.app/Contents/bin/PythonSlicer" ]; then \
		find . -type d -name tests -not -path "./.venv/*" -not -path "./build/*" -not -path "./__pycache__/*" | \
		xargs -I {} /Applications/Slicer.app/Contents/bin/PythonSlicer -m pytest {}; \
	else \
		echo "Slicer Python not found!"; \
		exit 1; \
	fi

# Run tests with specific pattern
test-pattern: build-testing
	@echo "Usage: make test-pattern PATTERN=AnnotateUltrasound"
	@if [ -z "$(PATTERN)" ]; then echo "Please specify PATTERN parameter"; exit 1; fi
	cd build && ctest -V -R $(PATTERN)

# Run pytest tests with coverage (Python-only)
test-coverage: find-slicer-python
	@echo "Running pytest tests with coverage..."
	@if [ -f "/Applications/Slicer.app/Contents/bin/PythonSlicer" ]; then \
		/Applications/Slicer.app/Contents/bin/PythonSlicer Testing/Python/run_slicer_tests.py --install-deps; \
	elif [ -f "/usr/local/bin/Slicer" ]; then \
		/usr/local/bin/Slicer --python-script Testing/Python/run_slicer_tests.py --install-deps; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/bin/PythonSlicer" ]; then \
		$$SLICER_HOME/bin/PythonSlicer Testing/Python/run_slicer_tests.py --install-deps; \
	else \
		echo "❌ Slicer not found. Please install Slicer first."; \
		exit 1; \
	fi

# Run all tests using Slicer-native python (Slicer-native)
test: test-py-slicer

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf build/

# Help target
help:
	@echo "Available targets:"
	@echo "  find-slicer-python - Check if Slicer is available"
	@echo "  install-module    - Build and install module into Slicer"
	@echo "  test-py-slicer    - Run pure Python tests in submodule tests/ directories using Slicer's Python (does not require Slicer modules to be loaded)"
	@echo "  test-slicer-modules - Run Slicer-dependent tests in Testing/Python/ and submodule Testing/Python/ directories (requires Slicer modules to be loaded)"
	@echo "  test-gui          - Run GUI tests (requires display, simulates user interactions)"
	@echo "  test-dicom        - Run DICOM loading tests (requires display, uses real DICOM data)"
	@echo "  test              - Run pytest tests (alias for test-py-slicer)"
	@echo "  test-pattern      - Run tests matching pattern (e.g., make test-pattern PATTERN=AnnotateUltrasound)"
	@echo "  test-coverage     - Run pytest tests with coverage (Python-only)"
	@echo "  clean             - Clean up generated files"
	@echo "  help              - Show this help message"
	@echo ""
	@echo "Note:"
	@echo "  test-py-slicer runs pure Python tests in all submodule tests/ directories using Slicer's Python."
	@echo "  test-slicer-modules runs Slicer-dependent tests in Testing/Python/ and submodule Testing/Python/ directories."
