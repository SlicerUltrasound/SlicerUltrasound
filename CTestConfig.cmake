# CTest configuration for SlicerUltrasound extension
# This file configures CTest behavior for the project

# Set the name of the project
set(CTEST_PROJECT_NAME "SlicerUltrasound")

# Set the build name
set(CTEST_BUILD_NAME "SlicerUltrasound-${CMAKE_SYSTEM_NAME}")

# Set the dashboard model (Experimental, Nightly, Continuous)
set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "my.cdash.org")
set(CTEST_DROP_LOCATION "/submit.php?project=SlicerUltrasound")
set(CTEST_DROP_SITE_CDASH TRUE)

# Set the number of parallel jobs for testing
set(CTEST_PARALLEL_LEVEL 4)

# Set the timeout for tests (in seconds)
set(CTEST_TEST_TIMEOUT 300)

# Configure test output
set(CTEST_OUTPUT_ON_FAILURE TRUE)
set(CTEST_USE_LAUNCHERS TRUE)

# Set environment variables for tests
set(CTEST_ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/lib/Slicer-4.11/qt-scripted-modules")

# Configure test labels
set(CTEST_LABELS_FOR_SUBPROJECTS TRUE)