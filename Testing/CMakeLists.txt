# Testing configuration for SlicerUltrasound extension
# This file configures the main testing infrastructure

if(BUILD_TESTING)
  # Add Python testing subdirectory
  add_subdirectory(Python)

  # Set test properties for all tests in this extension
  set_tests_properties(${EXTENSION_NAME} PROPERTIES
    LABELS "ultrasound;integration"
    TIMEOUT 300
  )
endif()