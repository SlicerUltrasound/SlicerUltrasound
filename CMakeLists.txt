cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(Ultrasound)
include(CTest)
#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/SlicerUltrasound/SlicerUltrasound")
set(EXTENSION_CATEGORY "Ultrasound")
set(EXTENSION_CONTRIBUTORS "Tamas Ungi (Queen's University), Tina Kapur (Brigham and Women's Hospital)")
set(EXTENSION_DESCRIPTION "Modules for ultrasound data processing, including anonymization, conversion between data formats and imaging modes, and annotations.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/SlicerUltrasound/SlicerUltrasound/main/Ultrasound.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/SlicerUltrasound/SlicerUltrasound/main/Screenshots/2024-04-14_AnonymizeUltrasound.png")
set(EXTENSION_DEPENDS SlicerIGT) # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
set(Slicer_USE_ADDITIONAL_LAUNCHER_SETTINGS OFF)
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Testing configuration
if(BUILD_TESTING)
  # Enable CTest
  enable_testing()

  # Set test timeout (5 minutes)
  set(DART_TESTING_TIMEOUT 300)

  # Configure test labels
  set(TEST_LABELS
    "unit"
    "integration"
    "gui"
    "slicer"
    "ultrasound"
  )

  # Add test labels to all tests
  foreach(label ${TEST_LABELS})
    add_test(NAME ${label} COMMAND ${CMAKE_COMMAND} -E echo "Test label: ${label}")
    set_tests_properties(${label} PROPERTIES LABELS ${label})
  endforeach()

endif()

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(AdjudicateUltrasound)
add_subdirectory(AnonymizeUltrasound)
add_subdirectory(AnnotateUltrasound)
add_subdirectory(MmodeAnalysis)
add_subdirectory(TimeSeriesAnnotation)
# add_subdirectory(SceneCleaner)
## NEXT_MODULE

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
