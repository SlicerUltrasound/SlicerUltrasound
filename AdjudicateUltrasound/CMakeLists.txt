#-----------------------------------------------------------------------------
set(MODULE_NAME AdjudicateUltrasound)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_ICON_PNGS
  "${CMAKE_CURRENT_SOURCE_DIR}/../AnnotateUltrasound/Resources/Icons/AnnotateUltrasound.png"
  "${CMAKE_CURRENT_SOURCE_DIR}/../AnnotateUltrasound/Resources/Icons/blueAdd.png"
  "${CMAKE_CURRENT_SOURCE_DIR}/../AnnotateUltrasound/Resources/Icons/blueEye.png"
  "${CMAKE_CURRENT_SOURCE_DIR}/../AnnotateUltrasound/Resources/Icons/blueFastForward.png"
  "${CMAKE_CURRENT_SOURCE_DIR}/../AnnotateUltrasound/Resources/Icons/blueFillNext.png"
  "${CMAKE_CURRENT_SOURCE_DIR}/../AnnotateUltrasound/Resources/Icons/blueFillPrevious.png"
  "${CMAKE_CURRENT_SOURCE_DIR}/../AnnotateUltrasound/Resources/Icons/BlueFillTrash.png"
  "${CMAKE_CURRENT_SOURCE_DIR}/../AnnotateUltrasound/Resources/Icons/blueRemove.png"
  "${CMAKE_CURRENT_SOURCE_DIR}/../AnnotateUltrasound/Resources/Icons/blueSave.png"
)

set(MODULE_PYTHON_RESOURCES
  ${MODULE_ICON_PNGS}
  "${CMAKE_CURRENT_SOURCE_DIR}/../AnnotateUltrasound/Resources/UI/*.ui"
  "${CMAKE_CURRENT_SOURCE_DIR}/../AnnotateUltrasound/Resources/annotation_labels.csv"
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Set test properties for the module test
  set_tests_properties(${MODULE_NAME}ModuleTest PROPERTIES
    LABELS "unit;slicer;ultrasound"
    TIMEOUT 300
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/lib/Slicer-5.8/qt-scripted-modules"
  )

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
