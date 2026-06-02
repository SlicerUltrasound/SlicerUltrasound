#-----------------------------------------------------------------------------
set(MODULE_NAME AnonymizeUltrasound)

#-----------------------------------------------------------------------------
# Bake git provenance (SHA + dirty flag) into _version.py so installed
# extensions — which ship without a .git directory — can still report the
# commit they were built from. Consumed by common/version_info.py.
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE MODULE_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _git_sha_result
    ERROR_QUIET
    )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} status --porcelain
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE _git_status
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _git_status_result
    ERROR_QUIET
    )
  if(NOT _git_sha_result EQUAL 0)
    set(MODULE_GIT_SHA "unknown")
  endif()
  if(_git_status STREQUAL "")
    set(MODULE_GIT_DIRTY "False")
  else()
    set(MODULE_GIT_DIRTY "True")
  endif()
else()
  set(MODULE_GIT_SHA "unknown")
  set(MODULE_GIT_DIRTY "False")
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/_version.py
  @ONLY
  )

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${CMAKE_CURRENT_BINARY_DIR}/_version.py
  common/__init__.py
  common/dicom_file_manager.py
  common/dicom_processor.py
  common/evaluation.py
  common/inference.py
  common/logging.py
  common/masking.py
  common/overview_generator.py
  common/progress_reporter.py
  common/uid_remap.py
  common/version_info.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
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
  add_subdirectory(Testing/Python)
endif()
