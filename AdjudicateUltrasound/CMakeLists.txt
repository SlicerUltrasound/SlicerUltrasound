#-----------------------------------------------------------------------------
set(MODULE_NAME AdjudicateUltrasound)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${CMAKE_CURRENT_SOURCE_DIR}/../AnnotateUltrasound/AnnotateUltrasound.py
  )

set(MODULE_ICON_PNGS
  Resources/Icons/AdjudicateUltrasound.png
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
  "${CMAKE_CURRENT_SOURCE_DIR}/../AnnotateUltrasound/Resources/UI/AnnotateUltrasound.ui"
  "${CMAKE_CURRENT_SOURCE_DIR}/../AnnotateUltrasound/Resources/annotation_labels.csv"
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )
