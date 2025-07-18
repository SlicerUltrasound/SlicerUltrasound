#-----------------------------------------------------------------------------
set(MODULE_NAME AnnotateUltrasound)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_ICON_PNGS
  Resources/Icons/AnnotateUltrasound.png
  Resources/Icons/blueAdd.png
  Resources/Icons/blueEye.png
  Resources/Icons/blueFastForward.png
  Resources/Icons/blueFillNext.png
  Resources/Icons/blueFillPrevious.png
  Resources/Icons/BlueFillTrash.png
  Resources/Icons/blueRemove.png
  Resources/Icons/blueSave.png
)

set(MODULE_PYTHON_RESOURCES
  ${MODULE_ICON_PNGS}
  Resources/UI/${MODULE_NAME}.ui
  Resources/annotation_labels.csv
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )
