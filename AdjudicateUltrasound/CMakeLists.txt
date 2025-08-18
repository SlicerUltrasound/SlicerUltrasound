#-----------------------------------------------------------------------------
set(MODULE_NAME AdjudicateUltrasound)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_ICON_PNGS
  Resources/Icons/AdjudicateUltrasound.png
  )

  set(MODULE_PYTHON_RESOURCES
  ${MODULE_ICON_PNGS}
  )

  #-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )
