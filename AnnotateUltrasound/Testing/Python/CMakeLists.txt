#slicer_add_python_unittest(SCRIPT ${MODULE_NAME}ModuleTest.py)

# Module test (unit tests)
slicer_add_python_unittest(SCRIPT AnnotateUltrasoundModuleTest.py)

# GUI test (integration tests)
slicer_add_python_unittest(SCRIPT AnnotateUltrasoundGUITest.py)

# Set test properties
set_tests_properties(AnnotateUltrasoundModuleTest PROPERTIES
  LABELS "unit;slicer;ultrasound"
  TIMEOUT 300
)

set_tests_properties(AnnotateUltrasoundGUITest PROPERTIES
  LABELS "integration;gui;slicer;ultrasound"
  TIMEOUT 600
  ENVIRONMENT "DISPLAY=:0"
)
