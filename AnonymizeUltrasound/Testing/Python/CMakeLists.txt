# Module test for AnonymizeUltrasound
slicer_add_python_unittest(SCRIPT ${MODULE_NAME}ModuleTest.py)

# Set test properties
set_tests_properties(${MODULE_NAME}ModuleTest PROPERTIES
  LABELS "unit;slicer;ultrasound"
  TIMEOUT 300
)
