
add_test(NAME pytest 
    COMMAND py.test ${CMAKE_CURRENT_SOURCE_DIR}/tests.py
    #WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
         )

set_tests_properties(pytest
    PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_SOURCE_DIR}/src:${PYTHONPATH}")


