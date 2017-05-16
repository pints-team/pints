
add_test(NAME pytest 
    COMMAND py.test ${CMAKE_CURRENT_SOURCE_DIR}/tests.py
    #WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
         )

set_tests_properties(pytest
    PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_SOURCE_DIR}/src:${PYTHONPATH}")


set(new_python_path ${CMAKE_BINARY_DIR}:${CMAKE_SOURCE_DIR}/src:$ENV{PYTHONPATH})
add_custom_target(run_tests 
    PYTHONPATH=${new_python_path} py.test ${CMAKE_CURRENT_SOURCE_DIR}/tests.py
    COMMENT "Running python tests using pytest..."
    VERBATIM
    )

