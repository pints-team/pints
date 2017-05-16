cmake_minimum_required(VERSION 2.8.3)
project(hobo)
FIND_PACKAGE(PythonLibs REQUIRED)
FIND_PACKAGE(Boost COMPONENTS python REQUIRED)

set(hobo_LIBRARIES ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set(hobo_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})


set(BUILD_SHARED_LIBS ON)
set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

set(source_files 
    src/cpp/electrochemistry/e_implicit_exponential_mesh.cpp
    )
set(header_files 
    src/cpp/electrochemistry/e_implicit_exponential_mesh.hpp
    src/cpp/electrochemistry/utilities.hpp
    )

set(header_dirs
    src/cpp/electrochemistry
    )

INCLUDE_DIRECTORIES(${header_dirs} ${hobo_INCLUDE_DIRS})

add_library(models ${source_files} ${header_files})
target_link_libraries(models ${hobo_LIBRARIES})

add_library(hobo_cpp src/cpp/python_wrapper.cpp)
target_link_libraries(hobo_cpp models ${hobo_LIBRARIES})

set_target_properties(hobo_cpp PROPERTIES PREFIX "" )

enable_testing()
add_subdirectory(tests)
