cmake_minimum_required(VERSION 2.8.3)
project(pints)
set(CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake"
                        ${CMAKE_MODULE_PATH})

set(Python_ADDITIONAL_VERSIONS 2.7 2.6 2.5 2.4 2.3 2.2 2.1 2.0)
FIND_PACKAGE(PythonLibs REQUIRED)
FIND_PACKAGE(Boost COMPONENTS python REQUIRED)
find_package(Eigen3 REQUIRED)

set(pints_LIBRARIES ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set(pints_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

set(BUILD_SHARED_LIBS ON)
set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )
list(APPEND CMAKE_CXX_FLAGS "-Wno-deprecated -std=c++11")

set(source_files 
    src/electrochemistry/e_implicit_exponential_mesh.cpp
    src/electrochemistry/seq_electron_transfer3_explicit.cpp
    )
set(header_files 
    src/electrochemistry/e_implicit_exponential_mesh.hpp
    src/electrochemistry/seq_electron_transfer3_explicit.hpp
    src/electrochemistry/utilities.hpp
    )

set(header_dirs
    src/electrochemistry
    )

INCLUDE_DIRECTORIES(${header_dirs} ${pints_INCLUDE_DIRS})

add_library(models ${source_files} ${header_files})
target_link_libraries(models ${pints_LIBRARIES})

add_library(pints_cpp src/python_wrapper.cpp)
target_link_libraries(pints_cpp models ${pints_LIBRARIES} )

set_target_properties(pints_cpp PROPERTIES PREFIX "" )

#enable_testing()
#add_subdirectory(tests)
