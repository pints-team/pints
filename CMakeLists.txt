cmake_minimum_required(VERSION 2.8.3)
project(hobo)
set(Python_ADDITIONAL_VERSIONS 2.7 2.6 2.5 2.4 2.3 2.2 2.1 2.0)
FIND_PACKAGE(PythonLibs REQUIRED)
FIND_PACKAGE(Boost COMPONENTS python REQUIRED)
find_package(Eigen3 REQUIRED)

set(hobo_LIBRARIES ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set(hobo_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})


set(BUILD_SHARED_LIBS ON)
set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

set(source_files 
    src/cpp/electrochemistry/e_implicit_exponential_mesh.cpp
    src/cpp/electrochemistry/seq_electron_transfer3_explicit.cpp
    )
set(header_files 
    src/cpp/electrochemistry/e_implicit_exponential_mesh.hpp
    src/cpp/electrochemistry/seq_electron_transfer3_explicit.hpp
    src/cpp/electrochemistry/utilities.hpp
    )

set(header_dirs
    src/cpp/electrochemistry
    )

INCLUDE_DIRECTORIES(${header_dirs} ${hobo_INCLUDE_DIRS})

add_library(models ${source_files} ${header_files})
target_link_libraries(models ${hobo_LIBRARIES})

add_library(hobo_cpp src/cpp/python_wrapper.cpp)
target_link_libraries(hobo_cpp models ${hobo_LIBRARIES} )

set_target_properties(hobo_cpp PROPERTIES PREFIX "" )

enable_testing()
add_subdirectory(tests)
