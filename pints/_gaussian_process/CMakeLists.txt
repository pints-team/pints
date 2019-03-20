cmake_minimum_required(VERSION 3.1)

project(aboria_wrapper)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1 )

set(include_dirs "")
set(lib_dir "${CMAKE_SOURCE_DIR}/../../third-party")

# Aboria 
list(APPEND include_dirs ${lib_dir}/Aboria/src)
list(APPEND include_dirs ${lib_dir}/Aboria/third-party)

# Eigen
add_definitions(-DHAVE_EIGEN)
list(APPEND include_dirs ${lib_dir}/eigen-git-mirror)

# Pybind11
add_subdirectory(${lib_dir}/pybind11 ${CMAKE_BINARY_DIR}/pybind11)

# Boost 
list(APPEND include_dirs ${lib_dir}/boost)

set(source_dir ".")

set(source_files 
  ${source_dir}/GaussianProcess.cpp
    )
set(header_files 
  ${source_dir}/GaussianProcess.hpp
    )

include_directories(${source_dir} ${include_dirs})

pybind11_add_module(aboria_wrapper ${source_files} ${header_files})

