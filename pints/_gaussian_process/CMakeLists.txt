cmake_minimum_required(VERSION 3.1)

project(aboria_wrapper)



set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1 )

set(include_dirs "")
set(libs "")
set(lib_dir "${CMAKE_SOURCE_DIR}/../../third-party")

set(CMAKE_MODULE_PATH "${lib_dir}/Aboria/cmake" ${CMAKE_MODULE_PATH})

# OpenMP
find_package(OpenMP REQUIRED)
add_definitions(-DHAVE_OPENMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# H2Lib
find_package(H2Lib REQUIRED)
list(APPEND libs ${H2Lib_LIBRARIES})
list(APPEND include_dirs ${H2Lib_INCLUDE_DIRS})
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${H2Lib_LINKER_FLAGS}")
add_definitions(-DHAVE_H2LIB)

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
  ${source_dir}/Kernels.cpp
  ${source_dir}/python_wrapper.cpp
    )
set(header_files 
  ${source_dir}/Kernels.hpp
  ${source_dir}/GaussianProcess.hpp
  ${source_dir}/GaussianProcess.tcc
  ${source_dir}/GaussianProcessMatrixFree.hpp
  ${source_dir}/GaussianProcessMatrixFree.tcc
  ${source_dir}/GaussianProcessDenseMatrix.hpp
  ${source_dir}/GaussianProcessDenseMatrix.tcc
    )

include_directories(${source_dir} ${include_dirs})

pybind11_add_module(gaussian_process ${source_files} ${header_files})
target_link_libraries(gaussian_process PRIVATE ${libs})

