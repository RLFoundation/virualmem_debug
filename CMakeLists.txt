cmake_minimum_required(VERSION 3.10)
project(cumem_test LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ROCm/HIP installation
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH})
else()
  set(ROCM_PATH "/opt/rocm")
endif()
message(STATUS "Using ROCm path: ${ROCM_PATH}")

# Add HIP include directories
include_directories(
  ${ROCM_PATH}/include
  ${ROCM_PATH}/hip/include
)

# Set HIP library paths
set(HIP_LIBRARY_DIR ${ROCM_PATH}/lib)
link_directories(${HIP_LIBRARY_DIR})

# Add compile definition for ROCm
add_definitions(-DUSE_ROCM)

# Copy compatibility header to local directory
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cumem_allocator_compat.h
  ${CMAKE_CURRENT_BINARY_DIR}/cumem_allocator_compat.h
  COPYONLY
)

# Extract ROCm implementation functions
add_library(cumem_functions OBJECT
  ${CMAKE_CURRENT_SOURCE_DIR}/cumem_functions.cpp
)

# Add include directories for cumem_functions
target_include_directories(cumem_functions PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${ROCM_PATH}/include
  ${ROCM_PATH}/hip/include
)

# Add the test executable
add_executable(cumem_test 
  ${CMAKE_CURRENT_SOURCE_DIR}/cumem_test.cpp
  $<TARGET_OBJECTS:cumem_functions>
)

# Add include directories for cumem_test
target_include_directories(cumem_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${ROCM_PATH}/include
  ${ROCM_PATH}/hip/include
)

# Link against HIP libraries
target_link_libraries(cumem_test
  amdhip64
)

# Set ROCm compile flags
target_compile_options(cumem_test PRIVATE
  -D__HIP_PLATFORM_AMD__
  -fPIC
)

# Set ROCm compile flags for the library
target_compile_options(cumem_functions PRIVATE
  -D__HIP_PLATFORM_AMD__
  -fPIC
) 