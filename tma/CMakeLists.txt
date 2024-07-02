cmake_minimum_required(VERSION 3.18)
project(MyCudaApp)

enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Path to the external CUTLASS library
set(CUTLASS_DIR "../external/cutlass")

# Repository directory
set(REPO_DIR "..")

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 90a)

# NVCC options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --expt-relaxed-constexpr -Xnvlink=--verbose -Xptxas=--verbose")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g -G")

# Include directories
include_directories(${CUTLASS_DIR}/include)
include_directories(${CUTLASS_DIR}/examples/common)
include_directories(${CUTLASS_DIR}/tools/util/include)
include_directories(${REPO_DIR}/include/utils)

# The main application
add_executable(main main.cu)

# CUDA libraries
target_link_libraries(main 
  cuda cudadevrt cudart_static cublas
)
# Clean target (optional, as CMake generates a clean target by default)
# add_custom_target(clean
#     COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake_clean.cmake
# )
