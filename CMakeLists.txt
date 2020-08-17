cmake_minimum_required(VERSION 3.10)
project(watertight)


set(CMAKE_VERBOSE_MAKEFILE 0)

# dependencies
find_package(CUDA 8.0 REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

find_package(Eigen3 3.3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(libs/glm/)
include_directories(libs/glm/)
include_directories(src)
include_directories(src/base/)
include_directories("${CUDA_INCLUDE_DIRS}")
# <-

# -> cuda compilation
set(CUDACXX "NVCC")
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -std=c++17")

enable_language(CUDA)
add_executable(watertight src/main.cpp src/voxelize_project.cpp src/watershed.cu src/sdf.cu)
# <-

