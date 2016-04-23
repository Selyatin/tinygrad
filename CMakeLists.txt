cmake_minimum_required(VERSION 3.5)
project(tinygrad)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(SOURCE_FILES main.cpp node.cpp node.h nodeimplementations.cpp nodeimplementations.h tensor.cpp tensor.h graph.cpp graph.h utils.cpp utils.h)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "/home/niko/Projektit/tinygrad/build")
add_executable(tinygrad ${SOURCE_FILES})
