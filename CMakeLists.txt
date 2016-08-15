cmake_minimum_required(VERSION 2.8)
project(tinygrad)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} "include/")

set(HEADER_FILES include/node.h include/nodeimplementations.h include/matrix.h include/graph.h include/utils.h include/predefined_models.h include/dataset.h)
set(SOURCE_FILES src/node.cpp src/nodeimplementations.cpp src/matrix.cpp src/graph.cpp src/utils.cpp src/predefined_models.cpp src/dataset.cpp)

add_executable(test_autoencoder ${SOURCE_FILES} examples/test_autoencoder.cpp)
add_executable(test_neural_network ${SOURCE_FILES} examples/test_neural_network.cpp)
add_executable(test_logistic_regression ${SOURCE_FILES} examples/test_logistic_regression.cpp)
