cmake_minimum_required(VERSION 3.31)
project(epic2)

set(CMAKE_CXX_STANDARD 20)

add_executable(epic2 main.cpp
        src/tensor.h
        src/nn_optimizer.h)
