cmake_minimum_required(VERSION 3.31)
project(epic2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(epic2 epic2/tests_epic2.cpp)

target_include_directories(epic2 PUBLIC epic2)
