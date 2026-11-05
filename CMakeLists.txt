# RL compatibility build. Simulator sources and tests belong to the engine fork.
cmake_minimum_required(VERSION 3.5)
project(SevenWondersRL LANGUAGES CXX)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/engine/CMakeLists.txt")
    message(FATAL_ERROR "Engine submodule missing. Run: git submodule update --init --recursive")
endif()

# Preserve build/7Wonders for existing training and evaluation entry points.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
enable_testing()
add_subdirectory(engine)
