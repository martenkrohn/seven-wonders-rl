# Downstream test integration; documented 2026-10-19. See NOTICE and docs/ORIGIN.md.
cmake_minimum_required(VERSION 3.5)

project(7Wonders LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(JSON_BuildTests OFF CACHE INTERNAL "")

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include_directories(include)

set(SOURCES
    main.cpp
    src/card.cpp
	src/player.cpp
    src/wonder.cpp
	src/filer.cpp
	)

set(HEADERS
    include/card.h
	include/player.h
	include/resources.h
    include/wonder.h
	include/filer.h
	include/nlohmann/json.hpp
	)

add_executable(7Wonders ${SOURCES} ${HEADERS})

# Test executable for choice-once-per-turn verification (simple, no card loading)
add_executable(test_choice_simple
    tests/test_choice_simple.cpp
    src/player.cpp
    src/wonder.cpp
    src/card.cpp
    src/filer.cpp
)

# Test executable for choice-once-per-turn verification (full integration)
add_executable(test_choice_once_per_turn
    tests/test_choice_once_per_turn.cpp
    src/card.cpp
    src/player.cpp
    src/wonder.cpp
    src/filer.cpp
)

file(MAKE_DIRECTORY ./build/io ./io ./logs)

# Register the existing regression executables for local checks and CI.
enable_testing()
add_test(NAME choice_simple COMMAND test_choice_simple)
add_test(NAME choice_once_per_turn COMMAND test_choice_once_per_turn)

add_executable(test_final_status tests/test_final_status.cpp
    src/card.cpp src/player.cpp src/wonder.cpp src/filer.cpp)
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/final_status_runtime")
add_test(NAME final_status COMMAND test_final_status)
set_tests_properties(final_status PROPERTIES
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/final_status_runtime" TIMEOUT 10)

add_executable(test_joint_action_resources tests/test_joint_action_resources.cpp
    src/card.cpp src/player.cpp src/wonder.cpp src/filer.cpp)
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/joint_action_resources_runtime")
add_test(NAME joint_action_resources COMMAND test_joint_action_resources)
set_tests_properties(joint_action_resources PROPERTIES
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/joint_action_resources_runtime" TIMEOUT 10)

# The regression programs use assert(), including when the engine is built Release.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_choice_simple PRIVATE -UNDEBUG)
    target_compile_options(test_choice_once_per_turn PRIVATE -UNDEBUG)
endif()
