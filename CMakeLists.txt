cmake_minimum_required(VERSION 3.10)
project(VectorDB)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /W4)
    add_definitions(-D_WIN32_WINNT=0x0A00)
else()
    add_compile_options(-O2 -Wall -Wextra)
    add_definitions(-D_WIN32_WINNT=0x0A00)
endif()

# Include directories
include_directories(include)

# Source files
file(GLOB_RECURSE SRC_FILES src/*.cpp)

# Executable
add_executable(db main.cpp ${SRC_FILES})

# Link libraries
if(WIN32)
    target_link_libraries(db ws2_32)
endif()
