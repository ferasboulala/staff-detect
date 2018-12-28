cmake_minimum_required(VERSION 2.8)

project(stav)

find_package(OpenCV REQUIRED)

include_directories(include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_STANDARD_11)
set(ALL_LIBRARIES)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

list(APPEND ALL_LIBRARIES ${OpenCV_LIBS})
list(APPEND ALL_LIBRARIES -lboost_system)
list(APPEND ALL_LIBRARIES -lboost_program_options)
list(APPEND ALL_LIBRARIES -lboost_filesystem)
list(APPEND ALL_LIBRARIES -lstdc++fs)
list(APPEND ALL_LIBRARIES -pthread)

file(GLOB PROJECT_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cc")

add_executable(stav ${PROJECT_SOURCES})
target_link_libraries(stav ${ALL_LIBRARIES})