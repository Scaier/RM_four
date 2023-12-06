cmake_minimum_required(VERSION 3.0.0)
project(task_final)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

aux_source_directory(./src ALL_SRCS)
include_directories(include ${OpenCV_INCLUDE_DIRS})
include_directories(include ${CERES_INCLUDE_DIRS})

set(SRC ./cc)

add_executable(task_final main.cpp ${ALL_SRCS})
target_link_libraries(task_final ${OpenCV_LIBS})
target_link_libraries(task_final ${CERES_LIBRARIES})