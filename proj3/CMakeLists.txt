cmake_minimum_required(VERSION 3.10)
project(proj3)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)

add_executable(main src/main.cpp src/segmentation.cpp src/database.cpp)

find_package(OpenCV REQUIRED)

# linking
target_link_libraries(main ${OpenCV_LIBS})