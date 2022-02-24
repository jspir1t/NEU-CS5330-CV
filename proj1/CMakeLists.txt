cmake_minimum_required(VERSION 3.10)
project(proj1)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)

add_executable(image_display src/imgDisplay.cpp)
add_executable(video_display src/filter.cpp src/vidDisplay.cpp)

find_package(OpenCV REQUIRED)


# linking
target_link_libraries(image_display ${OpenCV_LIBS})
target_link_libraries(video_display ${OpenCV_LIBS})