cmake_minimum_required(VERSION 3.10)
project(proj2)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)

add_executable(main src/main.cpp src/features.cpp)
#add_executable(video_display src/filter.cpp src/vidDisplay.cpp)

find_package(OpenCV REQUIRED)


# linking
target_link_libraries(main ${OpenCV_LIBS})
#target_link_libraries(video_display ${OpenCV_LIBS})