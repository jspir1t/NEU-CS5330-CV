cmake_minimum_required(VERSION 3.10)
project(proj4)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)

add_executable(main src/main.cpp src/utils.cpp)
add_executable(ar src/ar.cpp src/utils.cpp)
add_executable(feature src/feature.cpp src/utils.cpp)


find_package(OpenCV REQUIRED)

# linking
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(ar ${OpenCV_LIBS})
target_link_libraries(feature ${OpenCV_LIBS})