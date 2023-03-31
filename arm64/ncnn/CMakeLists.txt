cmake_minimum_required(VERSION 3.4.1)
SET(PROJ_NAME chacha)

project(${PROJ_NAME} LANGUAGES CXX)

message(${VCPKG_INSTALLED_DIR})
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
# set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-linux/share/opencv4")


find_package(ncnn REQUIRED)
find_package(argparse REQUIRED)
# find_package(OpenCV REQUIRED)

add_executable(${PROJ_NAME} main.cpp)

set_property(TARGET ${PROJ_NAME} PROPERTY CXX_STANDARD 17)
target_link_libraries(${PROJ_NAME} PRIVATE ncnn)
target_link_libraries(${PROJ_NAME} PRIVATE argparse::argparse)
# target_link_libraries(${PROJ_NAME} PRIVATE ${OpenCV_LIBS})
