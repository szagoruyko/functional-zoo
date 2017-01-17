cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(hdf5-example)
   
set(CMAKE_CXX_STANDARD 11)

find_package(HDF5 REQUIRED)

include_directories(${HDF5_INCLUDE_DIRS})

add_executable(hdf5_app main.cpp)
target_link_libraries(hdf5_app ${HDF5_LIBRARIES} hdf5_cpp)
