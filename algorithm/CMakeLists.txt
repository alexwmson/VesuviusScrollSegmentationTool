cmake_minimum_required(VERSION 3.22)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(vesuvius_segmentation_tool LANGUAGES CXX)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TIFF REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(nlohmann_json REQUIRED)
find_package(z5 REQUIRED)

add_library(vc_core
    src/vc/core/types/ChunkedTensor.cpp
    src/vc/core/util/QuadSurface.cpp
    src/vc/core/util/Slicing.cpp
    src/vc/core/util/Tiff.cpp
    src/vc/core/util/ChunkCache.cpp
    src/vc/core/util/Compositing.cpp
    src/vc/core/util/Geometry.cpp
    src/vc/core/util/PointIndex.cpp
    src/vc/core/util/SurfacePatchIndex.cpp
    src/vc/core/util/PlaneSurface.cpp
    src/vc/core/util/ABFFlattening.cpp
)

target_include_directories(vc_core 
    SYSTEM PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include 
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(vc_core
    PUBLIC
        OpenMP::OpenMP_CXX
        nlohmann_json::nlohmann_json
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_calib3d
        opencv_video
        z5
        tiff
)

add_executable(aw_segmentation_algorithm apps/aw_segmentation_algorithm.cpp)
target_link_libraries(aw_segmentation_algorithm
    PRIVATE
        vc_core
        Boost::program_options
        z5
)