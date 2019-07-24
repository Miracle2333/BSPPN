cmake_minimum_required(VERSION 2.8)

project(project_bsp)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")


file(GLOB SOURCES
    BSP/BSP.h
    BSP/BSPNode.h
    BSP/BSPTree.h
    BSP/Filter.h
    EdgeDetection/CannyEdge.h
    geom/math/Distance.h
    geom/math/Intersection.h
    geom/Line2.h
    geom/LinearRing2.h
    geom/LineSegment2.h
    geom/Point2.h
    geom/Polygon2.h
    geom/Vector2.h
    io/BoundingBoxReader.h
    io/ImageWriter.h
    RANSAC/RANSAC.h
    RANSAC/geom/LineSegment2_RANSAC.h
    RANSAC/geom/Point2_RANSAC.h
    util/Util.h
    main.cpp
)


#if(CMAKE_VERSION VERSION_LESS "2.8.11")
#  # Add OpenCV headers location to your include paths
#  include_directories(${OpenCV_INCLUDE_DIRS})
#endif()
include_directories(include ${OpenCV_INCLUDE_DIRS})

# Declare the executable target built from your sources
add_executable(${PROJECT_NAME} ${SOURCES})

# Link your application with OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

