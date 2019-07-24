#include "BSP/BSP.h"
#include "EdgeDetection/CannyEdge.h"
#include "io/BoundingBoxReader.h"
#include "io/ImageWriter.h"
#include "RANSAC/RANSAC.h"
#include <iostream>

// Memory Leak Detector (only in debug)
//#include <vld.h>

int main()
{
    std::vector<std::tuple<int, Polygon2 *>> boundingBoxes = BoundingBoxReader::ReadFile("/home/kang/pytorch-mask-rcnn/BSP/data/input/bounding_boxes.txt");
    
    //std::cout << "Hello World!";


    for (std::tuple<int, Polygon2 *> boundingBox : boundingBoxes)
    {
        std::vector<Point2 *> edgePoints = CannyEdge::DetectEdges("/home/kang/pytorch-mask-rcnn/BSP/data/input/temp.tif", std::get<1>(boundingBox), 100);
        //ImageWriter::Debug2Image("/home/kang/pytorch-mask-rcnn/BSP/data/input/temp.tif", std::get<1>(boundingBox), "/home/kang/pytorch-mask-rcnn/BSP/data/output/tmp/image_" + std::to_string(std::get<0>(boundingBox)) + "_2_CannyEdge.tif", edgePoints);

        std::vector<LineSegment2 *> lineSegments = RANSAC::DetectLineSegments(edgePoints, 1.5, 100, 1.5, 1.0);
        //ImageWriter::Debug2Image("/home/kang/pytorch-mask-rcnn/BSP/data/input/temp.tif", std::get<1>(boundingBox), "/home/kang/pytorch-mask-rcnn/BSP/data/output/tmp/image_" + std::to_string(std::get<0>(boundingBox)) + "_3_LineSegmentsForBSP.tif", lineSegments);

        std::vector<Polygon2 *> partitions = BSP::DeterminePartitions(boundingBox, lineSegments);
        //ImageWriter::Debug2Image("/home/kang/pytorch-mask-rcnn/BSP/data/input/temp.tif", std::get<1>(boundingBox), "/home/kang/pytorch-mask-rcnn/BSP/data/output/tmp/image_" + std::to_string(std::get<0>(boundingBox)) + "_6_BSP.tif", partitions);

        // Write result
        ImageWriter::VectorData2Image("/home/kang/pytorch-mask-rcnn/BSP/data/input/temp.tif", std::get<1>(boundingBox), "/home/kang/pytorch-mask-rcnn/BSP/data/output/image_" + std::to_string(std::get<0>(boundingBox)) +  ".tif", partitions);



        delete std::get<1>(boundingBox);
        for (Point2 *p : edgePoints)
        {
            delete p;
        }
        for (Polygon2 *p : partitions)
        {
            delete p;
        }


    }


    //std::system("PAUSE");
}


/*#include <iostream>

using namespace std;

int main()
{
    cout << "Hello World!" << endl;
    return 0;
}
*/
