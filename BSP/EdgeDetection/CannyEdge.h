#include "../geom/Point2.h"
#include "../geom/Polygon2.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <fstream>
#include <iostream>

class CannyEdge
{

public:	

	const static int ratio = 3;
	const static int kernel_size = 3;

	static std::vector<Point2 *> DetectEdges(const std::string image, Polygon2 *boundingBox = nullptr, const int threshold = 100)
	{
		cv::Mat src = cv::imread(image, cv::IMREAD_COLOR); // Load an image

		if (src.empty())
		{
			std::cout << "CannyEdge: Could not open or find the image!\n" << std::endl;
			std::system("PAUSE");
		}

		if (boundingBox != nullptr && boundingBox->ExteriorRing()->Size() >= 3)
		{
			for (unsigned int i = 0; i < boundingBox->ExteriorRing()->Size(); i++)
			{
				Point2 p = boundingBox->ExteriorRing()->PointAt(i);
				if (p.x < 0)
				{
					boundingBox->ExteriorRing()->ChangePointValues(i, Point2(0.0, boundingBox->ExteriorRing()->PointAt(i).y));
				}
				if (p.y < 0)
				{
					boundingBox->ExteriorRing()->ChangePointValues(i, Point2(boundingBox->ExteriorRing()->PointAt(i).x, 0.0));
				}
				if (p.x > src.cols - 1)
				{
					boundingBox->ExteriorRing()->ChangePointValues(i, Point2(src.cols - 1, boundingBox->ExteriorRing()->PointAt(i).y));
				}
				if (p.y > src.rows - 1)
				{
					boundingBox->ExteriorRing()->ChangePointValues(i, Point2(boundingBox->ExteriorRing()->PointAt(i).x, src.rows - 1));
				}
			}

			int min_x = std::numeric_limits<int>::max();
			int min_y =	std::numeric_limits<int>::max();
			int max_x =	-std::numeric_limits<int>::max();
			int max_y =	-std::numeric_limits<int>::max();

			for (Point2 &p : boundingBox->ExteriorRing()->Points())
			{
				min_x = std::min(min_x, (int)(p.x));
				min_y = std::min(min_y, (int)(p.y));
				max_x = std::max(max_x, (int)(p.x));
				max_y = std::max(max_y, (int)(p.y));
			}
			
			src = src(cv::Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1));
		}

		// Create a matrix of the same type and size as src (for dst)
		cv::Mat dst(src.size(), src.type());	

		// Convert to gray
		cv::Mat src_gray;
		cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

		// Reduce noise with a kernel 3x3
		cv::Mat detected_edges;
		cv::blur(src_gray, detected_edges, cv::Size(3, 3));

		// Canny detector
		cv::Canny(detected_edges, detected_edges, threshold, threshold*ratio, kernel_size);

		return Mat2Points(detected_edges);
	}

private:

	static std::vector<Point2 *> Mat2Points(const cv::Mat &output)
	{
		std::vector<Point2 *> result;

		for (int i = 0; i < output.rows; i++)
		{
			for (int j = 0; j < output.cols; j++)
			{
				if (output.at<uchar>(i, j) == 255)
				{
					result.push_back(new Point2(j, i));
				}
			}
		}

		return result;
	}

};
