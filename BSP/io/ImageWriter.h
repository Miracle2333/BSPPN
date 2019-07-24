#pragma once

#include "../geom/Polygon2.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

class ImageWriter
{

public:
	static void Debug2Image(const std::string image_input, const Polygon2 &boundingBox, const std::string image_output, std::vector<LineSegment2 *> lineSegments)
	{
		cv::Mat src = imread(image_input, cv::IMREAD_COLOR);
		if (src.empty())
		{
			std::cout << "ImageWriter: Could not open or find the image!\n" << std::endl;
		}

		if (boundingBox.ExteriorRing()->Size() == 4)
		{
			cv::Point min(std::min(std::min(std::min((int)(boundingBox.ExteriorRing()->PointAt(0).x + 0.5), (int)(boundingBox.ExteriorRing()->PointAt(1).x + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(2).x + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(3).x + 0.5)),
				std::min(std::min(std::min((int)(boundingBox.ExteriorRing()->PointAt(0).y + 0.5), (int)(boundingBox.ExteriorRing()->PointAt(1).y + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(2).y + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(3).y + 0.5)));
			cv::Point max(std::max(std::max(std::max((int)(boundingBox.ExteriorRing()->PointAt(0).x + 0.5), (int)(boundingBox.ExteriorRing()->PointAt(1).x + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(2).x + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(3).x + 0.5)),
				std::max(std::max(std::max((int)(boundingBox.ExteriorRing()->PointAt(0).y + 0.5), (int)(boundingBox.ExteriorRing()->PointAt(1).y + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(2).y + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(3).y + 0.5)));

			src = src(cv::Rect(min.x, min.y, max.x - min.x + 1, max.y - min.y + 1));
		}

		for (const LineSegment2 *ls : lineSegments)
		{
			cv::line(src, cv::Point(ls->P1().x, ls->P1().y), cv::Point(ls->P2().x, ls->P2().y), cv::Scalar(0, 255, 0), 1, 8);
		
		}

		imwrite(image_output, src);
	}

	static void Debug2Image(const std::string image_input, const Polygon2 &boundingBox, const std::string image_output, std::vector<Point2 *> points)
	{
		cv::Mat src = imread(image_input, cv::IMREAD_COLOR);
		if (src.empty())
		{
			std::cout << "ImageWriter: Could not open or find the image!\n" << std::endl;
		}

		if (boundingBox.ExteriorRing()->Size() == 4)
		{
			cv::Point min(std::min(std::min(std::min((int)(boundingBox.ExteriorRing()->PointAt(0).x + 0.5), (int)(boundingBox.ExteriorRing()->PointAt(1).x + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(2).x + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(3).x + 0.5)),
				std::min(std::min(std::min((int)(boundingBox.ExteriorRing()->PointAt(0).y + 0.5), (int)(boundingBox.ExteriorRing()->PointAt(1).y + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(2).y + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(3).y + 0.5)));
			cv::Point max(std::max(std::max(std::max((int)(boundingBox.ExteriorRing()->PointAt(0).x + 0.5), (int)(boundingBox.ExteriorRing()->PointAt(1).x + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(2).x + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(3).x + 0.5)),
				std::max(std::max(std::max((int)(boundingBox.ExteriorRing()->PointAt(0).y + 0.5), (int)(boundingBox.ExteriorRing()->PointAt(1).y + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(2).y + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(3).y + 0.5)));

			src = src(cv::Rect(min.x, min.y, max.x - min.x + 1, max.y - min.y + 1));
		}

		for (const Point2 *p : points)
		{
			src.at<cv::Vec3b>(cv::Point(p->x, p->y)) = cv::Vec3b(0, 255, 0);
		}

		imwrite(image_output, src);
	}

	static void Debug2Image(const std::string image_input, const Polygon2 &boundingBox, const std::string image_output, std::vector<Polygon2 *> polygons)
	{
		cv::Mat src = imread(image_input, cv::IMREAD_COLOR);
		if (src.empty())
		{
			std::cout << "ImageWriter: Could not open or find the image!\n" << std::endl;
		}

		if (boundingBox.ExteriorRing()->Size() == 4)
		{
			cv::Point min(std::min(std::min(std::min((int)(boundingBox.ExteriorRing()->PointAt(0).x + 0.5), (int)(boundingBox.ExteriorRing()->PointAt(1).x + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(2).x + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(3).x + 0.5)),
				std::min(std::min(std::min((int)(boundingBox.ExteriorRing()->PointAt(0).y + 0.5), (int)(boundingBox.ExteriorRing()->PointAt(1).y + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(2).y + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(3).y + 0.5)));
			cv::Point max(std::max(std::max(std::max((int)(boundingBox.ExteriorRing()->PointAt(0).x + 0.5), (int)(boundingBox.ExteriorRing()->PointAt(1).x + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(2).x + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(3).x + 0.5)),
				std::max(std::max(std::max((int)(boundingBox.ExteriorRing()->PointAt(0).y + 0.5), (int)(boundingBox.ExteriorRing()->PointAt(1).y + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(2).y + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(3).y + 0.5)));

			src = src(cv::Rect(min.x, min.y, max.x - min.x + 1, max.y - min.y + 1));
		}


		std::vector<std::vector<cv::Point>> polygons_cv;
		for (const Polygon2 *polygon : polygons)
		{
			polygons_cv.push_back(std::vector<cv::Point>());
			for (const Point2 &p : polygon->ExteriorRing()->Points())
			{
				polygons_cv.back().push_back(cv::Point(p.x, p.y));
			}
		}

		for (unsigned int polygon_num = 0; polygon_num < polygons_cv.size(); polygon_num++)
		{
			const cv::Point *pts = (const cv::Point *)cv::Mat(polygons_cv[polygon_num]).data;
			int npts = cv::Mat(polygons_cv[polygon_num]).rows;

			polylines(src, &pts, &npts, 1, true, cv::Scalar(0, 255, 0));
		}

		imwrite(image_output, src);
	}

	static void VectorData2Image(const std::string image_input, const Polygon2 &boundingBox, const std::string image_output, std::vector<Polygon2 *> polygons)
	{
		cv::Mat src = imread(image_input, cv::IMREAD_COLOR);
		if (src.empty())
		{
			std::cout << "ImageWriter: Could not open or find the image!\n" << std::endl;
		}

		if (boundingBox.ExteriorRing()->Size() == 4)
		{
			cv::Point min(std::min(std::min(std::min((int)(boundingBox.ExteriorRing()->PointAt(0).x + 0.5), (int)(boundingBox.ExteriorRing()->PointAt(1).x + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(2).x + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(3).x + 0.5)),
						  std::min(std::min(std::min((int)(boundingBox.ExteriorRing()->PointAt(0).y + 0.5), (int)(boundingBox.ExteriorRing()->PointAt(1).y + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(2).y + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(3).y + 0.5)));
			cv::Point max(std::max(std::max(std::max((int)(boundingBox.ExteriorRing()->PointAt(0).x + 0.5), (int)(boundingBox.ExteriorRing()->PointAt(1).x + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(2).x + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(3).x + 0.5)),
						  std::max(std::max(std::max((int)(boundingBox.ExteriorRing()->PointAt(0).y + 0.5), (int)(boundingBox.ExteriorRing()->PointAt(1).y + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(2).y + 0.5)), (int)(boundingBox.ExteriorRing()->PointAt(3).y + 0.5)));

			src = src(cv::Rect(min.x, min.y, max.x - min.x + 1, max.y - min.y + 1));
		}

		// Create a matrix of the same type and size as src (for output)
		cv::Mat result(src.size(), src.type());

		result = cv::Scalar::all(255);

		std::vector<std::vector<cv::Point>> polygons_cv;
		for (const Polygon2 *polygon : polygons)
		{
			polygons_cv.push_back(std::vector<cv::Point>());
			for (const Point2 &p : polygon->ExteriorRing()->Points())
			{
				polygons_cv.back().push_back(cv::Point(p.x, p.y));
			}
		}

		for (unsigned int polygon_num = 0; polygon_num < polygons_cv.size(); polygon_num++)
		{
			const cv::Point *pts = (const cv::Point *)cv::Mat(polygons_cv[polygon_num]).data;
			int npts = cv::Mat(polygons_cv[polygon_num]).rows;

			polylines(result, &pts, &npts, 1, true, cv::Scalar(0, 0, 0));
		}

		imwrite(image_output, result);

		//src.copyTo(result, result);	// overlayed with input image
		//imwrite(image_output, result);

		//imshow("Image", result);
		//waitKey(0);
	}
};