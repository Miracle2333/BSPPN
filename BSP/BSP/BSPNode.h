#pragma once

#include "../geom/math/Intersection.h"
#include "../geom/Line2.h"
#include "../geom/LineSegment2.h"
#include "../geom/Polygon2.h"

#include <algorithm>
#include <iostream>

class BSPNode
{

public:
	Polygon2 *polygon;
	Line2 *partitioning_line;	
	BSPNode *child_left, *child_right;

	// ---------------------------------------------------------------------------------------------------- //

	BSPNode(Polygon2 *polygon)
		: polygon(polygon), partitioning_line(nullptr), child_left(nullptr), child_right(nullptr) {}

	~BSPNode()
	{
		delete polygon;

		if (partitioning_line)
		{
			delete partitioning_line;
			delete child_left;
			delete child_right;
		}
	}

	// ---------------------------------------------------------------------------------------------------- //

	void Build_BSPTree(std::vector<LineSegment2 *> &partitioning_lines)
	{
		Sort(partitioning_lines);

		partitioning_line = partitioning_lines[0];

		Divide();

		if (partitioning_lines.size() > 1)
		{
			std::vector<LineSegment2 *> partitioning_lines_left = std::vector<LineSegment2 *>();
			std::vector<LineSegment2 *> partitioning_lines_right = std::vector<LineSegment2 *>();

			AssignSide(std::vector<LineSegment2 *>(partitioning_lines.begin() + 1, partitioning_lines.end()), partitioning_lines_left, partitioning_lines_right);

			if (partitioning_lines_left.size() > 0)
			{
				child_left->Build_BSPTree(partitioning_lines_left);
			}
			if (partitioning_lines_right.size() > 0)
			{
				child_right->Build_BSPTree(partitioning_lines_right);
			}
		}		
	}

	// ---------------------------------------------------------------------------------------------------- //

private:

	void AssignSide(std::vector<LineSegment2 *> partitioning_lines, std::vector<LineSegment2 *> &partitioning_lines_left, std::vector<LineSegment2 *> &partitioning_lines_right)
	{
		for (LineSegment2 *ls : partitioning_lines)
		{
			int pos = Position(*ls);

			if (pos == -2)
			{
				Point2 intersection;
				if (!Intersection::Intersection2D(*partitioning_line, Line2(ls->P1(), ls->P2() - ls->P1()), intersection))
				{
					std::cout << "Error: BSPNode -> AssignSide -> Case 1" << std::endl;
				}

				partitioning_lines_left.push_back(new LineSegment2(ls->P1(), intersection));
				partitioning_lines_right.push_back(new LineSegment2(ls->P2(), intersection));
				delete ls;
			}
			else if (pos == -1)
			{
				partitioning_lines_left.push_back(ls);
			}
			else if (pos == 0)
			{

			}
			else if (pos == 1)
			{
				partitioning_lines_right.push_back(ls);
			}
			else if (pos == 2)
			{
				Point2 intersection;
				if (!Intersection::Intersection2D(*partitioning_line, Line2(ls->P1(), ls->P2() - ls->P1()), intersection))
				{
					std::cout << "Error: BSPNode -> AssignSide -> Case 2" << std::endl;
				}

				partitioning_lines_left.push_back(new LineSegment2(ls->P2(), intersection));
				partitioning_lines_right.push_back(new LineSegment2(ls->P1(), intersection));
				delete ls;
			}
			else
			{
				std::cout << "Error: BSPNode -> AssignSide" << std::endl;
			}
		}	
	}

	void Divide()
	{
		bool pos_left = true;
		bool pos_right = true;
		for (const Point2 &p : polygon->ExteriorRing()->Points())
		{
			double pos = this->partitioning_line->EstimateSide2D(p);

			if (pos < 0)
			{
				pos_right = false;
			}
			else if (pos > 0)
			{
				pos_left = false;
			}

			if (!pos_left && !pos_right)
			{
				break;
			}
		}
		
		if (pos_left)
		{
			child_left = new BSPNode(new Polygon2(polygon));
			child_right = new BSPNode(new Polygon2());
		}
		else if (pos_right)
		{
			child_left = new BSPNode(new Polygon2());
			child_right = new BSPNode(new Polygon2(polygon));
		}
		else
		{
			std::vector<Point2> points = polygon->ExteriorRing()->Points();
			points.push_back(polygon->ExteriorRing()->PointAt(0));

			std::vector<Point2> points_left = std::vector<Point2>();
			std::vector<Point2> points_right = std::vector<Point2>();
			
			double pos = this->partitioning_line->EstimateSide2D(points[0]);
			if (pos < 0)
			{
				points_left.push_back(points[0]);
			}
			else if (pos == 0)
			{
				points_left.push_back(points[0]);
				points_right.push_back(points[0]);
			}
			else if (pos > 0)
			{
				points_right.push_back(points[0]);
			}
			

			for (int i = 1; i < points.size(); i++)
			{
				double pos_prev = pos;
				pos = this->partitioning_line->EstimateSide2D(points[i]);				

				if (pos < 0)
				{
					if (pos_prev > 0)
					{
						Point2 intersection;
						if (!Intersection::Intersection2D(*this->partitioning_line, Line2(polygon->ExteriorRing()->PointAt(i - 1), points[i] - polygon->ExteriorRing()->PointAt(i - 1)), intersection))
						{
							std::cout << "Error: BSPNode -> Divide -> Case 1" << std::endl;
						}

						points_left.push_back(intersection);
						points_right.push_back(intersection);
					}

					points_left.push_back(points[i]);					
				}
				else if (pos == 0)
				{
					points_left.push_back(points[i]);
					points_right.push_back(points[i]);
				}
				else if (pos > 0)
				{
					if (pos_prev < 0)
					{
						Point2 intersection;
						if (!Intersection::Intersection2D(*this->partitioning_line, Line2(polygon->ExteriorRing()->PointAt(i - 1), points[i] - polygon->ExteriorRing()->PointAt(i - 1)), intersection))
						{
							std::cout << "Error: BSPNode -> Divide -> Case 2" << std::endl;
						}

						points_left.push_back(intersection);
						points_right.push_back(intersection);
					}

					points_right.push_back(points[i]);					
				}
			}

			child_left = new BSPNode(new Polygon2(new LinearRing2(points_left)));
			child_right = new BSPNode(new Polygon2(new LinearRing2(points_right)));
		}
	}

	static bool IsLongerThan(LineSegment2 *ls1, LineSegment2 *ls2)
	{
		return (ls1->Length() > ls2->Length());
	}

	int Position(const LineSegment2 &ls)
	{
		double pos_p1 = partitioning_line->EstimateSide2D(ls.P1());
		double pos_p2 = partitioning_line->EstimateSide2D(ls.P2());

		if (pos_p1 < 0 && pos_p2 > 0)				// left, right
		{
			return -2;
		}
		else if ((pos_p1 < 0 && pos_p2 <= 0)  ||	// left/on, left/on
				 (pos_p1 <= 0 && pos_p2 < 0))
		{
			return -1;
		}
		else if (pos_p1 == 0 && pos_p2 == 0)		// on, on
		{
			return 0;
		}
		else if ((pos_p1 > 0 && pos_p2 >= 0) ||		// right/on, right/on
				 (pos_p1 >= 0 && pos_p2 > 0))
		{
			return 1;
		}
		else if (pos_p1 > 0 && pos_p2 < 0) 		// right, left
		{
			return 2;
		}
		else
		{
			return 42;
		}
	}

	void Sort(std::vector<LineSegment2 *> &partitioning_lines)
	{
		std::sort(partitioning_lines.begin(), partitioning_lines.end(), BSPNode::IsLongerThan);
	}

};