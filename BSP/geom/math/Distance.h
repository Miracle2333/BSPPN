#pragma once

#include "../Line2.h"

class Distance
{

public:

	static double Distance2D(const Point2 &point, const Line2 &line)
	{
		Vector2 a = Vector2(line.point.x, line.point.y);
		Vector2 v = Vector2(line.direction.x, line.direction.y);
		Vector2 p = Vector2(point.x, point.y);

		double t = DotProduct(v, p - a) / DotProduct(v, v);

		Vector2 pt = a + t * v;
		Vector2 vec = p - pt;

		return sqrt(DotProduct(vec, vec));
	}

	static double Distance2D(const Point2 &point, const Line2 &line, double *t)
	{
		*t = DotProduct(line.direction, point - line.point) / DotProduct(line.direction, line.direction);

		Vector2 pt = line.point + *t * line.direction;
		Vector2 vec = point - pt;

		return sqrt(DotProduct(vec, vec));
	}

	static double SignedDistance2D(const Point2 &point, const Line2 &line)
	{
		Point2 line_p1 = line.point;
		Point2 line_p2 = line.point + line.direction;

		return (line_p1.y - line_p2.y) * point.x + (line_p2.x - line_p1.x) * point.y + (line_p1.x*line_p2.y - line_p2.x*line_p1.y);	
	}

};