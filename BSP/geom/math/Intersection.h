#pragma once

#include "../Line2.h"

class Intersection
{

public:	

	static bool Intersection2D(const Line2 &l1, const Line2 &l2, Point2 &intersection)
	{
		Vector2 d1 = l1.direction;
		Vector2 d2 = l2.direction;

		Vector2 o1 = l1.point;
		Vector2 o2 = l2.point;

		double dot_d1_per_d2 = DotProduct(d1, d2.PerpendicularCCW());
		if (fabs(dot_d1_per_d2) < std::numeric_limits<double>::epsilon())
			return false;

		double s = DotProduct((o2 - o1), d2.PerpendicularCCW()) / dot_d1_per_d2;

		double dot_d2_per_d1 = DotProduct(d2, d1.PerpendicularCCW());

		double t = DotProduct((o1 - o2), d1.PerpendicularCCW()) / dot_d2_per_d1;

		intersection = o1 + s * d1;
		return true;	
	}

};