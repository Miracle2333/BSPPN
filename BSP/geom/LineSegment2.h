#pragma once

#include "Line2.h"

// Line segment is valid between t0 and t1
// Definition see Eberly book
class LineSegment2 : public Line2
{

public:
	double t0;
	double t1;

	// ---------------------------------------------------------------------------------------------------- //

public:
	inline LineSegment2()
		: Line2(), t0(std::numeric_limits<double>::max()), t1(-std::numeric_limits<double>::max()) {}

	inline LineSegment2(const Point2 &p1, const Point2 &p2) : Line2(p1, p2 - p1)
	{
		SetT0(p1);
		SetT1(p2);
	}

	// ---------------------------------------------------------------------------------------------------- //

	inline class Point2 Center() const
	{
		return (P1() + P2()) / 2.0;
	}	

	inline double Direction_Angle() const
	{
		return AngleDegree(Vector2(1.0, 0.0), direction);
	}

	inline double Direction_Angle360_CCW() const
	{
		return AngleDegree360_CCW(Vector2(1.0, 0.0), direction);
	}

	void HaveTsInAscendingOrder()
	{
		if (t0 > t1)
		{
			double tmp = t0;
			t0 = t1;
			t1 = tmp;
		}
	}

	bool IsParallel(const LineSegment2 &ls) const
	{
		return (AngleDegree(direction, ls.direction) < 0.0001) || (AngleDegree(direction, -ls.direction) < 0.0001);
	}

	inline bool IsValid() const
	{
		return t0 != std::numeric_limits<double>::max() && t1 != -std::numeric_limits<double>::max();
	}

	inline double Length() const
	{
		return abs(t1 - t0);
	}

	inline Point2 P1() const
	{
		return point + t0 * direction;
	}

	inline Point2 P2() const
	{
		return point + t1 * direction;
	}

	void Rotate_CCW(const double angle_degree)
	{
		Point2 p1 = P1();
		Point2 p2 = P2();

		Point2 center = Center();

		double s = sin(angle_degree * M_PI / 180); // angle in radians
		double c = cos(angle_degree * M_PI / 180); // angle in radians

		// translate points back to origin
		p1 -= center;
		p2 -= center;

		// rotate points
		double p1_x_new = p1.x * c - p1.y * s;
		double p1_y_new = p1.x * s + p1.y * c;
		double p2_x_new = p2.x * c - p2.y * s;
		double p2_y_new = p2.x * s + p2.y * c;

		// translate points back
		p1 = Point2(p1_x_new, p1_y_new) + center;
		p2 = Point2(p2_x_new, p2_y_new) + center;

		this->point = p1;
		this->direction = p2 - p1;
		Normalize();
		SetT0(p1);
		SetT1(p2);
	}

	void SetT0(const Point2 &ending_point)
	{
		if (direction.x != 0)
			t0 = (ending_point.x - point.x) / direction.x;
		else
			t0 = (ending_point.y - point.y) / direction.y;
	}

	void SetT1(const Point2 &ending_point)
	{
		if (direction.x != 0)
			t1 = (ending_point.x - point.x) / direction.x;
		else
			t1 = (ending_point.y - point.y) / direction.y;
	}

	void SwitchDirection()
	{
		direction = -direction;		
		t1 = -t1;
	}

};

// Writes Point2 p to output stream os.
inline std::ostream & operator<<(std::ostream &os, const LineSegment2 &ls)
{
	os << ls.P1() << "," << ls.P2();
	return os;
}