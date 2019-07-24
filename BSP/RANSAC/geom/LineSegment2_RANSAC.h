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

	inline LineSegment2(const Point2 &p1, const Point2 &p2) : Line2(p1, p2-p1)
	{
		SetT0(p1);
		SetT1(p2);
	}

// ---------------------------------------------------------------------------------------------------- //
	
	void SetT0(const Point2 &ending_point)
	{
		if (direction.x != 0)
			t0 = (ending_point.x-point.x) / direction.x;
		else
			t0 = (ending_point.y-point.y) / direction.y;
	}

	void SetT1(const Point2 &ending_point)
	{
		if (direction.x != 0)
			t1 = (ending_point.x-point.x) / direction.x;
		else
			t1 = (ending_point.y-point.y) / direction.y;
	}

	inline double Length() const
	{
		return t1 - t0;
	}

	inline bool IsValid() const
	{
		return t0 != std::numeric_limits<double>::max() && t1 != -std::numeric_limits<double>::max();
	}

	inline Point2 P1() const
	{
		return point + t0 * direction;
	}

	inline Point2 P2() const
	{
		return point + t1 * direction;
	}

	inline class Point2 Center() const
	{
		return (P1() + P2()) / 2.0;
	}

	// Returns a string representation of this line segment.
	inline std::string ToString()
	{
		return P1().ToString() + ", " + P2().ToString();
	}

};

// Writes Point2 p to output stream os.
inline std::ostream & operator<<(std::ostream &os, const LineSegment2 &ls)
{
	os << ls.P1() << "," << ls.P2();
	return os;
}