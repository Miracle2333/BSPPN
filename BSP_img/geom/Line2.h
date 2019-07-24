#pragma once

#include "Point2.h"

#include <vector>
#include <limits>

class Line2
{

public:
	Point2 point;
	Vector2 direction;

	std::vector<Point2 *> consensus_set;

	// ---------------------------------------------------------------------------------------------------- //

public:
	inline Line2()
		: point(std::numeric_limits<double>::max(), std::numeric_limits<double>::max()),
		direction(std::numeric_limits<double>::max(), std::numeric_limits<double>::max()),
		consensus_set(std::vector<Point2 *>()) {}

	inline Line2(Point2 point, Vector2 direction)
		: point(point), direction(direction), consensus_set(std::vector<Point2 *>())
	{
		Normalize();
	}

	// ---------------------------------------------------------------------------------------------------- //

	// < 0: left
	// = 0: on the line
	// > 0: right
	double EstimateSide2D(const Point2 p) const
	{
		return direction.x*(point.y - p.y) + direction.y*(p.x - point.x);
	}

	inline void Normalize()
	{
		direction.Normalize();
	}	

	inline Line2 ParallelLine(const double &distance) const
	{
		return Line2(point + distance * direction.PerpendicularCW(), direction);
	}

};

// ---------------------------------------------------------------------------------------------------- //

// Writes Line2 line to output stream os.
inline std::ostream & operator<<(std::ostream &os, const Line2 &line)
{
	os << line.point << " " << line.direction;
	return os;
}
