#pragma once

#include "Point2.h"

#include <vector>

class LinearRing2
{

private:
	std::vector<Point2> m_points;

	// ---------------------------------------------------------------------------------------------------- //

public:
	LinearRing2() {}

	LinearRing2(const Point2 first, const Point2 last)
	{
		m_points.push_back(first);
		m_points.push_back(last);
	}

	LinearRing2(const std::vector<Point2> points)
		: m_points(points) {}

	LinearRing2(const LinearRing2 *linearRing2)
	{
		for (const Point2 &p : linearRing2->Points())
			m_points.push_back(p);
	}

	// Destructor
	~LinearRing2() {}

	// ---------------------------------------------------------------------------------------------------- //	

	inline void AddPoint(const Point2 point)
	{
		m_points.push_back(point);
	}

	inline void ChangePointValues(int point_index, Point2 p)
	{
		m_points[point_index] = p;
	}

	inline Point2 PointAt(const int i) const
	{
		return m_points[i];
	}

	inline std::vector<Point2> Points() const
	{
		return m_points;
	}

	inline size_t Size() const
	{
		return m_points.size();
	}

};