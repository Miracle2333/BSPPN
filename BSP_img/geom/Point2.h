#pragma once

#include "Vector2.h"

class Point2 : public Vector2
{

public:
	bool used = false;

public:
	inline Point2() {}

	inline Point2(double x, double y)
		: Vector2(x, y) {}

	inline Point2(const Point2 &p)
		: Vector2(p.x, p.y) {}

	inline Point2(const Vector2 &v)
		: Vector2(v.x, v.y) {}

	// ---------------------------------------------------------------------------------------------------- //

	inline double & operator[](int i)
	{
		return ((double *)&x)[i];
	}

	inline double operator[](int i) const
	{
		return ((double *)&x)[i];
	}

	inline bool operator==(const Point2 &p) const
	{
		return x == p.x && y == p.y;
	}

	inline bool operator!=(const Point2 &p) const
	{
		return x != p.x || y != p.y;
	}

	inline Point2 operator+(const Vector2 &v) const
	{
		return Point2(x + v.x, y + v.y);
	}

	inline Point2 & operator+=(const Vector2 &v)
	{
		x += v.x;
		y += v.y;
		return *this;
	}

	inline Point2 operator-(const Vector2 &v) const
	{
		return Point2(x - v.x, y - v.y);
	}

	inline Point2 & operator-=(const Vector2 &v)
	{
		x -= v.x;
		y -= v.y;
		return *this;
	}

	inline Point2 operator-() const
	{
		return Point2(-x, -y);
	}

	// ---------------------------------------------------------------------------------------------------- //	

	inline std::string ToString() const
	{
		return std::to_string(x) + " " + std::to_string(y);
	}

};

inline double Det(double a, double b, double c, double d)
{
	return a * d - b * c;
}