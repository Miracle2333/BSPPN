#pragma once

#define _USE_MATH_DEFINES

#include <math.h>
#include <string>
#include <iostream>

class Vector2
{

public:
	double x, y;

	// ---------------------------------------------------------------------------------------------------- //

public:
	// Constructs a vector without explicit initialization.
	inline Vector2() {}

	// Constructs a vector with coordinates (x, y).
	inline Vector2(const double x, const double y)
		: x(x), y(y) {}

	// Constructs a copy of vector v.
	inline Vector2(const Vector2 &v)
		: x(v.x), y(v.y) {}

	// ---------------------------------------------------------------------------------------------------- //

	// Returns the pointer to coordinate i.
	inline double & operator[](int i)
	{
		return ((double *)&x)[i];
	}

	// Returns coordinate i.
	inline double operator[](int i) const
	{
		return ((double *)&x)[i];
	}

	// Returns true if this vector equals vector v.
	inline bool operator==(const Vector2 &v) const
	{
		const double EPSILON = 0.000000000000001;
		return abs(x - v.x) < EPSILON && abs(y - v.y) < EPSILON;
	}

	// Returns false if this vector equals vector v.
	inline bool operator!=(const Vector2 &v) const
	{
		return x != v.x || y != v.y;
	}

	// Returns true if this vector comes before vector v in a non-ambiguous sorting order.
	inline bool operator<(const Vector2 &v) const
	{
		if (x < v.x)
			return true;
		else if (x > v.x)
			return false;
		else
			return y < v.y;
	}

	// Returns the result vector of the addition of this vector and vector v.
	inline Vector2 operator+(const Vector2 &v) const
	{
		return Vector2(x + v.x, y + v.y);
	}

	// Adds vector v to this vector.
	inline Vector2 & operator+=(const Vector2 &v)
	{
		x += v.x;
		y += v.y;
		return *this;
	}

	// Returns the result vector of the subtraction of vector v from this vector.
	inline Vector2 operator-(const Vector2 &v) const
	{
		return Vector2(x - v.x, y - v.y);
	}

	// Subtracts vector v from this vector.
	inline Vector2 & operator-=(const Vector2 &v)
	{
		x -= v.x;
		y -= v.y;
		return *this;
	}

	// Returns the result vector of the multiplication of this vector with scalar s.
	inline Vector2 operator*(const double s) const
	{
		return Vector2(x*s, y*s);
	}

	// Multiplies this vector with scalar s.
	inline Vector2 & operator*=(const double s)
	{
		x *= s;
		y *= s;
		return *this;
	}

	// Returns the result vector of the division of this vector by scalar s.
	inline Vector2 operator/(const double s) const
	{
		double f = 1.0 / s;
		return Vector2(x*f, y*f);
	}

	// Divides this vector by scalar s.
	inline Vector2 & operator/=(const double s)
	{
		double f = 1.0 / s;
		x *= f;
		y *= f;
		return *this;
	}

	// Returns the negative vector of this vector.
	inline Vector2 operator-() const
	{
		return Vector2(-x, -y);
	}

	// ---------------------------------------------------------------------------------------------------- //

	// Returns the length of this vector.
	inline double Length() const
	{
		return sqrt(x * x + y * y);
	}

	// Normalizes this vector to the unit vector.
	inline Vector2 Normalize()
	{
		double norm = 1.0 / this->Length();
		x *= norm;
		y *= norm;
		return *this;
	}

	// Returns the clockwise perpendicular vector.
	inline Vector2 PerpendicularCW() const
	{
		return Vector2(y, -x);
	}

	// Returns the counterclockwise perpendicular vector.
	inline Vector2 PerpendicularCCW() const
	{
		return Vector2(-y, x);
	}

	// Returns a string representation of this vector.
	inline std::string ToString() const
	{
		return std::to_string(x) + " " + std::to_string(y);
	}

};

// ---------------------------------------------------------------------------------------------------- //

// Returns the result vector of the multiplication of scalar s with vector v.
inline Vector2 operator*(const double s, const Vector2 &v)
{
	return Vector2(s*v.x, s*v.y);
}

// Writes vector v to output stream os.
inline std::ostream & operator<<(std::ostream &os, const Vector2 &v)
{
	os << v.x << " " << v.y;
	return os;
}

// ---------------------------------------------------------------------------------------------------- //

// Returns a vector with the absolute values of v.
inline Vector2 Abs(const Vector2 &v)
{
	return Vector2(abs(v.x), abs(v.y));
}

// Returns the cross product of vector a and vector b.
inline Vector2 CrossProduct(const Vector2 &a, const Vector2 &b)
{
	return Vector2(a.y*b.x - a.x*b.y, a.x*b.y - a.y*b.x);
}

// Returns the dot product of vector a and vector b.
inline double DotProduct(const Vector2 &a, const Vector2 &b)
{
	return a.x*b.x + a.y*b.y;
}

// Returns the angle between vector a and vector b in degree.
inline double AngleDegree(const Vector2 &a, const Vector2 &b)
{
	double x = DotProduct(a, b) / (a.Length() * b.Length());

	if (x < -1.0)
		x = -1.0;
	else if (x > 1.0)
		x = 1.0;

	return 180.0 / M_PI * acos(x);
}

// Returns the angle between vector a and b in degree in counterclockwise orientation.
inline double AngleDegree360_CCW(const Vector2 &a, const Vector2 &b)
{
	// angle between 0 and 180
	double angle = AngleDegree(a, b);

	// angle between 0 and 360
	if (DotProduct(Vector2(b.y, -b.x), a) < 0.0)
		angle = 360.0 - angle;

	return angle;
}
