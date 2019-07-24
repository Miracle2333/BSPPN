#pragma once

#include "LinearRing2.h"

class Polygon2
{

private:
	LinearRing2 *m_exterior_ring;

	// ---------------------------------------------------------------------------------------------------- //

public:
	Polygon2()
		: m_exterior_ring(new LinearRing2()) {}

	Polygon2(LinearRing2 *exterior_ring)
		: m_exterior_ring(exterior_ring) {}

	Polygon2(const Polygon2 *polygon)
	{
		m_exterior_ring = new LinearRing2(polygon->m_exterior_ring);		
	}

	// Copy Constructor
	Polygon2(const Polygon2 &other)
	{
		m_exterior_ring = new LinearRing2(other.m_exterior_ring);
	}

	// Move Constructor
	Polygon2(Polygon2 &&other)
	{
		m_exterior_ring = other.m_exterior_ring;
		other.m_exterior_ring = nullptr;		
	}

	// Destructor
	~Polygon2()
	{
		delete m_exterior_ring;		
	}

	// ---------------------------------------------------------------------------------------------------- //

	// Move Assignment Operator
	void operator=(Polygon2 &&other)
	{
		m_exterior_ring = other.m_exterior_ring;
		other.m_exterior_ring = nullptr;
	}

	// ---------------------------------------------------------------------------------------------------- //

	inline LinearRing2 * ExteriorRing() const
	{
		return m_exterior_ring;
	}

	inline void SetExteriorRing(LinearRing2 *lr)
	{
		delete m_exterior_ring;
		m_exterior_ring = lr;
	}

};