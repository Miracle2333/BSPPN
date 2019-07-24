#pragma once

#include "../geom/LineSegment2.h"
#include "../geom/math/Distance.h"

#include <list>
#include <random>
#include <tuple>

class RANSAC
{

public:
	static std::vector<LineSegment2 *> DetectLineSegments(std::vector<Point2 *> &points,
														  const double maxDistance2line = 1.5, const int iteration_num = 100,
														  const double maximumConsecutivePointDistance = 1.5, const double minLength = 1.0)
	{
		std::vector<LineSegment2 *> result;

		do
		{
			result.push_back(new LineSegment2(GetLineSegment(points, maxDistance2line, iteration_num, maximumConsecutivePointDistance, minLength)));

			for (Point2 *p : result.back()->consensus_set)
			{
				p->used = true;
			}
		} while (result.back()->IsValid());
		
		delete result.back();
		result.pop_back();

		return result;
	}

	// ---------------------------------------------------------------------------------------------------- //

private:
	static LineSegment2 GetLineSegment(std::vector<Point2 *> &points, const double maxDistance2line, const unsigned int iteration_num,
									   const double maximumConsecutivePointDistance, const double minLength = 0.0)
	{
		LineSegment2 result = LineSegment2();		

		for (unsigned int i = 0; i < iteration_num; i++)
		{
			std::shuffle(points.begin(), points.end(), std::default_random_engine{});

			Line2 candidate_line = Line2(*points[0], *points[1] - *points[0]);
			for (Point2 *p : points)
			{
				if (!p->used && Distance::Distance2D(*p, candidate_line) <= maxDistance2line)
				{
					candidate_line.consensus_set.push_back(p);
				}
			}

			LineSegment2 candidate_maxLineSegment = GetMaxLineSegment(candidate_line, maximumConsecutivePointDistance, minLength);

			if (!result.IsValid() || candidate_maxLineSegment.Length() > result.Length())
			{
				result = candidate_maxLineSegment;
			}
		}

		return result;
	}

	// ---------------------------------------------------------------------------------------------------- //

	// Returns a sorted list of T values (perpendicular points on line)
	static std::vector<std::tuple<double, Point2 *>> CollectTs(const Line2 line)
	{
		std::vector<std::tuple<double, Point2 *>> ts;
		double t;

		for (Point2 *p : line.consensus_set)
		{
			Distance::Distance2D(*p, line, &t);
			ts.push_back(std::make_tuple(t, p));
		}
		
		std::sort(ts.begin(), ts.end(),
			[](const std::tuple<double, Point2 *> &a,
			   const std::tuple<double, Point2 *> &b) -> bool
		{
			return std::get<0>(a) < std::get<0>(b);
		});

		return ts;
	}

	static LineSegment2 GetMaxLineSegment(const Line2 &line, const double maximumConsecutivePointDistance, const double minLength = 0.0)
	{
		std::vector<LineSegment2> lineSegments = TraceLineSegmentsFromTs(line, CollectTs(line), maximumConsecutivePointDistance, minLength);

		LineSegment2 result = LineSegment2();
		double length_max = 0;
		for (const LineSegment2 &ls : lineSegments)
		{
			if (ls.Length() > length_max)
			{
				length_max = ls.Length();
				result = ls;
			}
		}

		return result;
	}

	// Returns a list of line segments, where each line segment is supported by a sequence of Ts (points) that 
	// are not further apart from each other than maximumConsecutivePointDistance. Or the other way round: the 
	// line is split where two consecutive Ts (points) are further apart than maximumConsecutivePointDistance. 
	// Only valid line segments are generated, i.e. zero length line segments are not generated (from a single 
	// point).
	static std::vector<LineSegment2> TraceLineSegmentsFromTs(const Line2 &line, const std::vector<std::tuple<double, Point2 *>> &ts, const double maximumConsecutivePointDistance, const double minLength)
	{
		std::vector<LineSegment2> lineSegments;

		auto ItS = ts.begin();
		while (ItS != ts.end())
		{
			auto ItE1 = ItS;
			auto ItE2 = ItS;
			ItE2++;

			while (ItE2 != ts.end() && std::get<0>(*ItE2) - std::get<0>(*ItE1) <= maximumConsecutivePointDistance)
			{
				ItE1++;
				ItE2++;
			}

			if (std::get<0>(*ItE1) - std::get<0>(*ItS) > 0 &&
				std::get<0>(*ItE1) - std::get<0>(*ItS) >= minLength)
			{
				lineSegments.push_back(LineSegment2(line.point + std::get<0>(*ItS) * line.direction, line.point + std::get<0>(*ItE1) * line.direction));
				lineSegments.back().consensus_set.clear();

				do
				{
					lineSegments.back().consensus_set.push_back(std::get<1>(*ItS));
					ItS++;
				} while (ItS != ItE1);
			}

			ItS = ItE2;
		}

		return lineSegments;
	}

};