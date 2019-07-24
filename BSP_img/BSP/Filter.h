#pragma once

#include "../geom/LineSegment2.h"
#include "../geom/math/Distance.h"

#include <cmath>
#include <map>
#include <set>
#include <tuple>
#include <vector>

class Filter
{

public:
	static void MergeParallelLineSegments(std::vector<LineSegment2 *> &lineSegments, const double distance_parallel_max, const double maximumConsecutivePointDistance = 0.0)
	{
		std::vector<LineSegment2 *> result;
		std::vector<LineSegment2 *> candidates = lineSegments;

		while (!candidates.empty())
		{
			LineSegment2 *longestLineSegment = GetLongestLineSegment(candidates);
			candidates.erase(std::find(candidates.begin(), candidates.end(), longestLineSegment));

			std::vector<LineSegment2 *> longestLineSegment_mergeCandidates;
			for (std::vector<LineSegment2 *>::iterator it = candidates.begin(); it != candidates.end(); )
			{
				if (IsMergeCandidate(longestLineSegment, *it, distance_parallel_max))
				{
					longestLineSegment_mergeCandidates.push_back(*it);
					it = candidates.erase(it);
				}
				else
				{
					it++;
				}
			}

			result.push_back(MergeLineSegmentWithLineSegments(longestLineSegment, longestLineSegment_mergeCandidates, maximumConsecutivePointDistance));

			// add line segments which have not been merged to the set of candidates
			for (LineSegment2 *ls : longestLineSegment_mergeCandidates)
			{
				candidates.push_back(ls);
			}
		}

		lineSegments = result;
	}

	// generate for each section an average line according (sections are determined according to longest line direction)
	static void QuantizeLineDirections(std::vector<LineSegment2 *> &lineSegments, const double quantization = 22.5)
	{
		double main_direction = fmod(AngleDegree(Vector2(1.0, 0.0), GetLongestLineSegment(lineSegments)->direction), quantization);

		for (LineSegment2 *ls : lineSegments)
		{
			ls->Rotate_CCW(-main_direction);
		}

		for (LineSegment2 *ls : lineSegments)
		{
			if (ls->Direction_Angle360_CCW() >= 180.0)
			{
				ls->SwitchDirection();
			}
		}

		// ---------------------------------------------------------------------------------------------------- //

		std::vector<std::vector<LineSegment2 *>> lineSegments_quantized((int)(180.0 / quantization));
		for (LineSegment2 *ls : lineSegments)
		{
			double ls_direction_angle = ls->Direction_Angle();

			int clf_section_num = (int)(ls_direction_angle / quantization);

			lineSegments_quantized[clf_section_num].push_back(ls);
		}

		// ---------------------------------------------------------------------------------------------------- //

		for (std::vector<LineSegment2 *> lineSegments_section : lineSegments_quantized)
		{
			double representative_angle = GetWeightedAverageLineDirection(lineSegments_section);

			for (LineSegment2 *ls : lineSegments_section)
			{
				double ls_direction_angle = ls->Direction_Angle();

				if (ls_direction_angle < representative_angle)
				{
					ls->Rotate_CCW(representative_angle - ls_direction_angle);
				}
				else if (ls_direction_angle > representative_angle)
				{
					ls->Rotate_CCW(360.0 - ls_direction_angle + representative_angle);
				}				
			}
		}

		// ---------------------------------------------------------------------------------------------------- //

		for (LineSegment2 *ls : lineSegments)
		{
			ls->Rotate_CCW(main_direction);
		}		
	}

private:
	static double GetWeightedAverageLineDirection(std::vector<LineSegment2 *> &lineSegments)
	{
		double result = 0;
		double summed_length = 0;

		for (const LineSegment2 *ls : lineSegments)
		{
			result += (ls->Direction_Angle() * ls->Length());
			summed_length += ls->Length();
		}

		return result / summed_length;
	}

	static LineSegment2 * GetLongestLineSegment(const std::vector<LineSegment2 *> &lineSegments)
	{
		double length_max = 0;

		LineSegment2 *result = nullptr;
		for (LineSegment2 *ls : lineSegments)
		{
			if (ls->Length() > length_max)
			{
				result = ls;
				length_max = ls->Length();
			}
		}
		return result;
	}

	static double GetRepresentativeAngle(const double &candidate_angle, const double &reference_angle = 0.0, const double quantization = 22.5)
	{
		double angle = candidate_angle - reference_angle;
		if (angle < 0)
		{
			angle += 360.0;
		}

		int multiplier = (int)(angle / quantization);
		if (fmod(angle, quantization) > quantization / 2.0)
		{
			multiplier++;
		}

		return fmod(reference_angle + multiplier * quantization, 360.0);
	}

	static bool IsMergeCandidate(const LineSegment2 *ls1, const LineSegment2 *ls2, const double distance_parallel_max)
	{
		if (!ls1->IsParallel(*ls2))
		{
			return false;
		}

		Line2 ls1_line = Line2(ls1->point, ls1->direction);
		if (Distance::Distance2D(ls2->P1(), ls1_line) > distance_parallel_max || Distance::Distance2D(ls2->P2(), ls1_line) > distance_parallel_max)
		{
			return false;
		}

		return true;
	}

	// if merged, reference and merged line segments will be deleted at the end
	static LineSegment2 * MergeLineSegmentWithLineSegments(LineSegment2 *reference, std::vector<LineSegment2 *> &lineSegments, const double maximumConsecutivePointDistance = 0.0)
	{
		if (lineSegments.size() == 0)
		{
			return reference;
		}

		std::vector<std::tuple<double, double, LineSegment2 *>> lineSegments_tsOnReferenceLine;
		for (LineSegment2 *ls : lineSegments)
		{
			double t0 = DotProduct(reference->direction, ls->P1() - reference->point) / DotProduct(reference->direction, reference->direction);
			double t1 = DotProduct(reference->direction, ls->P2() - reference->point) / DotProduct(reference->direction, reference->direction);

			if (t0 < t1)
			{
				lineSegments_tsOnReferenceLine.push_back(std::make_tuple(t0, t1, ls));
			}
			else
			{
				lineSegments_tsOnReferenceLine.push_back(std::make_tuple(t1, t0, ls));
			}
		}

		double t_min = std::min(reference->t0, reference->t1);
		double t_max = std::max(reference->t0, reference->t1);
		std::vector<LineSegment2 *> toBeMerged;

		std::sort(lineSegments_tsOnReferenceLine.begin(), lineSegments_tsOnReferenceLine.end(), [](const std::tuple<double, double, LineSegment2 *> &a, const std::tuple<double, double, LineSegment2 *> &b) -> bool
		{
			return std::get<0>(a) < std::get<0>(b);
		});

		// Check if there is any line segment longer than the reference
		for (std::vector<std::tuple<double, double, LineSegment2 *>>::iterator it = lineSegments_tsOnReferenceLine.begin(); it != lineSegments_tsOnReferenceLine.end(); )
		{
			if (std::get<0>(*it) < t_min && std::get<1>(*it) > t_max)
			{
				t_min = std::get<0>(*it);
				t_max = std::get<1>(*it);
				toBeMerged.push_back(std::get<2>(*it));
				it = lineSegments_tsOnReferenceLine.erase(it);
			}
			else
			{
				it++;
			}
		}

		// Check if the line segment can be extended in positive direction				
		for (std::vector<std::tuple<double, double, LineSegment2 *>>::iterator it = lineSegments_tsOnReferenceLine.begin(); it != lineSegments_tsOnReferenceLine.end(); )
		{
			if (std::get<0>(*it) + maximumConsecutivePointDistance >= t_min && std::get<0>(*it) - maximumConsecutivePointDistance <= t_max)
			{
				if (std::get<1>(*it) > t_max)
				{
					t_max = std::get<1>(*it);
				}				
				toBeMerged.push_back(std::get<2>(*it));
				it = lineSegments_tsOnReferenceLine.erase(it);
			}
			else
			{
				it++;
			}
		}

		// Check if the line segment can be extended in positive direction
		std::sort(lineSegments_tsOnReferenceLine.begin(), lineSegments_tsOnReferenceLine.end(), [](const std::tuple<double, double, LineSegment2 *> &a, const std::tuple<double, double, LineSegment2 *> &b) -> bool
		{
			return std::get<1>(a) > std::get<1>(b);
		});
		for (std::vector<std::tuple<double, double, LineSegment2 *>>::iterator it = lineSegments_tsOnReferenceLine.begin(); it != lineSegments_tsOnReferenceLine.end(); )
		{
			if (std::get<1>(*it) + maximumConsecutivePointDistance >= t_min && std::get<1>(*it) - maximumConsecutivePointDistance <= t_max)
			{
				if (std::get<0>(*it) < t_min)
				{
					t_min = std::get<0>(*it);
				}				
				toBeMerged.push_back(std::get<2>(*it));
				it = lineSegments_tsOnReferenceLine.erase(it);
			}
			else
			{
				it++;
			}
		}


		// Calculate new anchor point and averaging line	
		Point2 average_anchorPoint = reference->point * reference->Length();
		double summed_length = reference->Length();

		for (const LineSegment2 *ls : toBeMerged)
		{
			average_anchorPoint += (ls->point * ls->Length());
			summed_length += ls->Length();
		}
		average_anchorPoint /= summed_length;

		Line2 average_line = Line2(average_anchorPoint, reference->direction);		

		// calculate t_min and t_max on average_line
		t_min = DotProduct(average_line.direction, (reference->point + t_min * reference->direction) - average_line.point) / DotProduct(average_line.direction, average_line.direction);
		t_max = DotProduct(average_line.direction, (reference->point + t_max * reference->direction) - average_line.point) / DotProduct(average_line.direction, average_line.direction);

		LineSegment2 *result = new LineSegment2(average_line.point + t_min * average_line.direction, average_line.point + t_max * average_line.direction);
		for (const LineSegment2 *ls : toBeMerged)
		{
			result->consensus_set.insert(result->consensus_set.end(), ls->consensus_set.begin(), ls->consensus_set.end());
		}

		// Delete merged line segments
		delete reference;
		for (LineSegment2 *ls : toBeMerged)
		{
			lineSegments.erase(std::remove(lineSegments.begin(), lineSegments.end(), ls), lineSegments.end());
			delete ls;
		}

		return result;
	}
};