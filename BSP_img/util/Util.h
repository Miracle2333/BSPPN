#pragma once

#include "../geom/Point2.h"

#include <algorithm>
#include <vector>

namespace util
{

	class Util 
	{

	public:
		static bool PointInBoundingBox(const Point2 &point, Point2 p_min, Point2 p_max, const double enlarged = 0.0)
		{
			p_min -= Point2(enlarged, enlarged);
			p_max += Point2(enlarged, enlarged);

			std::vector<Point2> bb;
			bb.push_back(p_min);
			bb.push_back(Point2(p_max.x, p_min.y));
			bb.push_back(p_max);
			bb.push_back(Point2(p_min.x, p_max.y));

			bool r = false;
			for (int i = 0, j = bb.size() - 1; i < bb.size(); j = i++)
			{
				if ((((bb[i].y <= point.y) && (point.y < bb[j].y)) ||
					((bb[j].y <= point.y) && (point.y < bb[i].y))) &&
					(point.x < (bb[j].x - bb[i].x) * (point.y - bb[i].y) / (bb[j].y - bb[i].y) + bb[i].x))
				{
					r = !r;
				}
			}

			return r;
		}

		static std::vector<std::string> Split(const std::string &s, const std::string &delim, const bool keep_empty = true)
		{
			std::vector<std::string> result;
			if (delim.empty()) {
				result.push_back(s);
				return result;
			}
			std::string::const_iterator substart = s.begin(), subend;
			while (true) {
				subend = search(substart, s.end(), delim.begin(), delim.end());
				std::string temp(substart, subend);
				if (keep_empty || !temp.empty()) {
					result.push_back(temp);
				}
				if (subend == s.end()) {
					break;
				}
				substart = subend + delim.size();
			}
			return result;
		}

	};
}