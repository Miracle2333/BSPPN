#pragma once

#include "../geom/Polygon2.h"
#include "../util/Util.h"

#include <fstream>
#include <tuple>

class BoundingBoxReader
{

public:

	// id min_x min_y max_x max_y
	static std::vector<std::tuple<int, Polygon2 *>> ReadFile(const std::string path)
	{
		std::vector<std::tuple<int, Polygon2 *>> result;

		std::ifstream file(path, std::ios::in);

		std::string fileInput_bb;
		std::vector<std::string> fileInput_bb_data;

		while (file && !file.eof())
		{
			std::getline(file, fileInput_bb);			

			if (!fileInput_bb.empty())
			{
				fileInput_bb_data = util::Util::Split(fileInput_bb, " ", false);

				result.push_back(std::make_tuple(std::stoi(fileInput_bb_data[0]), new Polygon2()));
				std::get<1>(result.back())->ExteriorRing()->AddPoint(Point2((int)(atof(fileInput_bb_data[1].c_str()) + 0.5), (int)(atof(fileInput_bb_data[2].c_str()) + 0.5)));
				std::get<1>(result.back())->ExteriorRing()->AddPoint(Point2((int)(atof(fileInput_bb_data[3].c_str()) + 0.5), (int)(atof(fileInput_bb_data[2].c_str()) + 0.5)));
				std::get<1>(result.back())->ExteriorRing()->AddPoint(Point2((int)(atof(fileInput_bb_data[3].c_str()) + 0.5), (int)(atof(fileInput_bb_data[4].c_str()) + 0.5)));
				std::get<1>(result.back())->ExteriorRing()->AddPoint(Point2((int)(atof(fileInput_bb_data[1].c_str()) + 0.5), (int)(atof(fileInput_bb_data[4].c_str()) + 0.5)));				
			}
		}

		return result;
	}

};