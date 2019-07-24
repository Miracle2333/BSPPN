#pragma once

#include "BSPTree.h"
#include "Filter.h"
#include "../io/ImageWriter.h"

class BSP
{

public:
	// lineSegments are modified
	static std::vector<Polygon2 *> DeterminePartitions(const std::tuple<int, Polygon2 *> &boundingBox, std::vector<LineSegment2 *> &lineSegments)
	{
		Polygon2 * bb_local = new Polygon2();
		bb_local->ExteriorRing()->AddPoint(Point2(0.0, 0.0));
		bb_local->ExteriorRing()->AddPoint(Point2(std::get<1>(boundingBox)->ExteriorRing()->PointAt(1).x - std::get<1>(boundingBox)->ExteriorRing()->PointAt(0).x, 0.0));
		bb_local->ExteriorRing()->AddPoint(Point2(std::get<1>(boundingBox)->ExteriorRing()->PointAt(1).x - std::get<1>(boundingBox)->ExteriorRing()->PointAt(0).x,
			std::get<1>(boundingBox)->ExteriorRing()->PointAt(2).y - std::get<1>(boundingBox)->ExteriorRing()->PointAt(1).y));
		bb_local->ExteriorRing()->AddPoint(Point2(0.0, std::get<1>(boundingBox)->ExteriorRing()->PointAt(2).y - std::get<1>(boundingBox)->ExteriorRing()->PointAt(1).y));

		if (lineSegments.size() == 0)
		{		
			return std::vector<Polygon2 *>({ bb_local });
		}

		Filter::QuantizeLineDirections(lineSegments);
		//ImageWriter::Debug2Image("data/input/image.tif", std::get<1>(boundingBox), "data/output/tmp/image_" + std::to_string(std::get<0>(boundingBox)) + "_4_LineSegmentsBSP_CLF.tif", lineSegments);

		Filter::MergeParallelLineSegments(lineSegments, 2.5, 0.5);
		//ImageWriter::Debug2Image("data/input/image.tif", std::get<1>(boundingBox), "data/output/tmp/image_" + std::to_string(std::get<0>(boundingBox)) + "_5_LineSegmentsBSP_CLF_Merging.tif", lineSegments);

		BSPTree bsp_tree = BSPTree(bb_local, lineSegments);

		std::vector<Polygon2 *> result;		
		for (Polygon2 *p : bsp_tree.Partitions())
		{
			result.push_back(new Polygon2(p));
		}

		return result;
	}
};
