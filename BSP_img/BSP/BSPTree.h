#pragma once

#include "BSPNode.h"

class BSPTree
{

public:
	BSPNode *root;

	BSPTree(Polygon2 *polygon, std::vector<LineSegment2 *> &partitioning_lines)
		: root(new BSPNode(polygon))
	{
		root->Build_BSPTree(partitioning_lines);
	}

	~BSPTree()
	{
		delete root;
	}

	// ---------------------------------------------------------------------------------------------------- //

	std::vector<Polygon2 *> Partitions() const
	{
		std::vector<Polygon2 *> result = std::vector<Polygon2 *>();

		if (root->child_left == nullptr && root->child_right == nullptr)
		{
			result.push_back(root->polygon);
		}
		else
		{
			std::vector<Polygon2 *> partitions_left = Partitions(root->child_left);
			std::vector<Polygon2 *> partitions_right = Partitions(root->child_right);
			result.insert(result.end(), partitions_left.begin(), partitions_left.end());
			result.insert(result.end(), partitions_right.begin(), partitions_right.end());
		}

		return result;
	}

	std::vector<Polygon2 *> Partitions(const BSPNode *root) const
	{
		std::vector<Polygon2 *> result = std::vector<Polygon2 *>();

		if (root->child_left == nullptr && root->child_right == nullptr)
		{
			result.push_back(root->polygon);
		}
		else
		{
			std::vector<Polygon2 *> partitions_left = Partitions(root->child_left);
			std::vector<Polygon2 *> partitions_right = Partitions(root->child_right);
			result.insert(result.end(), partitions_left.begin(), partitions_left.end());
			result.insert(result.end(), partitions_right.begin(), partitions_right.end());
		}

		return result;
	}

};