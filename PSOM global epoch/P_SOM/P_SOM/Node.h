#ifndef Node_H
#define Node_H
#include <windows.h>
#include <vector>
#include <string>
#include "constants.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;


class Node
{

private:

	//this node's weights
	vector<double>    nodeWeights;

	

	//the edges of this node's cell. Each node, when draw to the client 
	//area, is represented as a rectangular cell. The color of the cell 
	//is set to the RGB value its weights represent.
	int               nodeLeftEdge;
	int               nodeTopEdge;
	int               nodeRightEdge;
	int               nodeBottomEdge;



public:

inline double RandFloat()
{ 
	return (rand()) / (RAND_MAX + 1.0); 
}
//its position within the lattice
	double   nodePositionX,nodePositionY;

	    Node(int lft, int rgt, int top, int bot, int NumWeights) :nodeLeftEdge(lft),
		nodeRightEdge(rgt),
		nodeBottomEdge(bot),
		nodeTopEdge(top)

	{
		//initialize the weights to small random variables
		for (int w = 0; w<NumWeights; ++w)
		{
			nodeWeights.push_back(RandFloat());
		}

		//calculate the node's center
		nodePositionX = nodeLeftEdge + (double)(nodeRightEdge - nodeLeftEdge) / 2;
		nodePositionY = nodeTopEdge + (double)(nodeBottomEdge - nodeTopEdge) / 2;
	}



	inline void   Render(HDC surface);

	inline double CalculateDistance(const vector<double> &InputVector);

	inline void   AdjustWeights(const vector<double> &vec,
		const double         LearningRate,
		const double         Influence);

	double X()const{ return nodePositionX; }
	double Y()const{ return nodePositionY; }

};

void Node::Render(HDC surface)
{

	//create a brush and pen of the correct color
	int red = (int)(nodeWeights[0] * 255);
	int green = (int)(nodeWeights[1] * 255);
	int blue = (int)(nodeWeights[2] * 255);

	HBRUSH brush = CreateSolidBrush(RGB(red, green, blue));
	HPEN   pen = CreatePen(PS_SOLID, 1, RGB(red, green, blue));

	HBRUSH OldBrush = (HBRUSH)SelectObject(surface, brush);
	HPEN   OldPen = (HPEN)SelectObject(surface, pen);

	Rectangle(surface, nodeLeftEdge, nodeTopEdge, nodeRightEdge, nodeBottomEdge);

	SelectObject(surface, OldBrush);
	SelectObject(surface, OldPen);

	DeleteObject(brush);
	DeleteObject(pen);

}


__device__ double Node::CalculateDistance(const vector<double> &InputVector)
{
	double distance = 0;

	for (int i = 0; i<nodeWeights.size(); ++i)
	{
		distance += (InputVector[i] - nodeWeights[i]) *
			(InputVector[i] - nodeWeights[i]);
	}

	return distance;
}

__device__ void Node::AdjustWeights(const vector<double> &target, const double LearningRate, const double Influence)
{
	for (int w = 0; w<target.size(); ++w)
	{
		nodeWeights[w] += LearningRate * Influence * (target[w] - nodeWeights[w]);
	}
}



#endif



