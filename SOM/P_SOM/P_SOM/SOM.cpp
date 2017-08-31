#include "SOM.h"


void SOM::Create(int cxClient,
	int cyClient,
	int CellsUp,
	int CellsAcross,
	int NumIterations)
{

	cellWidth = (double)cxClient / (double)CellsAcross;

	cellHeight = (double)cyClient / (double)CellsUp;

	numIterations = NumIterations;

	//create all the nodes
	for (int row = 0; row<CellsUp; ++row)
	{
		for (int col = 0; col<CellsAcross; ++col)
		{
			mSOM.push_back(Node(col*cellWidth,           //left
				(col + 1)*cellWidth,       //right
				row*cellHeight,          //top
				(row + 1)*cellHeight,      //bottom
				constSizeOfInputVector));   //num weights
		}
	}

	//this is the topological 'radius' of the feature map
	mapRadius = max(constWindowWidth, constWindowHeight) / 2;

	//used in the calculation of the neighbourhood width of m_dInfluence
	timeConstant = numIterations / log(mapRadius);
}

void SOM::Render(HDC surface)
{
	//render all the cells
	for (int nd = 0; nd<mSOM.size(); ++nd)
	{
		mSOM[nd].Render(surface);

	}

	SetBkMode(surface, TRANSPARENT);
	SetTextColor(surface, RGB(255, 255, 255));

	

	string s = "Iteration: " + intToString(iterationCount);
	TextOut(surface, 5, constWindowHeight - 40, s.c_str(), s.size());

	char* g_szWindowClassName = "MyWindowClass";
	string r = "Press 'R' to retrain";
	TextOut(surface, 260, constWindowHeight - 40, r.c_str(), s.size());

	/*
	s = "Learning: " + ftos(m_dLearningRate);
	TextOut(surface, 5, 20, s.c_str(), s.size());

	s = "Radius: " + ftos(m_dNeighbourhoodRadius);
	TextOut(surface, 5, 40, s.c_str(), s.size());
	*/
}

bool SOM::Epoch(const vector<vector<double> > &data)
{
	
				//make sure the size of the input vector matches the size of each node's 
	//weight vector
	if (data[0].size() != constSizeOfInputVector) return false;

	//return if the training is complete
	if (bDone) return true;


	//enter the training loop
	if (--numIterations > 0)
	{
		//the input vectors are presented to the network at random
		int ThisVector = RandInt(0, data.size() - 1);

		//present the vector to each node and determine the BMU
		winningNode = FindBestMatchingNode(data[ThisVector]);

		//calculate the width of the neighbourhood for this timestep
		neighbourhoodRadius = mapRadius * exp(-(double)iterationCount / timeConstant);

		//Now to adjust the weight vector of the BMU and its
		//neighbours

		//For each node calculate the m_dInfluence (Theta from equation 6 in
		//the tutorial. If it is greater than zero adjust the node's weights
		//accordingly
		for (int n = 0; n<mSOM.size(); ++n)
		{
			//calculate the Euclidean distance (squared) to this node from the
			//BMU
			double DistToNodeSq = (winningNode->X() - mSOM[n].X()) *
				(winningNode->X() - mSOM[n].X()) +
				(winningNode->Y() - mSOM[n].Y()) *
				(winningNode->Y() - mSOM[n].Y());

			double WidthSq = neighbourhoodRadius * neighbourhoodRadius;

			//if within the neighbourhood adjust its weights
			if (DistToNodeSq < (neighbourhoodRadius * neighbourhoodRadius))
			{

				//calculate by how much its weights are adjusted
				influence = exp(-(DistToNodeSq) / (2 * WidthSq));

				mSOM[n].AdjustWeights(data[ThisVector],
					learningRate,
					influence);
			}

		}//next node


		//reduce the learning rate
		learningRate = constStartLearningRate * exp(-(double)iterationCount / numIterations);

		++iterationCount;

	}

	else
	{
		bDone = true;
	
	}

	return true;
}
Node* SOM::FindBestMatchingNode(const vector<double> &vec)
{
	Node* winner = NULL;

	double LowestDistance = 999999;

	for (int n = 0; n<mSOM.size(); ++n)
	{
		double dist = mSOM[n].CalculateDistance(vec);

		if (dist < LowestDistance)
		{
			LowestDistance = dist;

			winner = &mSOM[n];
		}
	}

	return winner;
}

