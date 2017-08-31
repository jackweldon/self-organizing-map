#ifndef SOM_H
#define SOM_H
#include <windows.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <sstream>
#include <omp.h>
#include <string>
#include "constants.h"
#include <omp.h>
using namespace std;

#include "Node.h"


class SOM
{

private:

	
	//the neurons representing the Self Organizing Map
	vector<Node>       mSOM;

	//this holds the address of the winning node from the current iteration
	Node*              winningNode;

	//this is the topological 'radius' of the feature map
	double              mapRadius;

	//used in the calculation of the neighbourhood width of influence
	double              timeConstant;

	//the number of training iterations
	int                 numIterations;

	//keeps track of what iteration the epoch method has reached
	int                 iterationCount;

	//the current width of the winning node's area of influence
	double              neighbourhoodRadius;

	//how much the learning rate is adjusted for nodes within
	//the area of influence
	double              influence;

	double              learningRate;

	//set true when training is finished
	bool                bDone;

	//the height and width of the cells that the nodes occupy when 
	//rendered into 2D space.
	double              cellWidth;
	double              cellHeight;



	Node*    FindBestMatchingNode(const vector<double> &vec);

	inline    double Gaussian(const double dist, const double sigma);


public:
//#define RANDOM_TRAINING_SETS
#ifdef RANDOM_TRAINING_SETS

//	int    constMaxNumTrainingSets = 20;
	
//	int    constMinNumTrainingSets = 5;

#endif


	SOM() :cellWidth(0),
		cellHeight(0),
		winningNode(NULL),
		iterationCount(1),
		numIterations(0),
		timeConstant(0),
		mapRadius(0),
		neighbourhoodRadius(0),
		influence(0),
		learningRate(constStartLearningRate),
		bDone(false)
	{}
	void Create(int cxClient, int cyClient, int CellsUp, int CellsAcross, int NumIterations);

	//the data for the training
	
	 
	void Render(HDC surface);

	inline int getIteration(){return iterationCount;}

	bool Epoch(const vector<vector<double> > &data);
	
	bool FinishedTraining()const{ return bDone; }

	inline int	  RandInt(int x, int y) { return rand() % (y - x + 1) + x; }
	
	inline vector<Node> getSOM(){ return mSOM;}

	inline int getSize(){ return mSOM.size();}

	inline string intToString(int arg)
	{
		ostringstream buffer;

		//send the int to the ostringstream
		buffer << arg;

		//capture the string
		return buffer.str();
	}
};


#endif