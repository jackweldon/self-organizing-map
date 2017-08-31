#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include "Node.h"
#include "SOM.h"
#include <time.h>
#include "constants.h"
#include "resource.h"
#include <stdio.h>
#include <omp.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include "curand.h"

//the width of the matrix (not the number of total elements)
int N = 1600;
//grid and block size
dim3 block(16,16);
dim3 grid(N/block.x, N/block.y);

float cpuStart;
float cpuEnd;
char* g_szApplicationName = "Kohonen Self Organizing Map Demo";
char* g_szWindowClassName = "MyWindowClass";
vector<vector<double>> m_TrainingSet;

void CreateDataSet();
//pointer to a Self Organising Map
SOM*  pSOM = new SOM();
//the data for the training

//used to create the back buffer
static HDC   hdcBackBuffer;
static HBITMAP hBitmap;
static HBITMAP hOldBitmap;

LRESULT CALLBACK WindowProc(HWND   hwnd,
	UINT   msg,
	WPARAM wParam,
	LPARAM lParam)
{

	//these hold the dimensions of the client window area
	static int CxClient, CyClient;

	//used to create the back buffer
	//static HDC   hdcBackBuffer;
	//static HBITMAP hBitmap;
	//static HBITMAP hOldBitmap;


	switch (msg)
	{

	case WM_CREATE:
	{
		//to get get the size of the client window first we need  to create
		//a RECT and then ask Windows to fill in our RECT structure with
		//the client window size. Then we assign to CxClient and CyClient 
		//accordingly
		RECT rect;

		GetClientRect(hwnd, &rect);

		CxClient = rect.right;
		CyClient = rect.bottom;

		//seed random number generator
		srand((unsigned)time(NULL));

		//create a memory device context
		hdcBackBuffer = CreateCompatibleDC(NULL);

		//get the DC for the front buffer
		HDC hdc = GetDC(hwnd);

		hBitmap = CreateCompatibleBitmap(hdc,
			CxClient,
			CyClient);


		//select the bitmap into the memory device context
		hOldBitmap = (HBITMAP)SelectObject(hdcBackBuffer, hBitmap);

		//don't forget to release the DC
		ReleaseDC(hwnd, hdc);

		pSOM->Create(CxClient, CyClient, constNumCellsAcross,
			constNumCellsDown, constNumIterations);
		
	}

		break;
	case WM_KEYUP:
	{
		switch (wParam)
		{
		case VK_ESCAPE:
		{
			SendMessage(hwnd, WM_DESTROY, NULL, NULL);

			PostQuitMessage(0);
		}

			break;

		case 'R':
		{
			delete pSOM;

			pSOM = new SOM();
			pSOM->Create(CxClient, CyClient, constNumCellsAcross,
				constNumCellsDown, constNumIterations);
		}

			break;
		}
	}


	case WM_PAINT:
	{

		PAINTSTRUCT ps;

		BeginPaint(hwnd, &ps);

		//fill the backbuffer with white
		BitBlt(hdcBackBuffer,
			0,
			0,
			CxClient,
			CyClient,
			NULL,
			NULL,
			NULL,
			WHITENESS);

		pSOM->Render(hdcBackBuffer);


		//now blit the backbuffer to the front
		BitBlt(ps.hdc, 0, 0, CxClient, CyClient, hdcBackBuffer, 0, 0, SRCCOPY);

		EndPaint(hwnd, &ps);

	}

		break;

		//has the user resized the client area?
	case WM_SIZE:
	{
		//if so we need to update our variables so that any drawing
		//we do using cxClient and cyClient is scaled accordingly
		CxClient = LOWORD(lParam);
		CyClient = HIWORD(lParam);

		//now to resize the backbuffer accordingly. First select
		//the old bitmap back into the DC
		SelectObject(hdcBackBuffer, hOldBitmap);

		//don't forget to do this or you will get resource leaks
		DeleteObject(hBitmap);

		//get the DC for the application
		HDC hdc = GetDC(hwnd);

		//create another bitmap of the same size and mode
		//as the application
		hBitmap = CreateCompatibleBitmap(hdc,
			CxClient,
			CyClient);

		ReleaseDC(hwnd, hdc);

		//select the new bitmap into the DC
		SelectObject(hdcBackBuffer, hBitmap);

	}

		break;

	case WM_DESTROY:
	{

		//clean up our backbuffer objects
		SelectObject(hdcBackBuffer, hOldBitmap);

		DeleteDC(hdcBackBuffer);
		DeleteObject(hBitmap);

		// kill the application, this sends a WM_QUIT message  
		PostQuitMessage(0);
	}

		break;

	}//end switch

	//this is where all the messages not specifically handled by our 
	//winproc are sent to be processed
	return DefWindowProc(hwnd, msg, wParam, lParam);
}
wchar_t *convertCharArrayToLPCWSTR(const char* charArray)
{
	wchar_t* wString = new wchar_t[4096];
	MultiByteToWideChar(CP_ACP, 0, charArray, -1, wString, 4096);
	return wString;
}

string Convert(float number){
    std::ostringstream buff;
    buff<<number;
    return buff.str();   
}


__device__ int	  RandInt(int x, int y) { return 8 % (y - x + 1) + x; }

__device__ Node* device_findBestMatchingNode(const vector<double> &vec, vector<Node> mSOM)
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
__global__ void device_epoch(vector<vector<double>> data, int numIterations, vector<Node> mSOM,
	int mapRadius, int iterationCount, int timeConstant, int learningRate){
	
	//enter the training loop
	if (--numIterations > 0)
	{
		//the input vectors are presented to the network at random
		int ThisVector = RandInt(0, data.size() - 1);

		//present the vector to each node and determine the BMU
		Node* winningNode = device_findBestMatchingNode(data[ThisVector], mSOM);

		//calculate the width of the neighbourhood for this timestep
		int neighbourhoodRadius = mapRadius * exp(-(double)iterationCount / timeConstant);

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
				int influence = exp(-(DistToNodeSq) / (2 * WidthSq));

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
		//bDone = true;
	
	}
	//return true
}


int WINAPI WinMain(HINSTANCE hInstance,
	HINSTANCE hPrevInstance,
	LPSTR     szCmdLine,
	int       iCmdShow)
{
	CreateDataSet();
	//handle to our window
	HWND           hWnd;

	//our window class structure
	WNDCLASSEX     winclass;

	// first fill in the window class stucture
	winclass.cbSize = sizeof(WNDCLASSEX);
	winclass.style = CS_HREDRAW | CS_VREDRAW;
	winclass.lpfnWndProc = WindowProc;
	winclass.cbClsExtra = 0;
	winclass.cbWndExtra = 0;
	winclass.hInstance = hInstance;
	winclass.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON1));
	winclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	winclass.hbrBackground = NULL;
	winclass.lpszMenuName = NULL;
	winclass.lpszClassName = g_szWindowClassName;
	winclass.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON1));

	//register the window class
	if (!RegisterClassEx(&winclass))
	{
		MessageBox(NULL, TEXT("Registration Failed!"), TEXT("Error"), 0);

		//exit the application
		return 0;
	}

	//create a window with the *client* area specified.
	RECT rect;
	rect.left = 0;
	rect.top = 0;
	rect.bottom = constWindowHeight;
	rect.right = constWindowWidth;

	if (!AdjustWindowRectEx(&rect, CS_HREDRAW | CS_VREDRAW, true, NULL))
	{
		MessageBox(NULL, TEXT("Problem creating window"), TEXT("error!"), MB_OK);
		return 0;
	}

	//create the window and assign its ID to hwnd    
	hWnd = CreateWindowEx(NULL,                 // extended style
		g_szWindowClassName,  // window class name
		g_szApplicationName,  // window caption
		WS_OVERLAPPED | WS_VISIBLE | WS_CAPTION | WS_SYSMENU,
		GetSystemMetrics(SM_CXSCREEN) / 2 - constWindowWidth / 2,
		GetSystemMetrics(SM_CYSCREEN) / 2 - constWindowHeight / 2,
		rect.right,           // initial x size
		rect.bottom,          // initial y size
		NULL,                 // parent window handle
		NULL,                 // window menu handle
		hInstance,            // program instance handle
		NULL);                // creation parameters

	//make sure the window creation has gone OK
	if (!hWnd)
	{
		MessageBox(NULL, TEXT("CreateWindowEx Failed!"), TEXT("Error!"), 0);
	}

	//make the window visible
	ShowWindow(hWnd, iCmdShow);
	UpdateWindow(hWnd);

	// enter the message loop
	bool bDone = false;
	MSG msg;


			int numIterations =  constNumIterations;
			int mapRadius = max(constWindowWidth, constWindowHeight) / 2;
			int timeConstant = numIterations / log(mapRadius);
			int learningRate = constStartLearningRate;
			int iterationCount = 0;
			vector<Node> mSOM = pSOM->getSOM();
	cpuStart= omp_get_wtime();

	while (!bDone)
	{
		while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
			{
				// Stop loop if it's a quit message
				bDone = true;
			}

			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}

		if (!pSOM->FinishedTraining())//if still training map
		{
			

			device_epoch<<<grid,block>>>(m_TrainingSet, numIterations, mSOM, mapRadius, iterationCount, timeConstant,learningRate);
			
			/*vector<vector<double>> data, int numIterations,  vector<Node> mSOM, int mapRadius, 
					int iterationCount, int timeConstant, int learningRate*/


			//this will call WM_PAINT which will render the map
			
			InvalidateRect(hWnd, NULL, TRUE);
			UpdateWindow(hWnd);
		}

		if(pSOM->getIteration() == constNumIterations){
			cpuEnd =  omp_get_wtime();	
			bDone = true; 
		}
	}

		float cpuTime = (cpuEnd-cpuStart);//*1000;

		char str[256];
		char num[256];
		char it[256];
		sprintf_s(str, " CPU Time: %6f \n", cpuTime);
		sprintf_s(num, " Number of Nodes: %d \n",pSOM->getSize());
		sprintf_s(it, " Number of Iterations: %d \n",pSOM->getIteration());

		OutputDebugString("----------CPU TIME-----------\n");
		OutputDebugString(str);
		OutputDebugString("----------Number of Nodes-----------\n");
		OutputDebugString(num);
		OutputDebugString("----------Number of Iterations-----------\n");
		OutputDebugString(it);
		OutputDebugString("-----------------------------\n");
	


	delete pSOM;

	UnregisterClass (g_szWindowClassName, winclass.hInstance);

	return msg.wParam;
}


void Render(HDC surface)
{
	pSOM->Render(surface);
}

inline double RandFloat()		   { return (rand()) / (RAND_MAX + 1.0); }

void CreateDataSet()
{

#ifndef RANDOM_TRAINING_SETS

	//create a data set
	vector<double> red, green, blue, yellow, orange, purple, dk_green, dk_blue;
	//push to back of vector 
	red.push_back(1);
	red.push_back(0);
	red.push_back(0);

	green.push_back(0);
	green.push_back(1);
	green.push_back(0);

	dk_green.push_back(0);
	dk_green.push_back(0.5);
	dk_green.push_back(0.25);

	blue.push_back(0);
	blue.push_back(0);
	blue.push_back(1);

	dk_blue.push_back(0);
	dk_blue.push_back(0);
	dk_blue.push_back(0.5);

	yellow.push_back(1);
	yellow.push_back(1);
	yellow.push_back(0.2);

	orange.push_back(1);
	orange.push_back(0.4);
	orange.push_back(0.25);

	purple.push_back(1);
	purple.push_back(0);
	purple.push_back(1);

	m_TrainingSet.push_back(red);
	m_TrainingSet.push_back(green);
	m_TrainingSet.push_back(blue);
	m_TrainingSet.push_back(yellow);
	m_TrainingSet.push_back(orange);
	m_TrainingSet.push_back(purple);
	m_TrainingSet.push_back(dk_green);
	m_TrainingSet.push_back(dk_blue);


#else

	//choose a random number of training sets
	int NumSets = RandInt(constMinNumTrainingSets, constMaxNumTrainingSets);

	for (int s = 0; s<NumSets; ++s)
	{

		vector<double> set;

		set.push_back(RandFloat());
		set.push_back(RandFloat());
		set.push_back(RandFloat());

		m_TrainingSet.push_back(set);
	}

#endif
}