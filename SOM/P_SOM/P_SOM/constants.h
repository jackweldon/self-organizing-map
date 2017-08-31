#ifndef CONSTANTS_H
#define CONSTANTS_H



const int constWindowWidth = 800;
const int constWindowHeight = 800;

const int constNumCellsAcross = 80;
const int constNumCellsDown = 80;


//number of weights each node must contain. One for each element of 
//the input vector. In this example it is 3 because a color is
//represented by its red, green and blue components. (RGB)
const int     constSizeOfInputVector = 3;

//the number of epochs desired for the training
const int    constNumIterations = 1000;

//the value of the learning rate at the start of training
const double constStartLearningRate = 0.1;


#define RANDOM_TRAINING_SETS

#ifdef RANDOM_TRAINING_SETS

const int    constMaxNumTrainingSets = 20;
const int    constMinNumTrainingSets = 5;

#endif

#endif