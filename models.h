//
// Created by niko on 4/25/16.
//

#ifndef TINYGRAD_MODELS_H
#define TINYGRAD_MODELS_H

#include "tensor.h"
#include "graph.h"
#include "nodeimplementations.h"

class ClassifierLogisticRegression {
private:
    Tensor weights;
public:
    Graph g;
    NodeMultiplyRightWithMatrix *n1;
    NodeElementWiseSigmoidFunction *n2;
    NodeSingleBinaryCrossEntropy *n3;
    unsigned int data_dimensionality;
    Tensor* evaluate(Tensor* input);
    void sgd(Tensor *input, double target, double lr);
    ClassifierLogisticRegression(unsigned int data_dimensionality);
    ~ClassifierLogisticRegression(void);
};

/*
class ClassifierNeuralNetworkTwoLayers {
public:
    Graph g;
    unsigned int data_dimensionality;
    Tensor* evaluate(Tensor*);
    ClassifierNeuralNetworkTwoLayers(unsigned int data_dimensionality);
};
*/

#endif //TINYGRAD_MODELS_H
