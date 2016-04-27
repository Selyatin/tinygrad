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

class ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayerOneOutput {
private:
    Tensor w1;
    Tensor w2;
public:
    Graph g;
    NodeMultiplyRightWithMatrix *n1;
    NodeElementWiseSigmoidFunction *n2;
    NodeMultiplyRightWithMatrix *n3;
    NodeElementWiseSigmoidFunction *n4;
    NodeSingleBinaryCrossEntropy *n5;
    unsigned int data_dimensionality;
    unsigned int n_hidden;
    Tensor* evaluate(Tensor*);
    void sgd(Tensor *input, double target, double lr);
    ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayerOneOutput(unsigned int data_dimensionality, unsigned int n_hidden);
    ~ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayerOneOutput(void);
};

#endif //TINYGRAD_MODELS_H
