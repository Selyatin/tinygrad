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
    NodeBinaryCrossEntropy *n3;
    unsigned int data_dimensionality;
    Tensor* evaluate(Tensor* input);
    void sgd(Tensor *input, Tensor* target, double lr);
    ClassifierLogisticRegression(unsigned int data_dimensionality);
    ~ClassifierLogisticRegression(void);
};

class AutoencoderSigmoidActivationsOneHiddenLayer {
private:
    Tensor w1;
    Tensor w2;
public:
    Graph g;
    NodeMultiplyRightWithMatrix *n1;
    NodeElementWiseSigmoidFunction *n2;
    NodeMultiplyRightWithMatrix *n3;
    NodeSquaredError *n4;
    unsigned int data_dimensionality;
    unsigned int n_hidden;
    Tensor* evaluate(Tensor*);
    void sgd(Tensor *input, Tensor *target, double lr);
    AutoencoderSigmoidActivationsOneHiddenLayer(unsigned int data_dimensionality, unsigned int n_hidden);
    ~AutoencoderSigmoidActivationsOneHiddenLayer(void);
};

class ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayer {
private:
    Tensor w1;
    Tensor w2;
public:
    Graph g;
    NodeMultiplyRightWithMatrix *n1;
    NodeElementWiseSigmoidFunction *n2;
    NodeMultiplyRightWithMatrix *n3;
    NodeElementWiseSigmoidFunction *n4;
    NodeBinaryCrossEntropy *n5;
    unsigned int data_dimensionality;
    unsigned int n_hidden;
    Tensor* evaluate(Tensor*);
    void sgd(Tensor *input, Tensor *target, double lr);
    ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayer(unsigned int data_dimensionality, unsigned int n_hidden);
    ~ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayer(void);
};

#endif //TINYGRAD_MODELS_H
