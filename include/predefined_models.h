#ifndef TINYGRAD_MODELS_H
#define TINYGRAD_MODELS_H

#include "matrix.h"
#include "graph.h"
#include "nodeimplementations.h"

class ClassifierLogisticRegression {
private:
    TGMatrix weights;
    TGMatrix biases;
public:
    Graph g;
    NodeMultiplyRightWithMatrix *n1;
    NodeAddTGMatrix *n2;
    NodeElementWiseSigmoidFunction *n3;
    NodeBinaryCrossEntropy *n4;
    unsigned int data_dimensionality;
    TGMatrix* evaluate(TGMatrix* input);
    void sgd(TGMatrix *input, TGMatrix* target, double lr);
    ClassifierLogisticRegression(unsigned int data_dimensionality);
    ~ClassifierLogisticRegression(void);
};

class AutoencoderSigmoidActivationsOneHiddenLayer {
private:
    TGMatrix w1;
    TGMatrix w2;
public:
    Graph g;
    NodeMultiplyRightWithMatrix *n1;
    NodeElementWiseSigmoidFunction *n2;
    NodeMultiplyRightWithMatrix *n3;
    NodeSquaredError *n4;
    unsigned int data_dimensionality;
    unsigned int n_hidden;
    TGMatrix* evaluate(TGMatrix*);
    void sgd(TGMatrix *input, TGMatrix *target, double lr);
    AutoencoderSigmoidActivationsOneHiddenLayer(unsigned int data_dimensionality, unsigned int n_hidden);
    ~AutoencoderSigmoidActivationsOneHiddenLayer(void);
};

class ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayer {
private:
    TGMatrix w1;
    TGMatrix w2;
    TGMatrix b1;
    TGMatrix b2;
public:
    Graph g;
    NodeMultiplyRightWithMatrix *n1;
    NodeAddTGMatrix *n2;
    NodeElementWiseSigmoidFunction *n3;
    NodeMultiplyRightWithMatrix *n4;
    NodeAddTGMatrix *n5;
    NodeElementWiseSigmoidFunction *n6;
    NodeBinaryCrossEntropy *n7;
    unsigned int data_dimensionality;
    unsigned int n_hidden;
    TGMatrix* evaluate(TGMatrix*);
    void sgd(TGMatrix *input, TGMatrix *target, double lr);
    ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayer(unsigned int data_dimensionality, unsigned int n_hidden);
    ~ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayer(void);
};

class ClassifierNeuralNetworkSigmoidActivationsTwoHiddenLayers {
private:
    TGMatrix w1;
    TGMatrix w2;
    TGMatrix w3;
public:
    Graph g;
    NodeMultiplyRightWithMatrix *n1;
    NodeElementWiseSigmoidFunction *n2;
    NodeMultiplyRightWithMatrix *n3;
    NodeElementWiseSigmoidFunction *n4;
    NodeMultiplyRightWithMatrix *n5;
    NodeElementWiseSigmoidFunction *n6;
    NodeBinaryCrossEntropy *n7;
    unsigned int data_dimensionality;
    unsigned int n_hidden1;
    unsigned int n_hidden2;
    TGMatrix* evaluate(TGMatrix*);
    void sgd(TGMatrix *input, TGMatrix *target, double lr);
    ClassifierNeuralNetworkSigmoidActivationsTwoHiddenLayers(unsigned int data_dimensionality, unsigned int n_hidden1, unsigned int n_hidden2);
    ~ClassifierNeuralNetworkSigmoidActivationsTwoHiddenLayers(void);
};

#endif //TINYGRAD_MODELS_H
