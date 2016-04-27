//
// Created by niko on 4/25/16.
//

#include "models.h"

Tensor* ClassifierLogisticRegression::evaluate(Tensor* input){
    this->g.clean();
    return this->g.forward(input, this->n1, this->n2);
}

void ClassifierLogisticRegression::sgd(Tensor *input, double target, double lr){
    this->evaluate(input);
    this->n3->update_target(target);
    this->g.backward(this->n3, this->n1);
    for(int i=0; i<this->data_dimensionality; i++){
        this->weights.data[i] = this->weights.data[i] - lr * this->n1->grad[0]->data[i];
    }
}

ClassifierLogisticRegression::ClassifierLogisticRegression(unsigned int data_dimensionality) : weights(data_dimensionality, 1){
    this->data_dimensionality = data_dimensionality;
    this->weights.data = new double[data_dimensionality];
    this->weights.guarded = true;
    for(int i=0;i < data_dimensionality; i++){
        this->weights.data[i] = (double)rand() / RAND_MAX;
    }
    this->n1 = new NodeMultiplyRightWithMatrix(&this->weights);
    this->n2 = new NodeElementWiseSigmoidFunction;
    this->n3 = new NodeSingleBinaryCrossEntropy(0.0);
    this->g.add_node(this->n1);
    this->g.add_node(this->n2);
    this->g.add_node(this->n3);
    this->g.connect_to(0, 1);
    this->g.connect_to(1, 2);
}

ClassifierLogisticRegression::~ClassifierLogisticRegression(void){
    this->g.clean();
    delete[] this->weights.data;
    delete this->n1;
    delete this->n2;
    delete this->n3;
}

/*
 * A reference implementation of a neural network with one hidden layer, sigmoid
 * activations and one sigmoid output (single activation).
 */
Tensor* ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayerOneOutput::evaluate(Tensor* input){
    this->g.clean();
    return this->g.forward(input, this->n1, this->n4);
}

void ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayerOneOutput::sgd(Tensor *input, double target, double lr){
    this->evaluate(input);
    this->n5->update_target(target);
    this->g.backward(this->n5, this->n1);
    for(int i=0; i<this->w1.cols*this->w1.rows; i++){
        this->w1.data[i] = this->w1.data[i] - lr * this->n1->grad[0]->data[i];
    }
    this->g.clear_gradients();
    this->g.backward(this->n5, this->n3);
    for(int i=0; i<this->w2.cols*this->w2.rows; i++){
        this->w2.data[i] = this->w2.data[i] - lr * this->n3->grad[0]->data[i];
    }
}

ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayerOneOutput::ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayerOneOutput(
        unsigned int data_dimensionality, unsigned int n_hidden) : w1(data_dimensionality, n_hidden), w2(n_hidden, 1) {
    this->data_dimensionality = data_dimensionality;
    this->n_hidden = n_hidden;
    this->w1.guarded = true;
    this->w2.guarded = true;
    this->w1.data = new double[data_dimensionality*n_hidden];
    this->w2.data = new double[n_hidden];
    for(int i=0;i < data_dimensionality*n_hidden; i++){
        this->w1.data[i] = (double)rand() / RAND_MAX;
    }
    for(int i=0;i < n_hidden; i++){
        this->w2.data[i] = (double)rand() / RAND_MAX;
    }
    this->n1 = new NodeMultiplyRightWithMatrix(&this->w1);
    this->n2 = new NodeElementWiseSigmoidFunction;
    this->n3 = new NodeMultiplyRightWithMatrix(&this->w2);
    this->n4 = new NodeElementWiseSigmoidFunction;
    this->n5 = new NodeSingleBinaryCrossEntropy(0.0);
    this->g.add_node(this->n1);
    this->g.add_node(this->n2);
    this->g.add_node(this->n3);
    this->g.add_node(this->n4);
    this->g.add_node(this->n5);
    this->g.connect_to(0, 1);
    this->g.connect_to(1, 2);
    this->g.connect_to(2, 3);
    this->g.connect_to(3, 4);
}

ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayerOneOutput::~ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayerOneOutput(void){
    this->g.clean();
    delete[] this->w1.data;
    delete[] this->w2.data;
    delete this->n1;
    delete this->n2;
    delete this->n3;
    delete this->n4;
    delete this->n5;
}
