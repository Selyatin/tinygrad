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
    delete[] this->weights.data;
    delete this->n1;
    delete this->n2;
    delete this->n3;
}
