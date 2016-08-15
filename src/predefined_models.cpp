#include "predefined_models.h"
#include "utils.h"

TGMatrix* ClassifierLogisticRegression::evaluate(TGMatrix* input){
    this->g.clean();
    return this->g.forward(input, this->n1, this->n3);
}

void ClassifierLogisticRegression::sgd(TGMatrix *input, TGMatrix *target, double lr){
    this->evaluate(input);
    this->n4->update_target(target);

    this->g.backward(this->n4, this->n1);

    for(int i=0; i<this->data_dimensionality; i++){
        this->weights.data[i] = this->weights.data[i] - lr * this->weights.gradient->data[i];
    }

    this->biases.data[0] = this->biases.data[0] - lr * this->biases.gradient->data[0];
}

ClassifierLogisticRegression::ClassifierLogisticRegression(unsigned int data_dimensionality) : weights(data_dimensionality,1,true), biases(1,1,true){
    this->data_dimensionality = data_dimensionality;
    this->weights.guarded = true;
    this->biases.guarded = true;
    for(int i=0;i < data_dimensionality; i++){
        this->weights.data[i] = random_double(-2.0, 2.0);
    }
    this->biases.data[0] = random_double(-2.0, 2.0);
    this->n1 = new NodeMultiplyRightWithMatrix(&this->weights);
    this->n2 = new NodeAddTGMatrix(&this->biases);
    this->n3 = new NodeElementWiseSigmoidFunction;
    this->n4 = new NodeBinaryCrossEntropy();
    this->g.add_node(this->n1);
    this->g.add_node(this->n2);
    this->g.add_node(this->n3);
    this->g.add_node(this->n4);
    this->g.connect_to(0, 1);
    this->g.connect_to(1, 2);
    this->g.connect_to(2, 3);
}

ClassifierLogisticRegression::~ClassifierLogisticRegression(void){
    this->g.clean();
    delete this->n1;
    delete this->n2;
    delete this->n3;
    delete this->n4;
}

/*
 * A reference implementation of an autoencoder with one hidden layer and sigmoid activations.
 */
TGMatrix* AutoencoderSigmoidActivationsOneHiddenLayer::evaluate(TGMatrix* input){
    this->g.clean();
    return this->g.forward(input, this->n1, this->n3);
}

void AutoencoderSigmoidActivationsOneHiddenLayer::sgd(TGMatrix *input, TGMatrix *target, double lr){
    this->evaluate(input);
    this->n4->update_target(target);
    this->g.backward(this->n4, this->n1);
    for(int i=0; i<this->w1.size(); i++){
        this->w1.data[i] = this->w1.data[i] - lr * this->w1.gradient->data[i];
    }
    for(int i=0; i<this->w2.size(); i++){
        this->w2.data[i] = this->w2.data[i] - lr * this->w2.gradient->data[i];
    }
}

AutoencoderSigmoidActivationsOneHiddenLayer::AutoencoderSigmoidActivationsOneHiddenLayer(
        unsigned int data_dimensionality, unsigned int n_hidden) : w1(data_dimensionality, n_hidden, true), w2(n_hidden, data_dimensionality, true) {
    this->data_dimensionality = data_dimensionality;
    this->n_hidden = n_hidden;
    this->w1.guarded = true;
    this->w2.guarded = true;
    for(int i=0;i < data_dimensionality*n_hidden; i++){
        this->w1.data[i] = (double)rand() / RAND_MAX;
        this->w2.data[i] = (double)rand() / RAND_MAX;
    }
    this->n1 = new NodeMultiplyRightWithMatrix(&this->w1);
    this->n2 = new NodeElementWiseSigmoidFunction();
    this->n3 = new NodeMultiplyRightWithMatrix(&this->w2);
    this->n4 = new NodeSquaredError();
    this->g.add_node(this->n1);
    this->g.add_node(this->n2);
    this->g.add_node(this->n3);
    this->g.add_node(this->n4);
    this->g.connect_to(0, 1);
    this->g.connect_to(1, 2);
    this->g.connect_to(2, 3);
}

AutoencoderSigmoidActivationsOneHiddenLayer::~AutoencoderSigmoidActivationsOneHiddenLayer(void){
    this->g.clean();
    delete this->n1;
    delete this->n2;
    delete this->n3;
    delete this->n4;
}


/*
 * A reference implementation of a neural network with one hidden layer, sigmoid
 * activations and one sigmoid output (single activation).
 */
TGMatrix* ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayer::evaluate(TGMatrix* input){
    this->g.clean();
    return this->g.forward(input, this->n1, this->n6);
}

void ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayer::sgd(TGMatrix *input, TGMatrix *target, double lr){
    this->evaluate(input);
    this->n7->update_target(target);
    this->g.backward(this->n7, this->n1);

    for(int i=0; i<this->w1.size(); i++){
        this->w1.data[i] = this->w1.data[i] - lr * this->w1.gradient->data[i];
    }
    for(int i=0; i<this->w2.size(); i++){
        this->w2.data[i] = this->w2.data[i] - lr * this->w2.gradient->data[i];
    }
    for(int i=0; i<this->b1.size(); i++){
        this->b1.data[i] = this->b1.data[i] - lr * this->b1.gradient->data[i];
    }
    this->b2.data[0] = this->b2.data[0] - lr * this->b2.gradient->data[0];
}

ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayer::ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayer(
        unsigned int data_dimensionality, unsigned int n_hidden) : w1(data_dimensionality,n_hidden,true),
                                                                   w2(n_hidden,1,true),
                                                                   b1(1,n_hidden,true),
                                                                   b2(1,1,true){
    this->data_dimensionality = data_dimensionality;
    this->n_hidden = n_hidden;
    this->w1.guarded = true;
    this->w2.guarded = true;
    this->b1.guarded = true;
    this->b2.guarded = true;
    for(int i=0;i < data_dimensionality*n_hidden; i++){
        this->w1.data[i] = random_double(-2.0, 2.0);
    }
    for(int i=0;i < n_hidden; i++){
        this->w2.data[i] = random_double(-2.0, 2.0);
        this->b1.data[i] = random_double(-2.0, 2.0);
    }
    this->b2.data[0] = random_double(-2.0,2.0);
    this->n1 = new NodeMultiplyRightWithMatrix(&this->w1);
    this->n2 = new NodeAddTGMatrix(&this->b1);
    this->n3 = new NodeElementWiseSigmoidFunction();
    this->n4 = new NodeMultiplyRightWithMatrix(&this->w2);
    this->n5 = new NodeAddTGMatrix(&this->b2);
    this->n6 = new NodeElementWiseSigmoidFunction;
    this->n7 = new NodeBinaryCrossEntropy();
    this->g.add_node(this->n1);
    this->g.add_node(this->n2);
    this->g.add_node(this->n3);
    this->g.add_node(this->n4);
    this->g.add_node(this->n5);
    this->g.add_node(this->n6);
    this->g.add_node(this->n7);
    this->g.connect_to(0, 1);
    this->g.connect_to(1, 2);
    this->g.connect_to(2, 3);
    this->g.connect_to(3, 4);
    this->g.connect_to(4, 5);
    this->g.connect_to(5, 6);
}

ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayer::~ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayer(void){
    this->g.clean();
    delete this->n1;
    delete this->n2;
    delete this->n3;
    delete this->n4;
    delete this->n5;
    delete this->n6;
    delete this->n7;
}

/*
 * A reference implementation of a neural network with two hidden layers, sigmoid
 * activations and one sigmoid output (single activation).
 */
TGMatrix* ClassifierNeuralNetworkSigmoidActivationsTwoHiddenLayers::evaluate(TGMatrix* input){
    this->g.clean();
    return this->g.forward(input, this->n1, this->n6);
}

void ClassifierNeuralNetworkSigmoidActivationsTwoHiddenLayers::sgd(TGMatrix *input, TGMatrix *target, double lr){
    this->evaluate(input);
    this->n7->update_target(target);
    this->g.backward(this->n7, this->n1);
    for(int i=0; i<this->w1.size(); i++){
        this->w1.data[i] = this->w1.data[i] - lr * this->w1.gradient->data[i];
    }
    for(int i=0; i<this->w2.size(); i++){
        this->w2.data[i] = this->w2.data[i] - lr * this->w2.gradient->data[i];
    }
    for(int i=0; i<this->w3.size(); i++){
        this->w3.data[i] = this->w3.data[i] - lr * this->w3.gradient->data[i];
    }
}

ClassifierNeuralNetworkSigmoidActivationsTwoHiddenLayers::ClassifierNeuralNetworkSigmoidActivationsTwoHiddenLayers(
        unsigned int data_dimensionality, unsigned int n_hidden1, unsigned int n_hidden2) : w1(data_dimensionality, n_hidden1, true),
                                                                                            w2(n_hidden1, n_hidden2, true),
                                                                                            w3(n_hidden2, 1, true) {
    this->data_dimensionality = data_dimensionality;
    this->n_hidden1 = n_hidden1;
    this->n_hidden2 = n_hidden2;
    this->w1.guarded = true;
    this->w2.guarded = true;
    this->w3.guarded = true;
    for(int i=0;i < data_dimensionality*n_hidden1; i++){
        this->w1.data[i] = random_double(-2.0, 2.0);
    }
    for(int i=0;i < n_hidden1*n_hidden2; i++){
        this->w2.data[i] = random_double(-2.0, 2.0);
    }
    for(int i=0;i < n_hidden2; i++){
        this->w3.data[i] = random_double(-2.0, 2.0);
    }
    this->n1 = new NodeMultiplyRightWithMatrix(&this->w1);
    this->n2 = new NodeElementWiseSigmoidFunction();
    this->n3 = new NodeMultiplyRightWithMatrix(&this->w2);
    this->n4 = new NodeElementWiseSigmoidFunction();
    this->n5 = new NodeMultiplyRightWithMatrix(&this->w3);
    this->n6 = new NodeElementWiseSigmoidFunction();
    this->n7 = new NodeBinaryCrossEntropy();
    this->g.add_node(this->n1);
    this->g.add_node(this->n2);
    this->g.add_node(this->n3);
    this->g.add_node(this->n4);
    this->g.add_node(this->n5);
    this->g.add_node(this->n6);
    this->g.add_node(this->n7);
    this->g.connect_to(0, 1);
    this->g.connect_to(1, 2);
    this->g.connect_to(2, 3);
    this->g.connect_to(3, 4);
    this->g.connect_to(4, 5);
    this->g.connect_to(5, 6);
}

ClassifierNeuralNetworkSigmoidActivationsTwoHiddenLayers::~ClassifierNeuralNetworkSigmoidActivationsTwoHiddenLayers(void){
    this->g.clean();
    delete this->n1;
    delete this->n2;
    delete this->n3;
    delete this->n4;
    delete this->n5;
    delete this->n6;
    delete this->n7;
}
