//
// Created by niko on 4/21/16.
//

#ifndef TINYGRAD_NODEIMPLEMENTATIONS_H
#define TINYGRAD_NODEIMPLEMENTATIONS_H

#include <math.h>
#include <iostream>
#include "node.h"

class NodeMultiplyRightWithMatrix : public Node{
public:
    Tensor *mulmat;
    void calculate_value(void);
    void calculate_gradient(void);
    void update_matrix(Tensor *);
    void combine_upper_gradient(Tensor *upper_gradient);
    NodeMultiplyRightWithMatrix(Tensor*);
};

class NodeElementWisePower : public Node{
private:
    double power;
public:
    void calculate_value(void);
    void calculate_gradient(void);
    void combine_upper_gradient(Tensor *upper_gradient);
    NodeElementWisePower(double);
};

class NodeElementWiseSigmoidFunction : public Node{
public:
    void calculate_value(void);
    void calculate_gradient(void);
    void combine_upper_gradient(Tensor *upper_gradient);
    NodeElementWiseSigmoidFunction(void);
};

class NodeElementWiseLog : public Node{
public:
    void calculate_value(void);
    void calculate_gradient(void);
    void combine_upper_gradient(Tensor *upper_gradient);
    NodeElementWiseLog(void);
};

class NodeAddTensor : public Node{
public:
    Tensor *addition;
    void calculate_value(void);
    void calculate_gradient(void);
    void update_matrix(Tensor *);
    void combine_upper_gradient(Tensor *upper_gradient);
    NodeAddTensor(Tensor*);
};

class NodeSquaredError : public Node{
private:
    Tensor *target;
public:
    void calculate_value(void);
    void calculate_gradient(void);
    void update_target(Tensor*);
    void combine_upper_gradient(Tensor *upper_gradient);
    NodeSquaredError(void);
};

class NodeBinaryCrossEntropy : public Node{
private:
    Tensor *target;
public:
    void calculate_value(void);
    void calculate_gradient(void);
    void update_target(Tensor*);
    void combine_upper_gradient(Tensor *upper_gradient);
    NodeBinaryCrossEntropy(void);
};

#endif //TINYGRAD_NODEIMPLEMENTATIONS_H
