//
// Created by niko on 4/21/16.
//

#ifndef TINYGRAD_NODEIMPLEMENTATIONS_H
#define TINYGRAD_NODEIMPLEMENTATIONS_H

#include <math.h>
#include <iostream>
#include "node.h"



class NodeElementWiseAddConstant : public Node{
private:
    double addition;
public:
    void calculate_value(void);
    void calculate_gradient(void);
    NodeElementWiseAddConstant(double);
};

class NodeSumAllTensorElements : public Node{
public:
    void calculate_value(void);
    void calculate_gradient(void);
};

class NodeTransposeMatrix : public Node{
public:
    void calculate_value(void);
    void calculate_gradient(void);
};

class NodeMultiplyRightWithMatrix : public Node{
public:
    Tensor *mulmat;
    void calculate_value(void);
    void calculate_gradient(void);
    void update_matrix(Tensor *);
    NodeMultiplyRightWithMatrix(Tensor*);
};

class NodeElementWisePower : public Node{
private:
    double power;
public:
    void calculate_value(void);
    void calculate_gradient(void);
    NodeElementWisePower(double);
};

class NodeElementWiseSigmoidFunction : public Node{
public:
    void calculate_value(void);
    void calculate_gradient(void);
};

class NodeElementWiseLog : public Node{
public:
    void calculate_value(void);
    void calculate_gradient(void);
};

class NodeIdentity : public Node{
public:
    void calculate_value(void);
    void calculate_gradient(void);
};

class NodeElementWiseConstantMultiply : public Node{
private:
    double multiple;
public:
    void calculate_value(void);
    void calculate_gradient(void);
    NodeElementWiseConstantMultiply(double);
};

class NodeSingleSquaredError : public Node{
private:
    double target;
public:
    void calculate_value(void);
    void calculate_gradient(void);
    void update_target(double);
    NodeSingleSquaredError(double);
};

#endif //TINYGRAD_NODEIMPLEMENTATIONS_H
