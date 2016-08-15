#ifndef TINYGRAD_NODEIMPLEMENTATIONS_H
#define TINYGRAD_NODEIMPLEMENTATIONS_H

#include <math.h>
#include <iostream>
#include "node.h"

class NodeMultiplyRightWithMatrix : public Node{
public:
    TGMatrix *mulmat;
    void calculate_value(void);
    void calculate_gradient(void);
    void update_matrix(TGMatrix *);
    void combine_upper_gradient(TGMatrix *upper_gradient);
    NodeMultiplyRightWithMatrix(TGMatrix*);
};

class NodeElementWisePower : public Node{
private:
    double power;
public:
    void calculate_value(void);
    void calculate_gradient(void);
    void combine_upper_gradient(TGMatrix *upper_gradient);
    NodeElementWisePower(double);
};

class NodeElementWiseSigmoidFunction : public Node{
public:
    void calculate_value(void);
    void calculate_gradient(void);
    void combine_upper_gradient(TGMatrix *upper_gradient);
    NodeElementWiseSigmoidFunction(void);
};

class NodeElementWiseLog : public Node{
public:
    void calculate_value(void);
    void calculate_gradient(void);
    void combine_upper_gradient(TGMatrix *upper_gradient);
    NodeElementWiseLog(void);
};

class NodeAddTGMatrix : public Node{
public:
    TGMatrix *addition;
    void calculate_value(void);
    void calculate_gradient(void);
    void update_matrix(TGMatrix *);
    void combine_upper_gradient(TGMatrix *upper_gradient);
    NodeAddTGMatrix(TGMatrix*);
};

class NodeSquaredError : public Node{
private:
    TGMatrix *target;
public:
    void calculate_value(void);
    void calculate_gradient(void);
    void update_target(TGMatrix*);
    void combine_upper_gradient(TGMatrix *upper_gradient);
    NodeSquaredError(void);
};

class NodeBinaryCrossEntropy : public Node{
private:
    TGMatrix *target;
public:
    void calculate_value(void);
    void calculate_gradient(void);
    void update_target(TGMatrix*);
    void combine_upper_gradient(TGMatrix *upper_gradient);
    NodeBinaryCrossEntropy(void);
};

#endif //TINYGRAD_NODEIMPLEMENTATIONS_H
