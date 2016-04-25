//
// Created by niko on 4/17/16.
//

#ifndef TINYGRAD_NODE_H
#define TINYGRAD_NODE_H

#include <vector>
#include <utility>
#include <iostream>
#include <string>
#include <Eigen/Core>

#include "tensor.h"

class Node {
public:
    /*
     * How to combine the local gradient and the output gradient:
     *  0: element-wise product
     *  1: matrix multiplication
     */
    unsigned char grad_type;
    std::vector<Tensor*> grad;
    std::vector<Tensor*> buffer;
    std::vector<Tensor*> output;
    std::vector<Node*> in;
    std::vector<Node*> out;
    std::string name;
    virtual void calculate_value(void);
    virtual void calculate_gradient(bool last);
    unsigned int count_inputs(void);
    void connect_to(Node *);
    Node(void);
};


#endif //TINYGRAD_NODE_H
