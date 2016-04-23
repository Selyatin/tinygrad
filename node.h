//
// Created by niko on 4/17/16.
//

#ifndef TINYGRAD_NODE_H
#define TINYGRAD_NODE_H

#include <vector>
#include <utility>
#include <iostream>
#include <Eigen/Core>

#include "tensor.h"

class Node {
public:
    std::vector<Tensor*> grad;
    std::vector<Tensor*> buffer;
    std::vector<Tensor*> output;
    std::vector<Node*> in;
    std::vector<Node*> out;
    virtual void calculate_value(void);
    virtual void calculate_gradient(void);
    unsigned int count_inputs(void);
    void connect_to(Node *);
    Node(void);
};


#endif //TINYGRAD_NODE_H
