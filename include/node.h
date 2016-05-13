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
    Tensor *buffer;
    Tensor *output;
    Node* in;
    Node* out;
    std::string name;
    void free_buffer_gradient_tensor(void);
    void free_output_tensor(void);
    void free_buffer_tensor(void);
    virtual void calculate_value(void);
    virtual void calculate_gradient(void);
    virtual void combine_upper_gradient(Tensor *upper_gradient);
    void combine_gradient_tensors_to_lower(Tensor *upper, Tensor *lower, Tensor *lower_gradient_shape);
    void connect_to(Node *);
    Node(void);
    ~Node(void);
};


#endif //TINYGRAD_NODE_H
