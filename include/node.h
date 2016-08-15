#ifndef TINYGRAD_NODE_H
#define TINYGRAD_NODE_H

#include <vector>
#include <utility>
#include <iostream>
#include <string>
#include <Eigen/Core>

#include "matrix.h"

class Node {
public:
    /*
     * How to combine the local gradient and the output gradient:
     *  0: element-wise product
     *  1: matrix multiplication
     */
    unsigned char grad_type;
    TGMatrix *buffer;
    TGMatrix *output;
    Node* in;
    Node* out;
    std::string name;
    void free_buffer_gradient_TGMatrix(void);
    void free_output_TGMatrix(void);
    void free_buffer_TGMatrix(void);
    virtual void calculate_value(void);
    virtual void calculate_gradient(void);
    virtual void combine_upper_gradient(TGMatrix *upper_gradient);
    void combine_gradient_TGMatrices_to_lower(TGMatrix *upper, TGMatrix *lower, TGMatrix *lower_gradient_shape);
    void connect_to(Node *);
    Node(void);
    ~Node(void);
};


#endif //TINYGRAD_NODE_H
