//
// Created by niko on 4/21/16.
//

#ifndef TINYGRAD_TENSOR_H
#define TINYGRAD_TENSOR_H

class Tensor {
public:
    unsigned int cols, rows;
    double *data;
    bool guarded;
    Tensor(unsigned int, unsigned int);
};


#endif //TINYGRAD_TENSOR_H
