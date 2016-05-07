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
    unsigned int size(void);
    void copy_data_from_other_tensor(Tensor *src);
    Tensor(unsigned int rows, unsigned int cols, bool reserve_memory);
};


#endif //TINYGRAD_TENSOR_H
