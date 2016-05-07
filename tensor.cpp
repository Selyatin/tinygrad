//
// Created by niko on 4/21/16.
//

#include <stdexcept>
#include "tensor.h"

Tensor::Tensor(unsigned int rows, unsigned int cols, bool reserve_memory) {
    this->cols = cols;
    this->rows = rows;
    this->guarded = false;
    if (!reserve_memory)
        this->data = nullptr;
    else
        this->data = new double[this->size()];
}

unsigned int Tensor::size(void){
    return this->rows * this->cols;
}

void Tensor::copy_data_from_other_tensor(Tensor *src){
    if (src->cols == this->cols && src->rows == this->rows){
        std::copy(src->data, src->data+this->size(), this->data);
    } else
        throw std::invalid_argument("[Tensor::copy_data_from_tensor] The tensor dimensions do not match!");
}
