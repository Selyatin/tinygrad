//
// Created by niko on 4/21/16.
//

#include "tensor.h"

Tensor::Tensor(unsigned int rows, unsigned int cols) {
    this->data = nullptr;
    this->cols = cols;
    this->rows = rows;
    this->guarded = false;
}