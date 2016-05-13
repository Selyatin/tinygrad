//
// Created by niko on 4/21/16.
//

#include <stdexcept>
#include "tensor.h"

Tensor::Tensor(unsigned int rows, unsigned int cols, bool reserve_memory) {
    this->cols = cols;
    this->rows = rows;
    this->guarded = false;
    this->gradient = nullptr; // Store the chain rule of gradients with respect to this tensor.
    if (!reserve_memory)
        this->data = nullptr;
    else {
        this->data = new double[this->size()];
        for(int i=0;i<this->size();i++)
            this->data[i] = 0.0;
    }
}

Tensor::~Tensor(void){
    this->free_contents();
}

void Tensor::free_contents(void){
    if (this->data != nullptr){
        delete[] this->data;
        this->data = nullptr;
    }
    if (this->gradient != nullptr){
        if (this->gradient->data != nullptr){
            delete[] this->gradient->data;
            this->gradient->data = nullptr;
        }
        delete this->gradient;
        this->gradient = nullptr;
    }
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
