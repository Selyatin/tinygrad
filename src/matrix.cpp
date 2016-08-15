#include <stdexcept>
#include "matrix.h"

TGMatrix::TGMatrix(unsigned int rows, unsigned int cols, bool reserve_memory) {
    this->cols = cols;
    this->rows = rows;
    this->guarded = false;
    this->gradient = nullptr; // Store the chain rule of gradients with respect to this TGMatrix.
    if (!reserve_memory)
        this->data = nullptr;
    else {
        this->data = new double[this->size()];
        for(int i=0;i<this->size();i++)
            this->data[i] = 0.0;
    }
}

TGMatrix::~TGMatrix(void){
    this->free_contents();
}

void TGMatrix::free_contents(void){
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

unsigned int TGMatrix::size(void){
    return this->rows * this->cols;
}

void TGMatrix::copy_data_from_other_TGMatrix(TGMatrix *src){
    if (src->cols == this->cols && src->rows == this->rows){
        std::copy(src->data, src->data+this->size(), this->data);
    } else
        throw std::invalid_argument("[TGMatrix::copy_data_from_TGMatrix] The TGMatrix dimensions do not match!");
}
