//
// Created by niko on 4/22/16.
//

#ifndef TINYGRAD_UTILS_H
#define TINYGRAD_UTILS_H

#include "tensor.h"
#include <iostream>
#include <Eigen/Core>

double fRand_(double fMin, double fMax);
void print_tensor_as_eigen_matrix(Tensor *t, bool newline);
Tensor* copy_eigen_matrix_to_new_tensor(unsigned int, unsigned int, double*);
Tensor* create_guarded_tensor_with_random_elements(unsigned int h, unsigned int w, double low, double high);
Tensor* copy_tensor(Tensor *src);

#endif //TINYGRAD_UTILS_H
