#ifndef TINYGRAD_UTILS_H
#define TINYGRAD_UTILS_H

#include "matrix.h"
#include <iostream>
#include <string>
#include <Eigen/Core>

double random_double(double fMin, double fMax);
void print_TGMatrix_as_eigen_matrix(TGMatrix *t, bool newline);
void describe_TGMatrix(TGMatrix*, const std::string);
TGMatrix* create_guarded_TGMatrix_with_random_elements(unsigned int h, unsigned int w, double low, double high);

#endif //TINYGRAD_UTILS_H
