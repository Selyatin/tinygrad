//
// Created by niko on 4/22/16.
//

#include "utils.h"

Tensor* copy_eigen_matrix_to_new_tensor(unsigned int h, unsigned int w, double *s){
    Tensor *t_new = new Tensor(h, w);
    t_new->data = new double[h*w];
    memcpy(t_new->data, s, sizeof(double)*h*w);
    return t_new;
}

void print_tensor_as_eigen_matrix(Tensor *t){
    if (t == nullptr){
        std::cout << "[print_tensor_as_eigen_matrix] got a null pointer" << std::endl;
        return;
    }
    Eigen::MatrixXd m = Eigen::Map<Eigen::MatrixXd>(t->data, t->rows, t->cols);
    std::cout << m << std::endl;
}
