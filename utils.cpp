//
// Created by niko on 4/22/16.
//

#include "utils.h"

double fRand_(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

Tensor* copy_eigen_matrix_to_new_tensor(unsigned int h, unsigned int w, double *s){
    Tensor *t_new = new Tensor(h, w);
    t_new->data = new double[h*w];
    memcpy(t_new->data, s, sizeof(double)*h*w);
    return t_new;
}

void print_tensor_as_eigen_matrix(Tensor *t, bool newline){
    if (t == nullptr){
        std::cout << "[print_tensor_as_eigen_matrix] got a null pointer" << std::endl;
        return;
    }
    Eigen::MatrixXd m = Eigen::Map<Eigen::MatrixXd>(t->data, t->rows, t->cols);
    std::cout << m;
    if (newline)
        std::cout << std::endl;
}

Tensor* create_guarded_tensor_with_random_elements(unsigned int h, unsigned int w, double low, double high){
    Tensor *t = new Tensor(h, w);
    t->guarded = true;
    t->data = new double[h*w];
    for(int i=0;i < h*w; i++){
        t->data[i] = fRand_(low, high);
    }
    return t;
}
