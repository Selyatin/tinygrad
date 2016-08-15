#include "utils.h"

double random_double(double u, double k){
    return u + ((double)rand() / RAND_MAX) * (u - k);
}

void print_TGMatrix_as_eigen_matrix(TGMatrix *t, bool newline){
    if (t == nullptr){
        std::cout << "[print_TGMatrix_as_eigen_matrix] the provided TGMatrix is a null pointer!" << std::endl;
        return;
    }
    if (t->data == nullptr){
        std::cout << "[print_TGMatrix_as_eigen_matrix] the data (TGMatrix->data) of the provided TGMatrix is a null pointer!" << std::endl;
        return;
    }
    Eigen::MatrixXd m = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> >(t->data, t->rows, t->cols);
    std::cout << m;
    if (newline)
        std::cout << std::endl;
}

void describe_TGMatrix(TGMatrix* src, const std::string name){
    std::cout << "-----" << std::endl;
    if (src == nullptr)
        std::cout << "TGMatrix '" << name << "' is a null pointer!" << std::endl;
    else {
        std::cout << "TGMatrix '" << name << "' (" << src << ") rows: " << src->rows << ", cols: " << src->cols << std::endl;
        if (src->data == nullptr)
            std::cout << "The data of the TGMatrix is a null pointer!" << std::endl;
        else {
            print_TGMatrix_as_eigen_matrix(src, true);
        }
        if (src->gradient == nullptr)
            std::cout << "The gradient pointer of the TGMatrix is a null pointer!" << std::endl;
        else {
            if (src->gradient->data == nullptr)
                std::cout << "The data of the gradient TGMatrix is a null pointer!" << std::endl;
            else {
                std::cout << "Gradient:" << std::endl;
                print_TGMatrix_as_eigen_matrix(src->gradient, true);
            }
        }
    }
    std::cout << "-----" << std::endl;
}

TGMatrix* create_guarded_TGMatrix_with_random_elements(unsigned int h, unsigned int w, double low, double high){
    TGMatrix *t = new TGMatrix(h, w, true);
    t->guarded = true;
    for(int i=0;i < t->size(); i++){
        t->data[i] = random_double(low, high);
    }
    return t;
}
