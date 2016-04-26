#include <iostream>
#include "node.h"
#include "utils.h"
#include "models.h"

// fRand originates from http://stackoverflow.com/questions/2704521/generate-random-double-numbers-in-c
double fRand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int main() {
    std::cout << "tinygrad testing" << std::endl;

    srand(8458);

    std::cout << "Evaluating an implementation of logistic regression" << std::endl;
    unsigned int n_features = 4;
    ClassifierLogisticRegression clf(n_features);
    Tensor *input = create_guarded_tensor_with_random_elements(1, n_features, 0.0, 2.0); // input);

    int repeats = 100;
    double target;
    while(repeats > 0) {
        // Create a random input (x) and target label (y) for classification (two classes).
        if (repeats % 2 == 0){
            for(int i=0;i<n_features;i++){
                input->data[i] = fRand(0.0, 1.0);
            }
            target = 1.0;
        } else {
            for(int i=0;i<n_features;i++){
                input->data[i] = fRand(-1.0, 0.0);
            }
            target = 0.0;
        }

        // Evaluate a given input
        std::cout << "f(";
        print_tensor_as_eigen_matrix(input, false);
        std::cout << ") = ";
        print_tensor_as_eigen_matrix(clf.evaluate(input), true);

        // Compute stochastic gradient descend with a given learning rate
        clf.sgd(input, target, 0.1);

        repeats--;
    }

    return 0;
}