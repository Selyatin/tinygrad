#include <iostream>
#include "node.h"
#include "nodeimplementations.h"
#include "graph.h"
#include "utils.h"
#include "models.h"

using namespace std;

double fRand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int main() {
    cout << "tinygrad testing" << endl;

    srand(8458);

    /*
    cout << "Evaluating an implementation of a shallow neural network" << endl;
    Graph g;
    //NodeIdentity n_id1;
    NodeMultiplyRightWithMatrix n_mmul_w1(nullptr);
    NodeElementWiseSigmoidFunction n_sig1;
    NodeMultiplyRightWithMatrix n_mmul_w2(nullptr);
    NodeElementWiseSigmoidFunction n_sig2;
    NodeSingleBinaryCrossEntropy n_bce(1.0);

    //g.add_node(&n_id1);
    g.add_node(&n_mmul_w1);
    g.add_node(&n_sig1);
    g.add_node(&n_mmul_w2);
    g.add_node(&n_sig2);
    g.add_node(&n_bce);

    //g.connect_to(&n_id1, &n_mmul_w1);
    g.connect_to(&n_mmul_w1, &n_sig1);
    g.connect_to(&n_sig1, &n_mmul_w2);
    g.connect_to(&n_mmul_w2, &n_sig2);
    g.connect_to(&n_sig2, &n_bce);

    Tensor *t1 = create_guarded_tensor_with_random_elements(1, 2, 0.0, 2.0); // input
    Tensor *tw1 = create_guarded_tensor_with_random_elements(2, 4, -1.0, 1.0); // W1
    Tensor *tw2 = create_guarded_tensor_with_random_elements(4, 1, -1.0, 1.0); // W2

    n_mmul_w1.update_matrix(tw1);
    n_mmul_w2.update_matrix(tw2);

    int repeats = 200;
    while(repeats > 0) {
        if (repeats % 2 == 0){
            for(int i=0;i<(t1->rows*t1->cols);i++){
                t1->data[i] = fRand(1.0, 2.0);
            }
            n_bce.update_target(1.0);
        } else {
            for(int i=0;i<(t1->rows*t1->cols);i++){
                t1->data[i] = fRand(-2.0, -1.0);
            }
            n_bce.update_target(0.0);
        }

        cout << "f(";
        print_tensor_as_eigen_matrix(t1, false);
        cout << ") = ";
        print_tensor_as_eigen_matrix(g.forward(t1, &n_mmul_w1, &n_sig2), true);

        g.backward(&n_bce, &n_mmul_w1);

        //print_tensor_as_eigen_matrix(tw1, true);
        for(int i=0; i<(n_mmul_w1.mulmat->rows*n_mmul_w1.mulmat->cols); i++){
            n_mmul_w1.mulmat->data[i] = n_mmul_w1.mulmat->data[i] - 0.1 * n_mmul_w1.grad[0]->data[i];
        }
        //cout << endl;
        //print_tensor_as_eigen_matrix(tw1, true);

        //g.print_contents();

        g.clear_gradients();
        g.backward(&n_bce, &n_mmul_w2);

        for(int i=0; i<(n_mmul_w2.mulmat->rows*n_mmul_w2.mulmat->cols); i++) {
            n_mmul_w2.mulmat->data[i] = n_mmul_w2.mulmat->data[i] - 0.1 * n_mmul_w2.grad[0]->data[i];
        }

        //g.print_contents();

        g.clean();
        repeats--;
    }

    delete[] t1->data;
    delete[] tw1->data;
    delete[] tw2->data;
    delete t1;
    delete tw1;
    delete tw2;
    */


    cout << "Evaluating an implementation of logistic regression" << endl;
    unsigned int n_features = 4;
    ClassifierLogisticRegression clf(n_features);
    Tensor *t1 = create_guarded_tensor_with_random_elements(1, n_features, 0.0, 2.0); // input);

    int repeats = 100;
    double target;
    while(repeats > 0) {
        if (repeats % 2 == 0){
            for(int i=0;i<(t1->rows*t1->cols);i++){
                t1->data[i] = fRand(0.0, 1.0);
            }
            target = 1.0;
        } else {
            for(int i=0;i<(t1->rows*t1->cols);i++){
                t1->data[i] = fRand(-1.0, 0.0);
            }
            target = 0.0;
        }

        cout << "f(";
        print_tensor_as_eigen_matrix(t1, false);
        cout << ") = ";
        print_tensor_as_eigen_matrix(clf.evaluate(t1), true);

        clf.sgd(t1, target, 0.1);

        repeats--;
    }

    return 0;
}