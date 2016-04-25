#include <iostream>
#include "node.h"
#include "nodeimplementations.h"
#include "graph.h"
#include "utils.h"

using namespace std;

double fRand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int main() {
    cout << "tinygrad testing" << endl;

    srand(8458);

    cout << "Evaluating an implementation of a shallow neural network" << endl;
    Graph g;
    NodeIdentity n_id1;
    NodeMultiplyRightWithMatrix n_mmul_w1(nullptr);
    NodeElementWiseSigmoidFunction n_sig1;
    NodeMultiplyRightWithMatrix n_mmul_w2(nullptr);
    NodeElementWiseSigmoidFunction n_sig2;
    NodeSingleBinaryCrossEntropy n_bce(1.0);

    g.add_node(&n_id1);
    g.add_node(&n_mmul_w1);
    g.add_node(&n_sig1);
    g.add_node(&n_mmul_w2);
    g.add_node(&n_sig2);
    g.add_node(&n_bce);

    g.connect_to(&n_id1, &n_mmul_w1);
    g.connect_to(&n_mmul_w1, &n_sig1);
    g.connect_to(&n_sig1, &n_mmul_w2);
    g.connect_to(&n_mmul_w2, &n_sig2);
    g.connect_to(&n_sig2, &n_bce);

    Tensor *t1 = create_guarded_tensor_with_random_elements(1, 3, 0.0, 2.0); // input
    Tensor *tw1 = create_guarded_tensor_with_random_elements(3, 2, 0.0, 1.0); // W1
    Tensor *tw2 = create_guarded_tensor_with_random_elements(2, 1, 0.0, 1.0); // W2

    n_mmul_w1.update_matrix(tw1);
    n_mmul_w2.update_matrix(tw2);

    int repeats = 1;
    while(repeats > 0) {
        if (repeats % 2 == 0){
            for(int i=0;i<(t1->rows*t1->cols);i++){
                t1->data[i] = fRand(0.0, 2.0);
            }
            n_bce.update_target(1.0);
        } else {
            for(int i=0;i<(t1->rows*t1->cols);i++){
                t1->data[i] = fRand(0.0, 2.0) - 2.0;
            }
            n_bce.update_target(0.0);
        }

        cout << "f(";
        print_tensor_as_eigen_matrix(t1, false);
        cout << ") = ";
        print_tensor_as_eigen_matrix(g.forward(t1, &n_id1, &n_sig2), true);

        g.backward(&n_bce, &n_mmul_w1);

        n_mmul_w1.mulmat->data[0] = n_mmul_w1.mulmat->data[0] - 0.1 * n_mmul_w1.grad[0]->data[0];
        n_mmul_w1.mulmat->data[1] = n_mmul_w1.mulmat->data[1] - 0.1 * n_mmul_w1.grad[0]->data[1];
        n_mmul_w1.mulmat->data[2] = n_mmul_w1.mulmat->data[2] - 0.1 * n_mmul_w1.grad[0]->data[2];

        g.print_contents();

        g.clear_gradients();
        g.backward(&n_bce, &n_mmul_w2);

        n_mmul_w2.mulmat->data[0] = n_mmul_w2.mulmat->data[0] - 0.1 * n_mmul_w2.grad[0]->data[0];
        n_mmul_w2.mulmat->data[1] = n_mmul_w2.mulmat->data[1] - 0.1 * n_mmul_w2.grad[0]->data[1];

        g.clean();
        repeats--;
    }

    delete[] t1->data;
    delete[] tw1->data;
    delete[] tw2->data;
    delete t1;
    delete tw1;
    delete tw2;


    /*
    cout << "Evaluating an implementation of logistic regression" << endl;
    Graph g;
    NodeIdentity n_id1;
    NodeMultiplyRightWithMatrix n_mmul1(nullptr);
    NodeElementWiseSigmoidFunction n_sig1;
    NodeSingleBinaryCrossEntropy n_bce(1.0);

    g.add_node(&n_id1);
    g.add_node(&n_mmul1);
    g.add_node(&n_sig1);
    g.add_node(&n_bce);

    g.connect_to(&n_id1, &n_mmul1);
    g.connect_to(&n_mmul1, &n_sig1);
    g.connect_to(&n_sig1, &n_bce);

    Eigen::MatrixXd input_data = Eigen::MatrixXd::Constant(1, 3, 0.0);
    Tensor t1((unsigned int)input_data.rows(), (unsigned int)input_data.cols());
    t1.guarded = true;
    t1.data = input_data.data();

    Eigen::MatrixXd weights = Eigen::MatrixXd::Random(3, 1);
    Tensor t2((unsigned int)weights.rows(), (unsigned int)weights.cols());
    t2.guarded = true;
    t2.data = weights.data();
    n_mmul1.update_matrix(&t2);

    int repeats = 1000;
    while(repeats > 0) {
        if (repeats % 2 == 0){
            input_data << fRand(0.0, 2.0), fRand(0.0, 2.0), fRand(0.0, 2.0);
            n_bce.update_target(1.0);
        } else {
            input_data << fRand(0.0, 2.0)-2.0, fRand(0.0, 2.0)-2.0, fRand(0.0, 2.0)-2.0;
            n_bce.update_target(0.0);
        }

        cout << "f(" << input_data << ") = ";
        print_tensor_as_eigen_matrix(g.forward(&t1, &n_id1, &n_sig1));

        g.backward(&n_bce, &n_id1);

        n_mmul1.mulmat->data[0] = n_mmul1.mulmat->data[0] - 0.1 * n_mmul1.grad[0]->data[0];
        n_mmul1.mulmat->data[1] = n_mmul1.mulmat->data[1] - 0.1 * n_mmul1.grad[0]->data[1];

        g.clean();
        repeats--;
    }
    */

    return 0;
}