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

    srand(8462);

    cout << "Evaluating an implementation of logistic regression" << endl;
    Graph g;
    NodeIdentity n_id1;
    NodeMultiplyRightWithMatrix n_mmul1(nullptr);
    NodeElementWiseSigmoidFunction n_sig1;
    NodeSingleSquaredError n_sse(1.0);

    g.add_node(&n_id1);
    g.add_node(&n_mmul1);
    g.add_node(&n_sig1);
    g.add_node(&n_sse);

    g.connect_to(&n_id1, &n_mmul1);
    g.connect_to(&n_mmul1, &n_sig1);
    g.connect_to(&n_sig1, &n_sse);

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
            n_sse.update_target(1.0);
        } else {
            input_data << fRand(0.0, 2.0)-2.0, fRand(0.0, 2.0)-2.0, fRand(0.0, 2.0)-2.0;
            n_sse.update_target(0.0);
        }

        cout << "f(" << input_data << ") = ";
        print_tensor_as_eigen_matrix(g.forward(&t1, &n_id1, &n_sig1));

        g.backward(&n_sse, &n_id1);

        //g.print_contents();

        n_mmul1.mulmat->data[0] = n_mmul1.mulmat->data[0] - 0.1 * n_mmul1.grad[0]->data[0];
        n_mmul1.mulmat->data[1] = n_mmul1.mulmat->data[1] - 0.1 * n_mmul1.grad[0]->data[1];

        g.clean();

        repeats--;
    }

    return 0;
}