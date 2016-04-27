#include <iostream>
#include "node.h"
#include "utils.h"
#include "models.h"
#include "dataset.h"

int main() {
    std::cout << "Evaluating an implementation of a neural network" << std::endl;

    Dataset ds;
    ds.read_csv("dataset.txt");
    unsigned int n_features = ds.features;

    //ClassifierLogisticRegression clf(n_features);
    ClassifierNeuralNetworkSigmoidActivationsOneHiddenLayerOneOutput clf(n_features, 10);

    Tensor *input = create_guarded_tensor_with_random_elements(1, n_features, 0.0, 2.0);

    int epochs = 100;
    double target;
    while(epochs > 0) {
        double correct_classifications = 0.0;
        ds.random_swap(30); // Not exactly shuffling but at least some perturbation into example ordering

        // Read input (x) and target label (y) for classification (two classes).
        for(int example_index=0; example_index<ds.records; example_index++){
            target = ds.y[example_index];
            for(int feature=0; feature<ds.features; feature++){
                input->data[feature] = ds.x[example_index*ds.features+feature];
            }

            // Evaluate a given input
            Tensor *result = clf.evaluate(input);

            // Bookkeeping for computing the resulting accuracy
            if (result->data[0] > 0.5 && target == 1.0)
                correct_classifications++;
            if (result->data[0] < 0.5 && target == 0.0)
                correct_classifications++;

            // Compute stochastic gradient descend with a given learning rate
            clf.sgd(input, target, 0.05);
        }

        std::cout << "Classification accuracy: " << correct_classifications / ds.records << std::endl;
        epochs--;
    }

    delete[] input->data;
    delete input;

    return 0;
}