//
// Created by niko on 5/5/16.
//

#include <iostream>
#include "node.h"
#include "utils.h"
#include "models.h"
#include "dataset.h"

int main(int argc, char **argv) {
    std::cout << "Evaluating an implementation of an autoencoder" << std::endl;

    double learning_rate = argc > 1 ? atof(argv[1]) : 0.05;
    std::string dataset_filename = argc > 2 ? argv[2] : "dataset.txt";
    int epochs = argc > 3 ? atoi(argv[3]) : 100;

    Dataset ds;
    ds.read_csv(dataset_filename);
    unsigned int n_features = ds.features;

    unsigned int n_hidden = argc > 4 ? (unsigned int)atoi(argv[4]) : n_features;

    std::cout << "dataset: " << dataset_filename << std::endl;
    std::cout << "n_features: " << n_features << std::endl;
    std::cout << "n_hidden: " << n_hidden << std::endl;
    std::cout << "epochs: " << epochs << std::endl;
    std::cout << "learning rate: " << learning_rate << std::endl;

    // Learn a mapping f: R^d -> R
    AutoencoderSigmoidActivationsOneHiddenLayer ae(n_features, n_hidden);

    // Create a tensor for holding the input values (a matrix with n_features elements, R^d)
    Tensor *input = create_guarded_tensor_with_random_elements(1, n_features, 0.0, 1.0);

    // Create a tensor for holding the target values (a matrix with n_features element, R^d)
    Tensor *target = create_guarded_tensor_with_random_elements(1, n_features, 0.0, 1.0);
    target->data = new double[n_features];
    ae.n4->update_target(target);

    while(epochs > 0) {
        ds.random_swap(ds.records); // Not exactly shuffling but at least some perturbation into example ordering

        // Mean absolute error
        double avg_are = 0.0;

        // Read input (x) and target vector (x) for reconstruction.
        for(int example_index=0; example_index<ds.records; example_index++){

            // Update the input values
            for(int feature=0; feature<ds.features; feature++){
                input->data[feature] = ds.x[example_index*ds.features+feature];
                target->data[feature] = ds.x[example_index*ds.features+feature];
            }

            // Evaluate a given input
            Tensor *result = ae.evaluate(input);

            // Compute stochastic gradient descend with a given learning rate
            ae.sgd(input, target, learning_rate);

            // Add the absolute reconstruction error of the current training example
            double are = 0.0;
            for(int feature=0; feature<ds.features; feature++){
                are = are + fabs(input->data[feature] - result->data[feature]);
            }
            avg_are = avg_are + are / (double)n_features;
        }

        // Compute the mean absolute error
        avg_are = avg_are / (double)ds.records;

        // Mean absolute error between the inputs and the reconstructed inputs
        std::cout << "MAE: " << avg_are << std::endl;

        epochs--;
    }

    delete[] input->data;
    delete input;

    delete[] target->data;
    delete target;

    return 0;
}