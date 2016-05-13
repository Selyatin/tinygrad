//
// Created by niko on 5/5/16.
//

#include <iostream>
#include "node.h"
#include "utils.h"
#include "models.h"
#include "dataset.h"

int main(int argc, char **argv) {
    std::cout << "Evaluating a test implementation of a logistic regression" << std::endl;

    double learning_rate = argc > 1 ? atof(argv[1]) : 0.05;
    std::string dataset_filename = argc > 2 ? argv[2] : "dataset.txt";
    int epochs = argc > 3 ? atoi(argv[3]) : 100;

    Dataset ds;
    ds.read_csv(dataset_filename);
    unsigned int n_features = ds.features;

    std::cout << "dataset: " << dataset_filename << std::endl;
    std::cout << "n_features: " << n_features << std::endl;
    std::cout << "epochs: " << epochs << std::endl;
    std::cout << "learning rate: " << learning_rate << std::endl;

    // Learn a mapping f: R^d -> R
    ClassifierLogisticRegression clf(n_features);

    // Create a tensor for holding the input values (a matrix with n_features elements, R^d)
    Tensor *input = create_guarded_tensor_with_random_elements(1, n_features, -2.0, 2.0);

    // Create a tensor for holding the target values (a matrix with one element, R)
    Tensor *target = create_guarded_tensor_with_random_elements(1, 1, 0.0, 1.0);
    clf.n4->update_target(target);

    while(epochs > 0) {
        double correct_classifications = 0.0;
        ds.random_swap(ds.records); // Not exactly shuffling but at least some perturbation into example ordering

        // Read input (x) and target label (y) for classification (two classes).
        for(int example_index=0; example_index<ds.records; example_index++){

            // Update the target value
            target->data[0] = ds.y[example_index];

            // Update the input values
            for(int feature=0; feature<ds.features; feature++){
                input->data[feature] = ds.x[example_index*ds.features+feature];
            }

            // Evaluate a given input
            Tensor *result = clf.evaluate(input);

            // Bookkeeping for computing the resulting accuracy
            if (result->data[0] > 0.5 && target->data[0] == 1.0)
                correct_classifications++;
            if (result->data[0] < 0.5 && target->data[0] == 0.0)
                correct_classifications++;

            // Compute stochastic gradient descend with a given learning rate
            clf.sgd(input, target, learning_rate);
        }

        std::cout << "Classification accuracy: " << correct_classifications / ds.records << std::endl;
        epochs--;
    }

    input->free_contents();
    target->free_contents();

    delete input;
    delete target;

    return 0;
}