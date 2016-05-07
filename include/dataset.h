//
// Created by niko on 4/26/16.
//

#ifndef TINYGRAD_DATASET_H
#define TINYGRAD_DATASET_H

#include <string>

class Dataset {
public:
    unsigned int records;
    unsigned int features;
    double *x;
    double *y;
    void read_csv(std::string filename);
    void random_swap(unsigned int how_many);
    Dataset(void);
    ~Dataset(void);
};


#endif //TINYGRAD_DATASET_H
