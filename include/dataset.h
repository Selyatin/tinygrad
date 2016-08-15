#ifndef TINYGRAD_DATASET_H
#define TINYGRAD_DATASET_H

#include <string>

class Dataset {
public:
    unsigned int records;
    unsigned int features;
    double *x;
    double *y;
    void normalize(void);
    void read_csv(std::string filename);
    void random_swap(unsigned int how_many);
    Dataset(void);
    ~Dataset(void);
};


#endif //TINYGRAD_DATASET_H
