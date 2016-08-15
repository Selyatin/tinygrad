#ifndef TINYGRAD_MATRIX_H
#define TINYGRAD_MATRIX_H

class TGMatrix {
public:
    unsigned int cols, rows;
    double *data;
    bool guarded;
    TGMatrix *gradient;
    unsigned int size(void);
    void copy_data_from_other_TGMatrix(TGMatrix *src);
    void free_contents(void);
    TGMatrix(unsigned int rows, unsigned int cols, bool reserve_memory);
    ~TGMatrix(void);
};


#endif //TINYGRAD_MATRIX_H
