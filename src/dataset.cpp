#include "dataset.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <Eigen/Core>

using namespace Eigen;

Dataset::Dataset(void) {
    this->features = 0;
    this->records = 0;
    this->x = nullptr;
    this->y = nullptr;
}

Dataset::~Dataset(void){
    if (this->x != nullptr)
        delete[] this->x;
    if (this->y != nullptr)
        delete[] this->y;
}

void Dataset::normalize(void){
    MatrixXd m = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->x,this->records,this->features);
    RowVectorXd mean = m.colwise().mean();
    RowVectorXd var = (m.rowwise() - mean).array().square().colwise().mean().sqrt();
    m = (m.rowwise() - mean).array().rowwise() / var.array();
}

void Dataset::read_csv(std::string filename){
    std::ifstream ifs(filename);
    std::string row;
    while(std::getline(ifs, row)){
        this->records++;
        if (this->records == 1) {
            std::stringstream ss(row);
            std::string value;
            while (std::getline(ss, value, ',')) {
                this->features++;
            }
            this->features--; // Remove the target label
        }
    }
    ifs.clear();
    ifs.seekg(0, std::ios::beg);
    this->x = new double[this->records*this->features];
    this->y = new double[this->records];
    int index = 0;
    while(std::getline(ifs, row)){
        std::stringstream ss(row);
        std::string value;
        int feature_index = 0;
        while(std::getline(ss, value, ',')){
            if (feature_index < this->features)
                this->x[index * this->features + feature_index] = std::stod(value);
            else
                this->y[index] = std::stod(value);
            feature_index++;
        }
        index++;
    }
}

void Dataset::random_swap(unsigned int how_many){
    double swap_holder_x, swap_holder_y;
    for(int i=0; i<how_many; i++){
        int index_a = rand() % this->records;
        int index_b = rand() % this->records;
        swap_holder_y = this->y[index_a];
        for(int j=0; j<this->features; j++){
            swap_holder_x = this->x[index_b * this->features + j];
            this->x[index_b * this->features + j] = this->x[index_a * this->features + j];
            this->x[index_a * this->features + j] = swap_holder_x;
        }
        this->y[index_a] = this->y[index_b];
        this->y[index_b] = swap_holder_y;
    }
}
