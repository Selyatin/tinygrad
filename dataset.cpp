//
// Created by niko on 4/26/16.
//

#include "dataset.h"
#include <sstream>
#include <fstream>
#include <iostream>

Dataset::Dataset(void) {
    this->features = 0;
    this->records = 0;
    this->x = nullptr;
    this->y = nullptr;
}

void Dataset::read_csv(std::string filename){
    std::ifstream ifs(filename);
    std::string row;
    while(std::getline(ifs, row)){
        std::stringstream ss(row);
        std::string value;
        while(std::getline(ss, value, ',')){
            std::cout << value << std::endl;
        }
    }
}
