# tinygrad
Evaluate flow graphs and compute their gradients. Many methods can be implemented as a flow graph, which include the following examples:
- Neural networks
- Linear classifiers (perceptron, logistic regression, any model with a separating hyperplane)
- Dimensionality reduction

## Install Eigen and compile the example code
- Install Eigen system-wide (Arch Linux: "pacman -S eigen" or manually following the instructions at Eigen website)
- cd /to/tinygrad/repository/folder
- mkdir build
- cd build
- cmake "Unix Makefiles" ..
- make
- cd build
- ./tinygrad
