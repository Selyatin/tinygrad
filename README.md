# tinygrad
tinygrad evaluates flow graphs and computes their gradients. The scope of tinygrad is to provide a minimalistic C++ implementation of flow graphs (e.g. for embedded platforms). Therefore tinygrad is not a full framework, which implement GPU support and parallel/distributed computation. See Torch, Theano or TensorFlow for a full framework with bells and whistles. The current version of tinygrad utilizes [Eigen](http://eigen.tuxfamily.org/ "Eigen") for linear algebra. However, if needed, the required linear algebra can use any back-end (maybe roll your own for very restricted environments?).

See "examples" folder for using the predefined models (see predefined_models.h):
- Autoencoder
- Neural network
- Logistic regression


## Compile examples
Install Eigen:
- Arch Linux: "pacman -S eigen"
- Manual installation: follow the instructions at [Eigen website](http://eigen.tuxfamily.org/ "Eigen")

Compile the predefined example models:
- cd /to/tinygrad/repository/folder
- chmod +x build.sh
- ./build.sh