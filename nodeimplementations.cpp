//
// Created by niko on 4/21/16.
//

#include "nodeimplementations.h"
#include "utils.h"

/*
 * Compute a specified element-wise power for the elements of all tensors.
 */
void NodeElementWisePower::calculate_value(void){
    this->output.clear();
    for(int i=0;i<this->buffer.size();i++){
        Tensor *t_new = copy_eigen_matrix_to_new_tensor(this->buffer[i]->rows, this->buffer[i]->cols, this->buffer[i]->data);
        for(int j=0;j<this->buffer[i]->rows * this->buffer[i]->cols;j++){
            t_new->data[j] = pow(t_new->data[j], this->power);
        }
        this->output.push_back(t_new);
    }
}

void NodeElementWisePower::calculate_gradient(bool last){
    for(int i=0;i<this->buffer.size();i++){
        Tensor *t_new = copy_eigen_matrix_to_new_tensor(this->buffer[i]->rows, this->buffer[i]->cols, this->buffer[i]->data);
        for(int j=0;j<this->buffer[i]->rows * this->buffer[i]->cols;j++){
            t_new->data[j] = this->power * pow(t_new->data[j], this->power - 1.0);
        }
        this->grad.push_back(t_new);
    }
}

NodeElementWisePower::NodeElementWisePower(double power) : Node() {
    this->power = power;
}

/*
 * Sum the values of all elements of all tensors into a single scalar.
 */
void NodeSumAllTensorElements::calculate_value(void){
    this->output.clear();
    double total = 0.0;
    for(int i=0;i<this->buffer.size();i++){
        Tensor *t = this->buffer[i];
        Eigen::MatrixXd m = Eigen::Map<Eigen::MatrixXd>(t->data, t->rows, t->cols);
        total = total + m.sum();
    }
    Tensor *t_new = copy_eigen_matrix_to_new_tensor(1, 1, new double[1]);
    *t_new->data = total;
    this->output.push_back(t_new);
}

void NodeSumAllTensorElements::calculate_gradient(bool last){
    for(int i=0;i<this->buffer.size();i++){
        Tensor *t = this->buffer[i];
        Eigen::MatrixXd m = Eigen::MatrixXd::Constant(t->rows, t->cols, 1.0);
        Tensor *t_new = copy_eigen_matrix_to_new_tensor(t->rows, t->cols, m.data());
        this->grad.push_back(t_new);
    }
}

/*
 * Add a constant to the elements of all tensors.
 */
void NodeElementWiseAddConstant::calculate_value(void){
    this->output.clear();
    for(int i=0;i<this->buffer.size();i++){
        Tensor *t = this->buffer[i];
        Eigen::MatrixXd m = Eigen::Map<Eigen::MatrixXd>(t->data, t->rows, t->cols);
        Eigen::MatrixXd ms = Eigen::MatrixXd::Constant(t->rows, t->cols, this->addition);
        m = m + ms;
        Tensor *t_new = copy_eigen_matrix_to_new_tensor(t->rows, t->cols, m.data());
        this->output.push_back(t_new);
    }
}

void NodeElementWiseAddConstant::calculate_gradient(bool last){
    this->output.clear();
    for(int i=0;i<this->buffer.size();i++){
        Tensor *t = this->buffer[i];
        Eigen::MatrixXd ms = Eigen::MatrixXd::Constant(t->rows, t->cols, 1.0);
        Tensor *t_new = copy_eigen_matrix_to_new_tensor(t->rows, t->cols, ms.data());
        this->grad.push_back(t_new);
    }
}

NodeElementWiseAddConstant::NodeElementWiseAddConstant(double addition) : Node() {
    this->addition = addition;
}

/*
 * Compute the sigmoid/logistic function element-wise for the elements of the all tensors.
 */
void NodeElementWiseSigmoidFunction::calculate_value(void){
    this->output.clear();
    for(int i=0;i<this->buffer.size();i++){
        Tensor *t_new = copy_eigen_matrix_to_new_tensor(this->buffer[i]->rows, this->buffer[i]->cols, this->buffer[i]->data);
        for(int j=0;j<this->buffer[i]->rows * this->buffer[i]->cols;j++){
            t_new->data[j] = 1.0 / (1.0 + exp((-1.0)*t_new->data[j]));
        }
        this->output.push_back(t_new);
    }
}

void NodeElementWiseSigmoidFunction::calculate_gradient(bool last){
    for(int i=0;i<this->output.size();i++){
        Tensor *t_new = copy_eigen_matrix_to_new_tensor(this->output[i]->rows, this->output[i]->cols, this->output[i]->data);
        for(int j=0;j<this->output[i]->rows * this->output[i]->cols;j++){
            t_new->data[j] = t_new->data[j] * (1.0 - t_new->data[j]);
        }
        this->grad.push_back(t_new);
    }
}

/*
 * Multiply the elements of all tensors by a constant value.
 */
void NodeElementWiseConstantMultiply::calculate_value(void){
    this->output.clear();
    for(int i=0;i<this->buffer.size();i++){
        Tensor *t = this->buffer[i];
        Eigen::MatrixXd m = Eigen::Map<Eigen::MatrixXd>(t->data, t->rows, t->cols);
        m = m * this->multiple;
        Tensor *t_new = copy_eigen_matrix_to_new_tensor(t->rows, t->cols, m.data());
        this->output.push_back(t_new);
    }
}

void NodeElementWiseConstantMultiply::calculate_gradient(bool last){
    for(int i=0;i<this->buffer.size();i++){
        Tensor *t = this->buffer[i];
        Eigen::MatrixXd m = Eigen::MatrixXd::Constant(t->rows, t->cols, this->multiple);
        Tensor *t_new = copy_eigen_matrix_to_new_tensor(t->rows, t->cols, m.data());
        this->grad.push_back(t_new);
    }
}

NodeElementWiseConstantMultiply::NodeElementWiseConstantMultiply(double multiple) {
    this->multiple = multiple;
}

/*
 * Compute the logarithm element-wise the elements of a tensor.
 */
void NodeElementWiseLog::calculate_value(void){
    this->output.clear();
    for(int i=0;i<this->buffer.size();i++){
        Tensor *t_new = copy_eigen_matrix_to_new_tensor(this->buffer[i]->rows, this->buffer[i]->cols, this->buffer[i]->data);
        for(int j=0;j<this->buffer[i]->rows * this->buffer[i]->cols;j++){
            if (t_new->data[j] < 0.0)
                throw std::invalid_argument("Logarithm of a negative real value is not defined.");
            t_new->data[j] = log(t_new->data[j]);
        }
        this->output.push_back(t_new);
    }
}

void NodeElementWiseLog::calculate_gradient(bool last){
    for(int i=0;i<this->buffer.size();i++){
        Tensor *t_new = copy_eigen_matrix_to_new_tensor(this->buffer[i]->rows, this->buffer[i]->cols, this->buffer[i]->data);
        for(int j=0;j<this->buffer[i]->rows * this->buffer[i]->cols;j++){
            t_new->data[j] = 1.0 / t_new->data[j];
        }
        this->grad.push_back(t_new);
    }
}

/*
 * Identity operator
 */
void NodeIdentity::calculate_value(void){
    this->output.clear();
    for(int i=0;i<this->buffer.size();i++){
        Tensor *t_new = copy_eigen_matrix_to_new_tensor(this->buffer[i]->rows, this->buffer[i]->cols, this->buffer[i]->data);
        this->output.push_back(t_new);
    }
}

void NodeIdentity::calculate_gradient(bool last){
    Tensor *t1 = this->buffer[0];
    unsigned int h=t1->rows, w=t1->cols;
    Eigen::MatrixXd m = Eigen::MatrixXd::Constant(h, w, 1.0);
    Tensor *t_new = copy_eigen_matrix_to_new_tensor(h, w, m.data());
    this->grad.push_back(t_new);
}

/*
 * Multiply one matrix with a given matrix
 */
void NodeMultiplyRightWithMatrix::calculate_value(void){
    if (this->buffer.size() != 1)
        throw std::invalid_argument("NodeMultiplyWithMatrix: the number of matrices in buffer is not one.");

    this->output.clear();
    Tensor *t = this->buffer[0];
    unsigned int h1=t->rows, w1=t->cols, h2=this->mulmat->rows, w2=this->mulmat->cols;

    if (w1 != h2)
        throw std::invalid_argument("NodeMultiplyWithMatrix: the matrix dimensions do not match for multiply.");

    Eigen::MatrixXd m1 = Eigen::Map<Eigen::MatrixXd>(t->data, h1, w1);
    Eigen::MatrixXd m2 = Eigen::Map<Eigen::MatrixXd>(this->mulmat->data, h2, w2);
    Eigen::MatrixXd m3 = m1 * m2;

    Tensor *t_new = copy_eigen_matrix_to_new_tensor(h1, w2, m3.data());

    this->output.push_back(t_new);
}

void NodeMultiplyRightWithMatrix::calculate_gradient(bool last){
    // TODO: Implement matrix multiplication between t1 and the top gradient
    Tensor *t1 = last ? this->buffer[0] : this->mulmat;
    Tensor *t_new = copy_eigen_matrix_to_new_tensor(t1->rows, t1->cols, t1->data);
    this->grad.push_back(t_new);
}

void NodeMultiplyRightWithMatrix::update_matrix(Tensor *t){
    this->mulmat = t;
}

NodeMultiplyRightWithMatrix::NodeMultiplyRightWithMatrix(Tensor *t){
    this->mulmat = t;
    this->grad_type = 1;
}

/*
 * Transpose a matrix
 */
void NodeTransposeMatrix::calculate_value(void){
    if (this->buffer.size() != 1)
        throw std::invalid_argument("NodeMultiplyTwoMatrices: the number of matrices in buffer is not one.");

    this->output.clear();
    Tensor *t1 = this->buffer[0];
    unsigned int h=t1->rows, w=t1->cols;

    Eigen::MatrixXd m = Eigen::Map<Eigen::MatrixXd>(t1->data, h, w);
    Eigen::MatrixXd mt = m.transpose();

    // Note that the matrix dimensions change
    Tensor *t_new = copy_eigen_matrix_to_new_tensor(w, h, mt.data());

    this->output.push_back(t_new);
}

void NodeTransposeMatrix::calculate_gradient(bool last){
    Tensor *t1 = this->output[0];
    unsigned int h=t1->rows, w=t1->cols;
    Eigen::MatrixXd m = Eigen::MatrixXd::Constant(h, w, 1.0);
    Tensor *t_new = copy_eigen_matrix_to_new_tensor(h, w, m.data());
    this->grad.push_back(t_new);
}

/*
 * Compute the squared error between a real value and a target real value
 */
void NodeSingleSquaredError::calculate_value(void){
    this->output.clear();
    if (this->buffer.size() != 1)
        throw std::invalid_argument("[NodeSingleSquaredError] the number of buffered tensors is not one.");

    Tensor *t = this->buffer[0];
    double difference = 0.5 * pow((t->data[0] - this->target), 2.0);
    Eigen::MatrixXd m = Eigen::MatrixXd::Constant(1, 1, 0.0);
    m << difference;
    Tensor *t_new = copy_eigen_matrix_to_new_tensor(t->rows, t->cols, m.data());
    this->output.push_back(t_new);
}

void NodeSingleSquaredError::calculate_gradient(bool last){
    Tensor *t = this->buffer[0];
    double derivative = t->data[0] - this->target;
    Eigen::MatrixXd m = Eigen::MatrixXd::Constant(1, 1, 0.0);
    m << derivative;
    Tensor *t_new = copy_eigen_matrix_to_new_tensor(t->rows, t->cols, m.data());
    this->grad.push_back(t_new);
}

void NodeSingleSquaredError::update_target(double target){
    this->target = target;
}

NodeSingleSquaredError::NodeSingleSquaredError(double target) {
    this->target = target;
}

/*
 * Compute the cross entropy loss between a real value and a target real value
 */
void NodeSingleBinaryCrossEntropy::calculate_value(void){
    this->output.clear();
    if (this->buffer.size() != 1)
        throw std::invalid_argument("[NodeSingleBinaryCrossEntropy] the number of buffered tensors is not one.");

    Tensor *t = this->buffer[0];
    double cross_entropy = (-1.0) * (this->target*log(t->data[0]) + (1.0-this->target)*log(1.0-t->data[0]));
    Eigen::MatrixXd m = Eigen::MatrixXd::Constant(1, 1, 0.0);
    m << cross_entropy;
    Tensor *t_new = copy_eigen_matrix_to_new_tensor(t->rows, t->cols, m.data());
    this->output.push_back(t_new);
}

void NodeSingleBinaryCrossEntropy::calculate_gradient(bool last){
    Tensor *t = this->buffer[0];
    double derivative = ((1.0 - this->target) / (1.0 - t->data[0])) - (this->target / t->data[0]);
    Eigen::MatrixXd m = Eigen::MatrixXd::Constant(1, 1, 0.0);
    m << derivative;
    Tensor *t_new = copy_eigen_matrix_to_new_tensor(t->rows, t->cols, m.data());
    this->grad.push_back(t_new);
}

void NodeSingleBinaryCrossEntropy::update_target(double target){
    this->target = target;
}

NodeSingleBinaryCrossEntropy::NodeSingleBinaryCrossEntropy(double target) {
    this->target = target;
}
