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
    this->name = "NodeElementWisePower";
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

NodeSumAllTensorElements::NodeSumAllTensorElements(void){
    this->name = "NodeSumAllTensorElements";
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
    this->name = "NodeElementWiseAddConstant";
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

NodeElementWiseSigmoidFunction::NodeElementWiseSigmoidFunction(void){
    this->name = "NodeElementWiseSigmoidFunction";
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
    this->name = "NodeElementWiseConstantMultiply";
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

NodeElementWiseLog::NodeElementWiseLog(void){
    this->name = "NodeElementWiseLog";
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

NodeIdentity::NodeIdentity(void){
    this->name = "NodeIdentity";
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
    this->name = "NodeMultiplyRightWithMatrix";
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

NodeTransposeMatrix::NodeTransposeMatrix(void){
    this->name = "NodeTransposeMatrix";
}

/*
 * Compute the squared error between a real value and a target real value
 */
void NodeSquaredError::calculate_value(void){
    this->output.clear();
    if (this->target == nullptr)
        throw std::invalid_argument("[NodeSquaredError] the target tensor is not set!");

    if (this->buffer.size() != 1) {
        throw std::invalid_argument("[NodeSquaredError] the number of buffered tensors is not one.");
    }

    if (this->target->rows != this->buffer[0]->rows || this->target->cols != this->buffer[0]->cols){
        throw std::invalid_argument("[NodeSquaredError] the dimensions between the input and target do not match.");
    }

    Eigen::MatrixXd m_y = Eigen::Map<Eigen::MatrixXd>(this->target->data, this->target->rows, this->target->cols);
    Eigen::MatrixXd m_yhat = Eigen::Map<Eigen::MatrixXd>(this->buffer[0]->data, this->buffer[0]->rows, this->buffer[0]->cols);
    Eigen::MatrixXd m_diff = m_y - m_yhat;

    double difference = m_diff.norm();

    Eigen::MatrixXd m = Eigen::MatrixXd::Constant(1, 1, 0.0);
    m << difference;

    Tensor *t_new = copy_eigen_matrix_to_new_tensor(1, 1, m.data());
    this->output.push_back(t_new);
}

void NodeSquaredError::calculate_gradient(bool last){
    Tensor *t = this->buffer[0];
    Eigen::MatrixXd m_y = Eigen::Map<Eigen::MatrixXd>(this->target->data, this->target->rows, this->target->cols);
    Eigen::MatrixXd m_yhat = Eigen::Map<Eigen::MatrixXd>(t->data, t->rows, t->cols);
    Eigen::MatrixXd m_grad = m_yhat - m_y;

    Tensor *t_new = copy_eigen_matrix_to_new_tensor(t->rows, t->cols, m_grad.data());
    this->grad.push_back(t_new);
}

void NodeSquaredError::update_target(Tensor *target){
    this->target = target;
}

NodeSquaredError::NodeSquaredError(void) {
    this->name = "NodeSquaredError";
    this->target = nullptr;
    this->grad_type = 1;
}

/*
 * Compute the cross entropy loss between a real value and a target real value
 */
void NodeBinaryCrossEntropy::calculate_value(void){
    this->output.clear();
    if (this->buffer.size() != 1)
        throw std::invalid_argument("[NodeBinaryCrossEntropy] the number of buffered tensors is not one.");

    Tensor *y_target = this->target;
    Tensor *y_computed = this->buffer[0];

    if (y_target->rows != y_computed->rows || y_target->cols != y_computed->cols)
        throw std::invalid_argument("[NodeBinaryCrossEntropy] dimension mismatch between y and yhat.");

    Tensor *y_crossentropy = new Tensor(y_target->rows, y_target->cols);
    y_crossentropy->data = new double[y_target->rows*y_target->cols];

    for(int i=0;i<y_target->rows*y_target->cols;i++){
        y_crossentropy->data[i] = (-1.0) * (this->target->data[i]*log(y_computed->data[i]) + (1.0-this->target->data[i])*log(1.0-y_computed->data[i]));
    }

    this->output.push_back(y_crossentropy);
}

void NodeBinaryCrossEntropy::calculate_gradient(bool last){
    Tensor *y_target = this->target;
    Tensor *y_computed = this->buffer[0];

    if (y_target->rows != y_computed->rows || y_target->cols != y_computed->cols)
        throw std::invalid_argument("[NodeBinaryCrossEntropy] dimension mismatch between y and yhat.");

    Tensor *y_crossentropy_derivative = new Tensor(y_target->rows, y_target->cols);
    y_crossentropy_derivative->data = new double[y_target->rows*y_target->cols];

    for(int i=0;i<y_target->rows*y_target->cols;i++){
        y_crossentropy_derivative->data[i] = ((1.0 - y_target->data[i]) / (1.0 - y_computed->data[i])) - (y_target->data[i] / y_computed->data[i]);
    }

    this->grad.push_back(y_crossentropy_derivative);
}

void NodeBinaryCrossEntropy::update_target(Tensor *target){
    this->target = target;
}

NodeBinaryCrossEntropy::NodeBinaryCrossEntropy(void) {
    this->name = "NodeBinaryCrossEntropy";
    this->target = nullptr;
    this->grad_type = 1;
}
