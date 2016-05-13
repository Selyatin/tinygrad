//
// Created by niko on 4/21/16.
//

#include "nodeimplementations.h"
#include "utils.h"

using namespace Eigen;

/*
 * Compute a specified element-wise power for the elements of all tensors.
 */
void NodeElementWisePower::calculate_value(void){
    this->free_output_tensor();
    Tensor *t_new = new Tensor(this->buffer->rows, this->buffer->cols, true);

    //Tensor *t_new = copy_eigen_matrix_to_new_tensor(nullptr, this->buffer->rows, this->buffer->cols, this->buffer->data);

    for(int j=0;j<this->buffer->size();j++){
        t_new->data[j] = pow(this->buffer->data[j], this->power);
    }

    this->output = t_new;
}

void NodeElementWisePower::calculate_gradient(void){
    Tensor *current = this->buffer;

    if (current->gradient == nullptr)
        current->gradient = new Tensor(this->buffer->rows, this->buffer->cols, true);

    for(int j=0;j<current->gradient->size();j++)
        current->gradient->data[j] = this->power * pow(current->data[j], this->power - 1.0);
}

void NodeElementWisePower::combine_upper_gradient(Tensor *upper_gradient) {
    this->combine_gradient_tensors_to_lower(upper_gradient, this->buffer->gradient, this->buffer);
}

NodeElementWisePower::NodeElementWisePower(double power) : Node() {
    this->power = power;
    this->name = "NodeElementWisePower";
}

/*
 * Compute the sigmoid/logistic function element-wise for the elements of the all tensors.
 */
void NodeElementWiseSigmoidFunction::calculate_value(void){
    this->free_output_tensor();
    this->output = new Tensor(this->buffer->rows, this->buffer->cols, true);
    for(int j=0;j<this->buffer->size();j++){
        this->output->data[j] = 1.0 / (1.0 + exp((-1.0)*this->buffer->data[j]));
    }
}

void NodeElementWiseSigmoidFunction::calculate_gradient(void){
    this->free_buffer_gradient_tensor();
    Tensor *current_buffer = this->buffer;
    Tensor *current_output = this->output;
    current_buffer->gradient = new Tensor(current_buffer->rows, current_buffer->cols, true);

    for(int j=0;j<current_output->size();j++){
        current_buffer->gradient->data[j] = current_output->data[j] * (1.0 - current_output->data[j]);
    }
}

void NodeElementWiseSigmoidFunction::combine_upper_gradient(Tensor *upper_gradient) {
    this->combine_gradient_tensors_to_lower(upper_gradient, this->buffer->gradient, this->buffer);
}

NodeElementWiseSigmoidFunction::NodeElementWiseSigmoidFunction(void){
    this->name = "NodeElementWiseSigmoidFunction";
}

/*
 * Compute the logarithm element-wise the elements of a tensor.
 */
void NodeElementWiseLog::calculate_value(void){
    this->free_output_tensor();

    //Tensor *t_new = copy_eigen_matrix_to_new_tensor(nullptr, this->buffer->rows, this->buffer->cols, this->buffer->data);
    Tensor *o = new Tensor(this->buffer->rows, this->buffer->cols, true);

    for(int j=0;j<this->buffer->size();j++){
        if (this->buffer->data[j] < 0.0)
            throw std::invalid_argument("Logarithm of a negative real value is not defined.");
        o->data[j] = log(this->buffer->data[j]);
    }
    this->output = o;
}

void NodeElementWiseLog::calculate_gradient(void){
    this->free_buffer_gradient_tensor();
    this->buffer->gradient = new Tensor(this->buffer->rows, this->buffer->cols, true);

    for(int j=0;j<this->buffer->size();j++)
        this->buffer->gradient->data[j] = 1.0 / this->buffer->data[j];
}

void NodeElementWiseLog::combine_upper_gradient(Tensor *upper_gradient) {
    this->combine_gradient_tensors_to_lower(upper_gradient, this->buffer->gradient, this->buffer);
}

NodeElementWiseLog::NodeElementWiseLog(void){
    this->name = "NodeElementWiseLog";
}

/*
 * Multiply one matrix with a given matrix
 */
void NodeMultiplyRightWithMatrix::calculate_value(void){
    this->free_output_tensor();

    unsigned int h1=this->buffer->rows, w1=this->buffer->cols, h2=this->mulmat->rows, w2=this->mulmat->cols;

    if (w1 != h2)
        throw std::invalid_argument("NodeMultiplyWithMatrix: the matrix dimensions do not match for multiply.");

    MatrixXd m1 = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->buffer->data, h1, w1);
    MatrixXd m2 = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->mulmat->data, h2, w2);
    MatrixXd m3 = m1 * m2;

    //Tensor *t_new = copy_eigen_matrix_to_new_tensor(nullptr, h1, w2, m3.data());
    Tensor *o = new Tensor(h1, w2, true);
    memcpy(o->data, m3.data(), sizeof(double) * o->size());

    this->output = o;
}

void NodeMultiplyRightWithMatrix::calculate_gradient(void){
    this->free_buffer_gradient_tensor();

    this->buffer->gradient = new Tensor(this->mulmat->rows, this->mulmat->cols, true);
    memcpy(this->buffer->gradient->data, this->mulmat->data, sizeof(double) * this->mulmat->size());

    if (this->mulmat->gradient != nullptr){
        this->mulmat->gradient->free_contents();
        delete this->mulmat->gradient;
        this->mulmat->gradient = nullptr;
    }

    this->mulmat->gradient = new Tensor(this->buffer->rows, this->buffer->cols, true);
    memcpy(this->mulmat->gradient->data, this->buffer->data, sizeof(double) * this->buffer->size());
}

void NodeMultiplyRightWithMatrix::update_matrix(Tensor *t){
    this->mulmat = t;
}

void NodeMultiplyRightWithMatrix::combine_upper_gradient(Tensor *upper_gradient) {
    this->combine_gradient_tensors_to_lower(upper_gradient, this->buffer->gradient, this->buffer);
    this->combine_gradient_tensors_to_lower(upper_gradient, this->mulmat->gradient, this->mulmat);
}

NodeMultiplyRightWithMatrix::NodeMultiplyRightWithMatrix(Tensor *t){
    this->mulmat = t;
    this->grad_type = 1;
    this->name = "NodeMultiplyRightWithMatrix";
}

/*
 * Add a tensor into the input tensor.
 */
void NodeAddTensor::calculate_value(void){
    this->free_output_tensor();
    unsigned int h1=this->buffer->rows, w1=this->buffer->cols, h2=this->addition->rows, w2=this->addition->cols;

    if (h1 != h2 || w1 != w2)
        throw std::invalid_argument("NodeAddTensor: the matrix dimensions do not match for addition.");

    MatrixXd m1 = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->buffer->data, h1, w1);
    MatrixXd m2 = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->addition->data, h2, w2);
    MatrixXd m3 = m1 + m2;

    Tensor *o = new Tensor(h1, w2, true);
    memcpy(o->data, m3.data(), sizeof(double) * o->size());

    this->output = o;
}

void NodeAddTensor::calculate_gradient(void){
    this->free_buffer_gradient_tensor();

    this->buffer->gradient = new Tensor(this->buffer->rows, this->buffer->cols, true);
    for(int i=0;i<this->buffer->gradient->size();i++)
        this->buffer->gradient->data[i] = 1.0;

    if (this->addition->gradient != nullptr) {
        this->addition->gradient->free_contents();
        delete this->addition->gradient;
        this->addition->gradient = nullptr;
    }

    this->addition->gradient = new Tensor(this->addition->rows, this->addition->cols, true);
    for(int i=0;i<this->addition->gradient->size();i++)
        this->addition->gradient->data[i] = 1.0;
}

void NodeAddTensor::update_matrix(Tensor *t){
    this->addition = t;
}

void NodeAddTensor::combine_upper_gradient(Tensor *upper_gradient) {
    this->combine_gradient_tensors_to_lower(upper_gradient, this->buffer->gradient, this->buffer);
    this->combine_gradient_tensors_to_lower(upper_gradient, this->addition->gradient, this->addition);
}

NodeAddTensor::NodeAddTensor(Tensor *t){
    this->addition = t;
    this->grad_type = 0;
    this->name = "NodeAddTensor";
}

/*
 * Compute the squared error between a real value and a target real value
 */
void NodeSquaredError::calculate_value(void){
    this->free_output_tensor();
    if (this->target == nullptr)
        throw std::invalid_argument("[NodeSquaredError] the target tensor is not set!");

    if (this->target->rows != this->buffer->rows || this->target->cols != this->buffer->cols){
        throw std::invalid_argument("[NodeSquaredError] the dimensions between the input and target do not match.");
    }

    MatrixXd m_y = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->target->data, this->target->rows, this->target->cols);
    MatrixXd m_yhat = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->buffer->data, this->buffer->rows, this->buffer->cols);
    MatrixXd m_diff = m_y - m_yhat;

    double difference = m_diff.norm();

    Tensor *o = new Tensor(1, 1, true);
    o->data[0] = difference;

    this->output = o;
}

void NodeSquaredError::calculate_gradient(void){
    this->free_buffer_gradient_tensor();

    MatrixXd m_y = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->target->data, this->target->rows, this->target->cols);
    MatrixXd m_yhat = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->buffer->data, this->buffer->rows, this->buffer->cols);
    MatrixXd m_grad = m_yhat - m_y;

    this->buffer->gradient = new Tensor(this->buffer->rows, this->buffer->cols, true);
    memcpy(this->buffer->gradient->data, m_grad.data(), sizeof(double) * this->buffer->gradient->size());
}

void NodeSquaredError::update_target(Tensor *target){
    this->target = target;
}

void NodeSquaredError::combine_upper_gradient(Tensor *upper_gradient) {
    this->combine_gradient_tensors_to_lower(upper_gradient, this->buffer->gradient, this->buffer);
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
    this->free_output_tensor();

    Tensor *y_target = this->target;
    Tensor *y_input = this->buffer;

    if (y_target->rows != y_input->rows || y_target->cols != y_input->cols)
        throw std::invalid_argument("[NodeBinaryCrossEntropy] dimension mismatch between y and yhat.");

    Tensor *y_crossentropy = new Tensor(y_target->rows, y_target->cols, true);

    for(int i=0;i<y_target->size();i++){
        y_crossentropy->data[i] = (-1.0) * (this->target->data[i]*log(y_input->data[i]) + (1.0-this->target->data[i])*log(1.0-y_input->data[i]));
    }

    this->output = y_crossentropy;
}

void NodeBinaryCrossEntropy::calculate_gradient(void){
    this->free_buffer_gradient_tensor();

    this->buffer->gradient = new Tensor(this->buffer->rows, this->buffer->cols, true);

    for(int i=0;i<this->target->size();i++){
        this->buffer->gradient->data[i] = (((1.0 - this->target->data[i]) / (1.0 - this->buffer->data[i])) - (this->target->data[i] / this->buffer->data[i]));
    }
}

void NodeBinaryCrossEntropy::update_target(Tensor *target){
    this->target = target;
}

void NodeBinaryCrossEntropy::combine_upper_gradient(Tensor *upper_gradient) {
    this->combine_gradient_tensors_to_lower(upper_gradient, this->buffer->gradient, this->buffer);
}

NodeBinaryCrossEntropy::NodeBinaryCrossEntropy(void) {
    this->name = "NodeBinaryCrossEntropy";
    this->target = nullptr;
    this->grad_type = 1;
}
