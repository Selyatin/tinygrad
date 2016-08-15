#include "nodeimplementations.h"
#include "utils.h"

using namespace Eigen;

/*
 * Compute a specified element-wise power for the elements of all TGMatrices.
 */
void NodeElementWisePower::calculate_value(void){
    this->reserve_output_memory_once(this->buffer->rows, this->buffer->cols);
    for(int j=0;j<this->buffer->size();j++){
        this->output->data[j] = pow(this->buffer->data[j], this->power);
    }
}

void NodeElementWisePower::calculate_gradient(void){
    this->reserve_buffer_gradient_memory_once(this->buffer->rows, this->buffer->cols);
    for(int j=0;j<this->buffer->gradient->size();j++)
        this->buffer->gradient->data[j] = this->power * pow(this->buffer->data[j], this->power - 1.0);
}

void NodeElementWisePower::combine_upper_gradient(TGMatrix *upper_gradient) {
    this->combine_gradient_TGMatrices_to_lower(upper_gradient, this->buffer->gradient, this->buffer);
}

NodeElementWisePower::NodeElementWisePower(double power) : Node() {
    this->power = power;
    this->name = "NodeElementWisePower";
}

/*
 * Compute the sigmoid/logistic function element-wise for the elements of the all TGMatrices.
 */
void NodeElementWiseSigmoidFunction::calculate_value(void){
    this->reserve_output_memory_once(this->buffer->rows, this->buffer->cols);
    for(int j=0;j<this->buffer->size();j++){
        this->output->data[j] = 1.0 / (1.0 + exp((-1.0)*this->buffer->data[j]));
    }
}

void NodeElementWiseSigmoidFunction::calculate_gradient(void){
    TGMatrix *current_buffer = this->buffer;
    TGMatrix *current_output = this->output;

    this->reserve_buffer_gradient_memory_once(this->buffer->rows, this->buffer->cols);

    for(int j=0;j<current_output->size();j++){
        current_buffer->gradient->data[j] = current_output->data[j] * (1.0 - current_output->data[j]);
    }
}

void NodeElementWiseSigmoidFunction::combine_upper_gradient(TGMatrix *upper_gradient) {
    this->combine_gradient_TGMatrices_to_lower(upper_gradient, this->buffer->gradient, this->buffer);
}

NodeElementWiseSigmoidFunction::NodeElementWiseSigmoidFunction(void){
    this->name = "NodeElementWiseSigmoidFunction";
}

/*
 * Compute the logarithm element-wise the elements of a TGMatrix.
 */
void NodeElementWiseLog::calculate_value(void){
    this->reserve_output_memory_once(this->buffer->rows, this->buffer->cols);
    for(int j=0;j<this->buffer->size();j++){
        if (this->buffer->data[j] <= 0.0)
            throw std::invalid_argument("Logarithm is defined only for non-negative real values.");
        this->output->data[j] = log(this->buffer->data[j]);
    }
}

void NodeElementWiseLog::calculate_gradient(void){
    this->reserve_buffer_gradient_memory_once(this->buffer->rows, this->buffer->cols);
    for(int j=0;j<this->buffer->size();j++)
        this->buffer->gradient->data[j] = 1.0 / this->buffer->data[j];
}

void NodeElementWiseLog::combine_upper_gradient(TGMatrix *upper_gradient) {
    this->combine_gradient_TGMatrices_to_lower(upper_gradient, this->buffer->gradient, this->buffer);
}

NodeElementWiseLog::NodeElementWiseLog(void){
    this->name = "NodeElementWiseLog";
}

/*
 * Multiply one matrix with a given matrix
 */
void NodeMultiplyRightWithMatrix::calculate_value(void){
    unsigned int h1=this->buffer->rows, w1=this->buffer->cols, h2=this->mulmat->rows, w2=this->mulmat->cols;

    if (w1 != h2)
        throw std::invalid_argument("NodeMultiplyWithMatrix: the matrix dimensions do not match for multiply.");

    MatrixXd m1 = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->buffer->data, h1, w1);
    MatrixXd m2 = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->mulmat->data, h2, w2);
    MatrixXd m3 = m1 * m2;

    // Allocate memory only once for a matrix with correct dimensionality
    this->reserve_output_memory_once(h1, w2);

    memcpy(this->output->data, m3.data(), sizeof(double) * this->output->size());
}

void NodeMultiplyRightWithMatrix::calculate_gradient(void){
    this->reserve_buffer_gradient_memory_once(this->mulmat->rows, this->mulmat->cols);
    memcpy(this->buffer->gradient->data, this->mulmat->data, sizeof(double) * this->mulmat->size());

    if (this->mulmat->gradient == nullptr)
        this->mulmat->gradient = new TGMatrix(this->buffer->rows, this->buffer->cols, true);
    memcpy(this->mulmat->gradient->data, this->buffer->data, sizeof(double) * this->buffer->size());
}

void NodeMultiplyRightWithMatrix::update_matrix(TGMatrix *t){
    this->mulmat = t;
}

void NodeMultiplyRightWithMatrix::combine_upper_gradient(TGMatrix *upper_gradient) {
    this->combine_gradient_TGMatrices_to_lower(upper_gradient, this->buffer->gradient, this->buffer);
    this->combine_gradient_TGMatrices_to_lower(upper_gradient, this->mulmat->gradient, this->mulmat);
}

NodeMultiplyRightWithMatrix::NodeMultiplyRightWithMatrix(TGMatrix *t){
    this->mulmat = t;
    this->grad_type = 1;
    this->name = "NodeMultiplyRightWithMatrix";
}

/*
 * Add a TGMatrix into the input TGMatrix.
 */
void NodeAddTGMatrix::calculate_value(void){
    unsigned int h1=this->buffer->rows, w1=this->buffer->cols, h2=this->addition->rows, w2=this->addition->cols;

    if (h1 != h2 || w1 != w2)
        throw std::invalid_argument("NodeAddTGMatrix: the matrix dimensions do not match for addition.");

    MatrixXd m1 = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->buffer->data, h1, w1);
    MatrixXd m2 = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->addition->data, h2, w2);
    MatrixXd m3 = m1 + m2;

    this->reserve_output_memory_once(h1, w2);

    memcpy(this->output->data, m3.data(), sizeof(double) * this->output->size());
}

void NodeAddTGMatrix::calculate_gradient(void){
    this->reserve_buffer_gradient_memory_once(this->buffer->rows, this->buffer->cols);
    
    for(int i=0;i<this->buffer->gradient->size();i++)
        this->buffer->gradient->data[i] = 1.0;

    if (this->addition->gradient == nullptr)
        this->addition->gradient = new TGMatrix(this->addition->rows, this->addition->cols, true);
    
    for(int i=0;i<this->addition->gradient->size();i++)
        this->addition->gradient->data[i] = 1.0;
}

void NodeAddTGMatrix::update_matrix(TGMatrix *t){
    this->addition = t;
}

void NodeAddTGMatrix::combine_upper_gradient(TGMatrix *upper_gradient) {
    this->combine_gradient_TGMatrices_to_lower(upper_gradient, this->buffer->gradient, this->buffer);
    this->combine_gradient_TGMatrices_to_lower(upper_gradient, this->addition->gradient, this->addition);
}

NodeAddTGMatrix::NodeAddTGMatrix(TGMatrix *t){
    this->addition = t;
    this->grad_type = 0;
    this->name = "NodeAddTGMatrix";
}

/*
 * Compute the squared error between a real value and a target real value
 */
void NodeSquaredError::calculate_value(void){
    this->free_output_TGMatrix();
    if (this->target == nullptr)
        throw std::invalid_argument("[NodeSquaredError] the target TGMatrix is not set!");

    if (this->target->rows != this->buffer->rows || this->target->cols != this->buffer->cols){
        throw std::invalid_argument("[NodeSquaredError] the dimensions between the input and target do not match.");
    }

    MatrixXd m_y = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->target->data, this->target->rows, this->target->cols);
    MatrixXd m_yhat = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->buffer->data, this->buffer->rows, this->buffer->cols);
    MatrixXd m_diff = m_y - m_yhat;

    double difference = m_diff.norm();

    TGMatrix *o = new TGMatrix(1, 1, true);
    o->data[0] = difference;

    this->output = o;
}

void NodeSquaredError::calculate_gradient(void){
    MatrixXd m_y = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->target->data, this->target->rows, this->target->cols);
    MatrixXd m_yhat = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(this->buffer->data, this->buffer->rows, this->buffer->cols);
    MatrixXd m_grad = m_yhat - m_y;

    this->reserve_buffer_gradient_memory_once(this->buffer->rows, this->buffer->cols);
    memcpy(this->buffer->gradient->data, m_grad.data(), sizeof(double) * this->buffer->gradient->size());
}

void NodeSquaredError::update_target(TGMatrix *target){
    this->target = target;
}

void NodeSquaredError::combine_upper_gradient(TGMatrix *upper_gradient) {
    this->combine_gradient_TGMatrices_to_lower(upper_gradient, this->buffer->gradient, this->buffer);
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
    TGMatrix *y_target = this->target;
    TGMatrix *y_input = this->buffer;

    if (y_target->rows != y_input->rows || y_target->cols != y_input->cols)
        throw std::invalid_argument("[NodeBinaryCrossEntropy] dimension mismatch between y and yhat.");

    this->reserve_output_memory_once(y_target->rows, y_target->cols);

    for(int i=0;i<y_target->size();i++){
        this->output->data[i] = (-1.0) * (this->target->data[i]*log(y_input->data[i]) + (1.0-this->target->data[i])*log(1.0-y_input->data[i]));
    }
}

void NodeBinaryCrossEntropy::calculate_gradient(void){
    this->reserve_buffer_gradient_memory_once(this->buffer->rows, this->buffer->cols);

    for(int i=0;i<this->target->size();i++){
        this->buffer->gradient->data[i] = (((1.0 - this->target->data[i]) / (1.0 - this->buffer->data[i])) - (this->target->data[i] / this->buffer->data[i]));
    }
}

void NodeBinaryCrossEntropy::update_target(TGMatrix *target){
    this->target = target;
}

void NodeBinaryCrossEntropy::combine_upper_gradient(TGMatrix *upper_gradient) {
    this->combine_gradient_TGMatrices_to_lower(upper_gradient, this->buffer->gradient, this->buffer);
}

NodeBinaryCrossEntropy::NodeBinaryCrossEntropy(void) {
    this->name = "NodeBinaryCrossEntropy";
    this->target = nullptr;
    this->grad_type = 1;
}
