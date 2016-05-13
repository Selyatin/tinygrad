//
// Created by niko on 4/17/16.
//

#include "node.h"
#include "utils.h"

void Node::combine_gradient_tensors_to_lower(Tensor *upper, Tensor *lower, Tensor *lower_gradient_shape){
    if (this->grad_type == 0){
        Eigen::MatrixXd m_lg = Eigen::Map<Eigen::MatrixXd>(lower->data, lower->rows, lower->cols);
        Eigen::MatrixXd m_tg = Eigen::MatrixXd::Constant(lower->rows, lower->cols, 0.0);
        Eigen::MatrixXd m_ug = Eigen::Map<Eigen::MatrixXd>(upper->data, upper->rows, upper->cols);

        if (m_tg.rows() == m_ug.cols() && m_tg.cols() == m_ug.rows()){
            m_tg = m_tg + m_lg.cwiseProduct(m_ug.transpose());
        } else {
            m_tg = m_tg + m_lg.cwiseProduct(m_ug);
        }

        memcpy(lower->data, m_tg.data(), sizeof(double)*m_tg.size());
    }

    if (this->grad_type == 1){
        unsigned int target_h = lower_gradient_shape->rows;
        unsigned int target_w = lower_gradient_shape->cols;

        unsigned int m_lg_h = lower->rows;
        unsigned int m_lg_w = lower->cols;

        unsigned int m_ug_h = upper->rows;
        unsigned int m_ug_w = upper->cols;

        Eigen::MatrixXd m_tg_acc = Eigen::MatrixXd::Constant(target_h, target_w, 0.0);
        Eigen::MatrixXd m_ug = Eigen::Map<Eigen::MatrixXd>(upper->data, m_ug_h, m_ug_w);
        Eigen::MatrixXd m_lg = Eigen::Map<Eigen::MatrixXd>(lower->data, m_lg_h, m_lg_w);

        if (target_h == m_lg_h && target_w == m_ug_w && m_lg_w == m_ug_h){
            Eigen::MatrixXd m_tg = m_lg * m_ug;
            m_tg_acc = m_tg_acc + m_tg;
        }
        else if (target_h == m_ug_h && target_w == m_lg_w && m_ug_w == m_lg_h){
            Eigen::MatrixXd m_tg = m_ug * m_lg;
            m_tg_acc = m_tg_acc + m_tg;
        }
        else if (target_h == m_ug_h && target_w == m_lg_h && m_ug_w == m_lg_w){
            Eigen::MatrixXd m_tg = m_ug * m_lg.transpose();
            m_tg_acc = m_tg_acc + m_tg;
        }
        else if (target_h == m_lg_w && target_w == m_ug_w && m_lg_h == m_ug_h){
            Eigen::MatrixXd m_tg = m_lg.transpose() * m_ug;
            m_tg_acc = m_tg_acc + m_tg;
        }

        delete[] lower->data;
        lower->rows = target_h;
        lower->cols = target_w;
        lower->data = new double[lower->size()];
        memcpy(lower->data, m_tg_acc.data(), sizeof(double)*m_tg_acc.size());
    }

}

void Node::combine_upper_gradient(Tensor *upper_gradient){

}

void Node::calculate_value(void){
}

void Node::calculate_gradient(void){
}

void Node::free_output_tensor(void){
    if (this->output != nullptr) {
        this->output->free_contents();
        delete this->output;
        this->output = nullptr;
    }
}

void Node::free_buffer_tensor(void){
    if (this->buffer != nullptr) {
        this->buffer->free_contents();
        delete this->buffer;
        this->buffer = nullptr;
    }
}

void Node::free_buffer_gradient_tensor(void){
    if (this->buffer != nullptr){
        if (this->buffer->gradient != nullptr){
            this->buffer->gradient->free_contents();
            delete this->buffer->gradient;
            this->buffer->gradient = nullptr;
        }
    }
}

void Node::connect_to(Node *target){
    this->out = target;
    target->in = this;
}

Node::Node(void) {
    this->buffer = nullptr;
    this->output = nullptr;

    this->out = nullptr;
    this->in = nullptr;

    this->grad_type = 0;
    this->name = "Node";
}

Node::~Node(void){
    this->free_buffer_tensor();
    this->free_output_tensor();
}
