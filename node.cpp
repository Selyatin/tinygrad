//
// Created by niko on 4/17/16.
//

#include "node.h"


unsigned int Node::count_inputs(void) {
    return (unsigned int)this->in.size();
}

void Node::calculate_value(void){
}

void Node::calculate_gradient(bool last){
}

void Node::connect_to(Node *target){
    this->out.push_back(target);
    target->in.push_back(this);
}

Node::Node() {
    this->grad.clear();
    this->buffer.clear();
    this->output.clear();
    this->grad_type = 0;
}
