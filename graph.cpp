//
// Created by niko on 4/21/16.
//

#include "graph.h"
#include "utils.h"

Graph::Graph(void){
    this->nodes.clear();
}

void Graph::print_contents(void) {
    std::cout << "* Cleaning the graph" << std::endl;
    std::cout << "--- Nodes, buffers and outputs ---" << std::endl;
    for(int i=0; i<this->nodes.size(); i++) {
        Node *c = this->nodes[i];
        std::cout << "Node[" << i << "] at " << c << std::endl;
        for(Tensor *t : c->buffer){
            std::cout << "  Buffer tensor at " << t << " data addr=" << t->data << std::endl;
            print_tensor_as_eigen_matrix(t);
        }
        for(Tensor *t : c->output){
            std::cout << "  Output tensor at " << t << " data addr=" << t->data << std::endl;
            print_tensor_as_eigen_matrix(t);
        }
        for(Tensor *t : c->grad){
            std::cout << "  Gradient tensor at " << t << " data addr=" << t->data << std::endl;
            print_tensor_as_eigen_matrix(t);
        }
    }
    std::cout << "----------------------------------" << std::endl;
}

void Graph::clean(void){
    // Free the buffered tensors of the first node (not in the output of any node)
    for(Tensor *t : this->nodes[0]->buffer){
        if (t == nullptr)
            continue;
        if (t->guarded)
            continue;
        if (t->data != nullptr) {
            delete[] t->data;
            t->data = nullptr;
        }
    }

    // Free the output tensors of all nodes
    for(int i=0; i<this->nodes.size(); i++){
        Node *c = this->nodes[i];

        // Free the memory of the output tensors
        for(Tensor *t : c->output){
            if (t == nullptr)
                continue;
            if (t->guarded)
                continue;
            if (t->data != nullptr) {
                delete[] t->data;
                t->data = nullptr;
            }
        }
        for(int j=0;j<c->output.size();j++) {
            if (c->output[j] != nullptr && !c->output[j]->guarded) {
                delete c->output[j];
                c->output[j] = nullptr;
            }
        }

        // Free the memory of the gradient calculations
        for(Tensor *t : c->grad){
            if (t == nullptr)
                continue;
            if (t->guarded)
                continue;
            if (t->data != nullptr) {
                delete[] t->data;
                t->data = nullptr;
            }
        }
        for(int j=0;j<c->grad.size();j++) {
            if (c->grad[j] != nullptr && !c->grad[j]->guarded) {
                delete c->grad[j];
                c->grad[j] = nullptr;
            }
        }

        c->output.clear();
        c->buffer.clear();
        c->grad.clear();
    }
}

Tensor* Graph::forward(Tensor *input, Node *a, Node *b){
    if (this->nodes.size() == 0)
        return nullptr;

    Node initial;
    initial.output.push_back(input);
    Node *s = a;
    this->connect_to(&initial, s);

    if (s != nullptr){
        this->traverse.clear();
        this->traverse.push_back(std::make_pair(&initial, s));

        while(this->traverse.size() > 0){
            std::pair<Node*, Node*> c = this->traverse[0];
            this->traverse.erase(this->traverse.begin());

            for(int i=0;i<c.first->output.size();i++){
                c.second->buffer.push_back(c.first->output[i]);
            }

            if (c.second->buffer.size() == c.second->in.size()){
                c.second->calculate_value();
                for(int i=0;i<c.second->out.size();i++){
                    this->traverse.push_back(std::make_pair(c.second, c.second->out[i]));
                }
            }
        }

        // TODO: Might not be correct if the input node already has IN nodes (we clear them for the temporary)
        s->in.clear();
        initial.out.clear();
    }

    // TODO: The evaluation of the flow graph should end when the desired output node is computed.
    //          Currently every node is computed
    Tensor *value = b->output[0];

    return value;
}

void Graph::backward(Node *a, Node *b){
    if (this->nodes.size() == 0)
        return;

//    for(Node *c : this->nodes){
//        c->calculate_gradient();
//    }

    Node *s = a;
    if (s == nullptr)
        return;
    s->calculate_gradient();

    this->traverse.clear();
    for(int i=0;i<s->in.size(); i++){
        this->traverse.push_back(std::make_pair(s, s->in[i]));
    }

    while(this->traverse.size() > 0){
        std::pair<Node*, Node*> c = this->traverse[0];
        this->traverse.erase(this->traverse.begin());

        c.second->calculate_gradient();
        if (c.second->grad.size() == c.second->out.size()) {



            if (c.second->out.size() > 0){
                //std::cout << "Eest f'n GYYST! " << c.second << std::endl;
                unsigned int ingsum_size = c.second->out[0]->grad[0]->rows * c.second->out[0]->grad[0]->cols;
                double *ingsum = new double[ingsum_size];
                for (int k = 0; k < ingsum_size; k++) {
                    ingsum[k] = 0.0;
                }
                for (int j = 0; j < c.second->out.size(); j++) {
                    for (int jj = 0; jj < c.second->out[j]->grad.size(); jj++) {
                        for (int k = 0; k < ingsum_size; k++) {
                            ingsum[k] = ingsum[k] + c.second->out[j]->grad[jj]->data[k];
                        }
                    }
                }

                if (c.second->grad.size() > 1)
                    throw std::invalid_argument("The gradient back flow is not implemented for |gradtensors| > 1 yet.");

                if (ingsum_size != 1) {
                    for (int k = 0; k < ingsum_size; k++) {
                        c.second->grad[0]->data[k] = c.second->grad[0]->data[k] * ingsum[k];
                    }
                } else {
                    for (int kk = 0; kk < c.second->grad[0]->rows*c.second->grad[0]->cols; kk++) {
                        c.second->grad[0]->data[kk] = c.second->grad[0]->data[kk] * ingsum[0];
                    }
                }

                //std::cout << "[";
                //for (int k = 0; k < ingsum_size; k++) {std::cout << ingsum[k] << " ";}
                //std::cout << "]" << std::endl;

                delete[] ingsum;
            }




            for(int i = 0; i < c.second->in.size(); i++) {
                this->traverse.push_back(std::make_pair(c.second, c.second->in[i]));

                // TODO: Calculate the total gradient of c.second here
                // How to calculate the total gradient of a node:
                // Sum the gradients of out gradients and multiple with the local gradient
            }
        }

        /*
        for(int i=0;i<c.first->output.size();i++){
            c.second->buffer.push_back(c.first->output[i]);
        }

        if (c.second->buffer.size() == c.second->in.size()){
            c.second->calculate_value();
            for(int i=0;i<c.second->out.size();i++){
                this->traverse.push_back(std::make_pair(c.second, c.second->out[i]));
            }
        }
        */
    }
}

// The first node with no input nodes is the interface
// for commencing the feed-forward computation.
Node* Graph::get_input_node(void){
    for (int i=0; i<this->nodes.size(); i++){
        if (this->nodes[i]->count_inputs() == 0)
            return this->nodes[i];
    }
    return nullptr;
}

Node* Graph::get_output_node(void){
    for (int i=0; i<this->nodes.size(); i++){
        if (this->nodes[i]->out.size() == 0)
            return this->nodes[i];
    }
    return nullptr;
}

void Graph::add_node(Node *n){
    this->nodes.push_back(n);
}

void Graph::connect_to(int a, int b){
    this->connect_to(this->nodes[a], this->nodes[b]);
}

void Graph::connect_to(Node *a, Node *b){
    a->connect_to(b);
}
