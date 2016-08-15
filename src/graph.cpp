#include "graph.h"
#include "utils.h"
#include "nodeimplementations.h"

Graph::Graph(void){
    this->nodes.clear();
}

void Graph::print_contents(void) {
    std::cout << "--- Nodes, buffers and outputs ---" << std::endl;
    for(int i=0; i<this->nodes.size(); i++) {
        Node *c = this->nodes[i];
        std::cout << std::endl;
        std::cout << "Node[" << i << "] '" << c->name << "' at " << c << std::endl;
        if (c->buffer != nullptr){
            std::cout << "  Buffer TGMatrix at " << c->buffer << " data addr=" << c->buffer->data << std::endl;
            print_TGMatrix_as_eigen_matrix(c->buffer, true);
            if (c->buffer->gradient != nullptr){
                std::cout << "  Gradient TGMatrix of buffer TGMatrix (" << c->buffer << ") at " << c->buffer->gradient << " data addr=" << c->buffer->data << std::endl;
                print_TGMatrix_as_eigen_matrix(c->buffer->gradient, true);
            }
        }
        if (c->output != nullptr){
            std::cout << "  Output TGMatrix at " << c->output << " data addr=" << c->output->data << std::endl;
            print_TGMatrix_as_eigen_matrix(c->output, true);
        }
        std::cout << std::endl;
    }
    std::cout << "----------------------------------" << std::endl;
}

void Graph::clean(void){
    for(int i=0; i<this->nodes.size(); i++){
        Node *c = this->nodes[i];
        c->free_output_TGMatrix();
        c->free_buffer_TGMatrix();
    }
}

TGMatrix* Graph::forward(TGMatrix *input, Node *a, Node *b){
    if (this->nodes.size() == 0)
        return nullptr;

    Node *n = a;

    while(n != nullptr){
        if (n == a){
            if (n->buffer == nullptr)
                n->buffer = new TGMatrix(input->rows, input->cols, true);
            memcpy(n->buffer->data, input->data, sizeof(double)*input->size());
        } else {
            if (n->buffer == nullptr)
                n->buffer = new TGMatrix(n->in->output->rows, n->in->output->cols, true);
            memcpy(n->buffer->data, n->in->output->data, sizeof(double)*(n->in->output->size()));
        }

        n->calculate_value();

        n = n->out;
    }

    if (b->output == nullptr){
        throw std::invalid_argument("[Graph::forward] The output node did not compute anything. Maybe the output node is not connected to the input node?");
    }

    return b->output;
}

void Graph::backward(Node *a, Node *b){
    if (this->nodes.size() == 0)
        return;

    Node *n = a;

    while(n != nullptr){
        n->free_buffer_gradient_TGMatrix();
        n->calculate_gradient();

        if (n->buffer->gradient == nullptr)
            throw std::invalid_argument("[Graph::backward] The gradient TGMatrix 'n->buffer->gradient' is null!");

        if (n->out != nullptr) {
            if (n->out->buffer->gradient == nullptr)
                throw std::invalid_argument("[Graph::backward] The gradient TGMatrix 'n->out->buffer->gradient' is null!");
            n->combine_upper_gradient(n->out->buffer->gradient);
        }

        n = n->in;
    }
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
