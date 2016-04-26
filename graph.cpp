//
// Created by niko on 4/21/16.
//

#include "graph.h"
#include "utils.h"
#include "nodeimplementations.h"

Graph::Graph(void){
    this->nodes.clear();
}

void Graph::print_contents(void) {
    std::cout << "* Cleaning the graph" << std::endl;
    std::cout << "--- Nodes, buffers and outputs ---" << std::endl;
    for(int i=0; i<this->nodes.size(); i++) {
        Node *c = this->nodes[i];
        std::cout << std::endl;
        std::cout << "Node[" << i << "] '" << c->name << "' at " << c << std::endl;
        for(Tensor *t : c->buffer){
            std::cout << "  Buffer tensor at " << t << " data addr=" << t->data << std::endl;
            print_tensor_as_eigen_matrix(t, true);
        }
        for(Tensor *t : c->output){
            std::cout << "  Output tensor at " << t << " data addr=" << t->data << std::endl;
            print_tensor_as_eigen_matrix(t, true);
        }
        for(Tensor *t : c->grad){
            std::cout << "  Gradient tensor at " << t << " data addr=" << t->data << std::endl;
            print_tensor_as_eigen_matrix(t, true);
        }
        std::cout << std::endl;
    }
    std::cout << "----------------------------------" << std::endl;
}

void Graph::clear_gradients(void){
    for(int i=0; i<this->nodes.size(); i++) {
        Node *c = this->nodes[i];
        for (Tensor *t : c->grad) {
            if (t == nullptr)
                continue;
            if (t->guarded)
                continue;
            if (t->data != nullptr) {
                delete[] t->data;
                t->data = nullptr;
            }
        }
        for (int j = 0; j < c->grad.size(); j++) {
            if (c->grad[j] != nullptr && !c->grad[j]->guarded) {
                delete c->grad[j];
                c->grad[j] = nullptr;
            }
        }
        c->grad.clear();
    }
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

        s->in.pop_back();
        initial.out.clear();
    }

    Tensor *value = b->output[0];

    return value;
}

void Graph::backward(Node *a, Node *b){
    if (this->nodes.size() == 0)
        return;

    Node *s = a;
    if (s == nullptr)
        return;
    s->calculate_gradient(false);

    this->traverse.clear();
    for(int i=0;i<s->in.size(); i++){
        this->traverse.push_back(std::make_pair(s, s->in[i]));
    }

    while(this->traverse.size() > 0){
        std::pair<Node*, Node*> c = this->traverse[0];
        this->traverse.erase(this->traverse.begin());

        c.second->calculate_gradient(b == c.second);
        if (c.second->grad.size() == c.second->out.size()) {

            if (c.second->grad_type == 0 && c.second->out.size() > 0){
                Eigen::MatrixXd m_lg = Eigen::Map<Eigen::MatrixXd>(c.second->grad[0]->data, c.second->grad[0]->rows, c.second->grad[0]->cols);
                Eigen::MatrixXd m_tg = Eigen::MatrixXd::Constant(c.second->grad[0]->rows, c.second->grad[0]->cols, 0.0);

                for (int i = 0; i < c.second->out.size(); i++) {
                    Eigen::MatrixXd m_ug = Eigen::Map<Eigen::MatrixXd>(c.second->out[i]->grad[0]->data, c.second->out[i]->grad[0]->rows, c.second->out[i]->grad[0]->cols);
                    if (m_tg.rows() == m_ug.cols() && m_tg.cols() == m_ug.rows()){
                        m_tg = m_tg + m_lg.cwiseProduct(m_ug.transpose());
                    } else {
                        m_tg = m_tg + m_lg.cwiseProduct(m_ug);
                    }
                }
                Tensor *t_new = copy_eigen_matrix_to_new_tensor(c.second->grad[0]->rows, c.second->grad[0]->cols, m_tg.data());

                delete[] c.second->grad[0]->data;
                delete c.second->grad[0];
                c.second->grad.clear();

                c.second->grad.push_back(t_new);
            }

            if (c.second->grad_type == 1 && c.second->out.size() > 0){
                unsigned int target_h = 0;
                unsigned int target_w = 0;
                if (c.second != b) {
                    target_h = c.second->buffer[0]->rows;
                    target_w = c.second->buffer[0]->cols;
                } else {
                    target_h = dynamic_cast<NodeMultiplyRightWithMatrix*>(c.second)->mulmat->rows;
                    target_w = dynamic_cast<NodeMultiplyRightWithMatrix*>(c.second)->mulmat->cols;
                }

                Eigen::MatrixXd m_tg_acc = Eigen::MatrixXd::Constant(target_h, target_w, 0.0);
                unsigned int m_lg_h = c.second->grad[0]->rows;
                unsigned int m_lg_w = c.second->grad[0]->cols;

                for (int i = 0; i < c.second->out.size(); i++) {
                    unsigned int m_ug_h = c.second->out[0]->grad[i]->rows;
                    unsigned int m_ug_w = c.second->out[0]->grad[i]->cols;
                    Eigen::MatrixXd m_ug = Eigen::Map<Eigen::MatrixXd>(c.second->out[i]->grad[0]->data, m_ug_h, m_ug_w);
                    Eigen::MatrixXd m_lg = Eigen::Map<Eigen::MatrixXd>(c.second->grad[0]->data, m_lg_h, m_lg_w);

                    if (target_h == m_lg_h && target_w == m_ug_w && m_lg_w == m_ug_h){
                        Eigen::MatrixXd m_tg = m_lg * m_ug;
                        m_tg_acc = m_tg_acc + m_tg;
                    }
                    if (target_h == m_ug_h && target_w == m_lg_w && m_ug_w == m_lg_h){
                        Eigen::MatrixXd m_tg = m_ug * m_lg;
                        m_tg_acc = m_tg_acc + m_tg;
                    }
                    if (target_h == m_ug_h && target_w == m_lg_h && m_ug_w == m_lg_w){
                        Eigen::MatrixXd m_tg = m_ug * m_lg.transpose();
                        m_tg_acc = m_tg_acc + m_tg;
                    }
                    if (target_h == m_lg_w && target_w == m_ug_w && m_lg_h == m_ug_h){
                        Eigen::MatrixXd m_tg = m_lg.transpose() * m_ug;
                        m_tg_acc = m_tg_acc + m_tg;
                    }
                }

                delete[] c.second->grad[0]->data;
                delete c.second->grad[0];
                c.second->grad.clear();
                c.second->grad.push_back(copy_eigen_matrix_to_new_tensor((unsigned int)m_tg_acc.rows(), (unsigned int)m_tg_acc.cols(), m_tg_acc.data()));
            }

            if (b == c.second){
                // The desired gradient is calculated, let us break out of the loop.
                break;
            }

            for(int i = 0; i < c.second->in.size(); i++) {
                this->traverse.push_back(std::make_pair(c.second, c.second->in[i]));
            }

        }
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
