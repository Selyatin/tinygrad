#ifndef TINYGRAD_GRAPH_H
#define TINYGRAD_GRAPH_H

#include "node.h"

class Graph {
private:
    std::vector<Node*> nodes;
    std::vector<std::pair<Node*, Node*>> traverse;
public:
    void add_node(Node *n);
    void connect_to(int a, int b);
    void connect_to(Node *a, Node *b);
    void print_contents(void);
    TGMatrix* forward(TGMatrix *input, Node *a, Node *b);
    void backward(Node *a, Node *b);
    void clean(void);
    Graph(void);
};


#endif //TINYGRAD_GRAPH_H
