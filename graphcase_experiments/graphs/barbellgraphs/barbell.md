
# Barbell graph

the weighted directed barbell graph with node labels is created as follow. First a direct "semi"-complete graph is created as follow with are the connected via a path.

## semi-complete-graph
The directed semi-complete graph with m1 nodes (b1) having incoming 
    egdes from the other n nodes (b2). And having both incoming and outgoing
    edge between nodes within the same group {b1, b} with the following
    properties:

    edges weight between b1 and b2: 1
    edge weight between b1: 0.8
    edge_weight between b2: 0.6

    node label: b1 for nodes with incoming edges
                b2 for nodes with outgoing edges
    note attributes:
        b1: attr1 = 0.3, attr2 = 0.7
        b2: attr1 = 0.5, attr2 = 0.3

    Note that one note of b2 is used for connecting to the path this note is relabeled to b3.

<img src="https://github.com/tonyPo/graphcase_experiments/blob/main/graphcase_experiments/graphs/barbellgraphs/images/clique.png?raw=true" alt="Graph bell" width="400"/>

## connecting path

The directed semi_complete graphs are connected from a b2 node (relabelled to b3) 
with a path of size m2 having alternating direction starting from the center node with the following properties:

    weight : 0.7
    label : b4 for centernode, increasing numbering towards b3
    attr1 : 0.25
    attr2 : 0.75

<img src="https://github.com/tonyPo/graphcase_experiments/blob/main/graphcase_experiments/graphs/barbellgraphs/images/barbell.png?raw=true" alt="Barbell" width="400"/>





