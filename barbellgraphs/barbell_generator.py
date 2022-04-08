#%%
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np


def create_direted_complete(n):
    """
    Return a directed semi-complete graph with n nodes (b1) having incoming 
    egdes from the other n nodes (b2). And having both incoming and outgoing
    edge between nodes in the same group (b1, b2) with the following
    properties:

    edges weight between b1 and b2: 1
    edge weight between b1: 0.8
    edge_weight between b2: 0.6

    node label: b1 for nodes with incoming edges
                b2 for nodes with outgoing edges
    note attributes:
        b1: attr1 = 0.3, attr2 = 0.7
        b2: attr1 = 0.5, attr2 = 0.3

    :param n: number of nodes
    :returns:   a directed semi-complete labeled/attributed graph 
                with 2 node roles.
    """
    w_b1b2 = 1
    w_b1 = 0.8
    w_b2 = 0.6

    b1_attr1 = 0.3
    b1_attr2 = 0.7
    b2_attr1 = 0.5
    b2_attr2 = 0.3

    b1 = int(n/2)

    G = nx.DiGraph()  # create empty graph
    # add nodes with labels and attributes
    G.add_nodes_from(
        [(n, {"attr1": b1_attr1, "attr2": b1_attr2, "label": 'b1'}) for n in range(b1)]
        )
    G.add_nodes_from(
        [(n, {"attr1": b2_attr1, "attr2": b2_attr2, "label": 'b2'}) for n in range(b1, n)]
        )

    # add edges between b1 and b2
    for inc_n in range(b1):
        G.add_weighted_edges_from([(n, inc_n, w_b1b2) for n in range(b1, n)])

    # add edges within b1
    b1_combinations = itertools.permutations(range(b1), 2)
    G.add_weighted_edges_from([(l, r, w_b1) for l,r in b1_combinations])

    # add edges within b2
    b2_combinations = itertools.permutations(range(b1, n), 2)
    G.add_weighted_edges_from([(l, r, w_b2) for l,r in b2_combinations])

    return G


def create_directed_barbell(m1, m2):
    """
    Returns a directed barbell graph. The bells are directed semi_complete
    graphs os size m1 and are connected from a b2 node (relabelled to b3) 
    with a path of size m2 having alternating direction starting from 
    the center node in the  path 
    
    weight : 0.7
    label : b4 for centernode, increasing numbering towards b3
    attr1 : 0.25
    attr2 : 0.75


    :paran m1: number of nodes in the bells
    :param m2: number of nodes in the path. Note if this is an even number
                 then an addtional center node is added
    """
    attr1 = 0.25
    attr2 = 0.75
    weight=0.7

    bell1 = create_direted_complete(m1)
    bell2 = create_direted_complete(m2)
    G = nx.disjoint_union(bell1, bell2)

    # determine and add center node of the path
    if np.mod(m2, 2) == 0:
            m2 = m2 + 1
    centernode = int((m2-1)/2+1)
    b4 = centernode + 2*m1 - 1
    G.add_node(b4, attr1=attr1, attr2=attr2,  label="b4")
    path_lenght = centernode -1

    # relabel connecting nodes
    b3_1 = m1-1
    b3_2 = 2*m1-1
    G.nodes[b3_1]["label"] = 'b3'
    G.nodes[b3_2]["label"] = 'b3'

    # create path
    for i in range (path_lenght): 
        if i == path_lenght-1:
            target = [b3_1, b3_2]  # connect to the b3 nodes
        else :
            target = [b4-i-1, b4+i+1]
            for t in target: # add nodes to graph
                G.add_node(t, attr1=attr1, attr2=attr2,  label="b"+str(i+5))
        source = [b4-i, b4+i]

        if np.mod(i, 2) == 0:
            # swap direction
            target, source = source, target
        
        for u, v in zip(source, target):
            G.add_edge(u, v, weight=weight)

    return G

