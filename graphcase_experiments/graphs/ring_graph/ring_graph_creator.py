import random
import networkx as nx
import numpy as np
from ..barbellgraphs.barbell_generator import create_direted_complete



def create_star(size, dim_node=2, dim_edge=1):
    """
    creates a star of size n have one central role (s1) with two roles
    (s2, s3) in the end nodes, having the below properties. The edge are
    from s1 to s2 and from s3 to s1.

    s1 - attr1: 0.5     attr2: 0.75
    s2 - attr1: 0.3     attr2: 0.4
    s3 - attr1: 0.7     attr2: 0.6

    weight s1 - s2 = 0.9
    weight s1 - s3 = 0.5
    """
    attr1 = [0.5, 0.3, 0.7]
    attr2 = [0.75, 0.4, 0.6]
    weights = [0.9, 0.5]

    np.random.seed(4)
    node_attributes = np.random.uniform(0.0, 1.0, (len(attr1), dim_node))
    node_attributes[:,0] = attr1
    node_attributes[:,1] = attr2
    node_attributes = [{"attr"+str(n+1):v for n,v in enumerate(row)} for row in node_attributes]
    
    edge_attributes = np.random.uniform(0.0, 1.0, (len(weights), dim_edge))
    edge_attributes[:,0] = weights
    edge_attributes = [{"weight" if n==0 else "e_attr"+str(n):v for n,v in enumerate(row)} for row in edge_attributes]
  

    s1 = int(size/2)

    G = nx.DiGraph()  # create empty graph
    # add s1 - centernode
    G.add_node(s1, **node_attributes[0], label='s1')
    # add s2 nodes
    G.add_nodes_from([(n, {**node_attributes[1], "label": 's2'}) for n in range(s1)])
    # add s3 nodes
    G.add_nodes_from([(n, {**node_attributes[2], "label": 's3'}) for n in range(s1+1,size)])
    # add edge  s1 -> s2
    for n in range(s1):
        G.add_edge(s1, n , **edge_attributes[0])
    #add edges s2 -> s3
    for n in range(s1+1, size):
        G.add_edge(n, s1 , **edge_attributes[1])
    return (G, s1)


def create_tree(s=4, depth=3, dim_node=2, dim_edge=1):
    """
    Creates a directed rooted tree. Every node has s children, where the first s/2
    children have incoming edge from the parent and the sencd s/2 have outgoing edges 
    to the parent. The tree depth is equal to the specified depth. 

    rootnode
    attr1 = 0.2, attr2 = 0.3

    si:  attr1 = 0.4, attr2 = 0.9
    so: attr1 = 0.1, attr2 = 0.6

    wi = 0.7,
    wo = 0.5
    """
    ## [root, incoming nodes, outgoing nodes]
    attr1 = [0.2, 0.4, 0.1]
    attr2 = [0.3, 0.9, 0.6]

    # [incoming edge, outgoing edge]
    weights = [0.7, 0.5]

    np.random.seed(4)
    node_attributes = np.random.uniform(0.0, 1.0, (len(attr1), dim_node))
    node_attributes[:,0] = attr1
    node_attributes[:,1] = attr2
    node_attributes = [{"attr"+str(n+1):v for n,v in enumerate(row)} for row in node_attributes]
    
    edge_attributes = np.random.uniform(0.0, 1.0, (len(weights), dim_edge))
    edge_attributes[:,0] = weights
    edge_attributes = [{"weight" if n==0 else "e_attr"+str(n):v for n,v in enumerate(row)} for row in edge_attributes]

    G = nx.DiGraph()  # create empty graph
    # add t1 - rootnode
    G.add_node(0, **node_attributes[0], label='t1')

    # add next level nodes
    parents = [0]
    i_range = int(s/2)
    cnt = 1
    for d in range(1, depth):  # loop per level
        new_parents = []
        for p in parents:  # loop per parent node
            name_base = G.nodes[p]['label'] + "_" + str(d+1)
            # add incoming nodes and edges
            G.add_nodes_from(
                [(cnt+n, {**node_attributes[1], "label": name_base + "i"}) for n in range(i_range)]
                )
            for i in range(i_range):
                G.add_edge(p, cnt+i , **edge_attributes[0])

            # dd outgoing nodes and edges
            G.add_nodes_from(
                [(cnt+n, {**node_attributes[1], "label": name_base + "o"}) for n in range(i_range, s)]
                )
            for i in range(i_range, s):
                G.add_edge(cnt+i, p, **edge_attributes[1])
            new_parents = new_parents + [cnt + i for i in range(s)]
            cnt = cnt + s

        parents = new_parents

    return (G, 0)

def create_bell(s, dim_node=2, dim_edge=1):
    """
    Creates a semi connected bell and sets a transfer node

    Args:
        s (int) : size of the bell
    """

    G = create_direted_complete(s, dim_node, dim_edge)
    G.nodes[0]['label'] = 'b3'

    return (G,0)


def create_ring(p, n):
    """
    Creates a ring with all edge directed into the same direction.
    A figure is attached on the ring every p nodes. In total, every
    symbol is repeated n times. The nodes are labeled r1 to rp having the 
    following properties:

    weight = 1
    attr1 = 0.9
    attr2 = 0.6

    Args:
        p (int): path length between two sub-figures.
        n (int): The number of times a sub-figure is repeated.

    Returns: graph G having a ring with subgraphs.

    """
    symbol_dic = {}
    symbol_dic['star'] = create_star(11)
    symbol_dic['tree'] = create_tree(s=4, depth=3)
    symbol_dic['bell'] = create_bell(10)
    return _create_ring_sub(p, n, symbol_dic)

def _create_ring_sub(p, n, symbol_dic, dim_node=2, dim_edge=1):
    """
    Creates a ring with all edge directed into the same direction.
    A figure is attached on the ring every p nodes. In total, every
    symbol is repeated n times. The nodes are labeled r1 to rp having the 
    following properties:

    weight = 1
    attr1 = 0.9
    attr2 = 0.6

    Args:
        p (int): path length between two sub-figures.
        n (int): The number of times a sub-figure is repeated.

    Returns: graph G having a ring with subgraphs.

    """

    weight = 1
    attr1 = 0.9
    attr2 = 0.6

    np.random.seed(4)
    # create a list of edge attribute dicts. Note that first attibute is called weight.
    # note that p+1 row is used for connecting the symbols
    edge_attributes = np.random.uniform(0.0, 1.0, (p + 1, dim_edge))
    edge_attributes[:,0] = weight
    edge_attributes = [{"weight" if n==0 else "e_attr"+str(n):v for n,v in enumerate(row)} for row in edge_attributes]

    # create a node attribute dicts, same for all ring nodes.
    node_attributes = {"attr"+str(n+1): v for n,v in enumerate(np.random.uniform(0.0, 1.0, dim_node))}
    node_attributes['attr1'] = attr1
    node_attributes['attr2'] = attr2

    symbol_list = [s for s in symbol_dic.values()] * n
    random.Random(4).shuffle(symbol_list)

    ring_node_cnt = p * len(symbol_list)

    # create ring
    G = nx.DiGraph()  # create empty graph
    G.add_nodes_from([(n, {**node_attributes, "label":"r" + str(n % p)}) for n in range(ring_node_cnt)])
    for n in range(ring_node_cnt-1):
        G.add_edge(n, n+1 , **edge_attributes[n % p])
    G.add_edge(ring_node_cnt-1, 0 , **edge_attributes[p-1])

    # add symbols to ring
    for i, s in enumerate(symbol_list):
        # union graph
        G = nx.union(G, s[0], rename=(None, "sym" + str(i) + "_"))
        src = i * p
        dst = "sym" + str(i) + "_" + str(s[1])
        G.add_edge(src, dst , **edge_attributes[p])
    G2 = nx.convert_node_labels_to_integers(G, label_attribute="old_id")

    return G2
    

def create_ring2(p,n, dim_node, dim_edge):
    """
    Creates a ring with all edge directed into the same direction.
    A figure is attached on the ring every p nodes. In total, every
    symbol is repeated n times. The nodes are labeled r1 to rp having the 
    following properties:

    weight = 1
    attr1 = 0.9
    attr2 = 0.6
    remaining node and edge attributed are random

    Args:
        p (int): path length between two sub-figures.
        n (int): The number of times a sub-figure is repeated.
        dim_node (int): the number of node attributes.
        dim_edge (int): the number of edge attributes.

    Returns: graph G having a ring with subgraphs.
    """
    symbol_dic = {
        'star': create_star(11, dim_node, dim_edge),
        'tree': create_tree(s=4, depth=3, dim_node=dim_node, dim_edge=dim_edge),
        'bell': create_bell(10, dim_node, dim_edge)
    }
    return _create_ring_sub(p, n, symbol_dic, dim_node, dim_edge)

