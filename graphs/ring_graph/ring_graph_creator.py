import networkx as nx
from ..barbellgraphs.barbell_generator import create_direted_complete
import random

def create_star(n):
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
    attr1_s1 = 0.5
    attr2_s1 = 0.75
    attr1_s2 = 0.3
    attr2_s2 = 0.4
    attr1_s3 = 0.7
    attr2_s3 = 0.6

    w_s1s2 = 0.9
    w_s1s3 = 0.5

    s1 = int(n/2)

    G = nx.DiGraph()  # create empty graph
    # add s1 - centernode
    G.add_node(s1, attr1=attr1_s1, attr2=attr1_s2, label='s1')
    # add s2 nodes
    G.add_nodes_from([(n, {"attr1": attr1_s2, "attr2": attr2_s2, "label": 's2'}) for n in range(s1)])
    # add s3 nodes
    G.add_nodes_from([(n, {"attr1": attr1_s3, "attr2": attr2_s3, "label": 's3'}) for n in range(s1+1,n)])
    # add edge  s1 -> s2
    G.add_weighted_edges_from([(s1, n, w_s1s2) for n in range(s1)])
    #add edges s2 -> s3
    G.add_weighted_edges_from([(n, s1, w_s1s3) for n in range(s1+1, n)])
    return (G, s1)


def create_tree(s=4, depth=3):
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
    attr1_t1 = 0.2
    attr2_t1 = 0.3
    attr1_ti = 0.4
    attr2_ti = 0.9
    attr1_to = 0.1
    attr2_to = 0.6
    wi = 0.7
    wo = 0.5


    G = nx.DiGraph()  # create empty graph
    # add t1 - rootnode
    G.add_node(0, attr1=attr1_t1, attr2=attr2_t1, label='t1')

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
                [(cnt+n, {"attr1": attr1_ti, "attr2": attr2_ti, "label": name_base + "i"}) for n in range(i_range)]
                )
            G.add_weighted_edges_from([(p, cnt+n, wi) for n in range(i_range)])

            # dd outgoing nodes and edges
            G.add_nodes_from([(cnt+n, {"attr1": attr1_to, "attr2": attr2_to, "label": name_base + "o"}) 
                for n in range(i_range, s)]
                )
            G.add_weighted_edges_from([(cnt+n, p, wo) for n in range(i_range, s)])
            new_parents = new_parents = [cnt + i for i in range(s)]
            cnt = cnt + s

        parents = new_parents

    return (G, 0)

def create_bell(s):
    """
    Creates a semi connected bell and sets a transfer node

    Args:
        s (int) : size of the bell
    """

    G = create_direted_complete(s)
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


def _create_ring_sub(p, n, symbol_dic):
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

    symbol_list = [s for s in symbol_dic.values()] * n
    random.Random(4).shuffle(symbol_list)

    ring_node_cnt = p * len(symbol_list)

    # create ring
    G = nx.DiGraph()  # create empty graph
    G.add_nodes_from([(n, {"attr1": attr1, "attr2": attr2, "label":"r" + str(n % p)}) for n in range(ring_node_cnt)])
    G.add_weighted_edges_from([(n, n+1, weight ) for n in range(ring_node_cnt-1)])
    G.add_edge(ring_node_cnt-1, 0 , weight=weight)

    # add symbols to ring
    for i, s in enumerate(symbol_list):
        # union graph
        G = nx.union(G, s[0], rename=(None, "sym" + str(i) + "_"))
        src = i * p
        dst = "sym" + str(i) + "_" + str(s[1])
        G.add_edge(src, dst , weight=weight)
    G2 = nx.convert_node_labels_to_integers(G, label_attribute="old_id")

    return G2
    
