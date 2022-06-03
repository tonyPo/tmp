import random
import numpy as np
import networkx as nx
from graphcase_experiments.graphs.ring_graph.ring_graph_creator import create_ring, create_ring2

def apply_bounds(x):
    return max(0, min(1, x))


def add_edge(G, deleted_edges, delta):
    for (src, _),e_attr in deleted_edges:
        nodes = list(G.nodes())
        neighbours = list(G.neighbors(src))
        nodes.remove(src)  # remove the source node from the list for selecting dst node
        nodes = list(set(nodes) - set(neighbours))  # remove neighbours
        dst = random.choice(nodes)
        e_attr['weight'] = np.random.power(0.2) * delta
        G.add_edge(src, dst, **e_attr)

def sample_graph(G, fraction, delta, seed=1):
    """samples for every node and edge attribute a the fraction of nodes specified by the fraction
    and changes the attribute with a random delta between -delta en delta.
    If the edge weight becomes negative then the edge is completely removed.
    For every removed edge a new edge is add from a random node to a node sampled based on the path
    length. 

    Args:
        G (networkx graph): Graph on which the sampling is applied
        fraction (float): The number nodes for which the attributes are changed.
        delta: the lower and upper bound for the change in attribute value, sampled uniform
        seed: the seed number.

    Returns:
        Graph having with updated attributes.
    """
    node_attributes = [a for a in G.nodes[0].keys() if a.startswith("attr")]
    edge_attributes = list(list(G.edges(data=True))[0][-1].keys())
    random.seed(seed)
    np.random.seed(seed)

    #update nodes
    nodes = list(G.nodes())
    sample_size = int(len(nodes) * fraction)
    for attr in node_attributes: 
        sampled_nodes = random.sample(nodes, sample_size)
        deltas = np.random.uniform(-delta, delta, sample_size)
        val_dic = nx.get_node_attributes(G, attr)
        old_values = [val_dic[n] for n in sampled_nodes]
        new_values = {n: apply_bounds(deltas[i] + old_values[i]) for i,n in enumerate(sampled_nodes)}
        nx.set_node_attributes(G, new_values, attr)
        
    
    edges = list(G.edges())
    sample_size = int(len(edges) * fraction)

    # update edges
    for attr in edge_attributes:
        
        sampled_edges = random.sample(edges, sample_size)
        deltas = np.random.uniform(-delta, delta, sample_size)
        val_dic = nx.get_edge_attributes(G, attr)
        new_values = {n: apply_bounds(deltas[i] + val_dic[n]) for i,n in enumerate(sampled_edges)}
        nx.set_edge_attributes(G, new_values, attr)

    # replace edge with negative weights with new edges
    zero_values = [((e[0], e[1]), e[2]) for e in list(G.edges(data=True)) if e[2]['weight']==0]
    if len(zero_values)>0:
        G.remove_edges_from(list(zip(*zero_values))[0])
        add_edge(G, zero_values, delta) 

    # remove isolated nodes
    degrees = G.to_undirected(as_view=True).degree()
    isolated_nodes = [n for n, d in degrees if d==0]
    G.remove_nodes_from(isolated_nodes)
    G = nx.convert_node_labels_to_integers(G)
    return G

def create_sampled_ring_graphs():
    """create 10x 10x 10 graphs based on the ring graph with noise add
    using the sample graph function ranging the fraction from 0.1 to 1
    and the delta from 0.1 to 1. using 5 seeds for each.
    """
    n = 10  # number of times figures are repeated.
    p = 5  # path lenght between figures.
    path = 'graphcase_experiments/graphs/sampled_ring_graphs/'
    
    for f in range(1, 11):
        fraction = f/10.
        for d in range(1, 11):
            delta = d/10.
            for seed in range(10):
                G = create_ring(n=n, p=p)
                G = sample_graph(G, fraction, delta, seed=seed)
                filename = f'{path}fraction{fraction}_delta{delta}_seed{seed+10}.pickle'
                nx.write_gpickle(G, filename)


def create_sampled_ring_graphs_dim5():
    """create 10x 10x 10 graphs based on the ring graph with noise add
    using the sample graph function ranging the fraction from 0.1 to 1
    and the delta from 0.1 to 1. using 5 seeds for each.
    """
    n = 10  # number of times figures are repeated.
    p = 5  # path lenght between figures.
    path = 'graphcase_experiments/graphs/sampled_ring_graphs_dim5/'

    G_orig = create_ring2(n=n, p=p, dim_node=5, dim_edge=5)
    filename = f'{path}fraction0_delta0_seed10.pickle'
    nx.write_gpickle(G_orig, filename)
    
    for f in range(1, 11):
        fraction = f/10.
        for d in range(1, 11):
            delta = d/10.
            for seed in range(10):
                G = G_orig.copy()
                G = sample_graph(G, fraction, delta, seed=seed)
                filename = f'{path}fraction{fraction}_delta{delta}_seed{seed+10}.pickle'
                nx.write_gpickle(G, filename)


def checks(path):
    cnt = []
    e_cnt = []
    for file in os.listdir(root_path):
            if file.endswith('.pickle'):
                G = nx.read_gpickle(root_path + file)
                cnt.append(G.number_of_nodes())
                e_cnt.append(G.number_of_edges())

    print(f"max nodes cnt {max(cnt))}, min node cnt {min(cnt)}")
    print(f"max edge cnt {max(e_cnt))}, min edge cnt {min(e_cnt)}")

    plt.hist(cnt)