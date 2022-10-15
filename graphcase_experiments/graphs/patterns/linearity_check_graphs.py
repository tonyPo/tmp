#%%
import os
os.chdir("../../..")
os.getcwd()

#%%
import pickle
import networkx as nx
import numpy as np
from graphcase_experiments.graphs.ring_graph.ring_graph_plotter import plot_ring


#%%
"""
create a simple graph with one central node 2 outgoing nodes and 2 incoming nodes
the graph will be used to test the sub pattern can be identified in the embedding
"""


weights1 = [0.5, 0.9]
weights2 = [0.5, 0.9]
node_type_cnt = 10
dim_node = 5
dim_edge = 5


np.random.seed(4)

node_attributes = np.random.uniform(0.0, 1.0, (node_type_cnt, dim_node)) 
node_attributes = [{"attr"+str(n+1):v for n,v in enumerate(row)} for row in node_attributes]

def construct_weight_attr(weights):
    edge_attributes = np.random.uniform(0.0, 1.0, (len(weights), dim_edge))
    edge_attributes[:,0] = weights
    edge_attributes = [{"weight" if n==0 else "e_attr"+str(n):v for n,v in enumerate(row)} for row in edge_attributes]
    return edge_attributes

edge_attributes1 = construct_weight_attr(weights1)
edge_attributes2 = construct_weight_attr(weights2)

def create_base_graph(node_attributes):
    G = nx.DiGraph()  # create empty graph
    # add centernode
    G.add_node(0, **node_attributes[0], label='c1')
    return G

def add_varying_part(G, node_attributes, edge_attributes):
    # add  first incoming node string
    G.add_node(1, **node_attributes[1], label = 'i1') 
    G.add_node(2, **node_attributes[2], label = 'i1i1') 
    G.add_edge(1, 0, **edge_attributes[0])
    G.add_edge(2, 1, **edge_attributes[0])
    # add first outgoing node string
    G.add_node(3, **node_attributes[3], label = 'o1') 
    G.add_node(4, **node_attributes[4], label = 'o1o1') 
    G.add_edge(0, 3 , **edge_attributes[0])
    G.add_edge(3, 4 , **edge_attributes[0])
    

def add_fix_part1(G, node_attributes, edge_attributes):
    node_id = G.number_of_nodes()
    G.add_node(node_id, **node_attributes[node_id], label = 'i2') 
    G.add_node(node_id+1, **node_attributes[node_id+1], label = 'i2i1') 
    G.add_edge(node_id, 0 , **edge_attributes[1])
    G.add_edge(node_id+1, node_id , **edge_attributes[1])

    G.add_node(node_id+2, **node_attributes[node_id+2], label = 'o2') 
    G.add_node(node_id+3, **node_attributes[node_id+3], label = 'o2o1') 
    G.add_edge(0, node_id+2 , **edge_attributes[1])
    G.add_edge(node_id+2, node_id+3, **edge_attributes[1])

def add_fix_part2(G, node_attributes, edge_attributes):
    node_id = G.number_of_nodes()
    G.add_node(node_id, **node_attributes[node_id], label = 'i2') 
    G.add_node(node_id+1, **node_attributes[node_id+1], label = 'i2i1') 
    G.add_node(node_id+2, **node_attributes[node_id+2], label = 'i2i2') 
    G.add_edge(node_id, 0 , **edge_attributes[1])
    G.add_edge(node_id+1, node_id , **edge_attributes[0])
    G.add_edge(node_id+2, node_id , **edge_attributes[1])


path = 'graphcase_experiments/graphs/patterns/'

for i, w in enumerate([edge_attributes1, edge_attributes2]):
    G1_with = create_base_graph(node_attributes)
    add_varying_part(G1_with, node_attributes, w)
    add_fix_part1(G1_with, node_attributes, w)

    G1_without = create_base_graph(node_attributes)
    add_fix_part1(G1_without, node_attributes, w)

    G2_with = create_base_graph(node_attributes)
    add_varying_part(G2_with, node_attributes, w)
    add_fix_part2(G2_with, node_attributes, w)

    G2_without = create_base_graph(node_attributes)
    add_fix_part2(G2_without, node_attributes, w)

    for n in ['G1_with', 'G1_without', 'G2_with', 'G2_without']:
        with open(path + n + f'v{i}.pickle', 'wb') as handle:
            pickle.dump(globals()[n], handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%




# %%
