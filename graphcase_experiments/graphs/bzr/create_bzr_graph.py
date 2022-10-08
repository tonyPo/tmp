#%%
import os
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
path = 'graphcase_experiments/graphs/bzr/'
edge_file = path + 'BZR-MD.edges'
edge_weight_file = path + "BZR-MD.link_attrs"
edge_feature_file = path + "BZR-MD.link_labels"
node_labels_path = path + 'BZR-MD.node_labels'

edges = pd.read_csv(edge_file, sep=',', header=None)
weights = pd.read_csv(edge_weight_file, header=None)
edge_feature = pd.read_csv(edge_feature_file, header=None)
edge_feature = pd.get_dummies(edge_feature[0], prefix='edge')
edge_feature['weight'] = weights/ weights.max()

G = nx.DiGraph()
G.add_edges_from(zip(edges[0], edges[1]))

cols = edge_feature.columns
attrs ={}
for index, row in edge_feature.iterrows():
    attr = {k:row[k] for k in cols}
    s = edges.loc[index][0]
    d = edges.loc[index][1]
    attrs[(s,d)] = attr

nx.set_edge_attributes(G, attrs)
#%% Add node attribute based on weighted degree
degrees = dict(nx.degree(G, weight='weight'))
max_degree = max(degrees.values())
degrees = {k: v/max_degree for k,v in degrees.items()}
nx.set_node_attributes(G, degrees, name= "attr1")
#%% add node labels
node_labels = pd.read_csv(node_labels_path, header=None)
node_labels[1] = node_labels[1].astype(str).copy()
node_labels.loc[node_labels[1]=='7',1] = 'no'  # only 1 labels
node_labels.loc[node_labels[1]=='8',1] = 'no'  # only 2 labels

labels = {l[0]: l[1] + "_label" for i,l in node_labels.iterrows()}
nx.set_node_attributes(G, labels, name= "label")

#%% relabel last node to zero to make sure node ids start at zero
max_node_id = G.number_of_nodes()
cur_neighbors = list(G.neighbors(max_node_id))
nx.relabel_nodes(G, {max_node_id: 0}, copy=False)
new_neighbors = list(G.neighbors(0))
print(f"list are same: {set(cur_neighbors)==set(new_neighbors)}")
#%%
nx.write_gpickle(G, path+"bzr_graph", protocol=5)
# %%  checks on graph

#%% node label distribution
nodes = pd.DataFrame([n[1] for n in G.nodes(data=True)])


fig = plt.figure(figsize=(10,10))
attr = [c for c in nodes.columns]

for i, a in enumerate(attr):
        ax = fig.add_subplot(int(len(attr)/2), 2, i+1)
        ax.hist(nodes[a])
        ax.set_title(f"attr {a}") 

# %% edge label distribution
edges = pd.DataFrame([n[2] for n in G.edges(data=True)])

fig = plt.figure(figsize=(10,20))
attr = [c for c in edges.columns]

for i, a in enumerate(attr):
        ax = fig.add_subplot(int(len(attr)/2), 2, i+1)
        ax.hist(edges[a])
        ax.set_title(f"attr {a}") 

# %%
