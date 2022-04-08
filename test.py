#%%
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np

# %%
n=8
w_b1b2 = 1
w_b1 = 0.8
w_b2 = 0.6

# %%

b1 = int(n/2)

G = nx.DiGraph()  # create empty graph
# add nodes with labels and attributes
G.add_nodes_from([(n, {"attr1": 0.3, "attr2": 0.7, "label": 'b1'}) for n in range(b1)])
G.add_nodes_from([(n, {"attr1": 0.5, "attr2": 0.3, "label": 'b2'}) for n in range(b1, n)])

# add edges between b1 and b2
for inc_n in range(b1):
    G.add_weighted_edges_from([(n, inc_n, w_b1b2) for n in range(b1, n)])

# add edges within b1
b1_combinations = itertools.permutations(range(b1), 2)
G.add_weighted_edges_from([(l, r, w_b1) for l,r in b1_combinations])

# add edges within b2
b2_combinations = itertools.permutations(range(b1, n), 2)
G.add_weighted_edges_from([(l, r, w_b2) for l,r in b2_combinations])
# %%


plt.subplot(111)
pos = nx.circular_layout(G)
# pos = nx.spring_layout(G)
# pos = nx.kamada_kawai_layout(G)
color = [int(x[-1])/2 for _,x in sorted(nx.get_node_attributes(G,'label').items())]
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
options = {
    'node_color': color,
    'node_size': 300,
    'edgelist':edges, 
    'edge_color':weights,
    'width': 1,
    'with_labels': True,
    'pos': pos,
    'edge_cmap': plt.cm.prism,
    'cmap': plt.cm.Wistia,
    'arrowsize': 20
}
nx.draw_networkx(G, **options)
plt.show()

# %%

from barbellgraphs.barbell_generator import *
m1 = 4
m2 = 4

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
#%%
from importlib import reload  # Python 3.4+

import barbellgraphs.barbell_plotter
reload(barbellgraphs.barbell_plotter)


barbellgraphs.barbell_plotter.plot_directed_semi_complete_8()
barbellgraphs.barbell_plotter.plot_directed_barbell_8_8()



# %%
def plot_directed_barbell(G):
    """plots the directed semi complete grpah with size n
    """
    plt.subplot(111)
    pos = barbel_pos(G)
    color = [int(x[-1]) for _,x in nx.get_node_attributes(G,'label').items()]
    color = [float(i)/max(color) for i in color]
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    options = {
        'node_color': color,
        'node_size': 200,
        'edgelist':edges, 
        'edge_color':weights,
        'width': 1,
        'with_labels': True,
        'pos': pos,
        'edge_cmap': plt.cm.prism,
        # 'cmap': plt.cm.Wistia,
        'cmap': plt.cm.Set3_r,
        'arrowsize': 20,
        'font_size': 8
    }
    nx.draw_networkx(G, **options)
    plt.show()

plot_directed_barbell(G)
# %%
import math
from math import pi, cos, sin

def barbel_pos(G):
    nodes = sorted(G.nodes(data=True))
    circles = [n for n, a in nodes if a['label'] in ['b1', 'b2', 'b3']]
    line = [n for n,_ in nodes if n not in circles]
    cutoff= int(len(circles)/2)
    circle1 = circles[:cutoff]
    circle2 = circles[cutoff:]

    def add_circle(pos, circle, center):
        size = 0.2
        steps = len(circle)
        for i, n in enumerate(reversed(circle)):
            pos[n] = [center[0] + sin(2*pi / steps * i) * size,
                    center[1] + cos(2*pi / steps * i + pi) * size]

    def add_line(pos, line):
        path_length =int(len(line) / 2)
        x = .5
        for i in range(path_length+1):
            dx = (0.5 - 0.25) / path_length * i
            y = 0.1 + (0.50 - 0.1) / (path_length + 1) * i
            pos[line[path_length+i]] = [x - dx, y]
            pos[line[path_length-i]] = [x + dx, y]



    pos = {}
    add_circle(pos, circle1, [0.75, 0.75])
    add_circle(pos, circle2, [0.25, 0.75])
    add_line(pos, line)

    return pos



barbel_pos(G)
#%%%




# %%
