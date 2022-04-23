from math import pi, cos, sin
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from  graphcase_experiments.graphs.barbellgraphs.barbell_generator import *


def plot_directed_semi_complete(G):
    """plots the directed semi complete grpah with size n
    """
    plt.subplot(111)
    pos = nx.circular_layout(G)
    # pos = nx.spring_layout(G)
    # pos = nx.kamada_kawai_layout(G)
    color = [int(x[-1]) for _,x in sorted(nx.get_node_attributes(G,'label').items())]
    color = [float(i)/max(color) for i in color]
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
    plt.title('Directed semi complete clique')
    plt.show()

def barbel_pos(G):
    """ create a layout for the barbell graph for plotting
    """
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

def plot_directed_barbell(G, return_axes=False):
    """plots the barbell graph
    """
    _, ax = plt.subplots()
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
    nx.draw(G, **options, ax=ax)
    plt.title("Barbell graph")
    if return_axes:
        return (ax, color)
    else:
        plt.show()

def plot_embedding(G, embed):
    _, ax = plt.subplots(1,2, figsize=(20,5))
    #plot G
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
    nx.draw(G, **options, ax=ax[0])

    # plot embeding
    embed_df = pd.DataFrame(embed, columns=['id', 'embed1', 'embed2'])
    lbl_df = pd.DataFrame(
        [[i,x] for i,x in nx.get_node_attributes(G,'label').items()],
        columns=['id', 'label']
    )
    embed_df = pd.merge(embed_df, lbl_df, on='id', how='inner')

    color = [int(x[1][-1]) for x in embed_df['label'].items()]
    color = [float(i)/max(color) for i in color]

    ax[1].scatter(embed_df['embed1'], embed_df['embed2'], s=200., c=color, cmap=plt.cm.Set3_r)

    # set labels
    ids = embed_df.groupby(['label'])['id'].min()
    for lbl, i in ids.items():
        ax[1].annotate(lbl, (embed_df.loc[embed_df['id']==i]['embed1'], embed_df.loc[embed_df['id']==i]['embed2']))

    ax[1].set_xlabel("dim1")
    ax[1].set_ylabel("dim2")
    plt.title("Barbel graph: node coler represents the node role, label = node id")
    plt.show()


def plot_directed_semi_complete_8():
    G = create_direted_complete(8)
    plot_directed_semi_complete(G)

def plot_directed_barbell_8_8(): 
    G = create_directed_barbell(8, 8)
    plot_directed_barbell(G)
