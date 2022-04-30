
import networkx as nx
import matplotlib.pyplot as plt
import pydot

def plot_star(G):
    """
    plots a star pattern
    """
    plt.subplot(111)
    # pos = nx.circular_layout(G )
    pos = nx.nx_pydot.graphviz_layout(G, prog='neato')
    labels = [x for _,x in nx.get_node_attributes(G,'label').items()]
    node_labels = {n:x for n,x in nx.get_node_attributes(G,'label').items()}
    label_dic = {n:i for i,n in enumerate(set(labels))}
    color = [label_dic[x] for _,x in nx.get_node_attributes(G,'label').items()]
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    options = {
        'node_color': color,
        'node_size': 300,
        'labels': node_labels,
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


def tree_pos(G):
    """ create a layout for the tree graph for plotting
    """
    lvls = [(n, int((len(x)-2)/3)) for n,x in nx.get_node_attributes(G,'label').items()]
    max_lvl = int(max(lvls, key=lambda a:a[1])[1])
    pos = {}
    for l in range(max_lvl+1):
        y = 0.9 - 0.8 / (max_lvl) * l
        x_begin = 0.5 - 0.4 / max_lvl * l
        x_end = 1 - x_begin
        lvl_nodes = [n for n, d in lvls if d==l]
        for i, n in enumerate(lvl_nodes):
            if len(lvl_nodes) == 1:
                x = 0.5
            else:
                x = x_begin + (x_end - x_begin) / (len(lvl_nodes) - 1) * i
            pos[n] = [x, y]

    return pos

def plot_tree(G):
    plt.subplot(111)
    pos = tree_pos(G)
    labels = [x for _,x in nx.get_node_attributes(G,'label').items()]
    node_labels = {n:x for n,x in nx.get_node_attributes(G,'label').items()}
    label_dic = {n:i for i,n in enumerate(set(labels))}
    color = [label_dic[x] for _,x in nx.get_node_attributes(G,'label').items()]
    # node_labels = {n:x for n,x in nx.get_node_attributes(G,'label').items()}
    edges,edge_weights = zip(*nx.get_edge_attributes(G,'weight').items())
    options = {
        'node_color': color,
        'node_size': 100,
        # 'labels': node_labels,
        'edgelist':edges, 
        'edge_color':edge_weights,
        'width': 1,
        'with_labels': False,
        'pos': pos,
        'edge_cmap': plt.cm.prism,
        'cmap': plt.cm.tab20,
        'arrowsize': 20
    }
    nx.draw_networkx(G, **options)
    plt.show()


def plot_bell(G):
    plt.subplot(111)
    plt.figure(figsize=(7,7))
    # pos = nx.kamada_kawai_layout(G)
    pos = nx.nx_pydot.graphviz_layout(G, prog='neato')
    labels = [x for _,x in nx.get_node_attributes(G,'label').items()]
    node_labels = {n:x for n,x in nx.get_node_attributes(G,'label').items()}
    label_dic = {n:i for i,n in enumerate(set(labels))}
    color = [label_dic[x] for _,x in nx.get_node_attributes(G,'label').items()]
    edges,edge_weights = zip(*nx.get_edge_attributes(G,'weight').items())
    options = {
        'node_color': color,
        'node_size': 400,
        'labels': node_labels,
        'edgelist':edges, 
        'edge_color':edge_weights,
        'width': 1,
        'with_labels': True,
        'pos': pos,
        'edge_cmap': plt.cm.tab10,
        'cmap': plt.cm.summer,
        'arrowsize': 10
    }
    nx.draw_networkx(G, **options)
    plt.show()


def plot_ring(G):
    plt.subplot(111)
    plt.figure(figsize=(20,20))
    # pos = nx.kamada_kawai_layout(G)
    pos = nx.nx_pydot.graphviz_layout(G, prog='neato')
    labels = [x for _,x in nx.get_node_attributes(G,'label').items()]
    label_dic = {n:i for i,n in enumerate(set(labels))}
    color = [label_dic[x] for _,x in nx.get_node_attributes(G,'label').items()]
    edges,edge_weights = zip(*nx.get_edge_attributes(G,'weight').items())
    options = {
        'node_color': color,
        'node_size': 100,
        'edgelist':edges, 
        'edge_color':edge_weights,
        'width': 1,
        'with_labels': False,
        'pos': pos,
        'edge_cmap': plt.cm.prism,
        'cmap': plt.cm.tab20,
        'arrowsize': 10
    }
    nx.draw_networkx(G, **options)
    # 
    plt.show()