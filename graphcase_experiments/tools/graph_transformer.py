import networkx as nx

def to_undirected_node_attributes_only_graph(G, verbose=True):
    """converts a directed graph with edge attributes into an undirected graph where each edge
    is replaced by two nodes, e1 and e2. The e1 and e2 have as node attributes that are the same 
    as the original edge attibutes concatenated with the original node attributes set to zero.
    The nodes of the new graph have the original node attributes concatenated with the new 
    edge attributes set to zero.
    Two new attrbiutes 'is_in_node', 'is_out_node' are introduce to identify whether the node
    represent the incoming repsectively outvoming part of an edge.

    @params: directed graph with edge attributes.

    @returns: undirected graph with edge transformed to nodes.
    """
    G_new = nx.create_empty_copy(G, with_data=True)

    # retrieve the names of the node and edge attributes.
    node_attributes = list(G.nodes(data=True)[0].keys())
    edge_attributes = list(list(G.edges(data=True))[0][-1].keys())
    direction_attributes = ['is_in_node', 'is_out_node']
    if verbose:
        print(f"found the following node features {node_attributes}")
        print(f"found the following edge features {edge_attributes}")

    # add the edge attributes with zero values to the current nodes.
    for _, features in G_new.nodes(data=True):
        for e in edge_attributes + direction_attributes:
            print()
            features['edge_' + e] = 0


    # loop throught edges and convert to nodes.
    counter = G_new.number_of_nodes()  #number of nodes in the graph sofar
    node_attr_dict = dict(zip(node_attributes,[0]*len(node_attributes))) #dict with the node attr set to zero
    for s,d,e in G.edges(data=True):
        new_edge_attr_dict = {'edge_'+k: v for k,v in e.items()}
        direction = {'edge_is_in_node': 1, 'edge_is_out_node': 0}
        attributes = {**new_edge_attr_dict, **node_attr_dict, **direction}
        G_new.add_node(counter, **attributes)
        G_new.add_edge(s, counter)
        counter = counter + 1
        direction = {'edge_is_in_node': 0, 'edge_is_out_node': 1}
        attributes = {**new_edge_attr_dict, **node_attr_dict, **direction} 
        G_new.add_node(counter, **attributes)
        G_new.add_edge(counter - 1, counter)
        G_new.add_edge(counter, d)
        counter = counter + 1

    if verbose:
        ns = G.number_of_nodes()
        es = G.number_of_edges()
        print(f"original graph: {ns} nodes en {es} edges")
        print(f"new graph: {G_new.number_of_nodes()} nodes en {G_new.number_of_edges()} edges")
        print(f"expected {ns + 2 * es} nodes and {es * 3}")

    return G_new.to_undirected()
    

def plot_converted_graph(G):
    """ plots G for visual inspection
    """
    plt.subplot(111)
    #plot G
    pos = nx.nx_pydot.graphviz_layout(G, prog='neato')
    labels = [str(x) for _,x in nx.get_node_attributes(G,'label').items()]
    labels.sort()
    tmp = {n:i for i,n in enumerate(list(dict.fromkeys(labels)))}
    color_dic = {k:v/len(tmp.values()) for k,v in tmp.items()}
    color = [color_dic[str(x)] for _,x in nx.get_node_attributes(G,'label').items()]
    options = {
        'node_color': color,
        'node_size': 20,
        'width': 1,
        'with_labels': True,
        'pos': pos,
        'edge_cmap': plt.cm.prism,
        # 'cmap': plt.cm.Wistia,
        'cmap': plt.cm.Set3_r,
        'font_size': 8
    }
    nx.draw(G, **options)
    plt.show()
