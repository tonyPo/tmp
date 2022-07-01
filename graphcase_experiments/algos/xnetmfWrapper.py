import os
import sys
import shutil
import numpy as np
import networkx as nx
from abc import ABC
from graphcase_experiments.algos.baseWrapper import BaseWrapper
from graphcase_experiments.tools.graph_transformer import to_undirected_node_attributes_only_graph

currentdir = os.getcwd()
parentdir = os.path.dirname(os.path.dirname(currentdir))
parentdir = parentdir + '/xnetmf/REGAL'
sys.path.insert(0, parentdir) 

import config as xnet
from xnetmf import get_representations


class XnetmfWrapper(BaseWrapper):
    NAME = 'xnetmf'
    LOCATION = 'graphcase_experiments/algos/processing_files/xnetmf/'
    COMP_PARAMS ={
        'max_layer': 2, 
        'alpha': 0.01, 
        'k': 10, 
        'num_buckets': 2, 
        'normalize': True, 
        'gammastruc': 1, 
        'gammaattr': 1
    }
    ENRON_PARAMS = COMP_PARAMS
    def __init__(self, G, **kwargs):
        self.params = kwargs

    def fit(self, **kwargs):
        return None

    def calculate_embeddings(self, G):
        # create undirected adjency matrix
        adj = nx.adjacency_matrix(G.to_undirected(), nodelist = range(G.number_of_nodes()))

        # create node attribute matrix
        nodes = G.nodes(data=True)
        at = list(nodes[0].keys())
        at.remove('label')
        at.remove('old_id')
        attr = np.array([[n] + [a[k] for k in at] for n,a in nodes])
        attr = attr[attr[:,0].argsort()]

        # create graph object
        graph = xnet.Graph(adj, node_attributes = attr[:,1:])

        # create rep_method
        rep_method = xnet.RepMethod(**self.params)

        # create embedding
        embed = get_representations(graph, rep_method)
        ids = np.array(range(embed.shape[0]))

        return np.hstack([ids[:, None], embed])

class XnetmfWrapperWithGraphTransformation(XnetmfWrapper):
    NAME = 'xnetmf_with_transformation'
    LOCATION = 'graphcase_experiments/algos/processing_files/xnetmf/'
    COMP_PARAMS ={
        'max_layer': 6, 
        'alpha': 0.01, 
        'k': 10, 
        'num_buckets': 2, 
        'normalize': True, 
        'gammastruc': 1, 
        'gammaattr': 1
    }
    ENRON_PARAMS = COMP_PARAMS
    def calculate_embeddings(self, G):
        G_undirected = to_undirected_node_attributes_only_graph(G, verbose=False)
        return super().calculate_embeddings(G_undirected)
