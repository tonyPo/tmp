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


class BaselineWrapper(BaseWrapper):
    NAME = 'baseline'
    LOCATION = 'graphcase_experiments/algos/processing_files/baseline/'
    COMP_PARAMS ={
    }
    ENRON_PARAMS = COMP_PARAMS
    def __init__(self, G, **kwargs):
        self.params = kwargs

    def fit(self, **kwargs):
        return None

    def calculate_embeddings(self, G):
        # create node attribute matrix
        nodes = G.nodes(data=True)
        at = list(nodes[0].keys())
        at.remove('label')
        at.remove('old_id')
        attr = np.array([[n] + [a[k] for k in at] for n,a in nodes])
        attr = attr[attr[:,0].argsort()]

        return attr
