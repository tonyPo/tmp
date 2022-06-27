import os
import numpy as np
import networkx as nx
from abc import ABC
from graphcase_experiments.algos.baseWrapper import BaseWrapper

class MultilensWrapper(BaseWrapper):
    NAME = 'MultiLENS'
    LOCATION = 'graphcase_experiments/algos/processing_files/multilens'
    COMP_PARAMS ={
        '--dim': 128,
        '--L': 2,
        '--base': 2
    }
    ENRON_PARAMS = {
        '--dim': 128,
        '--L': 2,
        '--base': 2
    }
    def __init__(self, G, **kwargs):
        self.params = {}
        for k, v in kwargs.items():
            self.params[k] = v

    def fit(self, **kwargs):
        return None

    def calculate_embeddings(self, G):
        # define locations
        graph_file_path = MultilensWrapper.LOCATION + 'edge_list.tsv'
        category_file_path = MultilensWrapper.LOCATION + "categories.tsv"
        embedding_file_path = MultilensWrapper.LOCATION + 'multilens_embeddings.tsv'

        # convert graph to edge list and create node category list
        nx.write_weighted_edgelist(G, graph_file_path, delimiter='\t')
        with open(category_file_path, "w") as text_file:
            text_file.write(f"0\t0\t{G.number_of_nodes()}")

        # execute algoritm
        param_string=''
        for k,v in self.params.items():
            param_string = param_string + ' ' + k + ' ' + str(v)
        exit_status = os.system(f'source ~/opt/anaconda3/etc/profile.d/conda.sh;conda activate py2;python ../../multilens/MultiLENS/src/main.py --input {graph_file_path} --cat {category_file_path} --output {embedding_file_path}{param_string}')
        print(f"MultiLENS process finished with status {exit_status}")
        if exit_status!=0:
            exit()

        # load results
        embedding = np.genfromtxt(embedding_file_path, skip_header=1)
        return embedding