import os
import shutil
import numpy as np
import networkx as nx
from abc import ABC
from graphcase_experiments.algos.baseWrapper import BaseWrapper


class DrneWrapper(BaseWrapper):
    NAME = 'Drne'
    LOCATION = 'graphcase_experiments/algos/processing_files/drne/'
    COMP_PARAMS ={
        '-s': '16',
        '--undirected': 'False'
    }
    MOOC_PARAMS = {
        '-s': '16',
        '--undirected': 'False'
    }
    BZR_PARAMS = MOOC_PARAMS

    ENRON_PARAMS = {
        '-s': '128',
        '--undirected': 'False'
    }
    def __init__(self, G, **kwargs):
        self.params = {}
        for k, v in kwargs.items():
            self.params[k] = v

    def fit(self, **kwargs):
        return None

    def calculate_embeddings(self, G):
        # define locations
        graph_file_path = DrneWrapper.LOCATION + 'edge_list.tsv'
        embedding_path = DrneWrapper.LOCATION + 'drne_res'
        shutil.rmtree(embedding_path, ignore_errors=True)  # remove tree
        suffix_path = 'test'

        # convert graph to edge list and create node category list
        nx.write_edgelist(G, graph_file_path, delimiter='\t')

        # execute algoritm
        param_string=''
        for k,v in self.params.items():
            param_string = param_string + ' ' + k + ' ' + str(v)
        exit_status = os.system(f'source ~/opt/anaconda3/etc/profile.d/conda.sh;conda activate drne;python ../../drne/DRNE/src/main.py --data_path {graph_file_path} --save_path {embedding_path} --save_suffix {suffix_path}{param_string}')
        print(f"MultiLENS process finished with status {exit_status}")
        if exit_status!=0:
            exit()

        # load results
        embedding = np.load(embedding_path + '/' + os.listdir(embedding_path)[0] + '/embeddings.npy')
        ids = np.array(range(embedding.shape[0]))

        return np.hstack([ids[:, None], embedding])