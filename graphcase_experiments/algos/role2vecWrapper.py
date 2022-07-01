import os
import sys
import shutil
import numpy as np
import networkx as nx
from abc import ABC
from graphcase_experiments.algos.baseWrapper import BaseWrapper


currentdir = os.getcwd()
parentdir = os.path.dirname(os.path.dirname(currentdir))
parentdir = parentdir + '/role2vec/role2vec'
sys.path.insert(0, parentdir) 




class Role2VecWrapper(BaseWrapper):
    NAME = 'role2vec'
    LOCATION = 'graphcase_experiments/algos/processing_files/role2vec/'
    COMP_PARAMS ={
    }
    ENRON_PARAMS = COMP_PARAMS
    
    def __init__(self, G, **kwargs):
        self.params = {}
        for k, v in kwargs.items():
            self.params[k] = v

    def fit(self, **kwargs):
        return None

    def calculate_embeddings(self, G):
        graph_file_path = Role2VecWrapper.LOCATION + 'edge_list.csv'
        embedding_file_path = Role2VecWrapper.LOCATION + 'role2vec_embeddings.csv'

        # convert graph to edge list
        nx.write_edgelist(G, graph_file_path, delimiter=',', data=False)
      
        # execute algoritm
        param_string=''
        for k,v in self.params.items():
            param_string = param_string + ' ' + k + ' ' + str(v)
        exit_status = os.system(f'python ../../role2vec/role2vec/src/main.py --graph-input {graph_file_path} --output {embedding_file_path}{param_string}')
        print(f"Role2vec process finished with status {exit_status}")
        if exit_status!=0:
            exit()

        # load results
        embedding = np.genfromtxt(embedding_file_path, skip_header=1, delimiter=',')

        return embedding