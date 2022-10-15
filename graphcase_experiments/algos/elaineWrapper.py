import os
import shutil
import numpy as np
import networkx as nx
from abc import ABC
from graphcase_experiments.algos.baseWrapper import BaseWrapper


class ElaineWrapper(BaseWrapper):
    NAME = 'Elaine'
    LOCATION = 'graphcase_experiments/algos/processing_files/Elaine/'
    COMP_PARAMS ={
        '--epochs': '300',
        '--batch_size': '30'
    }
    ENRON_PARAMS = {
        '--epochs': '300',
        '--batch_size': '30'
    }
    BZR_PARAMS = {
        '--epochs': '300',
        '--batch_size': '1024'
    }
    MOOC_PARAMS = {
        '--epochs': '3',
        '--batch_size': '1024'
    }

    def __init__(self, G, **kwargs):
        self.params = {}
        for k, v in kwargs.items():
            self.params[k] = v

    def fit(self, **kwargs):
        return None

    def calculate_embeddings(self, G):
        input_path = ElaineWrapper.LOCATION + 'graph.pickle'
        output_path = ElaineWrapper.LOCATION + 'embed.npy'
        # save graph
        nx.write_gpickle(G, input_path, protocol=4)

        # execute algoritm
        param_string=''
        for k,v in self.params.items():
            param_string = param_string + ' ' + k + ' ' + str(v)
        exit_status = os.system(f'source ~/opt/anaconda3/etc/profile.d/conda.sh;conda activate drne;python ../../elaine/ELAINE-tensorflow/ELAINE2.py --input_path {input_path} --output_path {output_path}{param_string}')
        print(f"Elaine process finished with status {exit_status}")
        if exit_status!=0:
            exit()

        # load results
        embedding = np.load(output_path)
        ids = np.array(range(embedding.shape[0]))

        return np.hstack([ids[:, None], embedding])