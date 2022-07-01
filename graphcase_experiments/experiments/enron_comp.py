import mlflow
import pickle
import os
import networkx as nx
import pandas as pd
from graphcase_experiments.experiments.ring_exp import decode_name, proces_graph
from GAE.graph_case_controller import GraphAutoEncoder
from graphcase_experiments.algos.GraphCaseWrapper import GraphCaseWrapper
from graphcase_experiments.algos.MultiLENSwrapper import MultilensWrapper
from graphcase_experiments.algos.drneWrapper import DrneWrapper
from graphcase_experiments.algos.xnetmfWrapper import XnetmfWrapper, XnetmfWrapperWithGraphTransformation
from graphcase_experiments.algos.role2vecWrapper import Role2VecWrapper
from graphcase_experiments.algos.dgiWrapper import DGIWrapper, DGIWrapperWithGraphTransformation

"""Please run first:
- email_util to convert the email boxes into a table
- mail_reader to conver the table into a graph.
"""


PATH = 'graphcase_experiments/data/enron/'  #for the results
SOURCE_PATH = 'graphcase_experiments/graphs/enron/data/enron_graph.pickle'  #input graph

ALGO = [
    GraphCaseWrapper, 
    MultilensWrapper,
    DrneWrapper,
    XnetmfWrapper,
    XnetmfWrapperWithGraphTransformation,
    Role2VecWrapper,
    DGIWrapper,
    # DGIWrapperWithGraphTransformation
    ]
def calc_enron_performance(algos=ALGO, G=None, source_path=SOURCE_PATH, test_size = 0.75):
    mlflow.set_experiment("ring_comp_all")
    if G is None:
        G = nx.read_gpickle(source_path)
    with mlflow.start_run():
        mlflow.log_param('algos', algos)
        res_df = pd.DataFrame(columns=['algo', 'ami','f1_macro', 'f1_micro'])
        for algo in algos:
            algo_res = proces_graph(graph=G, params=algo.ENRON_PARAMS, algo=algo, test_size = 0.75)
            algo_res['algo'] = algo.NAME
            res_df = res_df.append(algo_res, ignore_index=True)
        
        res_df.to_csv(PATH + 'algo_res', index=False)

        smry_df = res_df.groupby(['algo'])['ami','f1_macro', 'f1_micro'].agg(['mean', 'std'])
        smry_df.to_csv(PATH + 'smry_res', index=False)

        #log artifacts
        mlflow.log_artifacts(PATH)

    return res_df, smry_df
