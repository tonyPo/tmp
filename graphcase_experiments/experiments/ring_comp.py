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



PATH = 'graphcase_experiments/data/comp/'
SOURCE_PATH = 'graphcase_experiments/graphs/sampled_ring_graphs/'

graphs = {
    'fractions': ['0.5'],
    'delta': ['0.3', '0.5', '0.7']
}


def ring_comp(algo=GraphAutoEncoder, params=None):
    mlflow.set_experiment("ring_comp_test")
    res_df = pd.DataFrame(columns=['fraction','delta','seed','ami','f1_macro', 'f1_micro'])

    with mlflow.start_run():
        # create graph
        root_path = os.fsdecode(SOURCE_PATH)
        for file in os.listdir(root_path):
            if file.endswith('.pickle'):
                fraction, delta, seed = decode_name(file)
                if (fraction in graphs['fractions']) and (delta in graphs['delta']) and (int(seed) < 23):
                    G = nx.read_gpickle(root_path + file)
                    res_run = {"fraction": fraction, "delta": delta, 'seed': seed, **proces_graph(graph=G, params=params, algo=algo)}
                    res_df = res_df.append(res_run, ignore_index=True)
        
        #store result for logging
        with open(PATH + 'ring_results_comp.pickle', 'wb') as handle:
            pickle.dump(res_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #log artifacts
        mlflow.log_artifact(PATH + 'ring_results_comp.pickle')

        return res_df

ALGO = [
    GraphCaseWrapper, 
    MultilensWrapper,
    DrneWrapper
    ]
def ring_comp_all_algos(algos=ALGO):
    res_df = pd.DataFrame(columns=['algo', 'fraction','delta','seed','ami','f1_macro', 'f1_micro'])
    for algo in algos:
        algo_res = ring_comp(algo, algo.COMP_PARAMS)
        algo_res['algo'] = algo.NAME
        res_df = res_df.append(algo_res, ignore_index=True)

    res_df.to_csv(PATH + 'algo_res', index=False)

    smry_df = res_df.groupby(['algo','fraction','delta'])['ami','f1_macro', 'f1_micro'].agg(['mean', 'std'])
    smry_df.to_csv(PATH + 'smry_res', index=False)

    return res_df, smry_df
