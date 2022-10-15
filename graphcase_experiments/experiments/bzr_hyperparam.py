#%%
if __name__ == '__main__':
    import os
    os.chdir("../..")
print(os.getcwd())

#%%
import mlflow
import pickle
import os
import networkx as nx
import pandas as pd
import numpy as np
import tensorflow as tf
from graphcase_experiments.experiments.ring_exp import decode_name, proces_graph
from GAE.graph_case_controller import GraphAutoEncoder
from graphcase_experiments.algos.GraphCaseWrapper import GraphCaseWrapper

#%%

dims = [2, 4, 8, 16, 32, 64, 128, 256]
# dims = [2, 4]
dims = np.asarray(dims)
dims = np.expand_dims(dims, 1)
dims = dims.repeat(3,axis=1)
dims = [[3, *r] for r in dims]

support_size = [1, 3, 5, 7, 9, 11,13, 15]
# support_size = [3, 5]
support_size = np.asarray(support_size)
support_size = np.expand_dims(support_size, 1)
support_size = support_size.repeat(2,axis=1)
layers = [1, 2, 3, 4, 5]
# layers = [1, 2]

grid ={
    'dims': dims,
    # 'support_size': support_size,
    # 'layers': layers
}

#%%

def calc_hyperparam_sensitivity(G, ref_params, test_size = 0.5, runs=1):
    mlflow.set_experiment("ring_comp_all")
    seeds = np.random.RandomState(0).randint(0, 1000, runs)
        
    overall_results = []
    for hpar,value_list in grid.items():
        res_df = pd.DataFrame(columns=['param', 'value', 'seed', 'ami','f1_macro', 'f1_micro', 'accuracy', 'balanced_accuracy'])
        for val in value_list:
            for s in seeds:
                params = ref_params.copy()
                if hpar == 'layers':
                    params['support_size'] = [params['support_size'][0]] * val
                    params['dims'] = params['dims'][:2] + [params['dims'][1]] * ((val-1)*2)
                else:
                    params[hpar] = val
                algo_res = proces_graph(graph=G, params=params, algo=GraphCaseWrapper, seed=s, test_size = test_size)
                algo_res['param'] = hpar
                algo_res['seed'] = s
                algo_res['value'] = str(val)
                res_df = res_df.append(algo_res, ignore_index=True)
       
                res_df.to_csv(PATH + f'hpar_{hpar}_details', index=False)

        smry_df = res_df.groupby(['param','value'])['ami','f1_macro', 'f1_micro',  'accuracy', 'balanced_accuracy'].agg(['mean', 'std'])
        smry_df.to_csv(PATH + f'hpar_{hpar}_smry', index=True)
        overall_results.append(smry_df)

    res_df = pd.concat(overall_results, axis=0)

    return res_df

# %%
if __name__ == '__main__':

    PATH = 'graphcase_experiments/data/results/hyper/'  #for the results
    SOURCE_PATH = 'graphcase_experiments/graphs/bzr/bzr_graph'  #input graph
    G = nx.read_gpickle(SOURCE_PATH)
    ref_params = {'batch_size': 1024,
        'hub0_feature_with_neighb_dim': 64,
        'verbose': False,
        'seed': 1,
        'encoder_labels': ['attr1', 'attr2'],
        'learning_rate': 0.0001,
        'act': tf.nn.sigmoid,
        'useBN': True,
        'dropout': 0.1,
        'support_size': [7, 7],
        'dims': [4, 64, 64, 64],
        'epochs': 1500
    }

    res_df = calc_hyperparam_sensitivity(G, ref_params, test_size = 0.5, runs=2)
    res_df
    # %%
