# Databricks notebook source
# PATH = '/dbfs/ton/graphcase/data/ring/'
# mlflow.set_experiment("/Users/antonius.b.a.poppe@nl.abnamro.com/graphcase_experiments")
# print("param/n")
# print(params)

# mlflow.set_experiment("/Users/antonius.b.a.poppe@nl.abnamro.com/ring_experiment_all_test")

# COMMAND ----------

#create graph
import networkx as nx
SOURCE_PATH = 'graphcase_experiments/graphs/enron/data/'
G = nx.read_gpickle(SOURCE_PATH + 'enron_sub_graph3.pickle')

# COMMAND ----------

from graphcase_experiments.tools.gridsearch import grid_search_graphcase
def search_params(trial):
    return {
        'learning_rate': trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
        'act': trial.suggest_categorical("act", ['relu', 'sigmoid', 'identity']),
        'useBN': trial.suggest_categorical("useBN", [True]),
        'dropout': trial.suggest_float("dropout", 0.1, 0.2),
        'support_size': trial.suggest_int("support_size", 7, 10),
        'dims': trial.suggest_int("dims", 32, 128)
    }
FIXED_PARAMS = {
       'batch_size': 30,
        'hub0_feature_with_neighb_dim': 128,
        'verbose': False,
        'seed': 1,
        'encoder_labels': ['attr_received_size', 'attr_cnt_to', 'attr_cnt_cc', 'attr_sent_size', 'attr_cnt_send'],
        'epochs': 1000,
        'trials': 200
}

                
PATH = '/dbfs/ton/graphcase/data/ring/'

grid_search_res = grid_search_graphcase(G, PATH, [search_params, FIXED_PARAMS])

# COMMAND ----------

grid_search_res[1]

# COMMAND ----------

from graphcase_experiments.experiments.enron_comp import calc_enron_performance
from graphcase_experiments.algos.GraphCaseWrapper import GraphCaseWrapper
from graphcase_experiments.tools.calculate_embed import calculate_graphcase_embedding
# ind, res = calc_enron_performance(algos= [GraphCaseWrapper],G=G, test_size=0.5, runs=1)
PATH = '/dbfs/ton/graphcase/data/ring/'
_, tbl = calculate_graphcase_embedding(
            G, PATH, params=GraphCaseWrapper.ENRON_PARAMS, verbose=True, algo=GraphCaseWrapper
        )

# COMMAND ----------

import graphcase_Experiments.experiments.bzr_hyperparam


# COMMAND ----------


