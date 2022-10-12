# Databricks notebook source
# MAGIC %pip install pydot
# MAGIC %pip install optuna
# MAGIC %pip install graphcase
# MAGIC %pip install pyvis

# COMMAND ----------

from graphcase_experiments.experiments import bzr_hyperparam

# COMMAND ----------

import os
os.chdir("antonius.b.a.poppe@nl.abnamro.com_old/tmp_graphcase")

# COMMAND ----------

import networkx as nx
import tensorflow as tf
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



# COMMAND ----------

# MAGIC %md
# MAGIC - params goed zetten
# MAGIC - runs op 10

# COMMAND ----------

res_df = bzr_hyperparam.calc_hyperparam_sensitivity(G, ref_params, PATH, test_size = 0.5, runs=5)
# res_df

# COMMAND ----------

res_df

# COMMAND ----------


