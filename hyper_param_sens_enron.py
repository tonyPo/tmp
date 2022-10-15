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
PATH = '/dbfs/mnt/dseedsi/users/ton/enron/'  #for the results
SOURCE_PATH = 'graphcase_experiments/graphs/enron/data/enron_sub_graph4.pickle'  #input graph
G = nx.read_gpickle(SOURCE_PATH)
ref_params = {'batch_size': 30,
    'hub0_feature_with_neighb_dim': 128,
    'verbose': False,
    'seed': 1,
    'encoder_labels': ['attr1', 'attr2'],
    'learning_rate': 0.0002,
    'act': tf.nn.sigmoid,
    'useBN': True,
    'dropout': 0.14,
    'support_size': [10, 10],
    'dims': [4, 112, 112, 112],
    'epochs': 1500
 }



# COMMAND ----------

res_df = bzr_hyperparam.calc_hyperparam_sensitivity(G, ref_params, PATH, test_size = 0.5, runs=5)
# res_df

# COMMAND ----------

res_df

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ls /dbfs/mnt/dseedsi/users/ton/enron

# COMMAND ----------

|
