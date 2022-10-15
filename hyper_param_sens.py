# Databricks notebook source
# MAGIC %pip install pydot
# MAGIC %pip install optuna
# MAGIC %pip install graphcase
# MAGIC %pip install pyvis
# MAGIC %pip install tensorflow==2.8.0

# COMMAND ----------

import tensorflow
tensorflow.__version__

# COMMAND ----------

from graphcase_experiments.experiments import bzr_hyperparam

# COMMAND ----------

import os
os.chdir("/Workspace/Repos/antonius.b.a.poppe@nl.abnamro.com_old/tmp_graphcase")

# COMMAND ----------

import networkx as nx
import tensorflow as tf
PATH = '/dbfs/mnt/dseedsi/users/ton/bzr/'  #for the results
SOURCE_PATH = 'graphcase_experiments/graphs/bzr/bzr_graph'  #input graph
G = nx.read_gpickle(SOURCE_PATH)
ref_params = {'batch_size': 1024,
    'hub0_feature_with_neighb_dim': 128,
    'verbose': False,
    'seed': 1,
    'encoder_labels': ['attr1', 'attr2'],
    'learning_rate': 0.000366887,
    'act': tf.nn.sigmoid,
    'useBN': True,
    'dropout': 0.1,
    'support_size': [7, 7],
    'dims': [3, 128, 128, 128],
    'epochs': 200
 }



# COMMAND ----------

res_df = bzr_hyperparam.calc_hyperparam_sensitivity(G, ref_params, PATH, test_size=0.5, runs=5)
# res_df

# COMMAND ----------

res_df

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ls /dbfs/mnt/dseedsi/users/ton/bzr/

# COMMAND ----------

import pandas as pd


# COMMAND ----------

G.number_of_nodes()

# COMMAND ----------

G.number_of_edges()

# COMMAND ----------

import pandas as pd
res = pd.DataFrame([v for k,v in G.nodes(data=True)])
res.groupby('label').mean()

# COMMAND ----------



# COMMAND ----------

pd.DataFrame([v for s,d,v in G.edges(data=True)]).describe()

# COMMAND ----------

import tensorflow
tensorflow.__version__

# COMMAND ----------


