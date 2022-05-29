#%%
from base64 import decode
from sqlite3 import paramstyle
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np

# %%
n=8

attr1_s1 = 0.5
attr2_s1 = 0.75
attr1_s2 = 0.3
attr2_s2 = 0.4
attr1_s3 = 0.7
attr2_s3 = 0.6

w_s1s2 = 0.9
w_s1s3 = 0.5

s1 = int(n/2)

G = nx.DiGraph()  # create empty graph
# add s1 - centernode
G.add_node(s1, attr1=attr1_s1, attr2=attr1_s2, label='s1')
# add s2 nodes
G.add_nodes_from([(n, {"attr1": attr1_s2, "attr2": attr2_s2, "label": 's2'}) for n in range(s1)])
# add s3 nodes
G.add_nodes_from([(n, {"attr1": attr1_s3, "attr2": attr2_s3, "label": 's3'}) for n in range(s1+1,n)])
# add edge  s1 -> s2
G.add_weighted_edges_from([(s1, n, w_s1s2) for n in range(s1)])
#add edges s2 -> s3
G.add_weighted_edges_from([(n, s1, w_s1s3) for n in range(s1+1, n)])

#%%

plt.subplot(111)
# pos = nx.circular_layout(G)
pos = nx.circular_layout(G )
# pos = nx.nx_pydot.pydot_layout(G)
color = [int(x[-1])/2 for _,x in nx.get_node_attributes(G,'label').items()]
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
options = {
    'node_color': color,
    'node_size': 300,
    'edgelist':edges, 
    'edge_color':weights,
    'width': 1,
    'with_labels': True,
    'pos': pos,
    'edge_cmap': plt.cm.prism,
    'cmap': plt.cm.Wistia,
    'arrowsize': 20
}
nx.draw_networkx(G, **options)
plt.show()

#%%

from ring_graph.ring_graph_creator import create_star
from ring_graph.ring_graph_plotter import plot_star

G = create_star(9)
plot_star(G)
# %%
import networkx as nx
import matplotlib.pyplot as plt
s = 4
depth = 3

attr1_t1 = 0.2
attr2_t1 = 0.3
attr1_ti = 0.4
attr2_ti = 0.9
attr1_to = 0.1
attr2_to = 0.6
wi = 0.7
wo = 0.5


G = nx.DiGraph()  # create empty graph
# add t1 - rootnode
G.add_node(0, attr1=attr1_t1, attr2=attr2_t1, label='t1')

# add next level nodes
parents = [0]
i_range = int(s/2)
cnt = 1
for d in range(1, depth):  # loop per level
    new_parents = []
    for p in parents:  # loop per parent node
        name_base = G.nodes[p]['label'] + "_" + str(d+1)
        # add incoming nodes and edges
        G.add_nodes_from(
            [(cnt+n, {"attr1": attr1_ti, "attr2": attr2_ti, "label": name_base + "i"}) for n in range(i_range)]
            )
        G.add_weighted_edges_from([(p, cnt+n, wi) for n in range(i_range)])

        # dd outgoing nodes and edges
        G.add_nodes_from([(cnt+n, {"attr1": attr1_to, "attr2": attr2_to, "label": name_base + "o"}) 
            for n in range(i_range, s)]
            )
        G.add_weighted_edges_from([(cnt+n, p, wo) for n in range(i_range, s)])
        new_parents = new_parents = [cnt + i for i in range(s)]
        cnt = cnt + s

    parents = new_parents
#%%
from graphcase_experiments.ring_graph.ring_graph_creator import create_tree, create_bell
from graphcase_experiments.ring_graph.ring_graph_plotter import plot_ring, plot_tree


G, _ = create_bell(8)
plot_tree(G)

# %%
plt.subplot(111)
# pos = nx.circular_layout(G)
pos = tree_pos(G)
# pos = nx.nx_pydot.pydot_layout(G)
color = [(len(x), x[-1]) for _,x in nx.get_node_attributes(G,'label').items()]
color = [n if d=='i' else n+1 for n,d in color]
color = [c/max(color) for c in color]
edges,edge_weights = zip(*nx.get_edge_attributes(G,'weight').items())
options = {
    'node_color': color,
    'node_size': 300,
    'edgelist':edges, 
    'edge_color':edge_weights,
    'width': 1,
    'with_labels': True,
    'pos': pos,
    'edge_cmap': plt.cm.prism,
    'cmap': plt.cm.tab20,
    'arrowsize': 20
}
nx.draw_networkx(G, **options)
plt.show()


# %%

def tree_pos(G):
    """ create a layout for the tree graph for plotting
    """
    lvls = [(n, int((len(x)-2)/3)) for n,x in nx.get_node_attributes(G,'label').items()]
    max_lvl = int(max(lvls, key=lambda a:a[1])[1])
    pos = {}
    for l in range(max_lvl+1):
        y = 0.9 - 0.8 / (max_lvl) * l
        x_begin = 0.5 - 0.4 / max_lvl * l
        x_end = 1 - x_begin
        lvl_nodes = [n for n, d in lvls if d==l]
        for i, n in enumerate(lvl_nodes):
            if len(lvl_nodes) == 1:
                x = 0.5
            else:
                x = x_begin + (x_end - x_begin) / (len(lvl_nodes) - 1) * i
            pos[n] = [x, y]

    return pos

tree_pos(G)


# %%
from graphcase_experiments.ring_graph.ring_graph_creator import create_tree, create_bell, create_star
import networkx as nx
import numpy as np
import random

n = 2
p = 3

weight = 1
attr1 = 0.9
attr2 = 0.6

symbol_dic = {}
symbol_dic['star'] = create_star(11)
symbol_dic['tree'] = create_tree(s=4, depth=3)
symbol_dic['bell'] = create_bell(10)
symbol_list = [s for s in symbol_dic.values()] * n
random.Random(4).shuffle(symbol_list)


ring_node_cnt = p * len(symbol_list)

# create ring
G = nx.DiGraph()  # create empty graph
G.add_nodes_from([(n, {"attr1": attr1, "attr2": attr2, "label":"r" + str(n % p)}) for n in range(ring_node_cnt)])
G.add_weighted_edges_from([(n, n+1, weight ) for n in range(ring_node_cnt-1)])
G.add_edge(ring_node_cnt-1, 0 , weight=weight)

# add symbols to ring
for i, s in enumerate(symbol_list):
    # union graph
    G = nx.union(G, s[0], rename=(None, "sym" + str(i) + "_"))
    src = i * p
    dst = "sym" + str(i) + "_" + str(s[1])
    G.add_edge(src, dst , weight=weight)



# %%
import matplotlib.pyplot as plt
import pydot

plt.subplot(111)
plt.figure(figsize=(20,20))
# pos = nx.kamada_kawai_layout(G)
pos = nx.nx_pydot.graphviz_layout(G, prog='neato')
labels = [x for _,x in nx.get_node_attributes(G,'label').items()]
label_dic = {n:i for i,n in enumerate(set(labels))}
color = [label_dic[x] for _,x in nx.get_node_attributes(G,'label').items()]
edges,edge_weights = zip(*nx.get_edge_attributes(G,'weight').items())
options = {
    'node_color': color,
    'node_size': 100,
    'edgelist':edges, 
    'edge_color':edge_weights,
    'width': 1,
    'with_labels': False,
    'pos': pos,
    'edge_cmap': plt.cm.prism,
    'cmap': plt.cm.tab20,
    'arrowsize': 10
}
nx.draw_networkx(G, **options)
# 
plt.show()

# %%

G, i = create_bell(10)
# %%



plt.subplot(111)
plt.figure(figsize=(7,7))
# pos = nx.kamada_kawai_layout(G)
pos = nx.nx_pydot.graphviz_layout(G, prog='neato')
labels = [x for _,x in nx.get_node_attributes(G,'label').items()]
node_labels = {n:x for n,x in nx.get_node_attributes(G,'label').items()}
label_dic = {n:i for i,n in enumerate(set(labels))}
color = [label_dic[x] for _,x in nx.get_node_attributes(G,'label').items()]
edges,edge_weights = zip(*nx.get_edge_attributes(G,'weight').items())
options = {
    'node_color': color,
    'node_size': 400,
    'labels': node_labels,
    'edgelist':edges, 
    'edge_color':edge_weights,
    'width': 1,
    'with_labels': True,
    'pos': pos,
    'edge_cmap': plt.cm.tab10,
    'cmap': plt.cm.summer,
    'arrowsize': 10
}
nx.draw_networkx(G, **options)
# 
plt.show()
# %%

from graphcase_experiments.ring_graph.ring_graph_creator import create_tree, create_bell, create_ring, create_star
from graphcase_experiments.ring_graph.ring_graph_plotter import plot_tree, plot_bell, plot_ring,plot_star


# G = create_ring(n=2, p=3)
# plot_ring(G)
# %%
G, _ = create_star(10)
plot_star(G)

#%%

G, _ = create_tree(s=4, depth=3)
plot_tree(G)
# %%
import sys
sys.path.insert(0, '/Users/tonpoppe/workspace/GraphCase/')
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from GAE.graph_case_controller import GraphAutoEncoder
import tensorflow as tf
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from graphcase_experiments.graphs.barbellgraphs.barbell_generator import create_directed_barbell
# %%
def gridsearch_graphCASE(G, parameters, fixed_parameters=None, max_evals=10):
    """
    Executes a gridsearch for algorithm A on graph G using hyperparameter space specified in parameters.
    H is a dictionary structure

    Args:
        G (networkx graph): The graph on which the algorithm is trained
        parameters (dict): Dictionary of the hyperparameter search space.

    returns:
        tuple with the best and trails object of the hyper opt grid search.
    """
    trials = Trials()

    def f(params):
        epochs = params.pop('epochs')
        gae = GraphAutoEncoder(G, **params, **fixed_parameters)
        hist = gae.fit(epochs=epochs, layer_wise=True)
        loss = hist[None].history['val_loss'][-1]
        return {'loss': loss, 
                'train_loss': hist[None].history['loss'],
                'val_loss': hist[None].history['val_loss']}

    best = fmin(f,
            space=parameters,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials)

    return (best, trials)
#%%
G = create_directed_barbell(10, 10)

fixed_params = {
    'support_size': [3, 3],
    'dims': [2, 6, 6, 4], 
    'batch_size': 9,
    'hub0_feature_with_neighb_dim': 2,
    'verbose': True,
    'seed': 1,
    'encoder_labels': ['attr1', 'attr2']
}

params = {
    'learning_rate': hp.loguniform('learning_rate', 0.00001, 0.01),
    'act': hp.choice('act', [tf.nn.relu, tf.nn.sigmoid, tf.identity ]),
    'useBN': hp.choice('useBN', [True, False]),
    'dropout': hp.uniform('dropout', 0, 0.3),
    'epochs': hp.choice('epochs', [1000])
}

best, trails = gridsearch_graphCASE(G, params, fixed_params, max_evals=0)

# %%
import optuna
import sys
sys.path.insert(0, '/Users/tonpoppe/workspace/GraphCase/')
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from GAE.graph_case_controller import GraphAutoEncoder
import tensorflow as tf
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from graphcase_experiments.graphs.barbellgraphs.barbell_generator import create_directed_barbell

G = create_directed_barbell(10, 10)

def objective(trial):
    # Define the search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    act = trial.suggest_categorical("act", [tf.nn.relu, tf.nn.sigmoid, tf.identity])
    useBN = trial.suggest_categorical("useBN", [True, False])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    support_size = trial.suggest_int("support_size", 2, 10)
    support_size = [support_size]*2
    dim = trial.suggest_int("dim", 2, 10)
    dims = [3]+[dim]*3

    epochs = 10

    fixed_params = {
        'batch_size': 9,
        'hub0_feature_with_neighb_dim': 2,
        'verbose': False,
        'seed': 1,
        'encoder_labels': ['attr1', 'attr2']
    }

    gae = GraphAutoEncoder(G, learning_rate=learning_rate, act=act, useBN=useBN,
        dropout=dropout, support_size=support_size, dims=dims, **fixed_params)
    hist = gae.fit(epochs=epochs, layer_wise=False)
    loss = hist[None].history['val_loss'][-1]    

    trial.set_user_attr("loss", hist[None].history['loss'])
    trial.set_user_attr("val_loss", hist[None].history['val_loss'])

    return loss

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=1),
    study_name = "barbell"
)
study.set_user_attr("dataset", "barbell")
study.optimize(objective, n_trials=5)
# %%
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

plot_optimization_history(study)
# %%
plot_intermediate_values(study)
# %%
plot_contour(study)
# %%
plot_contour(study, params=["dropout", "learning_rate"])
# %%

from graphcase_experiments.tools.gridsearch import grid_search_barbell_graphcase
study, best_params = grid_search_barbell_graphcase("graphcase_experiments/data/barbell")
print(best_params)
# %%
from graphcase_experiments.experiments.barbell_exp import barbell_exp
embed, G, tbl = barbell_exp(execute_grid_search=True)


# %%
from graphcase_experiments.graphs.barbellgraphs.barbell_plotter import plot_directed_barbell, plot_embedding
plot_embedding(G, embed[:G.number_of_nodes(),:])
# %%

from graphcase_experiments.experiments.ring_exp import ring_exp
embed, tbl, res = ring_exp(execute_grid_search=True)
# res['clustering']['ami']


# %% clustering 
# number of cluster
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
# def cluster_test(tbl):
n_clusters = tbl['label'].nunique()

# .KMeans(n_clusters=8, *, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')[source]¶
columns = [x for x in tbl.columns if x.startswith('embed')]
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(tbl[columns].to_numpy())
tbl['cluster'] = kmeans.labels_

# calculate the adjused mutual information
lbl_dic = {x: i for i, x in enumerate(tbl['label'].unique())}
tbl['label_id'] = [lbl_dic[x] for x in tbl['label']]
adjusted_mutual_info_score(tbl['label_id'], tbl['cluster'])


#%% intrinsic
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


def classify_svm(tbl):
    # set parameters for grid search
    test_size = 0.5
    param_grid = {
        'C': [1, 10], 'kernel': ('linear', 'rbf')
    }
    scoring = ['f1_macro', 'f1_micro']

    # prepaire data
    columns = [x for x in tbl.columns if x.startswith('embed')]
    X = tbl[columns].to_numpy()
    y = tbl['label_id'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # execute gridsearch and train classifier
    clf = GridSearchCV(SVC(random_state=42), param_grid, scoring=scoring, cv=3, refit='f1_macro', n_jobs=-1)
    clf.fit(X_train, y_train)

    # calculate f1 score on test set
    y_pred = clf.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')

    #create tabel
    tbl['pred_label'] = clf.predict(X)


    return {'f1_macro': f1_macro, 'f1_micro': f1_micro, 'pred_labels': tbl['pred_label']}

res = classify_svm(tbl)

# %%

# %%
from sklearn.manifold import MDS
mds = MDS(n_components=2)
embed_transformed = mds.fit_transform(embed)
# %%
import numpy as np
ids = embed[:,0]
tmp = embed[:,1:]
tmp = np.column_stack([ids, tmp])

# %%
from graphcase_experiments.graphs.ring_graph.ring_graph_creator import create_tree, create_bell, create_ring, create_star
from graphcase_experiments.tools.graph_sampler import sample_graph
import pandas as pd
import networkx as nx

G = create_ring(n=2, p=3)
df1  = pd.DataFrame(
    [[i, d['attr1'], d['attr2']] for i,d in G.nodes(data=True)],
    columns = ['id', 'a1', 'a2']
)
dfe1 =  pd.DataFrame(
    [[s,d, e['weight']] for s, d, e in G.edges(data=True)],
    columns = ['s', 'd', 'w1']
)
G = sample_graph(G, 0.3, 0.8, seed=1)

# %%
import pandas as pd
df2  = pd.DataFrame(
    [[i, d['attr1'], d['attr2']] for i,d in G.nodes(data=True)],
    columns = ['id', 'b1', 'b2']
)
dfe2 =  pd.DataFrame(
    [[s,d, e['weight']] for s, d, e in G.edges(data=True)],
    columns = ['s', 'd', 'w2']
)
# %%
tmp = pd.merge(df1, df2, on='id', how='inner')
tmp['d1'] = tmp['a1'] - tmp['b1']
tmp['d2'] = tmp['a2'] - tmp['b2']
tmp.loc[tmp['d1'] != 0].shape
# %%
tmp = pd.merge(dfe1, dfe2, left_on=['s','d'], right_on=['s','d'], how='inner')
tmp['d1'] = tmp['w1'] - tmp['w2']
print(f"{tmp.loc[tmp['d1'] != 0].shape} changed edges {tmp.shape} edges in total")

#%%

import matplotlib.pyplot as plt
plt.hist(tmp.loc[tmp['d1'] != 0]['d1'])
# %%


def apply_bounds(x):
    return max(0, min(1, x))

def sample_graph(G, fraction, delta, seed=1):
    """samples for every node and edge attribute a the fraction of nodes specified by the fraction
    and changes the attribute with a random delta between -delta en delta.
    If the edge weight becomes negative then the edge is completely removed.
    For every removed edge a new edge is add from a random node to a node sampled based on the path
    length. 

    Args:
        G (networkx graph): Graph on which the sampling is applied
        fraction (float): The number nodes for which the attributes are changed.
        delta: the lower and upper bound for the change in attribute value, sampled uniform
        seed: the seed number.

    Returns:
        Graph having with updated attributes.
    """
    node_attributes = ['attr1', 'attr2']
    edge_attributes = ['weight']
    random.seed(seed)
    np.random.seed(seed)


    for attr in node_attributes:
        sample_size = int(len(nodes) * fraction)
        sampled_nodes = random.sample(nodes, sample_size)
        deltas = np.random.uniform(-delta, delta, sample_size)
        val_dic = nx.get_node_attributes(G, attr)
        old_values = [val_dic[n] for n in sampled_nodes]
        new_values = {n: apply_bounds(deltas[i] + old_values[i]) for i,n in enumerate(sampled_nodes)}
        nx.set_node_attributes(G, new_values, attr)
        
    # 

#%%
import numpy as np
import random
fraction = 0.3
edge_attributes = ['weight']
delta = 0.1
seed=1

def apply_bounds(x):
    return max(0, min(1, x))

edges = list(G.edges())
sample_size = int(len(edges) * fraction)

for attr in edge_attributes:
    sampled_edges = random.sample(edges, sample_size)
    deltas = np.random.uniform(-delta, delta, sample_size)
    val_dic = nx.get_edge_attributes(G, attr)
    new_values = {n: apply_bounds(deltas[i] + val_dic[n]) for i,n in enumerate(sampled_edges)}

    if attr=='weight':
        # select and remove zero weight edges
        zero_values = [(k, new_values.pop(k)) for k,v in list(new_values.items()) if v ==0]
        # add new edges for the removed edges
        # add_edge(G, len(zero_values))

    nx.set_edge_attributes(G, new_values, attr)

# %%
import random
import numpy as np
size = 10

def add_edge(G, count, delta):
    while count > 0:
        src = random.choice(list(G.nodes()))
        nodes = list(G.nodes())
        nodes.remove(src)
        dst = random.choice(nodes)
        # check if edge is already present
        if not (src, dst) in list(G.edges()):
            # decrease count
            G.add_edge(src, dst, weight=np.random.uniform(-delta, delta))
            count = count -1

print(f"g has {len(list(G.edges()))} edges")
add_edge(G, 100, 0.1)
print(f"g has {len(list(G.edges()))} edges")
# %%
def sample_neighbour(G, n, ):
    neighbours = list(G.to_undirected(as_view=True).neighbors(n))
    neighbour = random.choice(neighbours)


#%%
from graphcase_experiments.tools.graph_sampler import create_sampled_ring_graphs
create_sampled_ring_graphs()
# %%

import os
SOURCE_PATH = 'graphcase_experiments/graphs/sampled_ring_graphs/'
root_path = os.fsdecode(SOURCE_PATH)
for file in os.listdir(root_path):
    print(file)
# %%

def decode_name(file):
    factor = file.split('fraction')[1].split('_')[0]
    delta = file.split('delta')[1].split('_')[0]
    seed = file.split('seed')[1].split('.')[0]
    return (factor, delta, seed)

f,d,s = decode_name(file)
# %%

from graphcase_experiments.experiments.ring_exp import ring_exp_all
import tensorflow as tf
params = {
    'batch_size': 30,
    'hub0_feature_with_neighb_dim': 128,
    'verbose': False,
    'seed': 1,
    'encoder_labels': ['attr1', 'attr2'],
    'learning_rate': 0.0003668872396300966,
    'act': tf.nn.sigmoid,
    'useBN': True,
    'dropout': 0.09859650451427784,
    'support_size': [7, 7],
    'dims': [3, 128, 128, 128],
}

res = ring_exp_all(params)
#%%
import pickle
local_path = 'graphcase_experiments/data/ring/ring_downstream_results_all.pickle'
with open(local_path, 'rb') as handle:
                res = pickle.load(handle)



# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3,1)
metrics = ['ami', 'f1_macro', 'f1_micro']
for i, m in enumerate(metrics):
    ax[i].set_title(m)
    ax[i].set_xlabel('fraction')

deltas = res['delta'].unique()
groupby_df = res.sort_values('fraction').groupby(['fraction', 'delta']).mean()
for d in deltas:
    serie = groupby_df.loc[groupby_df.index.get_level_values('delta')==d]
    for i, m in enumerate(metrics):
        ax[i].plot(list(serie[m]), label=d )

plt.legend()
plt.show()


# %%
from graphcase_experiments.experiments.ring_exp import plot_results
plot_results(res)
# %%

from graphcase_experiments.graphs.barbellgraphs.barbell_generator import create_directed_barbell
from graphcase_experiments.graphs.ring_graph.ring_graph_creator import create_ring
G = create_directed_barbell(10, 10)
G = create_ring(10, 10)
# %%
G.number_of_nodes()
print(f"0\t0\t31")
# %%
import networkx as nx
import os
import numpy as np
class MultilensWrapper:
    LOCATION = 'graphcase_experiments/algos/processing_files/multilens'
    def __init__(self, **kwargs):
        pass

    def calculate_embedding(self, G):
        # define locations
        graph_file_path = MultilensWrapper.LOCATION + 'edge_list.tsv'
        category_file_path = MultilensWrapper.LOCATION + "categories.tsv"
        embedding_file_path = MultilensWrapper.LOCATION + 'multilens_embeddings.tsv'

        nx.write_weighted_edgelist(G, graph_file_path, delimiter='\t')
        with open(category_file_path, "w") as text_file:
            text_file.write(f"0\t0\t{G.number_of_nodes()}")

        # execute algoritm
        exit_status = os.system(f'source ~/opt/anaconda3/etc/profile.d/conda.sh;conda activate py2;python ../../multilens/MultiLENS/src/main.py --input {graph_file_path} --cat {category_file_path} --output {embedding_file_path}')
        print(f"MultiLENS process finished with status {exit_status}")
        if exit_status!=0:
            exit()

        # load results
        embedding = np.genfromtxt(embedding_file_path, skip_header=1)
        return embedding


       

# %%
multilens = MultilensWrapper()
embed = multilens.calculate_embedding(G)
embed
# %%
import mlflow
import pickle
from mlflow.tracking import MlflowClient
from graphcase_experiments.graphs.barbellgraphs.barbell_generator import create_directed_barbell
from graphcase_experiments.graphs.barbellgraphs.barbell_plotter import plot_directed_barbell, plot_embedding
from graphcase_experiments.tools.calculate_embed import calculate_graphcase_embedding
from graphcase_experiments.tools.gridsearch import grid_search_graphcase
from GAE.graph_case_controller import GraphAutoEncoder

BEST_RUN_ID = 'e74faa6ce3384d8aa3cbd2744fc46bae'

# client = MlflowClient()
# local_path = client.download_artifacts(BEST_RUN_ID, "best_params_graphcase.pickle")
local_path = "/Users/tonpoppe/workspace/graphcase_experiments/mlruns/2/e74faa6ce3384d8aa3cbd2744fc46bae/artifacts/best_params_graphcase_barbell.pickle"
with open(local_path, 'rb') as handle:
                    params = pickle.load(handle)
param# %%

# %%

import tensorflow as tf
from graphcase_experiments.experiments.barbell_exp import barbell_exp
from GAE.graph_case_controller import GraphAutoEncoder

params = {'batch_size': 9,
 'hub0_feature_with_neighb_dim': 2,
 'verbose': False,
 'seed': 2,
 'encoder_labels': ['attr1', 'attr2'],
 'learning_rate': 0.004189781523436639,
 'act': tf.nn.sigmoid,
 'useBN': True,
 'dropout': 0.0745080843250766,
 'support_size': [6, 6],
 'dims': [3, 16, 16, 16],
 'epochs': 100}  #20000

embed, G, tbl = barbell_exp(execute_grid_search=False, algo=GraphAutoEncoder, params=params)
# %%
from graphcase_experiments.tools.embedding_plotter import plot_embedding
plot_embedding(G, embed)


# %%
import networkx as nx
color = [int(x[-1]) for _,x in nx.get_node_attributes(G,'label').items()]
color = [float(i)/max(color) for i in color]
# %%
color_dic = [(x, i) for i,x in nx.get_node_attributes(G,'label').items()]
color_dic.sort()
# tmp = {n:i for i,n in enumerate(list(dict.fromkeys(labels)))}
# label_dic = {k:v/len(tmp.values()) for k,v in tmp.items()}
# %%

labels = [x for _,x in nx.get_node_attributes(G,'label').items()]
labels.sort()
tmp = {n:i for i,n in enumerate(list(dict.fromkeys(labels)))}
label_dic = {k:v/len(tmp.values()) for k,v in tmp.items()}
color = [label_dic[x] for _, x in embed_df['label'].items()]
# %%

from graphcase_experiments.experiments.ring_comp import ring_comp
from GAE.graph_case_controller import GraphAutoEncoder

import tensorflow as tf
params = {
    'batch_size': 30,
    'hub0_feature_with_neighb_dim': 128,
    'verbose': False,
    'seed': 1,
    'encoder_labels': ['attr1', 'attr2'],
    'learning_rate': 0.0003668872396300966,
    'act': tf.nn.sigmoid,
    'useBN': True,
    'dropout': 0.09859650451427784,
    'support_size': [7, 7],
    'dims': [3, 128, 128, 128],
    'epochs': 5
}

res = ring_comp(GraphAutoEncoder, params)
# %%
from graphcase_experiments.experiments.ring_comp import ring_comp, ring_comp_all_algos

res_df, smry_df = ring_comp_all_algos()
res_df
# %%
from graphcase_experiments.algos.dgiWrapper import DGIWrapper
from graphcase_experiments.graphs.ring_graph.ring_graph_creator import create_ring
from stellargraph import StellarGraph, StellarDiGraph
import networkx as nx
import numpy as np
import pandas as pd

G = create_ring(10, 10)
kwargs = DGIWrapper.COMP_PARAMS
algo = DGIWrapper(G, **kwargs)

#%%
import pandas as pd
PATH = 'graphcase_experiments/data/comp/backup/'
tmp = pd.read_csv(PATH + 'algo_res')
tmp = pd.concat([tmp, res_df], ignore_index=True, axis=0)
tmp.to_csv(PATH + 'algo_res_27may22', index=False)
# %%
smry_df2 = tmp.groupby(['algo','fraction','delta'])['ami','f1_macro', 'f1_micro'].agg(['mean', 'std'])
smry_df2
# %%



from graphcase_experiments.graphs.ring_graph.ring_graph_creator import create_ring
from stellargraph import StellarGraph, StellarDiGraph
import networkx as nx
import numpy as np
import pandas as pd

G = create_ring(10, 10)
G.nodes[0]


nodes = G.nodes(data=True)
at = list(nodes[0].keys())
at.remove('label')
at.remove('old_id')
attr = np.array([[n] + [a[k] for k in at] for n,a in nodes])
attr = attr[attr[:,0].argsort()]
features_df = pd.DataFrame(attr[:,1:], index=attr[:,0], columns=at)

edges = G.edges(data=True)
edges_df = pd.DataFrame([[s, d] + [v for v in a.values()] for s,d,a in edges], columns=['source', 'target', 'weight'])

G_stellar = StellarDiGraph(features_df, edges_df)
print(G_stellar.info())

# %%
from stellargraph.mapper import (
    CorruptedGenerator,
    FullBatchNodeGenerator,
    GraphSAGENodeGenerator,
    HinSAGENodeGenerator,
    ClusterNodeGenerator,
)
from stellargraph import StellarGraph
from stellargraph.layer import GCN, DeepGraphInfomax, GraphSAGE, GAT, APPNP, HinSAGE

from stellargraph import datasets
from stellargraph.utils import plot_history

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from IPython.display import display, HTML

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import Model
# %%

# base_generator = FullBatchNodeGenerator(G_stellar, sparse=False)
# base_model = GCN(layer_sizes=[128], activations=["relu"], generator=base_generator)

base_generator = GraphSAGENodeGenerator(G_stellar, batch_size=1000, num_samples=[5,5])
base_model = GraphSAGE(
    layer_sizes=[32,32], activations=["relu", "relu"], generator=base_generator
)
corrupted_generator = CorruptedGenerator(base_generator)
gen = corrupted_generator.flow(G_stellar.nodes())

infomax = DeepGraphInfomax(base_model, corrupted_generator)
x_in, x_out = infomax.in_out_tensors()

model = Model(inputs=x_in, outputs=x_out)
model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=Adam(lr=1e-3))

# %%

epochs = 100
es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
history = model.fit(gen, epochs=epochs, verbose=0, callbacks=[es])
plot_history(history)
#%%

x_emb_in, x_emb_out = base_model.in_out_tensors()
if base_generator.num_batch_dims() == 2:
        x_emb_out = tf.squeeze(x_emb_out, axis=0)
emb_model = Model(inputs=x_emb_in, outputs=x_emb_out)
all_embeddings = emb_model.predict(base_generator.flow(G_stellar.nodes()))
all_embeddings.shape
# %%
