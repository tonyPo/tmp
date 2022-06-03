import mlflow
import pickle
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from graphcase_experiments.graphs.ring_graph.ring_graph_creator import create_ring
from graphcase_experiments.graphs.ring_graph.ring_graph_plotter import plot_ring
from graphcase_experiments.tools.calculate_embed import calculate_graphcase_embedding
from GAE.graph_case_controller import GraphAutoEncoder
from graphcase_experiments.tools.embedding_plotter import plot_embedding
from graphcase_experiments.tools.gridsearch import grid_search_graphcase
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from mlflow.tracking import MlflowClient

BEST_RUN_ID = '54d3e60cc3fc457c95218c29a561b0d6'
PATH = 'graphcase_experiments/data/ring/'
SOURCE_PATH = 'graphcase_experiments/graphs/sampled_ring_graphs/'
EPOCHS = 20
def search_params(trial):
    return {
        'learning_rate': trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True),
        'act': trial.suggest_categorical("act", ['relu', 'sigmoid', 'identity']),
        'useBN': trial.suggest_categorical("useBN", [True]),
        'dropout': trial.suggest_float("dropout", 0.0, 0.3),
        'support_size': trial.suggest_int("support_size", 2, 8),
        'dims': trial.suggest_categorical("dims", [64, 128]),
    }
FIXED_PARAMS = {
        'batch_size': 30,
        'hub0_feature_with_neighb_dim': 128,
        'verbose': False,
        'seed': 1,
        'encoder_labels': ['attr1', 'attr2'],
        'epochs': 200,
        'trials': 200
    }


def ring_exp(execute_grid_search=False, G=None):
    mlflow.set_experiment("ring_experiment")
    with mlflow.start_run():

        # create graph
        if not G:
            G = create_ring(5, 5)
        # plot_ring(G)
        res = {}

        if execute_grid_search:
            _, params = grid_search_graphcase(G, PATH, [search_params, FIXED_PARAMS])
        else:
            client = MlflowClient()
            local_path = client.download_artifacts(BEST_RUN_ID, "best_params_graphcase.pickle")
            with open(local_path, 'rb') as handle:
                params = pickle.load(handle)

        embed, tbl = calculate_graphcase_embedding(
            G, PATH, params=params, epochs=EPOCHS
        )

        #plot 2-d embedding results
        plot_embedding(G, embed[:G.number_of_nodes(),:], PATH)

        #run clustering
        res['clustering'] = cluster_test(tbl)
        mlflow.log_metric('clustering_ami', res['clustering']['ami'])
        
        #run classification
        res['classification'] = classify_svm(tbl)
        mlflow.log_metric('svl_f1_macro', res['classification']['f1_macro'])
        mlflow.log_metric('svl_f1_micro', res['classification']['f1_micro'])

        #store result for logging
        with open(PATH + 'ring_downstream_results.pickle', 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #log artifacts
        mlflow.log_artifacts(PATH)

    return (embed, G, tbl, res, params)

def cluster_test(tbl):
    """Performs a cluster analysis on the embedding dims and check the metrics
    with 
    """
    # perfrom clustering
    n_clusters = tbl['label'].nunique()
    columns = [x for x in tbl.columns if x.startswith('embed')]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(tbl[columns].to_numpy())
    tbl['cluster'] = kmeans.labels_

    # calculate the adjusted_mutual_info_score
    lbl_dic = {x: i for i, x in enumerate(tbl['label'].unique())}
    tbl['label_id'] = [lbl_dic[x] for x in tbl['label']]
    res = adjusted_mutual_info_score(tbl['label_id'], tbl['cluster'])

    return {'ami': res, 'clusters': tbl['label_id']}


def classify_svm(tbl):
    # set parameters for grid search
    test_size = 0.75
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

    #create table
    tbl['pred_label'] = clf.predict(X)

    return {'f1_macro': f1_macro, 'f1_micro': f1_micro, 'pred_labels': tbl['pred_label']}

def ring_exp_all(params):
    mlflow.set_experiment("ring_experiment_all_test")
    res_df = pd.DataFrame(columns=['fraction','delta','seed','ami','f1_macro', 'f1_micro'])

    #load graphCase parameters
    if not params:
        client = MlflowClient()
        local_path = client.download_artifacts(BEST_RUN_ID, "best_params_graphcase.pickle")
        with open(local_path, 'rb') as handle:
            params = pickle.load(handle)

    with mlflow.start_run():

        # loop though all files
        root_path = os.fsdecode(SOURCE_PATH)
        for file in os.listdir(root_path):
            if file.endswith('.pickle'):
                fraction, delta, seed = decode_name(file)
                if (seed =='10') and (delta=='0.3'):
                    G = nx.read_gpickle(root_path + file)
                    res_run = {"fraction": fraction, "delta": delta, 'seed': seed, **proces_graph(graph=G, params=params)}
                    res_df = res_df.append(res_run, ignore_index=True)
        
        #store result for logging
        with open(PATH + 'ring_downstream_results_all.pickle', 'wb') as handle:
            pickle.dump(res_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #log artifacts
        mlflow.log_artifact(PATH + 'ring_downstream_results_all.pickle')

        return res_df


def proces_graph(graph, params, algo=GraphAutoEncoder):
    res = {}
    _, tbl = calculate_graphcase_embedding(
            graph, PATH, params=params, verbose=False, algo=algo
        )
    
    #run clustering
    cluster_res = cluster_test(tbl)
    res['ami'] = cluster_res['ami']
    mlflow.log_metric('clustering_ami', cluster_res['ami'])

    #run classification
    svm_res = classify_svm(tbl)
    res['f1_macro'] = svm_res['f1_macro']
    res['f1_micro'] = svm_res['f1_micro']
    mlflow.log_metric('svm_f1_macro', res['f1_macro'])
    mlflow.log_metric('svm_f1_micro', res['f1_micro'])
 
    return res

def decode_name(file):
    factor = file.split('fraction')[1].split('_')[0]
    delta = file.split('delta')[1].split('_')[0]
    seed = file.split('seed')[1].split('.')[0]
    return (factor, delta, seed)

def plot_results(res):
    fig, ax = plt.subplots(3,1, figsize=(10,30))
    metrics = ['ami', 'f1_macro', 'f1_micro']
    for i, m in enumerate(metrics):
        ax[i].set_title(m)
        ax[i].set_xlabel('fraction')

    deltas = res['delta'].unique()
    deltas.sort()
    groupby_df = res.sort_values('fraction').groupby(['fraction', 'delta']).mean()
    for d in deltas:
        serie = groupby_df.loc[groupby_df.index.get_level_values('delta')==d]
        for i, m in enumerate(metrics):
            ax[i].plot(list(serie[m]), label=d )

    plt.legend(loc=(1.04,0), title = "max_delta")
    plt.show()
 
    
