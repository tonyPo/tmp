import mlflow
import pickle
import os
import networkx as nx
from graphcase_experiments.graphs.ring_graph.ring_graph_creator import create_ring
from graphcase_experiments.graphs.ring_graph.ring_graph_plotter import plot_ring
from graphcase_experiments.tools.calculate_embed import calculate_graphcase_embedding
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
EPOCHS = 1000
def search_params(trial):
    return {
        'learning_rate': trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True),
        'act': trial.suggest_categorical("act", ['relu', 'sigmoid', 'identity']),
        'useBN': trial.suggest_categorical("useBN", [True, False]),
        'dropout': trial.suggest_float("dropout", 0.0, 0.3),
        # 'support_size': trial.suggest_int("support_size", 32, 128, log=True),
        'support_size': trial.suggest_categorical("support_size", [32, 64, 128]),
        'dims': trial.suggest_int("dims", 2, 10)
    }
FIXED_PARAMS = {
        'batch_size': 30,
        'hub0_feature_with_neighb_dim': 128,
        'verbose': False,
        'seed': 1,
        'encoder_labels': ['attr1', 'attr2'],
        'epochs': 1000,
        'trials': 200
    }


def ring_exp(execute_grid_search=False, G=None):
    mlflow.set_experiment("ring_experiment")
    with mlflow.start_run():

        # create graph
        if not G:
            G = create_ring(5, 5)
        plot_ring(G)
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

    return (embed, G, tbl, res)

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

    #create table
    tbl['pred_label'] = clf.predict(X)

    return {'f1_macro': f1_macro, 'f1_micro': f1_micro, 'pred_labels': tbl['pred_label']}

def ring_exp_all():
    mlflow.set_experiment("ring_experiment_all_test")
    res={}

    #load graphCase parameters
    client = MlflowClient()
    local_path = client.download_artifacts(BEST_RUN_ID, "best_params_graphcase.pickle")
    with open(local_path, 'rb') as handle:
        params = pickle.load(handle)

    with mlflow.start_run():

        # loop though all files
        root_path = os.fsdecode(SOURCE_PATH)
        for file in os.listdir(root_path):
            if file.endswith('.pickle'):
                factor, delta, seed = decode_name(file)
                G = nx.read_gpickle(file)
                res[factor][delta][seed] = proces_graph(graph=G, params=params)
        
        #store result for logging
        with open(PATH + 'ring_downstream_results_all.pickle', 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #log artifacts
        mlflow.log_artifact(PATH + 'ring_downstream_results_all.pickle')


def proces_graph(graph, params):
    res = {}
    embed, tbl = calculate_graphcase_embedding(
            graph, PATH, params=params, epochs=EPOCHS
        )
    
    #run clustering
    res['clustering'] = cluster_test(tbl)
    mlflow.log_metric('clustering_ami', res['clustering']['ami'])

    #run classification
    res['classification'] = classify_svm(tbl)
    mlflow.log_metric('svl_f1_macro', res['classification']['f1_macro'])
    mlflow.log_metric('svl_f1_micro', res['classification']['f1_micro'])
 
    return res

def decode_name(file):
    factor = file.split('fraction')[1].split('_')[0]
    delta = file.split('delta')[1].split('_')[0]
    seed = file.split('seed')[1].split('.')[0]
    return (factor, delta, seed)