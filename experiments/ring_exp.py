import mlflow
import pickle
from graphcase_experiments.graphs.ring_graph.ring_graph_creator import create_ring
from graphcase_experiments.graphs.ring_graph.ring_graph_plotter import plot_ring
from graphcase_experiments.tools.calculate_embed import calculate_graphcase_embedding
from graphcase_experiments.tools.embedding_plotter import plot_embedding
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


BEST_RUN_ID = '54d3e60cc3fc457c95218c29a561b0d6'
PATH = 'graphcase_experiments/data/ring/'
EPOCHS = 10 #1000
def search_params(trial):
    return {
        'learning_rate': trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        'act': trial.suggest_categorical("act", ['relu', 'sigmoid', 'identity']),
        'useBN': trial.suggest_categorical("useBN", [True, False]),
        'dropout': trial.suggest_float("dropout", 0.0, 0.3),
        'support_size': trial.suggest_int("support_size", 2, 10),
        'dims': trial.suggest_int("dims", 2, 10)
    }
FIXED_PARAMS = {
        'batch_size': 9,
        'hub0_feature_with_neighb_dim': 4,
        'verbose': False,
        'seed': 1,
        'encoder_labels': ['attr1', 'attr2'],
        'epochs': 2, #1000
        'trials': 2 # 200
    }


def ring_exp(execute_grid_search=False):
    mlflow.set_experiment("ring_experiment_test")
    with mlflow.start_run():
        # create graph
        G = create_ring(5, 5)
        plot_ring(G)
        params = None
        res = {}

        if execute_grid_search:
            params = [search_params, FIXED_PARAMS]

        embed, G, tbl = calculate_graphcase_embedding(
            G, PATH, grid=params, run_id=BEST_RUN_ID, epochs=EPOCHS
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

