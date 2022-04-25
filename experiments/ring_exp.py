import mlflow
from graphcase_experiments.graphs.ring_graph.ring_graph_creator import create_ring
from graphcase_experiments.graphs.ring_graph.ring_graph_plotter import plot_ring
from graphcase_experiments.tools.calculate_embed import calculate_graphcase_embedding
from graphcase_experiments.tools.embedding_plotter import plot_embedding
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score


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
        'hub0_feature_with_neighb_dim': 2,
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

        #log artifacts
        mlflow.log_artifacts(PATH)

        #run clustering
        res['clustering'] = cluster_test(tbl)
        mlflow.log_metric('clustering_ami', res['clustering']['ami'])
        #run intrinsic_cor

        #run classification
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

def intrinsic_cor(tbl):
    return None

def classify_svm(tbl):
    return None

