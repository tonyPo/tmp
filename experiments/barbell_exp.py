import sys
import pickle
import mlflow
import pandas as pd
import networkx as nx
sys.path.insert(0, '/Users/tonpoppe/workspace/GraphCase/')
from graphcase_experiments.graphs.barbellgraphs.barbell_generator import create_directed_barbell
from graphcase_experiments.graphs.barbellgraphs.barbell_plotter import plot_directed_barbell, plot_embedding
from graphcase_experiments.tools.gridsearch import grid_search_barbell_graphcase, plot_loss
from GAE.graph_case_controller import GraphAutoEncoder
from mlflow.tracking import MlflowClient

def barbell_exp(execute_grid_search=False):
    mlflow.set_experiment("barbell_experiment_test")
    # execute gridsearch or load params
    if execute_grid_search:
        _, best_params = grid_search_barbell_graphcase()
    else:
        client = MlflowClient()
        local_path = client.download_artifacts('54d3e60cc3fc457c95218c29a561b0d6', "best_params_graphcase_barbell.pickle")
        with open(local_path, 'rb') as handle:
            best_params = pickle.load(handle)
    
    # create graph
    G = create_directed_barbell(10, 9)
    plot_directed_barbell(G)

    # train model and calculate embeddings
    epochs = best_params.pop('epochs')
    epochs = 10000
    gae = GraphAutoEncoder(G, **best_params)
    mlflow.autolog(silent=True) 
    hist = gae.fit(epochs=epochs, layer_wise=False)
    embed = gae.calculate_embeddings(G)

    #plot results training
    plot_loss(hist[None].history)

    #plot 2-d embedding results
    plot_embedding(G, embed[:G.number_of_nodes(),:])

    #plot table
    tbl = pd.DataFrame(embed[:29], columns=['id', 'embed1', ' embed2'])
    lbl_df = pd.DataFrame(
        [[i,x] for i,x in nx.get_node_attributes(G,'label').items()],
        columns=['id', 'label']
    )
    tbl = pd.merge(tbl, lbl_df, on='id', how='inner')
    tbl
    return (embed, G, tbl)
