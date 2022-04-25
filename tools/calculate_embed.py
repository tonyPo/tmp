import pickle
import mlflow
import pandas as pd
import networkx as nx
from graphcase_experiments.graphs.barbellgraphs.barbell_generator import create_directed_barbell
from graphcase_experiments.graphs.barbellgraphs.barbell_plotter import plot_directed_barbell, plot_embedding
from graphcase_experiments.tools.gridsearch import grid_search_graphcase, plot_loss
from GAE.graph_case_controller import GraphAutoEncoder
from mlflow.tracking import MlflowClient


def calculate_graphcase_embedding(G, path, grid=None, run_id=None, epochs=1000):    
    # execute gridsearch or load params
    if grid:
        _, best_params = grid_search_graphcase(G, path, grid)
    else:
        client = MlflowClient()
        local_path = client.download_artifacts(run_id, "best_params_graphcase.pickle")
        with open(local_path, 'rb') as handle:
            best_params = pickle.load(handle)
    
    # train model and calculate embeddings
    best_params.pop('epochs')
    gae = GraphAutoEncoder(G, **best_params)
    mlflow.autolog(silent=True) 
    hist = gae.fit(epochs=epochs, layer_wise=False)
    embed = gae.calculate_embeddings(G)

    #save model
    gae.save_weights(path + "model/saved_weights")

    #plot results training
    plot_loss(hist[None].history)

    #create table
    tbl = create_table(embed, G)
    tbl.to_csv(path + 'tabel.csv')
        
    return (embed, G, tbl)

def create_table(embed, G):
    columns = ['id'] + ['embed' + str(i) for i in range(embed.shape[1] -1)]
    tbl = pd.DataFrame(embed[:G.number_of_nodes()], columns=columns)
    lbl_df = pd.DataFrame(
        [[i,x] for i,x in nx.get_node_attributes(G,'label').items()],
        columns=['id', 'label']
    )
    tbl = pd.merge(tbl, lbl_df, on='id', how='inner')

    # create numeric label id
    lbl_dic = {x: i for i, x in enumerate(tbl['label'].unique())}
    tbl['label_id'] = [lbl_dic[x] for x in tbl['label']]
    return tbl