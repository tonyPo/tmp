import pickle
import mlflow
import pandas as pd
import networkx as nx
from graphcase_experiments.graphs.barbellgraphs.barbell_generator import create_directed_barbell
from graphcase_experiments.graphs.barbellgraphs.barbell_plotter import plot_directed_barbell, plot_embedding
from graphcase_experiments.tools.gridsearch import grid_search_graphcase, plot_loss
from GAE.graph_case_controller import GraphAutoEncoder



def calculate_graphcase_embedding(G, path, params, verbose=True, algo=GraphAutoEncoder, return_model=False):       
    # train model and calculate embeddings
    params = params.copy()  
    epochs = params.pop('epochs', None)
    algo_instance = algo(G, **params)
    mlflow.autolog(silent=True) 
    hist = algo_instance.fit(epochs=epochs, layer_wise=False)
    embed = algo_instance.calculate_embeddings(G)

    #save model
    if algo==GraphAutoEncoder:
        algo_instance.save_weights(path + "model/saved_weights")

    #plot results training
    if verbose and hist is not None:
      plot_loss(hist[None].history)

    #create table
    tbl = create_table(embed, G)
#     tbl.to_csv(path + 'tabel.csv')
    if return_model:
        return (embed, tbl, algo_instance)
    else:
        return (embed, tbl)

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