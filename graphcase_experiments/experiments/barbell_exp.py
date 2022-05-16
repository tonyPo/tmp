import mlflow
import pickle
from mlflow.tracking import MlflowClient
from graphcase_experiments.graphs.barbellgraphs.barbell_generator import create_directed_barbell
from graphcase_experiments.graphs.barbellgraphs.barbell_plotter import plot_directed_barbell, plot_embedding
from graphcase_experiments.tools.calculate_embed import calculate_graphcase_embedding
from graphcase_experiments.tools.gridsearch import grid_search_graphcase
from graphcase_experiments.tools.embedding_plotter import plot_embedding
from GAE.graph_case_controller import GraphAutoEncoder

BEST_RUN_ID = '54d3e60cc3fc457c95218c29a561b0d6'
PATH = 'graphcase_experiments/data/barbell/'
# EPOCHS = 20000
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
        'epochs': 1000,
        'trials': 200
    }

def barbell_exp(execute_grid_search=False, algo=GraphAutoEncoder, params=None):
    mlflow.set_experiment("barbell_experiment_test")
    with mlflow.start_run():
        # create graph
        G = create_directed_barbell(10, 9)
        plot_directed_barbell(G)
        params = params

        if execute_grid_search:
            _, params = grid_search_graphcase(G, PATH, [search_params, FIXED_PARAMS])
        else:
            if not params:
                client = MlflowClient()
                local_path = client.download_artifacts(BEST_RUN_ID, "best_params_graphcase.pickle")
                with open(local_path, 'rb') as handle:
                    params = pickle.load(handle)

        embed, tbl = calculate_graphcase_embedding(
            G, PATH, params=params, epochs=params['epochs'], algo=algo
        )

        #plot 2-d embedding results
        
        plot_embedding(G, embed[:G.number_of_nodes(),:], PATH)



        #log artifacts
        mlflow.log_artifacts(PATH)
    return (embed, G, tbl)



# def __barbell_exp(execute_grid_search=False):
#     mlflow.set_experiment("barbell_experiment_test")

#     # create graph
#     G = create_directed_barbell(10, 9)
#     plot_directed_barbell(G)

#     with mlflow.start_run():
#         # execute gridsearch or load params
#         if execute_grid_search:
#             _, best_params = grid_search_graphcase(G, PATH)
#         else:
#             client = MlflowClient()
#             local_path = client.download_artifacts(BEST_RUN_ID, "best_params_graphcase_barbell.pickle")
#             with open(local_path, 'rb') as handle:
#                 best_params = pickle.load(handle)
        

#         # train model and calculate embeddings
#         epochs = best_params.pop('epochs')
#         gae = GraphAutoEncoder(G, **best_params)
#         mlflow.autolog(silent=True) 
#         hist = gae.fit(epochs=EPOCHS, layer_wise=False)
#         embed = gae.calculate_embeddings(G)

#         #save model
#         gae.save_weights(PATH + "model/saved_weights")

#         #plot results training
#         plot_loss(hist[None].history)

#         #plot 2-d embedding results
#         plot_embedding(G, embed[:G.number_of_nodes(),:], PATH)

#         #plot table
#         tbl = pd.DataFrame(embed[:29], columns=['id', 'embed1', ' embed2'])
#         lbl_df = pd.DataFrame(
#             [[i,x] for i,x in nx.get_node_attributes(G,'label').items()],
#             columns=['id', 'label']
#         )
#         tbl = pd.merge(tbl, lbl_df, on='id', how='inner')
#         tbl.to_csv(PATH + 'tabel.csv')
#         mlflow.log_artifacts(PATH)
#     return (embed, G, tbl)
