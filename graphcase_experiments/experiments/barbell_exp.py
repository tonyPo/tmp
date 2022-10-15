import mlflow
import pickle
from mlflow.tracking import MlflowClient
from graphcase_experiments.graphs.barbellgraphs.barbell_generator import create_directed_barbell
from graphcase_experiments.graphs.barbellgraphs.barbell_plotter import plot_directed_barbell, plot_embedding
from graphcase_experiments.tools.calculate_embed import calculate_graphcase_embedding
from graphcase_experiments.tools.gridsearch import grid_search_graphcase
from graphcase_experiments.tools.embedding_plotter import plot_embedding3
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
        'epochs': 10,  #1000,
        'trials': 3  #200
    }

def barbell_exp(execute_grid_search=False, algo=GraphAutoEncoder, params=None, return_model=False):
    mlflow.set_experiment("barbell_experiment_test")
    with mlflow.start_run():
        # create graph
        G = create_directed_barbell(10, 9)
        plot_directed_barbell(G)
        params = params

        if execute_grid_search:
            _, params = grid_search_graphcase(G, PATH, [search_params, FIXED_PARAMS])
        else:
            if not params and algo==GraphAutoEncoder:
                client = MlflowClient()
                local_path = client.download_artifacts(BEST_RUN_ID, "best_params_graphcase.pickle")
                with open(local_path, 'rb') as handle:
                    params = pickle.load(handle)

        embed, tbl, mdl = calculate_graphcase_embedding(
            G, PATH, params=params, algo=algo, return_model=True
        )

        #plot 2-d embedding results
        
        plot_embedding3(tbl, PATH)



        #log artifacts
        mlflow.log_artifacts(PATH)
    if return_model:
        return (embed, G, tbl, mdl)
    else:
        return (embed, G, tbl)