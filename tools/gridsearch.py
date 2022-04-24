
import optuna
import sys
import pickle
import time
import mlflow
sys.path.insert(0, '/Users/tonpoppe/workspace/GraphCase/')
import tensorflow as tf
import matplotlib.pyplot as plt
from GAE.graph_case_controller import GraphAutoEncoder
from graphcase_experiments.graphs.barbellgraphs.barbell_generator import create_directed_barbell
from graphcase_experiments.graphs.barbellgraphs.barbell_plotter import plot_directed_barbell
from optuna.visualization import plot_contour
from optuna.visualization import plot_optimization_history
PATH = 'graphcase_experiments/data/barbell/'
ACT_DICT  = {
            'relu': tf.nn.relu,
            'sigmoid': tf.nn.sigmoid,
            'identity': tf.identity
        }

EPOCHS = 3 #1000
TRAILS = 2 #200
def objective(trial, G):
    # Define the search space
    params = {
        'learning_rate': trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        'act': trial.suggest_categorical("act", ['relu', 'sigmoid', 'identity']),
        'useBN': trial.suggest_categorical("useBN", [True, False]),
        'dropout': trial.suggest_float("dropout", 0.0, 0.3),
        'support_size': trial.suggest_int("support_size", 2, 10),
        'dims': trial.suggest_int("dims", 2, 10)
    }
    params = optuna_to_model_params_converter(params)

    epochs = EPOCHS

    fixed_params = {
        'batch_size': 9,
        'hub0_feature_with_neighb_dim': 2,
        'verbose': False,
        'seed': 1,
        'encoder_labels': ['attr1', 'attr2']
    }

    # create and train model
    gae = GraphAutoEncoder(G, **params, **fixed_params)
    hist = gae.fit(epochs=epochs, layer_wise=False)
    loss = hist[None].history['val_loss'][-1]    

    #set attributes
    trial.set_user_attr("loss", hist[None].history['loss'])
    trial.set_user_attr("val_loss", hist[None].history['val_loss'])
    trial.set_user_attr("fixed_params", fixed_params)

    return loss

def grid_search_graphCase(G):
    """Applies a grid search with hte graphCase algorithm on  graph G
    """

    # run grid search
    start_time = time.time()
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=1),
        study_name = "barbell"
    )
    study.set_user_attr("dataset", "barbell")
    study.optimize(lambda trial: objective(trial, G), n_trials=TRAILS)
    duration = time.time() - start_time
    print(f"grid search took {duration:.1f} seconds to run, {duration / TRAILS} per run")

    # create plots
    plot_optimization_history(study)
    plot_contour(study, params=["dropout", "learning_rate"])
    plot_loss(study.best_trial.user_attrs)

    return study

def grid_search_barbell_graphcase():
    G = create_directed_barbell(10, 9)
    plot_directed_barbell(G)
    mlflow.set_tag("type", "gridsearch") 
    study = grid_search_graphCase(G)

    # retrieve best parameters
    best_params = {
        **study.best_trial.user_attrs['fixed_params'],
        **study.best_trial.params,
        'epochs': len(study.best_trial.user_attrs['loss'])
    }
    best_params = optuna_to_model_params_converter(best_params)



    with open(PATH + 'gridsearch_graphcase_barbell.pickle', 'wb') as handle:
        pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PATH + 'best_params_graphcase_barbell.pickle', 'wb') as handle:
        pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    return (study, best_params)

def plot_loss(loss_dict):
    plt.plot(loss_dict['loss'], label='loss')
    plt.plot(loss_dict['val_loss'], label='val_loss')
    plt.yscale("log")
    plt.xlabel('epoch')
    plt.ylabel('loss (log)')
    plt.legend()
    plt.show()

def optuna_to_model_params_converter(params):
    """Converts the parameter from the Optuna grod search into parameters that can be fed into the model
    """
    model_params = params.copy()
    model_params['act'] = ACT_DICT[params['act']]
    model_params['support_size'] = [params['support_size']]*2
    model_params['dims'] = [3]+[params['dims']]*3
    return model_params

