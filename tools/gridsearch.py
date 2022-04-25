
import optuna
import sys
import pickle
import time
import mlflow
import tensorflow as tf
import matplotlib.pyplot as plt
from GAE.graph_case_controller import GraphAutoEncoder
from graphcase_experiments.graphs.barbellgraphs.barbell_generator import create_directed_barbell
from graphcase_experiments.graphs.barbellgraphs.barbell_plotter import plot_directed_barbell
from optuna.visualization import plot_contour
from optuna.visualization import plot_optimization_history
ACT_DICT  = {
            'relu': tf.nn.relu,
            'sigmoid': tf.nn.sigmoid,
            'identity': tf.identity
        }

def objective(trial, G, grid, epochs):
    # Define the search space
    params = optuna_to_model_params_converter(grid[0](trial))
    fixed_params = grid[1]

    # create and train model
    gae = GraphAutoEncoder(G, **params, **fixed_params)
    hist = gae.fit(epochs=epochs, layer_wise=False)
    loss = hist[None].history['val_loss'][-1]    

    #set attributes
    trial.set_user_attr("loss", hist[None].history['loss'])
    trial.set_user_attr("val_loss", hist[None].history['val_loss'])
    trial.set_user_attr("fixed_params", fixed_params)

    return loss


def grid_search_graphcase(G, PATH, grid):
    mlflow.set_tag("type", "gridsearch") 

    # run grid search
    start_time = time.time()
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    trials = grid[1].pop('trials')
    epochs = grid[1].pop('epochs')
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=1),
        study_name = "barbell"
    )
    study.set_user_attr("dataset", "barbell")
    study.optimize(lambda trial: objective(trial, G, grid, epochs), n_trials=trials)
    duration = time.time() - start_time
    print(f"grid search took {duration:.1f} seconds to run, {duration / trials} per run")

    # create plots
    plot_optimization_history(study)
    plot_contour(study, params=["dropout", "learning_rate"])
    plot_loss(study.best_trial.user_attrs)

    # retrieve best parameters
    best_params = {
        **study.best_trial.user_attrs['fixed_params'],
        **study.best_trial.params,
        'epochs': len(study.best_trial.user_attrs['loss'])
    }
    best_params = optuna_to_model_params_converter(best_params)

    with open(PATH + 'gridsearch_graphcase.pickle', 'wb') as handle:
        pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PATH + 'best_params_graphcase.pickle', 'wb') as handle:
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

