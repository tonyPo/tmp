
#%%
import pickle
import math
from graphcase_experiments.experiments.barbell_exp import barbell_exp
from graphcase_experiments.graphs.barbellgraphs.barbell_generator import create_directed_barbell
from graphcase_experiments.graphs.barbellgraphs.barbell_plotter import plot_directed_barbell, plot_embedding
from graphcase_experiments.tools.calculate_embed import calculate_graphcase_embedding
from graphcase_experiments.tools.gridsearch import grid_search_graphcase
from graphcase_experiments.tools.embedding_plotter import plot_embedding, plot_embedding2, plotly_embedding
from GAE.graph_case_controller import GraphAutoEncoder
import tensorflow as tf
from GAE.graph_case_controller import GraphAutoEncoder
from scipy.stats import t
import pandas as pd

#%% functions
def calc_confidence_interval(sd, n=10):
    t_student =  t.ppf( 0.975, n-1)
    return t_student * sd / math.sqrt(n)

def print_bold(val, max_val, col):
    if val in list(max_val[col]):
        return "\textbf{" + val + "}"
    else:
        return val

metrics = ['ami', 'f1_micro', 'f1_macro']
metric_name = {
    'ami score': 'ami score',
    'f1_micro score': 'F1 micro score',
    'f1_macro score': 'F1 macro score'
}

algo_names = {
    "xnetmf_with_transformation": "xNetMF transformed",
    "xnetmf": 'xNetMF', 
    "Drne": 'DRNE',
    'GraphCASE': 'GraphCase', 
    "MultiLENS": 'Multi-LENS',
    "role2vec": "Role2Vec",
}

#%% synthetic graphs

path = 'graphcase_experiments/data/results/ringgraph/'
filename = path + 'benchmark_results.pickle'
latex_file = path + 'benchmark_results.tex'
tstudent_d9_05 = 2.262


with open(filename, 'rb') as handle:
    pdf = pickle.load(handle)

#%%
for m in metrics:
    pdf[m+'_ci'] = pdf.loc[:,('ami', 'std')].apply(calc_confidence_interval)
    pdf[m+' score'] = pdf.apply(lambda row: f"{row[(m, 'mean')]:.3f} \u00B1 {row[m+'_ci'][0]:.3f}", axis=1)

# drop unnecessary levels
pdf = pdf.droplevel(level='fraction')
pdf = pdf.droplevel(level=1, axis=1)

# rename 
pdf.index.names = ['Algorithm', 'Delta']
pdf = pdf.rename(columns=metric_name)
res = pdf.rename(mapper=algo_names, axis=0)

# select relevant colums
res = res[['ami score', 'F1 micro score']]
res = res.reset_index().sort_values(["Delta", 'Algorithm'])

# print higest score bold
max_values = res.groupby('Delta').max()

for c in ['ami score', 'F1 micro score']:
    res[c] = res[c].apply(lambda x:print_bold(x, max_values, c))
res

# save a latex
res.to_latex(latex_file, 
    index=False, 
    caption="AMI score of the embedding clustering and F1 scores on the embedding classification of the synthetic graph for threee noise levels",
    escape=False
)


# %%  BZR + ENRON
# -------------------------------------------------------------
path = 'graphcase_experiments/data/results/reallife/'
filename_enron = path + 'enron_results.pickle'
filename_bzr = path + 'bzr_results.pickle'
latex_file = path + 'reallife_results.tex'

with open(filename_enron, 'rb') as handle:
    enron = pickle.load(handle)

with open(filename_bzr, 'rb') as handle:
    bzr = pickle.load(handle)

enron['Dataset'] = 'Enron'
bzr['Dataset'] = 'BZR'
res = pd.concat([enron, bzr], axis=0)
res = res.set_axis([res.Dataset, res.index])

for m in metrics:
    res[m+'_ci'] = res.loc[:,('ami', 'std')].apply(calc_confidence_interval)
    res[m+' score'] = res.apply(lambda row: f"{row[(m, 'mean')]:.3f} \u00B1 {row[m+'_ci'][0]:.3f}", axis=1)


# drop unnecessary levels
res = res.droplevel(level=1, axis=1)

# rename 
res.index.names = ['Dataset', 'Algorithm']
res = res.rename(columns=metric_name)
res = res.rename(mapper=algo_names, axis=0)
# select relevant colums
res = res[['ami score', 'F1 micro score']]
res = res.sort_values(['Dataset', 'Algorithm'])

# print higest score bold
max_values = res.groupby('Dataset').max()

for c in ['ami score', 'F1 micro score']:
    res[c] = res[c].apply(lambda x:print_bold(x, max_values, c))

res

#  save a latex
res.to_latex(latex_file, 
    index=True, 
    caption="AMI score of the embedding clustering and F1 scores on the embedding classification of the enron and BZR graph with 95\% confidence interval levels",
    escape=False
)
# %%
