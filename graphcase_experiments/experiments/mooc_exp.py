
#%% imports
# import os
# os. chdir("../..")
# os.getcwd()
#%%
import mlflow
import pickle
import os
import xgboost as xgb
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier

from graphcase_experiments.tools.calculate_embed import calculate_graphcase_embedding
from GAE.graph_case_controller import GraphAutoEncoder
from graphcase_experiments.algos.elaineWrapper import ElaineWrapper
from graphcase_experiments.algos.GraphCaseWrapper import GraphCaseWrapper
from graphcase_experiments.algos.MultiLENSwrapper import MultilensWrapper
from graphcase_experiments.algos.drneWrapper import DrneWrapper
from graphcase_experiments.algos.xnetmfWrapper import XnetmfWrapper, XnetmfWrapperWithGraphTransformation
from graphcase_experiments.algos.role2vecWrapper import Role2VecWrapper
from graphcase_experiments.algos.dgiWrapper import DGIWrapper, DGIWrapperWithGraphTransformation
from graphcase_experiments.algos.baselineWrapper import BaselineWrapper
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score, auc

SOURCE_PATH = 'graphcase_experiments/graphs/mooc/'  #input graph
SAVE_PATH = 'graphcase_experiments/data/mooc/'
MOOC_ACTION_FILE = 'mooc_actions.tsv'
MOOC_FEATURES_FILE = 'mooc_action_features.tsv'
MOOC_LABEL_FILE = 'mooc_action_labels.tsv'
FEATURE_NAMES = ['FEATURE0', 'FEATURE1', 'FEATURE2', 'FEATURE3']
percentage = 20


#%% load edge, add and normalize features
def get_edge_data():
    """load edge, add and normalize features"""
    # load and join data
    mooc_action = pd.read_csv(SOURCE_PATH + MOOC_ACTION_FILE, sep='\t')
    features = pd.read_csv(SOURCE_PATH + MOOC_FEATURES_FILE, sep='\t')
    edges = mooc_action.merge(features, on='ACTIONID', how='inner')
    print(f"edges: {edges.shape[0]} mooc_action {mooc_action.shape[0]}")

    # normalize log of features
    df = edges[FEATURE_NAMES]
    df = df + 1  # add1 to avoid log scaling of 0
    df = df.apply(lambda x: np.log10(x))
    normalized_features=(df-df.min())/(df.max()-df.min())
    edges = edges[['ACTIONID', 'USERID', 'TARGETID', 'TIMESTAMP']].merge(normalized_features, left_index=True, right_index=True)
    edges['src'] = "U" + edges['USERID'].astype(str)
    edges['dst'] = "T" + edges['TARGETID'].astype(str)
    return edges

def mooc_train_test_split(percentage, edges):
    """select test and trainset"""
    total_cnt = edges.shape[0]

    # select test edges based on percentage with highest timestamp
    test_size = int(total_cnt / 100 * percentage)
    edges_test = edges.nlargest(test_size, 'TIMESTAMP')
    test_cnt = edges_test.shape[0]

    # select train edges 
    train_size = total_cnt - test_size
    edges_train = edges.nsmallest(train_size, 'TIMESTAMP')
    train_cnt = edges_train.shape[0]

    print(f"total count is {total_cnt}, test_cnt is {test_cnt} and train_cnt is {train_cnt}: check_sum: {total_cnt - test_cnt - train_cnt}")

    return (edges_train, edges_test)

def prep_edges(edges):
    """add edges weight and select relevant columns"""
    min_value = edges['TIMESTAMP'].min()
    max_value = edges['TIMESTAMP'].max()
    edges['weight'] = (edges['TIMESTAMP'] - min_value) / (max_value - min_value)  # use timestamp as edge weight

    #prefix userid and target ids
    edges['src'] = "U" + edges['USERID'].astype(str)
    edges['dst'] = "T" + edges['TARGETID'].astype(str)
    edges = edges [['src', 'dst', 'weight', 'ACTIONID'] + FEATURE_NAMES]

    edges = edges.sort_values('weight', ascending=True)
    # add feature with the number of action between user and mooc
    edges['FEATURE_COUNT'] =edges.groupby(['src','dst'])['weight'].transform('count') +1
    edges['FEATURE_COUNT'] = edges['FEATURE_COUNT'].apply(lambda x: np.log10(x))
    edges['FEATURE_COUNT']=(edges['FEATURE_COUNT']-edges['FEATURE_COUNT'].min())/(edges['FEATURE_COUNT'].max()-edges['FEATURE_COUNT'].min())

    edges = edges.drop_duplicates(subset=['src','dst'], keep='last')
    print(f"there are {edges.shape[0]} edges after selecting only the last one")
    return edges

def get_mooc_graph(train_edges):
    # remove ACTIONID filed
    graph_edges = train_edges.drop("ACTIONID", axis='columns')
    print(f"edge colunms {graph_edges.columns}")
    G = nx.from_pandas_edgelist(graph_edges, 'src', 'dst', edge_attr=True, create_using=nx.DiGraph)
    print(f"edge check: df:{graph_edges.shape[0]}, graph edges: {G.number_of_edges()}")
    print(f"node check: users+ courses:{graph_edges['src'].nunique() + graph_edges['dst'].nunique()}, nodes: {G.number_of_nodes()}")

    #update node attribute
    for n in G.nodes(data=True):
        if n[0].startswith("U"):
            n[1]['attr_is_user']=1
        else:
            n[1]['attr_is_user']=0

    G = nx.convert_node_labels_to_integers(G, label_attribute='label')
    return G

def prep_mdl_input(df_actions, df_embed):
    labels = pd.read_csv(SOURCE_PATH + MOOC_LABEL_FILE, sep='\t')
    labels =labels.drop_duplicates(subset='ACTIONID')
    train_svm = df_actions.merge(df_embed, left_on='src', right_on='label')
    train_svm = train_svm.merge(df_embed, left_on='dst', right_on='label', suffixes=("_src", "_dst"))
    train_svm = train_svm.merge(labels, on='ACTIONID', how='inner')
    train_svm['label_id'] = train_svm['LABEL']
    rename_dict = {n: "embed_edge"+str(i) for i,n in enumerate(FEATURE_NAMES + ['FEATURE_COUNT'])}
    train_svm.rename(columns=rename_dict, inplace=True)
    return train_svm

def train_lgbm(tbl, seed=1):
    # prepaire data
    columns = [x for x in tbl.columns if x.startswith('embed')]
    X_train = tbl[columns].to_numpy()
    y_train = tbl['label_id'].to_numpy()

    # execute gridsearch and train classifier
    weight = y_train
    clf = HistGradientBoostingClassifier(random_state=seed)
    clf.fit(X_train, y_train)

    return clf

def train_classifier(tbl, seed=1):
    # prepaire data
    columns = [x for x in tbl.columns if x.startswith('embed')]
    X_train = tbl[columns].to_numpy()
    y_train = tbl['label_id'].to_numpy()

    counts = tbl['label_id'].value_counts().values
    weight = counts[0] / counts[1]
    weights = tbl['label_id'].apply(lambda x: x * weight +1)

    clf = xgb.XGBClassifier(random_state=seed)
    clf.fit(X_train, y_train, sample_weight=weights)
    return clf

def train_svm(tbl, seed=1):
    # prepaire data
    columns = [x for x in tbl.columns if x.startswith('embed')]
    print(f"the following columns are used for training {columns}")
    X_train = tbl[columns].to_numpy()
    y_train = tbl['label_id'].to_numpy()

    clf = SVC(random_state=seed, kernel='rbf')
    clf.fit(X_train, y_train)
    return clf

def evaluate_clf(clf, df_test_mdl):
    
    columns = [x for x in df_test_mdl.columns if x.startswith('embed')]
    X_test = df_test_mdl[columns].to_numpy()
    y_test = df_test_mdl['label_id'].to_numpy()

    # calculate f1 score on test set
    y_pred = clf.predict(X_test)
    y_proba = pd.DataFrame(clf.predict_proba(X_test))

    # calculate AUC PR and avg precision
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba[1])
    roc = pd.DataFrame.from_dict({'precision': precision, 'recall': recall})
    roc = roc.sort_values('recall')
    auc_pr = auc(roc['recall'], roc['precision'])
    ap = average_precision_score(y_test, y_proba[1])

    # metrics for multi label, F1
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred, adjusted=True )

    return {'f1_macro': f1_macro, 'f1_micro': f1_micro,  
        'balanced_accuracy': balanced_accuracy, 'y_pred':pd.DataFrame(y_pred),
        'y_proba': y_proba, 'auc_pr': auc_pr, 'avg_precision': ap
    }

def proces_graph(G, seed, algo, params, train, test):
    _, embed = calculate_graphcase_embedding(
            G, SAVE_PATH, params=params, verbose=False, algo=algo
        )
    df_train_mdl = prep_mdl_input(train, embed)
    clf = train_classifier(df_train_mdl, seed)
    # clf = train_svm(df_train_mdl, seed)
    df_test_mdl = prep_mdl_input(test, embed)
    return evaluate_clf(clf, df_test_mdl)

# %%
ALGO = [
    GraphCaseWrapper, 
    # MultilensWrapper,
    # DrneWrapper,
    # XnetmfWrapper,
    # XnetmfWrapperWithGraphTransformation,
    # Role2VecWrapper,
    # DGIWrapper,
    # BaselineWrapper,
    # ElaineWrapper
    # DGIWrapperWithGraphTransformation
    ]

def calc_mooc_performance(algos=ALGO, runs=1):
    # prep data
    seeds = np.random.RandomState(0).randint(0, 1000, runs)
    edges = get_edge_data()
    train, test = mooc_train_test_split(20, edges)
    train_edges = prep_edges(train)
    test_edges = prep_edges(test)
    G = get_mooc_graph(train_edges)
    res_df = pd.DataFrame(columns=['algo', 'seed', 'auc_pr', 'avg_precision'])

    for algo in algos:
        for i, s in enumerate(seeds):
            params = algo.MOOC_PARAMS
            if 'seed' in params:
                params['seed'] = s
            print(f"start processing {algo.NAME} with seed {i} of {runs}")
            algo_res = proces_graph(G, s, algo, params, train_edges.copy(), test_edges.copy())
            print(f"end processing {algo.NAME} with seed {i} of {runs}")
            algo_res['algo'] = algo.NAME
            algo_res['seed'] = s
            res_df = res_df.append(algo_res, ignore_index=True)
    
    res_df.to_csv(SAVE_PATH + 'algo_res', index=False)

    smry_df = res_df.groupby(['algo'])['auc_pr', 'avg_precision'].agg(['mean', 'std'])
    smry_df.to_csv(SAVE_PATH + 'smry_res', index=False)

    return res_df, smry_df

#%%
# details, smry = calc_mooc_performance(algos=ALGO, runs=1)
# smry
# %%

# edges = get_edge_data()
# train, test = mooc_train_test_split(20, edges)
# train_edges = prep_edges(train)
# G = get_mooc_graph(train_edges)
# _, embed = calculate_graphcase_embedding(
#         G, SAVE_PATH, params=GraphCaseWrapper.MOOC_PARAMS, verbose=True, algo=GraphCaseWrapper
#     )
# df_train_mdl = prep_mdl_input(train, embed)

# # %%

# tbl = df_train_mdl
# # prepaire data
# columns = [x for x in tbl.columns if x.startswith('embed')]
# X_train = tbl[columns].to_numpy()
# y_train = tbl['label_id'].to_numpy()

# counts = tbl['label_id'].value_counts().values
# weight = counts[0] / counts[1]
# weights = tbl['label_id'].apply(lambda x: x*weight +1)

# import xgboost as xgb
# clf = xgb.XGBClassifier()
# clf.fit(X_train, y_train, sample_weight=weights)
# # # execute gridsearch and train classifier

# clf = HistGradientBoostingClassifier(random_state=seed)
# clf.fit(X_train, y_train)
# %%
# df_test_mdl = prep_mdl_input(test, embed)
# evaluate_clf(clf, df_test_mdl)
# %%

# edges = get_edge_data()
# train, test = mooc_train_test_split(20, edges)
# train_edges = prep_edges(train)
# G = get_mooc_graph(train_edges)

# _, embed = calculate_graphcase_embedding(
#         G, SAVE_PATH, params=GraphCaseWrapper.MOOC_PARAMS, verbose=False, algo=GraphCaseWrapper
#     )
# # %%

# tmp1 = prep_mdl_input(train, embed)
# tmp2 = prep_mdl_input(train_edges, embed)
# %%



