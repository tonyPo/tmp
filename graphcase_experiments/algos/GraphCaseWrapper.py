
import tensorflow as tf
from GAE.graph_case_controller import GraphAutoEncoder

class GraphCaseWrapper(GraphAutoEncoder):
    NAME = 'GraphCASE'
    LOCATION = 'graphcase_experiments/algos/processing_files/other'
    COMP_PARAMS = {
        'batch_size': 30,
        'hub0_feature_with_neighb_dim': 128,
        'verbose': False,
        'seed': 1,
        'encoder_labels': ['attr1', 'attr2'],
        'learning_rate': 0.0003668872396300966,
        'act': tf.nn.sigmoid,
        'useBN': True,
        'dropout': 0.09859650451427784,
        'support_size': [7, 7],
        'dims': [3, 128, 128, 128],
        'epochs': 200,
    }
    ENRON_PARAMS = {
        'batch_size': 30,
        'hub0_feature_with_neighb_dim': 128,
        'verbose': False,
        'seed': 1,
        'encoder_labels': ['attr1', 'attr2'],
        'learning_rate': 0.0002, #9.98e-5,
        'act': tf.nn.sigmoid,
        'useBN': True,
        'dropout': 0.14,
        'support_size': [10, 10],
        'dims': [3, 112, 112, 112],
        'epochs': 1500,
    }
    ENRON_PARAMS_OLD = {
        'batch_size': 30,
        'hub0_feature_with_neighb_dim': 128,
        'verbose': False,
        'seed': 1,
        'encoder_labels': ['attr1', 'attr2'],
        'learning_rate': 0.0003668872396300966,
        'act': tf.nn.sigmoid,
        'useBN': True,
        'dropout': 0.09859650451427784,
        'support_size': [7, 7],
        'dims': [5, 128, 128, 128],
        'epochs': 300,
    }

    MOOC_PARAMS = {
        'batch_size': 1024,
        'hub0_feature_with_neighb_dim': 16,
        'verbose': False,
        'seed': 1,
        'encoder_labels': ['attr1', 'attr2'],
        'learning_rate': 0.0005,
        'act': tf.nn.sigmoid,
        'useBN': True,
        'dropout': 0.1,
        'support_size': [5, 5],
        'dims': [4, 8, 16, 16],
        'epochs': 400,  #400
    }

    BZR_PARAMS = {
        'batch_size': 1024,
        'hub0_feature_with_neighb_dim': 64,
        'verbose': False,
        'seed': 1,
        'encoder_labels': ['attr1', 'attr2'],
        'learning_rate': 0.0001,
        'act': tf.nn.sigmoid,
        'useBN': True,
        'dropout': 0.1,
        'support_size': [7, 7],
        'dims': [4, 64, 64, 64],
        'epochs': 1500,
    }

    def __init__(self, G, **kwargs):
        node_attributes = [a for a in G.nodes[0].keys() if a.startswith("attr")]
        kwargs['encoder_labels'] = node_attributes
        super().__init__(G, **kwargs)
