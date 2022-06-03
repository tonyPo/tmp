
import tensorflow as tf
from GAE.graph_case_controller import GraphAutoEncoder

class GraphCaseWrapper(GraphAutoEncoder):
    NAME = 'GraphCASE'
    LOCATION = 'graphcase_experiments/algos/processing_files/other'
    COMP_PARAMS ={
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

    def __init__(self, G, **kwargs):
        node_attributes = [a for a in G.nodes[0].keys() if a.startswith("attr")]
        kwargs['encoder_labels'] = node_attributes
        super().__init__(G, **kwargs)
