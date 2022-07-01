import numpy as np
import tensorflow as tf
import pandas as pd
from graphcase_experiments.algos.baseWrapper import BaseWrapper
from graphcase_experiments.tools.graph_transformer import to_undirected_node_attributes_only_graph
from stellargraph import StellarDiGraph
from stellargraph.layer import DeepGraphInfomax, GraphSAGE
from stellargraph.mapper import (
    CorruptedGenerator,
    GraphSAGENodeGenerator,
)
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class DGIWrapper(BaseWrapper):
    NAME = 'DGI'
    COMP_PARAMS ={
        'batch_size': 128,
        'num_samples': [7, 7],
        'layer_sizes': [128, 128],
        'activations': ["relu", "relu"],
        'epochs': 200
    }
    ENRON_PARAMS = COMP_PARAMS
    def __init__(self, G, **kwargs):
        self.G_stellar = self.__constructor_stellargraph(G)

        self.params = {}
        self.params['layer_sizes'] = kwargs.get('layer_sizes', [128, 128])
        self.params['activations'] = kwargs.get('activations', ["relu", "relu"])

        self.gen_params = {}
        self.gen_params['batch_size'] = kwargs.get('batch_size', 128)
        self.gen_params['num_samples'] = kwargs.get('num_samples', [5, 5])

        self.epochs = kwargs.get('epochs', 100)

        self.base_generator = GraphSAGENodeGenerator(self.G_stellar, **self.gen_params)
        self.base_model = GraphSAGE(**self.params, generator=self.base_generator)
        corrupted_generator = CorruptedGenerator(self.base_generator)
        self.gen = corrupted_generator.flow(self.G_stellar.nodes())

        infomax = DeepGraphInfomax(self.base_model, corrupted_generator)
        x_in, x_out = infomax.in_out_tensors()

        self.model = Model(inputs=x_in, outputs=x_out)
        self.model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=Adam(lr=1e-3))
        

    def fit(self, **kwargs):
        es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
        history = self.model.fit(self.gen, epochs=self.epochs, verbose=0, callbacks=[es])
        return history

    def calculate_embeddings(self, G):
        x_emb_in, x_emb_out = self.base_model.in_out_tensors()
        if self.base_generator.num_batch_dims() == 2:
                x_emb_out = tf.squeeze(x_emb_out, axis=0)
        emb_model = Model(inputs=x_emb_in, outputs=x_emb_out)
        embed = emb_model.predict(self.base_generator.flow(self.G_stellar.nodes()))
        ids = np.array(range(embed.shape[0]))
        
        return np.hstack([ids[:, None], embed])

    def __constructor_stellargraph(self, G):
        # create feature matrix
        features = G.nodes(data=True)
        feature_names = list(features[0].keys())
        feature_names.remove('label')
        feature_names.remove('old_id')
        attr = np.array([[n] + [a[k] for k in feature_names] for n,a in features])
        attr = attr[attr[:,0].argsort()]
        features_df = pd.DataFrame(attr[:,1:], index=attr[:,0], columns=feature_names)

        # create edge list with attributes
        edges = G.edges(data=True)
        edge_attr_names = list(list(G.edges(data=True))[0][-1].keys())
        edges_df = pd.DataFrame([[s, d] + [v for v in a.values()] for s,d,a in edges], columns=['source', 'target'] + edge_attr_names)

        G_stellar = StellarDiGraph(features_df, edges_df)
        return G_stellar


class DGIWrapperWithGraphTransformation(DGIWrapper):
    NAME = 'DGI_with_transformation'
    COMP_PARAMS ={
        'batch_size': 128,
        'num_samples': [7, 7, 7, 7, 7, 7],
        'layer_sizes': [128, 128, 128, 128, 128, 128],
        'activations': ["relu", "relu", "relu", "relu", "relu", "relu"],
        'epochs': 200
    }

    ENRON_PARAMS = COMP_PARAMS
    def __init__(self, G, **kwargs):
        G_undirected = to_undirected_node_attributes_only_graph(G, verbose=False)
        super().__init__(G, **kwargs)


