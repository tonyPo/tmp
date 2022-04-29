from math import pi, cos, sin
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.manifold import MDS

def plot_embedding(G, embed, path=None):
    # reduce dimensions of embeding to two
    if embed.shape[1] > 3:
        mds = MDS(n_components=2)
        ids = embed[:,0]
        embed = mds.fit_transform(embed[:,1:])
        embed = np.column_stack([ids, embed])

    fig, ax = plt.subplots(1,1)
    # plot embeding
    embed_df = pd.DataFrame(embed, columns=['id', 'embed1', 'embed2'])
    lbl_df = pd.DataFrame(
        [[i,x] for i,x in nx.get_node_attributes(G,'label').items()],
        columns=['id', 'label']
    )
    embed_df = pd.merge(embed_df, lbl_df, on='id', how='inner')

    labels = [x for _,x in nx.get_node_attributes(G,'label').items()]
    tmp = {n:i for i,n in enumerate(set(labels))}
    label_dic = {k:v/len(tmp.values()) for k,v in tmp.items()}
    color = [label_dic[x] for _, x in embed_df['label'].items()]
  
    ax.scatter(embed_df['embed1'], embed_df['embed2'], s=200., c=color, cmap=plt.cm.Set3_r)

    # add legend and title 
    markers = [plt.Line2D([0,0],[0,0], color=plt.cm.Set3_r(c), marker='o', linestyle='') for c in label_dic.values()]
    plt.legend(markers, label_dic.keys(), numpoints=1, loc=(1.04,0))
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    plt.title("Barbel graph: node coler represents the node role, label = node id")
    
    # save fig
    if path:
        fig.savefig(path + 'embed_plot_graphCASE.png', dpi=300, format='png')
    plt.show()