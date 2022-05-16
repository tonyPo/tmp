from abc import ABC

class BaseWrapper(ABC):
    def __init__(self, **kwargs):
        pass

    def fit(**kwargs):
        pass

    def calculate_embedding(G):
        pass



# GraphAutoEncoder(G, **params)
# gae.fit(epochs=epochs, layer_wise=False)
# embed = gae.calculate_embeddings(G)

class MultilensWrapper(BaseWrapper):
    def __init__(self, **kwargs):
        pass

    def fit(**kwargs):
        pass

    def calculate_embedding(self, G):
        # define locations
        graph_file_path = MultilensWrapper.LOCATION + 'edge_list.tsv'
        category_file_path = MultilensWrapper.LOCATION + "categories.tsv"
        embedding_file_path = MultilensWrapper.LOCATION + 'multilens_embeddings.tsv'

        nx.write_weighted_edgelist(G, graph_file_path, delimiter='\t')
        with open(category_file_path, "w") as text_file:
            text_file.write(f"0\t0\t{G.number_of_nodes()}")

        # execute algoritm
        exit_status = os.system(f'source ~/opt/anaconda3/etc/profile.d/conda.sh;conda activate py2;python ../../multilens/MultiLENS/src/main.py --input {graph_file_path} --cat {category_file_path} --output {embedding_file_path}')
        print(f"MultiLENS process finished with status {exit_status}")
        if exit_status!=0:
            exit()

        # load results
        embedding = np.genfromtxt(embedding_file_path, skip_header=1)
        return embedding