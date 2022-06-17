import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import functions as F
import networkx as nx

conf = SparkConf().setAppName('appName').setMaster('local')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)


class Enron_to_graph:
    LBL_PATH = '/Users/tonpoppe/workspace/graphcase_experiments/graphcase_experiments/graphcase_experiments/graphs/enron/data/jobtitles_creamer.txt'
    def __init__(self, path):
        self.df = spark.read.format('parquet').load(path)
        self.email = self.extract_individual_edges(self.df)
        self.lbl = self.get_labels(self.email)
        self.nodes = self.extract_nodes(self.email)
        self.edges = self.extract_edges(self.email)
        self.G = self.create_graph()

    def extract_individual_edges(self, df):
        cols =df.columns
        # explode from
        df = (df.select(F.explode("from_address"), *cols)
                    .withColumn('from_address', F.col('col'))
                    .filter("from_address is not null")
                    .drop('col')
            )
        # explode to
        to_row = (df.filter("cnt_to > 0")
                    .select(F.explode("to_address"), *cols)
                    .withColumn("is_to", F.lit(1))
                    .withColumn('recipient', F.col('col'))
                    .drop('col')
            )
        # explode cc
        cc_row = (df.filter("cnt_cc > 0")
                    .select(F.explode("cc_address"), *cols)
                    .withColumn("is_cc", F.lit(1))
                    .withColumn('recipient', F.col('col'))
                    .drop('col')
            )
        res = to_row.unionByName(cc_row, allowMissingColumns=True)

        # explode bcc
        bcc_row = (df.filter("cnt_bcc > 0")
                    .select(F.explode("bcc_address"), *cols)
                    .withColumn("is_bcc", F.lit(1))
                    .withColumn('recipient', F.col('col'))
                    .drop('col')
            )

        # sometimes the same names appear in the BCC as in the to for some reason, need to remove these duplicate entries.
        bcc_row = (bcc_row.join(cc_row.select('message_id','from_address', 'recipient', 'is_cc'), 
                                ['message_id','from_address', 'recipient'], 
                                'left')
                    .filter("is_cc is null")         
                    )

        res = (res.unionByName(bcc_row, allowMissingColumns=True)    
                    .fillna(0)
        )

        return res

    def extract_nodes(self, df):
        # recipient side
        to_nodes = (df.groupBy('recipient').agg(
                    F.sum('email_size').alias("attr_received_size"), 
                    F.sum('is_to').alias("attr_cnt_to"), 
                    F.sum('is_cc').alias("attr_cnt_cc"), 
                    F.sum('is_bcc').alias("attr_cnt_bcc")
                    )
                    .withColumnRenamed('recipient', 'email_address')
                )
        # from side
        from_nodes = (df.select('message_id', 'email_size', 'from_address')
                    .dropDuplicates()
                    .groupBy('from_address').agg(
                    F.sum('email_size').alias("attr_sent_size"), 
                    F.count('message_id').alias("attr_cnt_send"), 
                    )
                    .withColumnRenamed('from_address', 'email_address')
                )
        
        nodes = (to_nodes.join(from_nodes, 'email_address', 'outer')
                        .withColumn('organisation', F.split(F.col("email_address"),'@').getItem(1))
                        .withColumn('attr_is_enron', F.when(F.col('organisation')=='enron.com', 1).otherwise(0))
                        .fillna(0)
        )

        nodes = nodes.join(self.lbl, 'email_address', 'left').fillna(0)

        return nodes

    def extract_edges(self, df):
        edges = (df
                .withColumnRenamed('from_address', 'source')
                .withColumnRenamed('recipient', 'target')
                .groupBy(['source', 'target']).agg(
                    F.sum('is_to').alias('cnt_to'),
                    F.sum('is_cc').alias('cnt_cc'),
                    F.sum('is_bcc').alias('cnt_bcc'),
                    F.sum(F.when(F.col('is_to')==1, F.col('email_size'))).alias('size_to'),
                    F.sum('email_size').alias('weight')
                )
                .filter("source != target")
        )
        return edges

    def get_labels(self, df):
        self.lbls = pd.read_csv(Enron_to_graph.LBL_PATH, sep='\t')
        lbl_df = (spark.createDataFrame(self.lbls[['lower', 'group']])
                    .withColumnRenamed('lower', 'mailbox')
        )

        # extract mailbox name and join with label from sent mail
        df = (df.filter("folder like '%sent%'")
            .select("from_address", 'fname')
            .withColumn("fname_last", F.regexp_extract(F.col('fname'), '(?<=Downloads/)(.)*', 0))
            .withColumn("mailbox", F.lower(F.split(F.col('fname_last'), '/').getItem(1)))
            .select('mailbox', 'from_address')
            .dropDuplicates()
            .join(lbl_df, 'mailbox', 'left')
            .withColumnRenamed("from_address", 'email_address')
            .withColumnRenamed('group', 'label')
            .withColumn('attr_is_core', F.lit(1))
            .select('email_address', 'label', 'attr_is_core')
        ) 
        return df

    def create_graph(self):
        nodes = self.nodes.toPandas()
        edges = self.edges.toPandas()
        node_tuples = [(r['email_address'], {c: r[c] for c in r.index}) for i,r in nodes.iterrows()]

        edge_attr = [c for c in edges.columns if c not in ['source', 'target']]
        edge_tuples = [(r['source'], r['target'], {c: r[c] for c in edge_attr}) for i,r in edges.iterrows()]
        G = nx.DiGraph()
        G.add_nodes_from(node_tuples)
        G.add_edges_from(edge_tuples)
        return G



    def get_unlabelled(self, lbl_df):
        return df.select(F.col('label').isNull())
        

if __name__ == '__main__':
    import os
    print(os.getcwd())
    # enron_path = "/Users/tonpoppe/Downloads/enron_parsed.parquet"  #King
    enron_path = "/Users/tonpoppe/Downloads/enron_parsed_all"  #all 
    graph_path = '/Users/tonpoppe/workspace/graphcase_experiments/graphcase_experiments/graphcase_experiments/graphs/enron/data/enron_graph.pickle'
    enron = Enron_to_graph(enron_path)
    nx.write_gpickle(enron.G, graph_path)