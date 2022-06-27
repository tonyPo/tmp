import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
import networkx as nx

conf = SparkConf().setAppName('appName').setMaster('local')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)


class Enron_to_graph:
    LBL_PATH = '/Users/tonpoppe/workspace/graphcase_experiments/graphcase_experiments/graphcase_experiments/graphs/enron/data/email_to_lbl.csv'
    def __init__(self, path):
        self.df = spark.read.format('parquet').load(path)  # load emails
        self.lbl = self.get_labels()  # load label and mailbox info
        self.email = self.extract_individual_edges(self.df) 
        self.email = self.update_email_address(self.email)
        self.nodes = self.extract_nodes(self.email)
        self.add_labels()
        self.edges = self.extract_edges(self.email)
        self.nodes_norm = self.normalise_nodes()
        self.edges_norm = self.normalise_edges()  
        self.G, self.G_sub = self.create_graph()

    def extract_individual_edges(self, df):
        df = df.dropDuplicates(['mail_id'])
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
                    # F.sum('is_bcc').alias("attr_cnt_bcc")
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
                        # .withColumn('attr_is_enron', F.when(F.col('organisation')=='enron.com', 1).otherwise(0))
                        .fillna(0)
        )
        return nodes

    def update_email_address(self, df):
        ''' combine multiple email address related to same mail box'''   
        #update from_address
        lbls = self.lbl.select('email_address', 'node_level')
        df = (df
            .join(lbls.withColumnRenamed("email_address", 'from_address'), 'from_address', 'left')
            .withColumn('from_address', F.when(F.col('node_level').isNull(), F.col('from_address'))
                                        .otherwise(F.col('node_level'))
                        )
            .drop('node_level')
        )
        #update recipient
        df = (df
            .join(lbls.withColumnRenamed("email_address", 'recipient'), 'recipient', 'left')
            .withColumn('recipient', F.when(F.col('node_level').isNull(), F.col('recipient'))
                                        .otherwise(F.col('node_level'))
                        )
            .drop('node_level')
        )
        return df

    def add_labels(self): 
        ''' add labels to nodes  '''  
        lbls = (self.lbl
            .select('node_level', 'label', 'isCorrect')
            .dropDuplicates() 
            .withColumnRenamed("node_level", 'email_address')
        )
        self.nodes = (self.nodes.join(lbls, 'email_address', 'left')
            .fillna("no_label", subset='label')
            .withColumnRenamed('isCorrect', 'isCore')
            .fillna(0, subset='isCore')
        )

    def extract_edges(self, df):
        edges = (df
                .withColumnRenamed('from_address', 'source')
                .withColumnRenamed('recipient', 'target')
                .groupBy(['source', 'target']).agg(
                    F.sum('is_to').alias('weight'),
                    F.sum('is_cc').alias('cnt_cc'),
                    # F.sum('is_bcc').alias('cnt_bcc'),
                    F.sum(F.when(F.col('is_to')==1, F.col('email_size'))).alias('size_to'),
                    F.sum('email_size').alias('size')
                )
                .filter("source != target")
        )
        return edges

    def get_labels(self):
        email_lbls = pd.read_csv(Enron_to_graph.LBL_PATH, sep=',')
        email_lbls_df = (spark.createDataFrame(email_lbls)
                        .filter("isCorrect = 1")
        )
        return email_lbls_df

    def normalise_nodes(self):
        # get attribute names to normalise
        attr = [c for c in self.nodes.columns if c.startswith("attr")]
        for a in attr:  # apply log scaling
            df = self.nodes.withColumn(a, F.when(F.col(a)==0, 0).otherwise(F.log10(a)))

        assembler = VectorAssembler().setInputCols(attr).setOutputCol("features")
        transformed = assembler.transform(df)
        scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
        scalerModel =  scaler.fit(transformed.filter("isCore=1").select("features"))
        scaledData = scalerModel.transform(transformed)
        scaledData = scaledData.withColumn("tmp", vector_to_array('scaledFeatures'))

        names = {x + "_scaled": x for x in attr}
        scaledData = scaledData.select(["email_address", 'isCore', 'label']+[F.col('tmp')[i].alias(names[c]) for i,c in enumerate(names.keys())])
        return scaledData

    def normalise_edges(self):
        attr = [c for c in self.edges.columns if c not in ['target', 'source']]
        for a in attr:  # apply log scaling
            df = self.edges.withColumn(a, F.when(F.col(a)==0, 0).otherwise(F.log10(a)))

        nodes = self.nodes.filter('isCore=1')
        internals = (self.edges
            .join(nodes.select(F.col('email_address').alias('source')), 'source', 'inner')
            .join(nodes.select(F.col('email_address').alias('target')), 'target', 'inner')
            .withColumn('isInternal', F.lit(1))
            .select("source", 'target', 'isInternal')
        )
        df =df.join(internals, ["source", 'target'], 'left').fillna(0)
    
        assembler = VectorAssembler().setInputCols(attr).setOutputCol("features")
        transformed = assembler.transform(df)
        scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
        scalerModel =  scaler.fit(transformed.filter("isInternal=1").select("features"))
        scaledData = scalerModel.transform(transformed)
        scaledData = scaledData.withColumn("tmp", vector_to_array('scaledFeatures'))

        names = {x + "_scaled": x for x in attr}
        scaledData = scaledData.select(["source", 'target']+[F.col('tmp')[i].alias(names[c]) for i,c in enumerate(names.keys())])
        return scaledData

    def create_graph(self):
        nodes = self.nodes_norm.toPandas()
        edges = self.edges_norm.toPandas()
        node_attr = [c for c in nodes.columns if c.startswith("attr")]+ ['label']
        node_tuples = [(r['email_address'], {c: r[c] for c in node_attr}) for i,r in nodes.iterrows()]

        edge_attr = [c for c in edges.columns if c not in ['source', 'target']]
        edge_tuples = [(r['source'], r['target'], {c: r[c] for c in edge_attr}) for i,r in edges.iterrows()]
        G = nx.DiGraph()
        G.add_nodes_from(node_tuples)
        G.add_edges_from(edge_tuples)

        core_nodes = list(self.lbl.select('node_level').toPandas()['node_level'])
        G_sub = G.subgraph(core_nodes)
        G_sub = nx.convert_node_labels_to_integers(G_sub, label_attribute = 'old_id')
        G = nx.convert_node_labels_to_integers(G, label_attribute = 'old_id')
        return G, G_sub

    def get_unlabelled(self, lbl_df):
        return df.select(F.col('label').isNull())
        

if __name__ == '__main__':
    import os
    print(os.getcwd())
    # enron_path = "/Users/tonpoppe/Downloads/enron_parsed.parquet"  #King
    enron_path = "/Users/tonpoppe/Downloads/enron_parsed_all3"  #all 
    graph_path = '/Users/tonpoppe/workspace/graphcase_experiments/graphcase_experiments/graphcase_experiments/graphs/enron/data/enron_graph.pickle'
    sub_graph_path = '/Users/tonpoppe/workspace/graphcase_experiments/graphcase_experiments/graphcase_experiments/graphs/enron/data/enron_sub_graph.pickle'
    enron = Enron_to_graph(enron_path)
    nx.write_gpickle(enron.G, graph_path)
    nx.write_gpickle(enron.G_sub, sub_graph_path)


def create_email_to_lbl_mapping():
    """
    The mapping used from creamer is a mapping from name to position/group. The group is used a label.
    This means that the name needs to be mapped to and E-mail account. This is done by first mapping 
    the name to a mailbox, i.e. folder in the Enron mail download. Then we look at the sent folder of
    the mailboxes to extract the used email addresses. Sometime the results into email address to are
    not used by the owner of the mailbox, because he or she has move the mail to the sent box. Therefore
    we have done a manual check on feasibility by looking at the fraction and name information.
    Below the code for creating this mapping
    """
    enron_path = "/Users/tonpoppe/Downloads/enron_parsed_all2"  #all
    enron = Enron_to_graph(enron_path)

    # load mapping from mailbox to label (group)
    lbls = pd.read_csv(Enron_to_graph.LBL_PATH, sep='\t')
    lbls['shortname'] = lbls['shortname'].astype(str)
    lbl_df = (spark.createDataFrame(lbls[['shortname', 'group']])
                        .withColumnRenamed('shortname', 'mailbox')
            )

    # retrieve the email addresses + count used in the sent folders
    lbl_adj = (enron.email.filter("folder like '%sent%'")
                .select("from_address", 'fname')
                .withColumn("fname_last", F.regexp_extract(F.col('fname'), '(?<=Downloads/)(.)*', 0))
                .withColumn("mailbox", F.lower(F.split(F.col('fname_last'), '/').getItem(1)))
                .join(lbl_df, 'mailbox', 'left')
                .withColumnRenamed("from_address", 'email_address')
                .withColumnRenamed('group', 'label')
            ) 

    # retrieve count and total count
    lbl_adj_cnt_per_mailbox = lbl_adj.groupBy('mailbox').agg(
                            F.count('mailbox').alias('total_count'),
                            F.countDistinct('email_address').alias("email_count")
                            )
    lbl_adj_cnt_per_email = lbl_adj.groupBy('email_address', 'mailbox', 'label').count()
    email_to_lbl = (lbl_adj_cnt_per_email
                    .join(lbl_adj_cnt_per_mailbox, 'mailbox', 'inner')
                    .withColumn('fraction', F.col('count')/F.col('total_count'))
                    .orderBy(['mailbox', 'fraction'], ascending=False)
                    )

    # email_to_lbl.filter("email_count > 1").show(20)
    # %%
    email_to_lbl.write.csv("/Users/tonpoppe/Downloads/email_to_lbl.csv", sep='\t')
