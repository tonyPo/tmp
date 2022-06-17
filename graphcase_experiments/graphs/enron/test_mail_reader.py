#%%
file = '/Users/tonpoppe/Downloads/maildir/keiser-k/inbox/2.'
# %%

from email.parser import BytesParser
from email import policy
with open(file, 'rb') as fp:
    name = fp.name  # Get file name
    msg = BytesParser(policy=policy.default).parse(fp)

    msg.get('CC')
# %%
msg.get_all('to')
# %%
msg.get_all('from')
# %%
import os
os.getcwd()
# %%

from graphcase_experiments.graphs.enron.email_util import EmailInfo, EmailWalker
file = "/Users/tonpoppe/Downloads/testenron/king-j/inbox/33."
res = EmailInfo(file, "test")
res
#%%
root = '/Users/tonpoppe/Downloads/maildir/keiser-k/inbox'
emailWalker = EmailWalker(root)
pdf = emailWalker.parse_mails(verbose=False)

# %%

import os
os.chdir("../../..")

#%% load parsed emials

import pandas as pd
pdf = pd.read_pickle('/Users/tonpoppe/Downloads/enron_parsed_test')
pdf.shape
#%%
import pickle
with open("/Users/tonpoppe/Downloads/enron_parsed_test", "rb") as fp:
    b = pickle.load(fp)

#%% check if there are duplicate idee

pdf['message_id'].describe()
# %%
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import functions as F

conf = SparkConf().setAppName('appName').setMaster('local')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# %%

#%%
from_file
file ="/Users/tonpoppe/Downloads/enron_parsed.parquet"
# pdf.to_parquet(file)
df = spark.read.format('parquet').load(file)

#%%
def extract_individual_edges(df):
    #explode to_adress
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

#%%
tmp = extract_individual_edges(df)
tmp.count()

    
# %%

def extract_nodes(df):
    # recipient side
    to_nodes = (df.groupBy('recipient').agg(
                F.sum('email_size').alias("received_size"), 
                F.sum('is_to').alias("cnt_to"), 
                F.sum('is_cc').alias("cnt_cc"), 
                F.sum('is_bcc').alias("cnt_bcc")
                )
                .withColumnRenamed('recipient', 'email_address')
            )
    # from side
    from_nodes = (df.select('message_id', 'email_size', 'from_address')
                .dropDuplicates()
                .groupBy('from_address').agg(
                F.sum('email_size').alias("sent_size"), 
                F.count('message_id').alias("cnt_send"), 
                )
                .withColumnRenamed('from_address', 'email_address')
            )
    
    nodes = (to_nodes.join(from_nodes, 'email_address', 'outer')
                    .withColumn('organisation', F.split(F.col("email_address"),'@').getItem(1))
                    .withColumn('is_enron', F.when(F.col('organisation')=='enron.com', 1).otherwise(0))
                    .fillna(0)
    )
    
    return nodes

#%%
tmp2 = extract_nodes(tmp)
# %%

def extract_edges(df):
    edges = (df
            .withColumnRenamed('from_address', 'src')
            .withColumnRenamed('recipient', 'dst')
            .groupBy(['src', 'dst']).agg(
                F.sum('is_to').alias('cnt_to'),
                F.sum('is_cc').alias('cnt_cc'),
                F.sum('is_bcc').alias('cnt_bcc'),
                F.sum(F.when(F.col('is_to')==1, F.col('email_size'))).alias('size_to'),
                F.sum('email_size').alias('weight')
            )
            .filter("src != dst")
    )
    return edges
#%%
edges = extract_edges(tmp)
    

# %%
import os
os.getcwd()
# %%
import pandas as pd
# path = '/Users/tonpoppe/workspace/graphcase_experiments/graphcase_experiments/graphcase_experiments/graphs/enron/data'
email_file = 'graphcase_experiments/graphs/enron/data/employees.txt'
job_lbl = (spark.read.csv(email_file, sep='\t')
            .withColumn('name', F.split(F.col('_c1'),"\s{4,}").getItem(0))
            .withColumn('function', F.split(F.col('_c1'),"\s{4,}", 2).getItem(1))
            .withColumn('function_type', F.split(F.col('_c1'),"\s{2,}").getItem(1))
            .withColumn('function_type2', F.split(F.col('_c1'),"\s{2,}").getItem(2))
            .withColumn('function_type', F.when(F.col('function_type2').contains('Chief Operating Officer'), F.lit('board member'))
                                                .otherwise(F.col('function_type'))
                        )
            .withColumn('name', F.when(F.col('name')=='xxx', F.regexp_replace('_c0', "\.", " "))
                                                .otherwise(F.col('name'))
            )
                        
)
job_lbl.groupBy('function_type').count().show(20, truncate=False)

# %%

import pandas as pd
lbl_path = 'graphcase_experiments/graphs/enron/data/jobtitles_creamer.txt'
lbls = pd.read_csv(lbl_path, sep='\t')
# %%
import os
os.chdir("../../..")
from graphcase_experiments.graphs.enron.mail_reader import spark, Enron_to_graph


# %%
enron_path = "/Users/tonpoppe/Downloads/enron_parsed.parquet"  #King
# enron_path = "/Users/tonpoppe/Downloads/enron_parsed_all.parquet"  #all
enron = Enron_to_graph(enron_path)
# %%
import pyspark.sql.functions as F
import networkx as nx
import pandas as pd

nodes = enron.nodes.toPandas()
edges = enron.edges.toPandas()
node_tuples = [(r['email_address'], {c: r[c] for c in r.index}) for i,r in nodes.iterrows()]

edge_attr = [c for c in edges.columns if c not in ['source', 'target']]
edge_tuples = [(r['source'], r['target'], {c: r[c] for c in edge_attr}) for i,r in edges.iterrows()]
G = nx.DiGraph()
G.add_nodes_from(node_tuples)
G.add_edges_from(edge_tuples)
# %%
# check for duplicate nodes
enron.nodes.groupBy('email_address').count().filter("count > 1").join(enron.nodes, 'email_address', 'inner').show(10, truncate=False)
# %% check duplicate edges
enron.edges.groupBy(['source', 'target']).count().filter("count > 1").show(5)
# %% get core group
core = [r for r in list(enron.G.nodes(data=True)) if r[1]['attr_is_core']==1]


# %%
