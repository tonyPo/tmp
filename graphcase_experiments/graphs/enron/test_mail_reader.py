#%% set up
import os
import pyspark.sql.functions as F
import networkx as nx
import pandas as pd
os.chdir("../../..")

from graphcase_experiments.graphs.enron.mail_reader import spark, Enron_to_graph


#%% deduplicate mails
# path = "/Users/tonpoppe/Downloads/enron_parsed_all3"
# def de_duplicate_mail_ids(path):
#     df = spark.read.format('parquet').load(path)
#     df = (df
#             .withColumn('mail_id', F.regexp_replace(F.col("mail_id"), "[^A-Z0-9_]", ""))
#             .dropDuplicates(subset=['mail_id'])
#             .write.format('parquet').save(path + '_dedup')
#         )
# de_duplicate_mail_ids(path)

# %%
# enron_path = "/Users/tonpoppe/Downloads/enron_parsed.parquet"  #King
enron_path = "/Users/tonpoppe/Downloads/enron_parsed_all3_dedup"  #all
enron = Enron_to_graph(enron_path)
#%%
tmp_path = '/Users/tonpoppe/Downloads/'
# enron.email.write.format('parquet').save(tmp_path + 'enron_email')
email = spark.read.format('parquet').load(tmp_path + 'enron_email')
#%%

tmp_path = '/Users/tonpoppe/Downloads/'
enron.nodes.write.format('parquet').save(tmp_path + 'nodes', mode='overwrite')
nodes = spark.read.format('parquet').load(tmp_path + 'nodes')
#%%
tmp_path = '/Users/tonpoppe/Downloads/'
enron.edges.write.format('parquet').save(tmp_path + 'edges', mode='overwrite')
eedges = spark.read.format('parquet').load(tmp_path + 'edges')


# %%
# check for duplicate nodes
enron.nodes.groupBy('email_address').count().filter("count > 1").join(enron.nodes, 'email_address', 'inner').show(10, truncate=False)
# %% 
# check duplicate edges
enron.edges.groupBy(['source', 'target']).count().filter("count > 1").show(5)
# %% 
# get core group
core = [{'id':r[0], **r[1]} for r in list(enron.G.nodes(data=True)) if r[1]['label']!='no_label']
core_group = pd.DataFrame.from_dict(core)
graph_core_cnt = core_group.shape
sub_graph_cnt = enron.G_sub.number_of_nodes()
lbl_cnt = enron.lbl.select('node_level').drop_duplicates().count()
print(f"there are {lbl_cnt} unique person identified in the label definition and {graph_core_cnt[0]} and {sub_graph_cnt} core members in the graphs")


# %% check if there are message ids
mails = enron.df.withColumn('mail_id', F.regexp_replace(F.col("mail_id"), "[^A-Z0-9_]", ""))
dup_id_cnt = mails.groupBy('message_id').count().filter("count > 1").join(mails, 'message_id', 'inner').orderBy('message_id').count()
print(f"There are {dup_id_cnt} duplicate message ids")
if dup_id_cnt > 0:
        mails.groupBy('message_id').count().filter("count > 1").join(mails, 'message_id', 'inner').orderBy('message_id').show(5, truncate=False)
#%% number of duplicate email
mails = enron.df.withColumn('mail_id', F.regexp_replace(F.col("mail_id"), "[^A-Z0-9_]", ""))
dub_mail = mails.groupBy('mail_id').count().filter("count > 1").join(mails, 'mail_id', 'inner')
dup_id_cnt = dub_mail.count()
print(f"There are {dup_id_cnt} mails identified and filtered out")
if dup_id_cnt>0:
        print("the most frequest duplicate mail is")
        dub_mail.orderBy("count", ascending=False).show(4)
        print("least_frequest duplicate mails are")
        dub_mail.orderBy("count", "mail_id", ascending=True).show(4)



#%%
dup_after_filter = enron.email.groupBy('mail_id', 'recipient').count().filter("count > 1")
cnt = dup_after_filter.count()
print (f"there are {cnt} duplicate mail id after filtering")
if cnt > 0:
        dup_after_filter.join(tmp, ['mail_id', 'recipient'], 'inner').orderBy('mail_id').show(5, truncate=False)

#%% single id check
mind = 'F01D20000950000800EOESN302000'
df_cnt = enron.df.withColumn('mail_id', F.regexp_replace(F.col("mail_id"), "[^A-Z0-9_]", "")).filter(f"mail_id = '{mind}'").groupBy('fname').count().count()
mr_cnt = enron.email_raw.filter(f"mail_id = '{mind}'").groupBy('fname').count().filter("count > 1").count()
m_cnt = enron.email.filter(f"mail_id = '{mind}'").groupBy('recipient').count().filter("count > 1").count()
print(f"df has {df_cnt} dups, raw has {mr_cnt} dups and mail has {m_cnt}")

# tmp = enron.email.drop_duplicates(subset=['fname']).groupBy('mail_id').count().filter("count > 1").show(4, truncate=False)
# %% check internal emails 
lbl = enron.lbl
email_core = (email
        .join(lbl.select('node_level'), email.from_address==lbl.node_level, 'inner')
        .withColumnRenamed('node_level', 'from_node_level')
        .join(lbl.select('node_level'), email.recipient == lbl.node_level, 'inner')
        .withColumnRenamed('node_level', 'to_node_level')
)
print(f"there are {email_core.count()} singe edges in the core dataset")
# email.show(4)



# %%
LBL_PATH = '/Users/tonpoppe/workspace/graphcase_experiments/graphcase_experiments/graphcase_experiments/graphs/enron/data/email_to_lbl.csv'
email_lbls = pd.read_csv(LBL_PATH, sep=',')
email_lbls_df = (spark.createDataFrame(email_lbls)
                .filter("isCorrect = 1")
)

edges = enron.edges
nodes = enron.nodes
#%%
print(f"there are {nodes.count()} rows in the nodes df")
print(f"there are {enron.nodes.groupBy('email_address').count().count()} unique email address in the nodes df")
print(f"there are {edges.count()} rows in the edge df")
print(f"there are {edges.groupBy(['source','target']).count().count()} unique combinations in the edge df")

tmp = (edges
        .join(nodes.select(F.col('email_address').alias('source')), 'source', 'inner')
        .join(nodes.select(F.col('email_address').alias('target')), 'target', 'inner')
)
print(f"there are {tmp.count()} aggegated edges left in the graph")
#%%
nodes_internal = nodes.filter("isCore = 1")
tmp2 =  (edges
        .join(nodes_internal.select(F.col('email_address').alias('source')), 'source', 'inner')
        .join(nodes_internal.select(F.col('email_address').alias('target')), 'target', 'inner')
)
print(f"there are {tmp2.count()} aggegated core edges left in the graph")
# tmp.show(4, truncate=False)
# %%

#%% print distribution
nodes = enron.nodes.filter("isCore = 1").toPandas()
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,70))
attr = [c for c in nodes.columns if c.startswith("attr")]

for i, a in enumerate(attr):
        ax = fig.add_subplot(len(attr), 1, i+1)
        ax.hist(nodes[a])
        ax.set_title(f"attr {a}")

plt.xscale('log') 
#%%


# %% Check labels

#############################################
## sanity check on hernandez-j
###########################################
nm = "hernandez-j"

# number of parsed emails
df_cnt = enron.df.filter(f"fname like '%{nm}%'").count()
print(f"there are {df_cnt} mails geparsed from the {nm} mail box")

# number of single edges out
out_cnt = enron.email.filter(f"from_address like '%{nm}%'").count()
in_cnt = enron.email.filter(f"recipient like '%{nm}%'").count()
print(f"there are {in_cnt} incoming and {out_cnt} outgoing emails for {nm}")
#%%
#check the number of edges
out_cnt = enron.edges.filter(f"source like '%{nm}%'").count()
in_cnt = enron.edges.filter(f"target like '%{nm}%'").count()
print(f"there are {in_cnt} incoming and {out_cnt} outgoing edges for {nm}")

#check the number of edges in the graph
node_id = [n for n,a in enron.G.nodes(data=True) if nm in a['old_id']][0]
out_cnt = enron.G.in_degree[node_id]
in_cnt = enron.G.out_degree[node_id]
print(f"there are {in_cnt} incoming and {out_cnt} outgoing degree in G {nm}")

#%%
res = enron.extract_individual_edges(hern_sent)

# %% sanity checks on number of parsed emails

############# need to check if emails and df have some number of emials approximately. 
# small diference due to duplicatesis allowed.
enron.email.filter("fname like '/Users/tonpoppe/Downloads/maildir3/hernandez-j/sent_items/%'").dropDuplicates(['fname']).select('fname').show(10, truncate=False)
#%%
enron.df.filter("fname like '/Users/tonpoppe/Downloads/maildir3/hernandez-j/sent_items/%'").dropDuplicates(['fname']).select('fname').show(10, truncate=False)
# %%
enron.email_raw.filter("fname like '/Users/tonpoppe/Downloads/maildir3/hernandez-j/sent_items/%'").dropDuplicates(['fname']).select('fname').show(10, truncate=False)
# %%




res.filter("fname like '/Users/tonpoppe/Downloads/maildir3/hernandez-j/sent_items/%'").dropDuplicates(['fname']).select('fname', 'mail_id').show(10, truncate=False)
#  
# %%
tmp = enron.df.filter("fname like '/Users/tonpoppe/Downloads/maildir3/hernandez-j/sent_items/%'")
tmp.count()
# %%
tmp.groupBy('mail_id').count().show(9)
# %%

tmp.dropDuplicates(['mail_id']).count()
# %%
self.mail_id = msg.get('date') + msg.get('from') + msg.get('to') + msg.get('cc') + msg.get('subject')



#%%
df = enron.df
df = (df
            .withColumn('mail_id', F.regexp_replace(F.col("mail_id"), "[^A-Z0-9_]", ""))
            .dropDuplicates(['mail_id'])
        )
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

#         # explode bcc
# bcc_row = (df.filter("cnt_bcc > 0")
#                     .select(F.explode("bcc_address"), *cols)
#                     .withColumn("is_bcc", F.lit(1))
#                     .withColumn('recipient', F.col('col'))
#                     .drop('col')
#             )

#         # sometimes the same names appear in the BCC as in the to for some reason, need to remove these duplicate entries.
# bcc_row = (bcc_row.join(cc_row.select('message_id','from_address', 'recipient', 'is_cc'), 
#                                 ['message_id','from_address', 'recipient'], 
#                                 'left')
#                     .filter("is_cc is null")         
#                     )

# res = (res.unionByName(bcc_row, allowMissingColumns=True)    
#                     .fillna(0)
#         )


# %%
res.filter(f"mail_id = '{mind}'").groupBy('fname').count().show(12, truncate=False)
# %%

# %%
nodes = enodes.filter("email_address in ('scott-s', 'bass-e')").show(5, truncate=False)

# %%

eedges = enron.edges
# %%
n_bass = enodes.filter("email_address in ('bass-e')")
counter_nodes = (enodes
        .withColumnRenamed('email_address', 'target')
        .withColumnRenamed('label', 'counter_label')
)

n_bass = (n_bass.join(eedges, n.email_address==eedges.source, 'inner')
        .orderBy('weight', ascending=False)
        # .limit(15)
)

n_bass.filter("target not like '%@%'").show(7)
# %%
n_scott = enodes.filter("email_address in ('scott-s')")
counter_nodes = (enodes
        .withColumnRenamed('email_address', 'target')
        .withColumnRenamed('label', 'counter_label')
)

n_scott = (n_scott.join(eedges, n.email_address==eedges.source, 'inner')
        .join(counter_nodes, 'target', 'left')
        .orderBy('weight', ascending=False)

        # .limit(15)
)

n_scott.filter("target not like '%@%'").show(7)
#%% slecht 33 dezelfde target, met verschillende weights

res = (n_bass
    .select('target', F.col('weight').alias('bass_weight'))
    .join(n_scott.select('target', F.col('weight').alias('scott_weight')), 'target', 'inner')
)
res.show(20)
# %%
n_trader = enodes.filter("label = 'trader'")
counter_nodes = (enodes
        .withColumnRenamed('email_address', 'target')
        .withColumnRenamed('label', 'counter_label')
)

n_trader = (n_trader.join(eedges, n.email_address==eedges.source, 'inner')
        .join(counter_nodes, 'target', 'left')
        .select('email_address', 'label', 'counter_label', 'weight')

        # .limit(15)
)

n_trader = n_trader.groupBy('email_address').pivot('counter_label').agg(
        F.sum('weight').alias('weight'),
        # F.count('email_address').alias('count')
).fillna(0)
n_trader.show(35)
# n_trader.filter("target not like '%@%'")


# %%
n_trader_in = enodes.filter("label = 'trader'")
counter_nodes = (enodes
        .withColumnRenamed('email_address', 'source')
        .withColumnRenamed('label', 'counter_label')
)

n_trader_in = (n_trader_in.join(eedges, n.email_address==eedges.target, 'inner')
        .join(counter_nodes, 'source', 'left')
        .select('email_address', 'label', 'counter_label', 'weight')

        # .limit(15)
)

n_trader_in = n_trader_in.groupBy('email_address').pivot('counter_label').agg(
        F.sum('weight').alias('weight'),
        # F.count('email_address').alias('count')
).fillna(0)
n_trader_in.show(35)
# %%
