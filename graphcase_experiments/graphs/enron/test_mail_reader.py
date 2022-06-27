#%% set up
import os
import pyspark.sql.functions as F
import networkx as nx
import pandas as pd
os.chdir("../../..")

from graphcase_experiments.graphs.enron.mail_reader import spark, Enron_to_graph



# %%
# enron_path = "/Users/tonpoppe/Downloads/enron_parsed.parquet"  #King
enron_path = "/Users/tonpoppe/Downloads/enron_parsed_all3"  #all
enron = Enron_to_graph(enron_path)
# %%
# check for duplicate nodes
enron.nodes.groupBy('email_address').count().filter("count > 1").join(enron.nodes, 'email_address', 'inner').show(10, truncate=False)
# %% check duplicate edges
enron.edges.groupBy(['source', 'target']).count().filter("count > 1").show(5)
# %% get core group
core = [{'id':r[0], **r[1]} for r in list(enron.G.nodes(data=True)) if r[1]['attr_is_core']==1]
core_group = pd.DataFrame.from_dict(core)
core.shape


# %%
mails = enron.df
lbl = enron.lbl.filter("label is not null")

mails.groupBy('message_id').count().filter("count > 1").join(mails, 'message_id', 'inner').orderBy('message_id').show(5, truncate=False)
# %%
email = enron.email
email = (email
        .join(lbl.select('email_address'), email.from_address==lbl.email_address, 'inner')
        .withColumnRenamed('email_address', 'from_email_address')
        .join(lbl.select('email_address'), email.recipient == lbl.email_address, 'inner')
        .withColumnRenamed('email_address', 'to_email_address')
)
email.show(4)

# %% mailbox mapping stats

cnt_mapped = enron.lbl.filter("label is not null").count()
cnt_missing = enron.lbl.filter("label is null").count()
print(f"{cnt_mapped} mailbox have a label and {cnt_missing} mailboxes are not mapped")
#%%
# enron.lbl.filter("label is null").show(10, truncate=False)

lbls = pd.read_csv(Enron_to_graph.LBL_PATH, sep='\t')
lbls['shortname'] = lbls['shortname'].astype(str)
lbl_df = (spark.createDataFrame(lbls[['shortname', 'group']])
                    .withColumnRenamed('shortname', 'mailbox')
        )

        # extract mailbox name and join with label from sent mail
df = (enron.email.filter("folder like '%sent%'")
            .select("from_address", 'fname')
            .withColumn("fname_last", F.regexp_extract(F.col('fname'), '(?<=Downloads/)(.)*', 0))
            .withColumn("mailbox", F.lower(F.split(F.col('fname_last'), '/').getItem(1)))
            .dropDuplicates()
            .join(lbl_df, 'mailbox', 'left')
            .withColumnRenamed("from_address", 'email_address')
            .withColumnRenamed('group', 'label')
            .withColumn('attr_is_core', F.lit(1))
        ) 
df.filter("label is null").show(10, truncate=False)       


# %% print list of email address mapped to the same mailbox

df2 = (df.select('mailbox', 'email_address')
            .dropDuplicates()
            .groupBy('mailbox').count()
            .filter("count > 2")
            .join(df.select('mailbox', 'email_address').dropDuplicates(),
                    'mailbox', 'inner')
)

df2.show(10, truncate=False)
# %%

emails = (enron.nodes
            .select('email_address', 'attr_cnt_send')
            .join(df2, 'email_address', 'inner')
)
emails.orderBy('mailbox').show(10)
# %% create mapping from E-mail address to label

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

email_to_lbl.filter("email_count > 1").show(20)
# %%
email_to_lbl.write.csv("/Users/tonpoppe/Downloads/email_to_lbl.csv", sep='\t')

# %%
LBL_PATH = '/Users/tonpoppe/workspace/graphcase_experiments/graphcase_experiments/graphcase_experiments/graphs/enron/data/email_to_lbl.csv'
email_lbls = pd.read_csv(LBL_PATH, sep=',')
email_lbls_df = (spark.createDataFrame(email_lbls)
                .filter("isCorrect = 1")
)
# %% internal edges

edges = enron.edges
nodes = enron.nodes
tmp = (edges
        .join(nodes.select(F.col('email_address').alias('source')), 'source', 'inner')
        .join(nodes.select(F.col('email_address').alias('target')), 'target', 'inner')
)

tmp.show(20, truncate=False)
# %%
emails = enron.email
res = emails.filter("from_address = 'anita.fam@enron.com' and recipient = 'janice.moore@enron.com'")

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

G = enron.G_sub

G.nodes[0]
# %%
list(G.edges(data=True))[0]
# %%
