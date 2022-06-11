def extract_individual_edges(df):
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

    