#!/usr/bin/env python
#
###########################################
#
# File: spark_data_management.py
# Author: Ra Inta
#
# Description: An exploration of how to manage data using Spark, 
# in particular Spark SQL via PySpark.
#
# Created: March 25, 2021 Ra Inta
# Last Modified: 20210330, R.I.
#
###########################################


# setup init
import findspark
findspark.init()

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

import os

spark = SparkSession.builder.appName("SparkDataManagement").getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Alter this path as appropriate
DATA_DIR = "D:\Shared\Webinars\PySpark\data"

TRANSACTIONS_DIR = os.path.join(DATA_DIR, "transactions")
USERS_DIR = os.path.join(DATA_DIR, "users")

OUTPUT_DIRECTORY = os.path.join(DATA_DIR, "transaction_report")

transactions_df = spark.read.csv(TRANSACTIONS_DIR, header=True, inferSchema=True) 

# Hold on -- WE JUST IMPORTED ALL OF THOSE CSV FILES!
# How many files?
len(os.listdir(TRANSACTIONS_DIR))

# How many partitions in the underlying RDD?
transactions_df.rdd.getNumPartitions()

# What does the schema look like?
transactions_df.printSchema()

# What kind of DataFrame is it?
type(transactions_df)

# The first row
transactions_df.first()

# The first three
transactions_df.limit(3)

# Hey, that didn't print out!
# Spark is lazy. We need to tell it we're serious about doing this:
transactions_df.limit(3).show()

# Columns
transactions_df.columns

# Note that Spark SQL DataFrames have a strict schema
transactions_df.printSchema()

# Relating Spark SQL DataFrame to a pandas DataFrame
transactions_df.limit(3).toPandas()

col_names = "ABCDE"
n_cols = len(col_names)
n_rows = 10

pandas_df = pd.DataFrame({x: np.random.randint(1, 10, n_rows) for x in col_names}) 

pandas_to_spark_df = spark.createDataFrame(pandas_df)


###### End of data management I



# ETL in one line! Sure, it *looks* like two...
ETL_FILE = os.path.join(OUTPUT_DIRECTORY, "ETL.csv")

spark.read.csv(TRANSACTIONS_DIR, header=True, inferSchema=True)\
    .filter("transaction_amount > 0.01")\
        .groupBy("year_month")\
        .sum()\
            .coalesce(1)\
            .write.mode("append")\
                .option("header", True)\
                    .csv(ETL_FILE)
            
transactions_df.filter("transaction_amount > 0.01")\
        .groupBy("year_month")\
        .sum().columns


# Dealing with schemas
from pyspark.sql.types import StructField, StructType 
from pyspark.sql.types import StringType, IntegerType, FloatType

user_schema = StructType([StructField("user_id", IntegerType(),True),
                         StructField("first_name", StringType(),True),
                         StructField("last_name", StringType(),True),
                         StructField("email", StringType(),True),
                         StructField("password_hashed", StringType(),True),
                         StructField("address", StringType(),True),
                         StructField("city", StringType(),True),
                         StructField("state", StringType(),True),
                         StructField("employer", StringType(),True),
                         StructField("age", IntegerType(),True),
                         StructField("credit_card_num", StringType(),True),
                         StructField("credit_card_exp", StringType(),True),
                         StructField("security_answer", StringType(),True),
                         StructField("account_balance", FloatType(),True),
                         StructField("gender", StringType(),True)])

users_df = spark.read.csv(USERS_DIR, header=True, schema=user_schema)

users_df.printSchema()

users_df.head(3)

# How much data are missing?
from pyspark.sql.functions import isnull, lit, col

users_df.describe().show()
rows = users_df.count()
summary = users_df.describe().filter(col("summary") == "count")
summary.select(*((lit(rows)-col(c)).alias(c) for c in users_df.columns)).show()


users_df.count()

users_df.describe().show()

users_df[["age", "user_id"]].show(5)

users_df.select('age', 'user_id').show(5)

users_df.filter(transactions_df['age'] > 30).show(5)

users_df.filter(transactions_df['age'] > 30).describe('age').show(5)
#%%
transactions_df["age"].cast("float")

transactions_df.dtypes  # still the original

import pyspark.sql.functions as s_f

transactions_df = transactions_df.withColumn('Baseline_date_dt', 
                              	s_f.to_date(transactions_df['Baseline_date'], 
                                           'MM/dd/yyyy'))

transactions_df.select('Baseline_date', 'Baseline_date_dt').show(10)
#%%
## Add New Variables as function of others
transactions_df = transactions_df.withColumn('log_income',
                         s_f.log(transactions_df['Income']))

transactions_df.select('Income', 'log_income').show(10)
#%%
transactions_df.select('Gender').distinct().show()

#
transactions_df.groupBy('Gender').count().show()
#

transactions_df = \
    transactions_df.withColumn('Gender',
                         s_f.when(transactions_df['Gender'] != '999',
                                  transactions_df['Gender'])\
                                  .otherwise(None))

transactions_df.groupBy('Gender').count().show() 
#%%
transactions_df.select('Education').distinct().show()
#%%
transactions_df = transactions_df.withColumn('Education', 
                                 s_f.lower(transactions_df['Education']))

# we can lowercase everything, this will solve it!
transactions_df.select('Education').distinct().show()
#%%
transactions_df\
    .select('age')\
    .dropna()\
    .approxQuantile('age', [0.5], 0)

#%%
transactions_df\
    .select('age')\
    .approxQuantile('age', [0.25, 0.5, 0.75], 0)
    
transactions_df\
    .select('age')\
    .approxQuantile('age', [0.25, 0.5, 0.75], 0)
    
#%%

transactions_df\
    .select('age')\
    .describe()\
    .show()

    
#%%
transactions_df = \
    transactions_df.withColumn('age', 
                         s_f.when( transactions_df['age'] < 100, 
                                  transactions_df['age'])\
                             .otherwise(None))
        
#%%

transactions_df.select('age').describe().show()
#%%
transactions_df.groupBy('gender').avg('Income').show()

#%%
transactions_df.groupBy('Gender', 'Education').avg('Income').show()

#%%
transactions_df.groupBy('Gender').pivot('Education').avg('Income').show()
    

#%%
transactions_df.crosstab("Education", "Gender").show()

#%%
transactions_df.corr('age', 'Income') # Pearson correlation coef

#%%
transactions_df.corr("age", "value_1") # Pearson correlation coef

#%%
transactions_df.createOrReplaceTempView('transactions')


spark.sql('SELECT user_id, account_balance FROM transactions WHERE transaction_amount > 0.5').show()

#%%
query = 'SELECT corr(account_balance, transaction_amount) AS corr FROM transactions '
query += 'WHERE transaction_date > "2019-06-01"'

spark.sql(query).show()

#%%
query = 'SELECT MIN(age) AS min, max(age) AS max FROM person '
query += 'WHERE age > 55'

spark.sql(query).show()
#%%
# create spark dataframe
FILE_PATH = os.path.join(DATA_DIR, 'cluster_demo.csv')

df_cluster = spark.read.csv(FILE_PATH,
                           header=True,
                           inferSchema=True) 

df_cluster.show(5)
#%%
df_cluster = df_cluster.withColumnRenamed("site_id", "Clust_id")

#%%
## left join
df_left = \
    transactions_df.join(df_cluster,
                   on='Clust_id',
                   how='left_outer')

#%%    
## right join
df_right = \
    transactions_df.join(df_cluster,
                   on='Clust_id',
                   how='right_outer')
#%%    
## inner join
df_inner = \
    transactions_df.join(df_cluster,
                   on='Clust_id',
                   how='inner')


###########################################
# End of spark_data_management.py
###########################################