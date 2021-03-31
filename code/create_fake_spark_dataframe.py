#!/usr/bin/env python
#
###########################################
#
# File: create_fake_spark_dataframe.py
# Author: Ra Inta
# Description:
# Created: April 04, 2019
# Last Modified: 20210330, R.I.
#
###########################################

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, DateType

from pyspark.sql.functions import udf

from mimesis import Person, Address, Business, Payment, Text

from scipy.stats import pareto
import pandas as pd
import numpy as np

import os

import findspark
findspark.init()

spark = SparkSession.builder.appName('generate_user_data').getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Note: we don't ever store user passwords as clear text!!!
# To emulate salting and hashing the user passwords:
import hashlib, uuid

# However, we should really use a dedicated password hashing
# package, such as passlib. However, this is out of scope
# for this demo (it would require further installation).
# import passlib

import datetime as dt

np.random.seed(42)  # To make our analysis reproducible

person = Person()
address = Address()
business = Business()
payment = Payment()
text = Text()

##################################################
### Define a couple of convenience functions:
##################################################

def hashed_passwd(passwd):
    """We should never entertain the idea of storing users' passwords
    as plaintext. This function performs a basic salting and hashing
    of a password input. This function should *never* be used in a
    production setting; if you need to securely store salted and hashed
    passwords, use a dedicated package such as passlib."""
    salt = uuid.uuid4().hex
    return hashlib.sha512(passwd.encode('utf-8')
                          + salt.encode('utf-8')).hexdigest()


def account_balance():
    """Generate account balances according to a Pareto distribution.
    We should expect balances to be distributed as with other income
    distributions.  The power exponent is chosen here to replicate
    the 80-20 rule."""
    return float(pareto.rvs(1.161))


def generate_transaction_amount(account_balance):
    mean_fraction = 0.05  # Average transaction, percentage of balance
    min_transaction = 0.01
    max_fraction = 0.3
    rnd_amplitude = np.random.randn() + mean_fraction
    # Clip to maximum fraction of balance
    if rnd_amplitude > max_fraction:
        rnd_amplitude = max_fraction
    transaction_amount = rnd_amplitude*account_balance
    # Enforce minimum absolute transaction
    if transaction_amount < min_transaction:
        transaction_amount = min_transaction
    return transaction_amount

    
generate_transaction_amount_udf = udf(generate_transaction_amount, 
                                      returnType=DoubleType())


def generate_transaction_date(max_date=dt.datetime.today(), 
                              min_date=dt.datetime(2019, 1, 1, 12, 0)):
    rnd_date = min_date + dt.timedelta(seconds=np.random.rand()\
                                       *(max_date-min_date).total_seconds())
    return rnd_date


generate_transaction_date_udf = udf(generate_transaction_date, DateType())

##################################################

##################################################
### Generate a DataFrame of user information
##################################################
# Generate 20,000 rows of the following:
# user_id, first_name, last_name, email, password, employer, address,
# birth_date, credit_card_num, credit_card_exp, security_answer,
# account_balance


#csv_name = "/home/ra/host/BH_Analytics/spark/courseware/data/gapminder.csv"
# df = sqlContext.read.csv("/home/ra/host/BH_Analytics/spark/courseware/data/gapminder.csv", header=True)

user_schema = ["user_id", "first_name", "last_name", "email",
        "password_hashed", "address", "city", "state", "employer", "age",
        "credit_card_num", "credit_card_exp",
        "security_answer", "account_balance" ]

data_length = 2_000

user_df = spark.createDataFrame([[x, person.name(), person.surname(), person.email(),
            hashed_passwd(person.password()), address.address(),
            address.city(), address.state(),
            business.company(), person.age(), payment.credit_card_number(),
            payment.credit_card_expiration_date(), text.word(),
                         account_balance()] for x in range(data_length)], 
                                schema=user_schema)

# transaction_df = user_df.select("user_id", "account_balance")

test_df = user_df\
    .select("user_id", "account_balance")\
        .sample(withReplacement=True, fraction=1.0,
                seed=42)

test_df = test_df.withColumn("transaction_amount", 
                   generate_transaction_amount_udf(test_df["account_balance"]))

test_df = test_df.withColumn("transaction_date", 
                   generate_transaction_date_udf())

print("Summary of transaction data:")
test_df.describe().show()

print("Sample of transaction data:")
test_df.show(4)

# Examine logical and physical plans:
# test_df.explain(extended=True)

# Load relevant objects
# Load relevant objects
# sc = SparkContext('local')  # Context should already be set; otherwise uncomment
#sqlContext = SQLContext(spark)

#sc.setLogLevel("WARN")

# log_txt = sc.textFile("/home/ra/host/BH_Analytics/spark/courseware/data/gapminder.csv")

# header = log_txt.first()

# #filter out the header, make sure the rest looks correct
# log_txt = log_txt.filter(lambda line: line != header)
# log_txt.take(10)

# #  [u'0\\tdog\\t20160906182001\\tgoogle.com', u'1\\tcat\\t20151231120504\\tamazon.com']

# temp_var = log_txt.map(lambda k: k.split("\\t"))

# # here's where the changes take place
# # this creates a dataframe using whatever pyspark feels like 
# # using (I think string is the default). the header.split is providing the names of the columns
# log_df=temp_var.toDF(header.split("\\t"))
# log_df.show()

# +------+------+--------------+----------+
# |field1|field2|        field3|    field4|
# +------+------+--------------+----------+
# |     0|   dog|20160906182001|google.com|
# |     1|   cat|20151231120504|amazon.com|
# +------+------+--------------+----------+
#note log_df.schema
#StructType(List(StructField(field1,StringType,true),StructField(field2,StringType,true),StructField(field3,StringType,true),StructField(field4,StringType,true)))

# now lets cast the columns that we actually care about to dtypes we want
# log_df = log_df.withColumn("field1Int", log_df["field1"].cast(IntegerType()))
# log_df = log_df.withColumn("field3TimeStamp", log_df["field1"].cast(TimestampType()))

# log_df.show()

# +------+------+--------------+----------+---------+---------------+
# |field1|field2|        field3|    field4|field1Int|field3TimeStamp|
# +------+------+--------------+----------+---------+---------------+
# |     0|   dog|20160906182001|google.com|        0|           null|
# |     1|   cat|20151231120504|amazon.com|        1|           null|
# +------+------+--------------+----------+---------+---------------+

# log_df.schema

# StructType(List(StructField(field1,StringType,true),StructField(field2,StringType,true),StructField(field3,StringType,true),StructField(field4,StringType,true),StructField(field1Int,IntegerType,true),StructField(field3TimeStamp,TimestampType,true)))

#now let's filter out the columns we want
# log_df.select(["field1Int","field3TimeStamp","field4"]).show()

# +---------+---------------+----------+
# |field1Int|field3TimeStamp|    field4|
# +---------+---------------+----------+
# |        0|           null|google.com|
# |        1|           null|amazon.com|
# +---------+---------------+----------+

# sc.stop()


#!/usr/bin/env python
#
###########################################
#
# File: mimesis_demo.py
# Author: Ra Inta
# Description:
#
# Created: March 30, 2019
# Last Modified: March 30, 2019
#
###########################################


##################################################

##################################################
### Perform some Exploratory Data Analysis
##################################################

# user_df.sample(5)

# user_df.describe()

# Note the median balance is 1.8, while the mean is 5.3
# Recall we generated a heavily skewed distribution!

# We designed it according to the famous "80-20 rule"
# The top twenty percent own 80% of the balances.
# Let's test it. Take the 80th percentile:
# critical80 = np.quantile(user_df["account_balance"], 0.8)
## 4.013269256450965

# the_few = user_df.loc[user_df["account_balance"] > critical80,
#                       "account_balance"].sum()

# tot_balance = user_df["account_balance"].sum()

# the_few/tot_balance
## 0.7298469832819879
# So here, the top 20% 'only' have 73% of the account balance

# Some limitations of mimesis
# If you want realistic distributions of certain numerical variables
# then you should simulate populations yourself. E.g.:

# user_df["age"].plot(kind="kde")

# The way ages are generated are not exactly samples of any real population!
# This will depend on the underlying demographic dynamics.


##################################################


##################################################
### Export data to Excel and print summary
##################################################

# print("Account balance for top 20% of users: {} \nFraction of total \
#       balance owned by top 20%: {}%\n".format(critical80,
#                                               100*the_few/tot_balance))

# output_file_name = "synthetic_user_data.xlsx"

# user_df.to_excel(output_file_name, index=False)

# print("Synthetic user data output to: {}\n".format(output_file_name))

##################################################


spark.stop()

###########################################
# End of create_fake_spark_dataframe.py
###########################################
