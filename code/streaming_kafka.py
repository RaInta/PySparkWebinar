#!/usr/bin/env python
#
###########################################
#
# File: streaming_kafka.py
# Author: Ra Inta
# Description:
# Created: January 18, 2021
# Last Modified: January 18, 2021
#
###########################################

from pyspark.sql import SparkSession

spark = SparkSession.builder\
    .master("local")\
    .appName("KafkaSpark")\
    .getOrCreate()

df = spark\
    .read\
    .format("kafka")\
    .option("kafka.bootstrap.servers", "localhost:9092")\
    .option("subscribe", "quickstart-events")\
    .load()

df = df.withColumn('key_str', df['key'].cast('string').alias('key_str'))\
    .drop('key')\
    .withColumn('value_str', df['value'].cast('string').alias('key_str'))\
    .drop('value')

df.show(20)


###########################################
# End of streaming_kafka.py
###########################################
