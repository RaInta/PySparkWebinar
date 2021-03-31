#!/usr/bin/env python
#
###########################################
#
# File: count_raven.py
# Author: Ra Inta
#
# Description: Similar functionality as count_green_eggs_ham.py
# except this makes use of the SparkContext.textFile()
# convenience function to easily obtain data from HDFS format
# and convert to a RDD of strings. You can just as easily convert
# back to a HDFS store.
#
# Created: March 25, 2021 Ra Inta
# Last Modified: 20210330, R.I.
#
###########################################

import findspark
findspark.init()

import os
import shutil

from operator import add

from pyspark.sql import SparkSession

# The following initiates the Spark Session; check out 
# http://localhost:4040/  in your browser to check the status
spark = SparkSession.builder.appName("CountRaven").getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

DATA_DIR = "D:\Shared\Webinars\PySpark\data\word_count"

INPUT_FILE = os.path.join(DATA_DIR, "the_raven_lower.txt")

OUTPUT_DIRECTORY = os.path.join(DATA_DIR, "count_raven_output")

# The following will delete the entire OUTPUT_DIRECTORY
# Make sure you don't change it to something you're attached to
if os.path.exists(OUTPUT_DIRECTORY):
    try:
        shutil.rmtree(OUTPUT_DIRECTORY)
    except OSError as e:
        print(f"Error: {e} occurred trying to delete {OUTPUT_DIRECTORY}")

#####################################################################
### The next two lines are where the 'real' work is done
### Of course it could also have been one (long) line 
#####################################################################

lines = spark.read.text(INPUT_FILE).rdd.map(lambda r: r[0])

sortedCount = lines.flatMap(lambda x: x.split())\
    .map(lambda x: (str(x), 1))\
        .reduceByKey(add)\
            .sortBy(lambda x: x[1], ascending=False)

# Note that this will output a directory, rather than a single file. Why?
# Because RDDs are partitioned to a 'reasonable' size. 
# Each of these partitions will become a file.
sortedCount.saveAsTextFile(OUTPUT_DIRECTORY)  

# You can check this by repartitioning the RDD:
# n_repartitions = 4    
# OUTPUT_DIRECTORY_REPART = OUTPUT_DIRECTORY + f"_repartitioned{4}"
# sortedCount.repartition(n_repartitions).saveAsTextFile(OUTPUT_DIRECTORY) 

# We can bring all the data from the resulting RDD back to the driver
# with the .collect() action. 
# NOTE: THERE IS A GOOD REASON WHY SPARK IS DISTRIBUTED. PROCEED WITH CAUTION.
output = sortedCount.collect()

print("\n")

for rank, (word, count) in enumerate(output[:20], start=1):
    print(f"# {rank:<2}: {word:<12} ({count})")

print("\n")
    
spark.stop()

# Of course we could have done this quicker--and in one line--using Unix
# utilities:
# ! sed 's/\s\s*/\n/g' "the_raven_lower.txt" | sort | uniq -c | sort -nr > count_raven_output.txt



#############################################
### The following strips punctuation and case
### for 'The Raven' (or any text).
#############################################
#
# You will need to install the NLTK (Natural Language ToolKit) library
# from nltk.tokenize import word_tokenize
# from string import punctuation

# def process_text_file(input_text, output_text):
#     while os.path.exists(output_text):
#         print(f"\nFile {output_text} already exists!")
#         output_text = input("Please enter another filename: (Ctrl-C to exit): ")

#     with open(input_text, encoding='utf-8') as f1:
#         text_string = f1.readlines()
    
#     text_tokenized = [[w.lower() for w in word_tokenize(x) if w not in 
#           list(punctuation) + ['“', '”', '‘', '’']] for x in text_string]
    
#     processed_text = [" ".join(x) for x in text_tokenized]
    
#     processed_text = "\n".join(processed_text)
    
#     with open(output_text, "w", encoding="utf-8") as f1:
#         f1.write(processed_text)

# ORIGINAL_FILE = os.path.join(DATA_DIR, "the_raven.txt")

# process_text_file(ORIGINAL_FILE, INPUT_FILE)

###########################################
# End of count_raven.py
###########################################
