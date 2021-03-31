#!/usr/bin/env python
#
###########################################
#
# File: spark_machine_learning.py
# Author: Ra Inta
#
# Description: How to do Machine Learning with Spark's MLlib, via PySpark
#
# Created: March 25, 2021 Ra Inta
# Last Modified: 20210330, R.I.
#
###########################################


import findspark
findspark.init()

import numpy as np
import pandas as pd
import os

import pyspark

# To generate labels column from categorical data
from pyspark.ml.feature import OneHotEncoder, StringIndexer

# To generate features vector
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('spark_ml').getOrCreate()

DATA_DIR = "Desktop/Deloitte_Online_PySpark/data" 

# Double check this on the Virtual Machine!

#DATA_DIR = r"D:\Shared\Deloitte\Spark_Apr2021\Deloitte_Online_PySpark_updated\data"

FILE_PATH = os.path.join(DATA_DIR, "manufacturing", "CNC_Tool_Wear", "cnc_experiments_all.csv")

cnc_data = spark.read.csv(FILE_PATH, header=True, inferSchema=True)

cnc_class_labels = StringIndexer(inputCol='tool_condition', outputCol='label')

cnc_data = cnc_class_labels.fit(cnc_data).transform(cnc_data)
#%%
# These are the numerical columns:
cnc_feature_labels = cnc_data.columns[0:47] + cnc_data.columns[49:51]

cnc_feature_vector = VectorAssembler(
    inputCols=cnc_feature_labels,
    outputCol="features")  # Note the strict naming convention here

cnc_data = cnc_feature_vector.transform(cnc_data)

print("\nConverted features and labels for CNC wear dataset:")

cnc_data.sample(0.001).select("features", "label").show(10)
#%%
cnc_train, cnc_test = cnc_data.randomSplit([0.75, 0.25], seed=42)

print(f"\nRows of data for training: {cnc_train.count()}, testing: {cnc_test.count()}\n") 

#%%
from pyspark.ml.classification import LogisticRegression

logReg = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

logRegModel = logReg.fit(cnc_train)

print(f"Coefficients: {logRegModel.coefficients}; Intercept: {logRegModel.intercept}")
#%%
from pyspark.ml.evaluation import BinaryClassificationEvaluator

validation = logRegModel.transform(cnc_test)\
    .select("rawPrediction", "label")

evaluator = BinaryClassificationEvaluator()\
    .setMetricName("areaUnderROC")

print(f"\nValidation accuracy: {evaluator.evaluate(validation)*100}%\n" )

#%%

FILE_PATH = os.path.join(DATA_DIR, "manufacturing", "concrete", 
                         "concrete.csv")

concrete_data = spark.read.csv(FILE_PATH,
                               header=True,
                               inferSchema=True)

#%%
concrete_data = concrete_data.withColumnRenamed(concrete_data.columns[-1], 
                                                "label")

concrete_feature_labels = concrete_data.columns[0:8]

concrete_feature_vector = VectorAssembler(
    inputCols=concrete_feature_labels,
    outputCol="features")  
# Note the strict naming conventions here

#%%
concrete_train, concrete_test = concrete_data.randomSplit([0.75, 0.25], 
                                                          seed=42)

print(f"\nRows of data for training: {concrete_train.count()}, testing: {concrete_test.count()}\n")
#%%
# instantiate
from pyspark.ml.regression import LinearRegression

linReg = LinearRegression(maxIter=10, 
                          regParam=0.3, 
                          elasticNetParam=0.8, 
                          solver='normal')

# fit
from pyspark.ml.pipeline import Pipeline

lin_pipeline = Pipeline(stages=[concrete_feature_vector, linReg])

linRegModel = lin_pipeline.fit(concrete_train)

print(f"Coefficients: {linRegModel.stages[-1].coefficients}")

print(f"Intercept: {linRegModel.stages[-1].intercept}")

#%%
trainingSummary = linRegModel.stages[-1].summary

print(f"numIterations: {trainingSummary.totalIterations}")

print(f"objectiveHistory: {trainingSummary.objectiveHistory}")

#

trainingSummary.residuals.show(10)

print(f"RMSE: {trainingSummary.rootMeanSquaredError}")

print(f"r2: {trainingSummary.r2}") 

#%%
from pyspark.ml.evaluation import RegressionEvaluator

concrete_test = concrete_feature_vector.transform(concrete_test)

validation = linRegModel.stages[-1].transform(concrete_test).select("prediction", "label")

evaluator = RegressionEvaluator()

print(f"\nRMSE: {evaluator.evaluate(validation)}\n" )

#%%
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

paramGrid = ParamGridBuilder()\
    .addGrid(linReg.regParam, [0.1, 0.01, 5.0])\
    .addGrid(linReg.elasticNetParam, np.arange(0.0, 1.25, 0.25))\
    .addGrid(linReg.fitIntercept, [False, True])\
    .build() 
    
#%%
trainValSplit = TrainValidationSplit(estimator=linReg,
                           estimatorParamMaps=paramGrid,
                           evaluator=RegressionEvaluator(),
                           trainRatio=0.75) 

concrete_train = concrete_feature_vector.transform(concrete_train)

linReg_tvs = trainValSplit.fit(concrete_train)
#%%

linReg_tvs.transform(concrete_test)\
    .select("features", "label", "prediction")\
    .show(10)

#%%
linReg_valid = linReg_tvs.transform(concrete_test).select("prediction", "label")

evaluator = RegressionEvaluator()

print(f"\nRMSE: {evaluator.evaluate(linReg_valid)}\n" ) 

print(f"Model coefficients: {linReg_tvs.bestModel.coefficients}")

print(f"Model intercept: {linReg_tvs.bestModel.intercept}")

#%%
print(f"Chosen regularization parameter: {linReg_tvs.bestModel._java_obj.getRegParam()}")

print(f"Chosen elastic net parameter: {linReg_tvs.bestModel._java_obj.getElasticNetParam()}")

#%%
from pyspark.ml.tuning import CrossValidator

crossValid = CrossValidator(estimator=linReg,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=5) 

# WARNING: This may take a long time to run on the VM!
linReg_cv = crossValid.fit(concrete_train)

linReg_cv.transform(concrete_test)\
    .select("features", "label", "prediction")\
    .show()
    


#%%
validation = linReg_cv.transform(concrete_test).select("prediction", "label")

evaluator = RegressionEvaluator()

print(f"\nRMSE: {evaluator.evaluate(validation)}\n" )

#%%
print(f"\nRMSE: {evaluator.evaluate(validation)}\n" )

print(f"Model coefficients: {linReg_tvs.bestModel.coefficients}")

print(f"Model intercept: {linReg_tvs.bestModel.intercept}")

#%%

print(f'Best Param (regParam): {linReg_cv.bestModel._java_obj.getRegParam()}')

print(f'Best Param (elasticNetParam): {linReg_cv.bestModel._java_obj.getElasticNetParam()}')

#%%

from pyspark.ml.classification import MultilayerPerceptronClassifier

mlp_architecture = [len(cnc_feature_labels), 128, 128, 2]

mlp_template = MultilayerPerceptronClassifier(
    maxIter=100, 
    layers=mlp_architecture, 
    blockSize=128, 
    seed=13579)

# WARNING: This may take a long time to run on the VM!
mlp_model = mlp_template.fit(cnc_train)

validation = mlp_model.transform(cnc_test).select("rawPrediction", "label")

evaluator = BinaryClassificationEvaluator()

print(f"\nValidation accuracy: {evaluator.evaluate(validation)*100}%\n") 

spark.stop()

###########################################
# End of spark_machine_learning.py
###########################################
