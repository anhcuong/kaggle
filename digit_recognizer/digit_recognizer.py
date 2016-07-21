import pandas
import sklearn
import sys
import numpy as np
import xgboost as xgb
from pandas import Series, DataFrame
from sklearn import cross_validation
from common.CommonClassification import CommonClassification

train_df = pandas.read_csv("data/train.csv")
training_set = train_df.drop(["label"], axis=1)
target_set = train_df["label"]

test_df = pandas.read_csv("data/test.csv")

commonAlgo = CommonClassification(training_set, target_set, test_df)
commonAlgo.start()
