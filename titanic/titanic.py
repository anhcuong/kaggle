import pandas
import sklearn
import sys
import numpy as np
import xgboost as xgb
from pandas import Series, DataFrame
from sklearn import cross_validation
from common.CommonClassification import CommonClassification

train_df = pandas.read_csv("data/train.csv")
train_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
train_df = train_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].mean())
embarked_mode = train_df["Embarked"].mode()[0]
train_df["Embarked"] = train_df["Embarked"].fillna(embarked_mode)
train_df["Embarked"] = train_df["Embarked"].astype("category")
train_df["Embarked"] = train_df["Embarked"].cat.codes
train_df["Sex"] = train_df["Sex"].astype("category")
train_df["Sex"] = train_df["Sex"].cat.codes
training_set = train_df.drop(["Survived"], axis=1)
target_set = train_df["Survived"]

test_df = pandas.read_csv("data/test.csv")
test_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
test_df["Age"] = test_df["Age"].fillna(test_df["Age"].mean())
test_df["Embarked"] = test_df["Embarked"].fillna(test_df["Embarked"].mode()[0])
test_df["Embarked"] = test_df["Embarked"].astype("category")
test_df["Embarked"] = test_df["Embarked"].cat.codes
test_df["Sex"] = test_df["Sex"].astype("category")
test_df["Sex"] = test_df["Sex"].cat.codes

commonAlgo = CommonClassification(training_set, target_set, test_df)
commonAlgo.start()
