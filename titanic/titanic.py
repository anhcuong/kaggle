import pandas
import sklearn
import sys

from pandas import Series, DataFrame
import numpy as np
import xgboost as xgb
from sklearn import cross_validation

sys.path.append("../")
from common.CommonClassification import CommonClassification

train_df = pandas.read_csv("data/train.csv")
train_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
train_df = train_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].mean())
train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])
train_df["Embarked"] = train_df["Embarked"].astype("category")
train_df["Embarked"] = train_df["Embarked"].cat.codes
train_df["Sex"] = train_df["Sex"].astype("category")
train_df["Sex"] = train_df["Sex"].cat.codes
training_set = train_df.drop(["Survived"], axis=1)
target_set = train_df["Survived"]

# K-fold cross validation
kf = cross_validation.KFold(len(training_set), n_folds=3)
for train_index, test_index in kf:
    print "=============================================="
    x_train, x_test = training_set.ix[train_index], training_set.ix[test_index]
    y_train, y_test = target_set.ix[train_index], target_set.ix[test_index]        
    commonAlgo = CommonClassification(x_train, y_train, x_test, y_test)
    commonAlgo.start()
    print "=============================================="
    
