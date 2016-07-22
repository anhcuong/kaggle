import pandas
import sklearn
import sys
import numpy as np
import xgboost as xgb
from pandas import Series, DataFrame
from sklearn import cross_validation
from common.CommonClassification import CommonClassification
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

train_df = pandas.read_csv("data/train.csv")
training_set = train_df.drop(["label"], axis=1)
target_set = train_df["label"]
test_df = pandas.read_csv("data/test.csv")
commonAlgo = CommonClassification(training_set, target_set, test_df)


def evaluate():
    commonAlgo.start_evaluate()


def get_result():
    print("==*== Random forest ==*==")
    rf = RandomForestClassifier(n_estimators=100)
    model, score = commonAlgo.fit(rf)
    print "Train score: {score}".format(
        score=score
    )
    commonAlgo.predict(model, "output/rf.csv")

    print("==*== xgBoost ==*==")
    xgb = XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        silent=True,
        nthread=-1
    )
    model, score = commonAlgo.fit(xgb)
    print "Train score: {score}".format(
        score=score
    )
    commonAlgo.predict(model, "output/xgb.csv")

evaluate()
get_result()