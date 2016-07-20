from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                             AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from xgboost import XGBClassifier
from pandas import Series


class CommonClassification(object):
    def __init__(self, x_training_set, y_training_set, x_test_set):
        self._x_training_set = x_training_set
        self._y_training_set = y_training_set
        self._x_test_set = x_test_set

    def start(self):
        # self.random_forest_classifier()
        # self.gradient_boosting_classifier()
        # self.ada_boost_classifier()
        # self.k_neighbor_classifier()
        # self.decision_tree_classifier()
        # self.linear_svc()
        # self.svc()
        # self.quadratic_discriminant_analysis()
        # self.gaussian()
        self.xgboost()

    def random_forest_classifier(self):
        print("==*== Running Random Forest classifier ==*==")
        rf = RandomForestClassifier(n_estimators=100)
        self.get_result(rf)

    def gradient_boosting_classifier(self):
        print("==*== Running Gradient Boosting classifier ==*==")
        gradient_boosting = GradientBoostingClassifier(
            n_estimators=100,
            loss='deviance',
            max_features='sqrt'
        )
        self.get_result(gradient_boosting)

    def ada_boost_classifier(self):
        print("==*== Running AdaBoost classifier ==*==")
        adaboost = AdaBoostClassifier(n_estimators=100)
        self.get_result(adaboost)

    def k_neighbor_classifier(self):
        print("==*== Running K Neighbor classifier ==*==")
        knn = KNeighborsClassifier(
            n_neighbors=3,
            algorithm='auto',
            n_jobs=-1
        )
        self.get_result(knn)

    def decision_tree_classifier(self):
        print("==*== Running Decision Tree classifier ==*==")
        decision_tree = DecisionTreeClassifier(max_features='sqrt')
        self.get_result(decision_tree)

    def linear_svc(self):
        print("==*== Running Linear Support Vector Classification ==*==")
        lsvc = LinearSVC(max_iter=10000)
        self.get_result(lsvc)

    def svc(self):
        print("==*== Running Support Vector Classification ==*==")
        svc = SVC(
            C=1,
            kernel='poly',
            degree=3,
            gamma='auto'
        )
        self.get_result(svc)

    def quadratic_discriminant_analysis(self):
        print("==*== RunningQuadratic Discriminant Analysis ==*==")
        qda = QuadraticDiscriminantAnalysis()
        self.get_result(qda)

    def gaussian(self):
        print("==*== Gaussian Naive Bayes ==*==")
        gaussian = GaussianNB()
        self.get_result(gaussian)

    def xgboost(self):
        print("==*== xgBoost ==*==")
        xgb = XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            silent=True,
            nthread=-1
        )
        self.k_fold_cross_validation(xgb)

    def get_result(self, model):
        model.fit(self._x_training_set, self._y_training_set)
        training_score = model.score(self._x_training_set,
                                     self._y_training_set)
        print("Training Score: {score}".format(score=training_score))

    def fit(self, model, training_set, target_set):
        model.fit(training_set, target_set)
        training_score = model.score(training_set, target_set)
        return model, training_score


    def predict(self, model, x_test_set):
        return model.predict(x_test_set)

    def k_fold_cross_validation(self, model):
        N = len(self._x_training_set)
        kf = cross_validation.KFold(N, n_folds=3)
        training_score = 0
        y_test = Series()
        for train_index, test_index in kf:
            x_train, x_test = (self._x_training_set.ix[train_index],
                self._x_training_set.ix[test_index])
            y_train = self._y_training_set.ix[train_index]
            model, score = self.fit(model, x_train, y_train)
            training_score = training_score + score
            result = self.predict(model, x_test)
            print result
            print type(result.tolist())
            print test_index
            y_test.ix[test_index] = result.tolist()
            print y_test
        print y_test
        print("Average Training Score: {score}".format(score=training_score/3))