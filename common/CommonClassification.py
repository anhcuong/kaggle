from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation, metrics
from xgboost import XGBClassifier


class CommonClassification(object):
    def __init__(self, x_training_set, y_training_set, x_test_set):
        self._x_training_set = x_training_set.values
        self._y_training_set = y_training_set.values
        self._x_test_set = x_test_set.values

    def start(self):
        self.random_forest_classifier()
        self.gradient_boosting_classifier()
        self.ada_boost_classifier()
        self.k_neighbor_classifier()
        self.decision_tree_classifier()
        self.linear_svc()
        # self.svc()
        self.quadratic_discriminant_analysis()
        self.gaussian()
        self.xgboost()

    def random_forest_classifier(self):
        print("==*== Running Random Forest classifier ==*==")
        rf = RandomForestClassifier(n_estimators=100)
        self.k_fold_cross_validation(rf)

    def gradient_boosting_classifier(self):
        print("==*== Running Gradient Boosting classifier ==*==")
        gradient_boosting = GradientBoostingClassifier(
            n_estimators=100,
            loss='deviance',
            max_features='sqrt'
        )
        self.k_fold_cross_validation(gradient_boosting)

    def ada_boost_classifier(self):
        print("==*== Running AdaBoost classifier ==*==")
        adaboost = AdaBoostClassifier(n_estimators=100)
        self.k_fold_cross_validation(adaboost)

    def k_neighbor_classifier(self):
        print("==*== Running K Neighbor classifier ==*==")
        knn = KNeighborsClassifier(
            n_neighbors=3,
            algorithm='auto',
            n_jobs=-1
        )
        self.k_fold_cross_validation(knn)

    def decision_tree_classifier(self):
        print("==*== Running Decision Tree classifier ==*==")
        decision_tree = DecisionTreeClassifier(max_features='sqrt')
        self.k_fold_cross_validation(decision_tree)

    def linear_svc(self):
        print("==*== Running Linear Support Vector Classification ==*==")
        lsvc = LinearSVC(max_iter=10000)
        self.k_fold_cross_validation(lsvc)

    def svc(self):
        print("==*== Running Support Vector Classification ==*==")
        svc = SVC(
            C=1,
            kernel='poly',
            degree=3,
            gamma='auto'
        )
        self.k_fold_cross_validation(svc)

    def quadratic_discriminant_analysis(self):
        print("==*== RunningQuadratic Discriminant Analysis ==*==")
        qda = QuadraticDiscriminantAnalysis()
        self.k_fold_cross_validation(qda)

    def gaussian(self):
        print("==*== Gaussian Naive Bayes ==*==")
        gaussian = GaussianNB()
        self.k_fold_cross_validation(gaussian)

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

    def fit(self, model, training_set, target_set):
        model.fit(training_set, target_set)
        training_score = model.score(training_set, target_set)
        return model, training_score

    def predict(self, model, x_test_set):
        return model.predict(x_test_set)

    def k_fold_cross_validation(self, model):
        N = len(self._x_training_set)
        kf = cross_validation.KFold(N, n_folds=3, shuffle=True, random_state=4)
        average_scores = cross_validation.cross_val_score(
            estimator=model,
            X=self._x_training_set,
            y=self._y_training_set,
            cv=kf
        )
        y_predict = cross_validation.cross_val_predict(
            estimator=model,
            X=self._x_training_set,
            y=self._y_training_set,
            cv=kf
        )
        overall_score = metrics.accuracy_score(self._y_training_set, y_predict)
        print "Overall score: {score}".format(
            score="{0:.2f}".format(overall_score)
        )
        print "Average score: {score} +/- {std}, Max: {max}".format(
            score="{0:.2f}".format(average_scores.mean()),
            std="{0:.2f}".format(average_scores.std()),
            max="{0:.2f}".format(average_scores.max())
        )
