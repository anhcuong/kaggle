from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                             AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


class CommonClassification(object):
    def __init__(self, x_training_set, y_training_set, x_test_set, y_test_set):
        self._x_training_set = x_training_set
        self._y_training_set = y_training_set
        self._x_test_set = x_test_set
        self._y_test_set = y_test_set

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

    def get_result(self, model):
        model.fit(self._x_training_set, self._y_training_set)
        training_score = model.score(self._x_training_set,
                                     self._y_training_set)
        test_score = model.score(self._x_test_set,
                                 self._y_test_set)
        print("Training Score: {score}".format(score=training_score))
        print("Test Score: {score}".format(score=test_score))

    def predict(self, model, x_test_set):
        return model.predict(x_test_set)
