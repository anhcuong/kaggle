from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor \
                             AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


class CommonRegression(Object):
    def __init__(x_training_set, y_training_set, x_test_set, y_test_set):
        self._x_training_set = x_training_set
        self._y_training_set = y_training_set
        self._x_test_set = x_test_set
        self._y_test_set = y_test_set

    def start(self):
        self.linear_regression()
        self.random_forest_regression()
        self.logistic_regression()
        self.gradient_boosting_regression()
        self.ada_boost_regression()

    def linear_regression(self):
        print "==*== Running Linear Regression ==*=="
        linear = LinearRegression()
        self.get_result(linear)

    def logistic_regression(self):
        print "==*== Running Logistic Regression ==*=="
        logistic = LogisticRegression()
        self.get_result(logistic)

    def random_forest_regression(self):
        print "==*== Running Random Forest Regression ==*=="
        rf = RandomForestRegressor(n_estimator=50)
        self.get_result(rf)

    def gradient_boosting_regression(self):
        print "==*== Running Gradient Boosting Regression ==*=="
        gradient_boosting = GradientBoostingRegressor(n_estimator=100,
                                                      loss='ls',
                                                      max_features="sqrt")
        self.get_result(gradient_boosting)

    def ada_boost_regression(self):
        print "==*== Running AdaBoost Regression ==*=="
        adaboost = AdaBoostRegressor(n_estimator=100,
                                     base_estimator=DecisionTreeRegressor,
                                     loss='linear')
        self.get_result(adaboost)

    def get_result(self, model):
        model.fit(self._x_training_set, self._y_training_set)
        training_score = model.score(self._x_training_set,
                                     self._y_training_set)
        test_score = model.score(self._x_test_set,
                                 self._y_test_set)
        print "Training Score: {score}".format(score=training_score)
        print "Test Score: {score}".format(score=test_score)

    def predict(self, model, x_test_set):
        return model.predict(x_test_set)
