import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

class InjuryModel():
    def __init__(self, model_type):
        self.model_type = model_type
        self.feature_importances_ = None

    def fit(self, X, y):
        """
        Fit the training data X and labels y to either a Gradient Boosted model
        or a Random Forest model.
        The parameters are the results of prior gridsearching.
        """
        if self.model_type == 'GBC':
            max_depth = 2
            n_estimators = 1150
            learning_rate = 0.005
            subsample = 0.5
            self.model = GradientBoostingClassifier()
        elif self.model_type == 'RFC':
            max_depth=5
            min_samples_split=2
            n_estimators=1000
            self.model = RandomForestClassifier()

        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_

    def predict(self, X_test):
        y_hat = self.model.predict(X_test)
        return y_hat
