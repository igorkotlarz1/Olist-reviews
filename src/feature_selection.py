from sklearn.feature_selection import RFECV, RFE, VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import pandas as pd

class Selector:
    def __init__(self, scale_weight: float, seed: int):
        self.scale_weight = scale_weight
        self.SEED = seed

        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.SEED)

        self.support_ = None
        self.cv_results = None
        self.features_to_keep_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        xgb_rfe = XGBClassifier(scale_pos_weight=self.scale_weight, n_jobs=-1, random_state=self.SEED)
        
        rfecv = RFECV(estimator=xgb_rfe, step=1, cv=self.cv, scoring='f1', verbose=1)
        rfecv.fit(X, y)

        self.support_ = rfecv.support_
        self.cv_results = rfecv.cv_results_

        self.features_to_keep_ = X.columns[self.support_]

        return self          

    def transform(self, X):
        return X[self.features_to_keep_]