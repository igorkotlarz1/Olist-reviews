from sklearn.feature_selection import RFECV, RFE, VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import pandas as pd

class Selector:
    """
    Class responsible for feature selection using Recursive Feature Elimination with Cross-Validation (RFECV) method
    and the XGB model.
    """
    def __init__(self, scale_weight: float, seed: int) -> None:
        """
        Initializes selector object.

        Args:
            scale_weight (float): Imbalanced classes weight (scale_pos_weight for the XGB model).
            seed (int): Random seed for reproducibility.
        """
        self.scale_weight = scale_weight
        self.SEED = seed

        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.SEED)

        self.support_ = None
        self.cv_results = None
        self.features_to_keep_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'Selector':
        """
        Trains the RFECV process, adjusting the optimal set of features.

        Args:
            X (pd.DataFrame): The training input samples.
            y (pd.DataFrame): The target labels.

        Returns:
            Selector: The trained Selector object (self).
        """
        xgb_rfe = XGBClassifier(scale_pos_weight=self.scale_weight, n_jobs=-1, random_state=self.SEED)
        
        rfecv = RFECV(estimator=xgb_rfe, step=1, cv=self.cv, scoring='f1', verbose=1)
        rfecv.fit(X, y)

        self.support_ = rfecv.support_
        self.cv_results = rfecv.cv_results_

        self.features_to_keep_ = X.columns[self.support_]

        return self          

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies previously fitted mask, leaving only the most important features.

        Args:
            X (pd.DataFrame): DF with features to be selected.

        Returns:
            pd.DataFrame: DF with the most important features preserved.
        """
        return X[self.features_to_keep_]