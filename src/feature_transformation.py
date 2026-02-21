import pandas as pd
import numpy as np

from category_encoders import TargetEncoder

class Transformer:
    def __init__(self, smoothing: int = 10):
        self.smoothing = smoothing
        self.encoder = TargetEncoder(cols=['category'], smoothing = self.smoothing)
        self.quantiles = {}

    def _log_transform(self, X: pd.DataFrame):

        X_transformed = X.copy()
        X_transformed['log_total_price'] = np.log1p(X['total_price'])
        X_transformed['log_total_freight'] = np.log1p(X['total_freight'])

        X_transformed.drop(['total_price','total_freight'], axis=1, inplace=True)

        return X_transformed

    def _get_clip_quantiles(self, X_train: pd.DataFrame):
        lower_seller = X_train['seller_disp_diff'].quantile(0.001)
        lower_estimated = X_train['estimated_delivery_diff'].quantile(0.001)

        upper_delivery = X_train['delivery_days'].quantile(0.999)
        upper_estimated = X_train['estimated_delivery_diff'].quantile(0.999)
        upper_processing = X_train['processing_days'].quantile(0.999)
        upper_seller = X_train['seller_disp_diff'].quantile(0.999)

        self.quantiles = {
            'lower_seller' : lower_seller,
            'lower_estimated' : lower_estimated,
            'upper_delivery' : upper_delivery,
            'upper_estimated' : upper_estimated,
            'upper_processing' : upper_processing,
            'upper_seller' : upper_seller
        }

    def _clip_outliers(self, X: pd.DataFrame):

        X_clipped = X.copy()
        
        lower_seller = self.quantiles['lower_seller']
        lower_estimated = self.quantiles['lower_estimated']

        upper_delivery = self.quantiles['upper_delivery']
        upper_estimated = self.quantiles['upper_estimated']
        upper_processing = self.quantiles['upper_processing']
        upper_seller = self.quantiles['upper_seller']

        X_clipped['delivery_days'] = X_clipped['delivery_days'].clip(upper=upper_delivery)
        X_clipped['estimated_delivery_diff'] = X_clipped['estimated_delivery_diff'].clip(lower = lower_estimated, upper = upper_estimated)
        X_clipped['processing_days'] = X_clipped['processing_days'].clip(upper = upper_processing)
        X_clipped['seller_disp_diff'] = X_clipped['seller_disp_diff'].clip(lower = lower_seller, upper = upper_seller)

        
        return X_clipped

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):

        self._get_clip_quantiles(X)
        self.encoder.fit(X, y)

        return self

    def transform(self, X: pd.DataFrame):
        
        X_log = self._log_transform(X)
        X_clip = self._clip_outliers(X_log)
        X_encoded = self.encoder.transform(X_clip)

        return X_encoded
    
    def fit_transform(self, X: pd.DataFrame, y:pd.DataFrame):

        self.fit(X, y)
        return self.transform(X)
