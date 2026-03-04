import pandas as pd
import numpy as np

from category_encoders import TargetEncoder
from sklearn.exceptions import NotFittedError

import joblib
import os

class Transformer:
    """
    A feature transforming and feature engineering class.
    Responsible for log transformations, clipping outliers (based on training quantiles), and performing target encoding on categorical variable.
    """
    def __init__(self, smoothing: int = 10) -> None:
        """
        Initializes the Transformer object

        Args:
            smoothing (int, optional): The smoothing parameter for the TargetEncoder. Defaults to 10.
        """
        self.smoothing = smoothing
        self.encoder = None
        self.quantiles = {}

    def _log_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a logarithmic transformation to price and freight columns. 

        Args:
            X (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: A new DataFrame with log-transformed price and freight columns (original columns removed).
        """
        X_transformed = X.copy()
        X_transformed['log_total_price'] = np.log1p(X['total_price'])
        X_transformed['log_total_freight'] = np.log1p(X['total_freight'])

        X_transformed.drop(['total_price','total_freight'], axis=1, inplace=True)

        return X_transformed

    def _get_clip_quantiles(self, X_train: pd.DataFrame) -> None:
        """
        Calculates and stores the 1st and 99th percentiles for specific features to be used for outlier clipping.

        Args:
            X_train (pd.DataFrame): The training dataset used to define quantiles.
        """
        lower_seller = X_train['seller_disp_diff'].quantile(0.01)
        lower_estimated = X_train['estimated_delivery_diff'].quantile(0.01)

        upper_delivery = X_train['delivery_days'].quantile(0.99)
        upper_estimated = X_train['estimated_delivery_diff'].quantile(0.99)
        upper_processing = X_train['processing_days'].quantile(0.99)
        upper_seller = X_train['seller_disp_diff'].quantile(0.99)

        self.quantiles = {
            'lower_seller' : lower_seller,
            'lower_estimated' : lower_estimated,
            'upper_delivery' : upper_delivery,
            'upper_estimated' : upper_estimated,
            'upper_processing' : upper_processing,
            'upper_seller' : upper_seller
        }

    def _clip_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clips the outliers in the dataset based on the previously calculated quantiles.
        Args:
            X (pd.DataFrame): The input dataset to be clipped.

        Returns:
            pd.DataFrame: A DataFrame with extreme values capped.
        """
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

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'Transformer':
        """
        Fits the transformer to the training data. 
        Calculates clipping boundaries and trains the TargetEncoder.

        Args:
            X (pd.DataFrame): The training input samples.
            y (pd.DataFrame): The target labels.

        Returns:
            Transformer: The trained Transformer object (self).
        """
        self._get_clip_quantiles(X)

        self.encoder = TargetEncoder(cols=['category'], smoothing = self.smoothing)
        self.encoder.fit(X[['category']], y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input data applying log transformations, clipping, and target encoding.

        Args:
            X (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: A fully transformed dataset ready for modelling.
        """
        X_log = self._log_transform(X)
        X_clip = self._clip_outliers(X_log)
        X_encoded = X_clip.copy()

        X_encoded['category'] = self.encoder.transform(X_encoded['category'])

        return X_encoded
    
    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """Fits the transformer to the data and transforms it in one step.

        Args:
            X (pd.DataFrame): The training input samples.
            y (pd.DataFrame): The target labels.

        Returns:
            pd.DataFrame: A fully transformed dataset ready for modelling.
        """
        self.fit(X, y)
        return self.transform(X)

    def save(self, filename: str = 'target_encoder.pkl') -> None:
        """
        Dumps the fitted TargetEncoder object to a pickle file.

        Args:
            filename (str, optional): Name of the file to save. Defaults to 'target_encoder.pkl'.

        Raises:
            NotFittedError: If the transformer hasn't been fitted before saving.
        """
        if self.encoder is None:
            raise NotFittedError('You must fit the Target Encoder first!')

        if not filename.endswith('.pkl'):
            filename += '.pkl'

        path = os.path.join('..','models', filename)
        joblib.dump(self.encoder, path)

        print(f'Saved the target encoder to {filename}')

    def load(self, filename: str = 'target_encoder.pkl') -> None:
        """
        Loads the fitted TargetEncoder object from a pickle file.

        Args:
            filename (str, optional): Name of the file to load. Defaults to 'target_encoder.pkl'.

        Raises:
            ValueError: If the specified file cannot be found in the models/ directory.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        path = os.path.join('..','models', filename)

        if not os.path.exists(path):
            raise ValueError(f'Could not find the file: {filename}!')
        
        self.encoder = joblib.load(path)