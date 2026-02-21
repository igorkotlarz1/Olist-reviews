import optuna
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import numpy as np
import pandas as pd

import joblib
import os

class Model:
    def __init__(self, model_type, seed, n_trials=100):

        model_type = model_type.lower()
        if model_type not in ['xgb', 'rf']:
            raise ValueError("You need to specify the model type: either 'xgb' or 'rf'!")

        self.model_type = model_type
        self.SEED = seed
        self.n_trials = n_trials

        self.best_model_ = None
        self.best_params_ = None
        self.scale_weight_ = None

    def _check_fitted(self):
        if self.best_model_ is None:
            raise NotFittedError('You must fit the model first!')

    def _get_model_params(self, trial: optuna.trial.Trial):
        if self.model_type == 'xgb':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 2, 15),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'scale_pos_weight': self.scale_weight,
                'tree_method': 'hist',
                'device': 'cuda',               
                'random_state': self.SEED
            }

        elif self.model_type == 'rf':
            return {
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),               
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
                'max_features': trial.suggest_float('max_features', 0.4, 0.8),
                'class_weight': 'balanced',
                'n_jobs': -1,              
                'random_state': self.SEED
            }

    def _objective(self, trial: optuna.trial.Trial, X_train: pd.DataFrame, y_train: pd.DataFrame):
        params = self._get_model_params(trial)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.SEED)
        scores = []

        for train_idx, validation_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[validation_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[validation_idx]

            if self.model_type == 'xgb':
                model = XGBClassifier(**params)
            elif self.model_type == 'rf':
                model = RandomForestClassifier(**params)
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred)

            scores.append(score)

        return np.mean(scores)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self._check_fitted()

        self.best_model_.fit(X, y)

        return self.best_model_

    def tune_and_fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.scale_weight_ = (y==0).sum() / (y==1).sum()

        study = optuna.create_study(study_name='XGB_study', direction='maximize')
        study.optimize(lambda trial: self._objective(trial, X, y), n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params_ = study.best_params
        
        print('Best model params:', self.best_params_)

        final_params = self._get_model_params(optuna.trial.FixedTrial(self.best_params_))

        if self.model_type == 'xgb':
            self.best_model_ = XGBClassifier(**final_params)
        elif self.model_type == 'rf':
            self.best_model_ = RandomForestClassifier(**final_params)

        self.best_model_.fit(X, y)

        return self.best_model_

    def save(self, filename: str = 'best_model.pkl'):
        self._check_fitted()

        if not filename.endswith('.pkl'):
            filename += '.pkl'

        path = os.path.join('..','models', filename)
        joblib.dump(self.best_model_, path)

        print(f'Saved the best {self.model_type.upper()} model to {filename}')

    def load(self, filename: str = 'best_model.pkl'):
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        path = os.path.join('..','models', filename)

        if not os.path.exists(path):
            raise ValueError(f'Could not find the file: {filename}!')
        
        self.best_model_ = joblib.load(path)
        
    def predict(self, X: pd.DataFrame):
        self._check_fitted()
        return self.best_model_.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        self._check_fitted()
        return self.best_model_.predict_proba(X)