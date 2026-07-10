import os
import sys

# Add the project root to sys.path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model = load_object(os.path.join("artifacts", "model.pkl"))
        self.scaler_path = os.path.join("artifacts", "scaler.pkl")
        self.scaler = load_object(self.scaler_path) if os.path.exists(self.scaler_path) else None
        self.threshold = self._load_threshold()

    def _load_threshold(self):
        threshold_path = os.path.join("artifacts", "threshold.npy")

        if os.path.exists(threshold_path):
            return float(np.load(threshold_path))

        return 0.8

    def _prepare_features(self, features):
        expected_features = getattr(self.model, "feature_names_in_", None)
        if expected_features is not None:
            expected_features = list(expected_features)

        if self.scaler is None:
            return features.values if hasattr(features, "values") else features

        if isinstance(self.scaler, ColumnTransformer):
            transformed_data = self.scaler.transform(features)
            if hasattr(transformed_data, "toarray"):
                transformed_data = transformed_data.toarray()
            return transformed_data

        categorical_columns = ['Faculty', 'Financial_Support_Source', 'Admission_Year']
        # keep only the categorical columns that are actually present in the incoming dataframe
        present_categorical = [c for c in categorical_columns if c in features.columns]
        if present_categorical:
            features = pd.get_dummies(features, columns=present_categorical, drop_first=True)

        if expected_features is not None:
            for col in expected_features:
                if col not in features.columns:
                    features[col] = 0
            features = features[expected_features]

        transformed_data = self.scaler.transform(features)
        if hasattr(transformed_data, "toarray"):
            transformed_data = transformed_data.toarray()

        return transformed_data

    def predict(self, features):
        try:
            data = self._prepare_features(features)
            if hasattr(self.model, "predict_proba"):
                y_prob = self.model.predict_proba(data)[:, 1]
                return (y_prob >= self.threshold).astype(int)

            return self.model.predict(data)
        except Exception as e:
            raise CustomException(e, sys)

    def predict_proba(self, features):
        try:
            data = self._prepare_features(features)

            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(data)[:, 1]

            return self.model.predict(data)
        except Exception as e:
            raise CustomException(e, sys)

    def get_threshold(self):
        return self.threshold

    def set_threshold(self, new_threshold):
        self.threshold = float(new_threshold)

    def get_top_features(self, features, top_n=5):
        try:
            data = self._prepare_features(features)

            if not hasattr(self.model, "feature_importances_"):
                return []

            importances = self.model.feature_importances_

            if hasattr(data, "columns"):
                feature_names = data.columns
            elif self.scaler is not None and hasattr(self.scaler, "get_feature_names_out"):
                feature_names = self.scaler.get_feature_names_out()
            else:
                feature_names = [f"feature_{index}" for index in range(len(importances))]

            indices = np.argsort(importances)[::-1][:top_n]

            return [(feature_names[i], round(importances[i], 4)) for i in indices]
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_dataframe(self):
        try:
            return pd.DataFrame([self.data])
        except Exception as e:
            raise CustomException(e, sys)
