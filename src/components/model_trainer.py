import os
import sys

# Add the project root to sys.path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, save_object
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier



@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    scaler_file_path: str = os.path.join("artifacts", "scaler.pkl")
    threshold_file_path: str = os.path.join("artifacts", "threshold.npy")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    target_column_name: str = "Dropout_Risk"
    default_threshold: float = 0.4


class ModelTrainer:
    def __init__(self, force_retrain: bool = False):
        self.model_trainer_config = ModelTrainerConfig()
        self.force_retrain = force_retrain

    def _load_existing_model(self):
        model_path = self.model_trainer_config.trained_model_file_path
        if os.path.exists(model_path):
            logging.info(f"Existing trained model found at {model_path}. Loading saved model instead of retraining.")
            return load_object(model_path)
        return None

    def _load_saved_threshold(self):
        threshold_path = self.model_trainer_config.threshold_file_path
        if os.path.exists(threshold_path):
            return float(np.load(threshold_path))
        return self.model_trainer_config.default_threshold

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Starting Model Training")
            
            # Split features and target from arrays
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]
            
            existing_model = None
            if not self.force_retrain:
                existing_model = self._load_existing_model()

            if existing_model is not None:
                logging.info("Skipping retraining; evaluating saved model on current test set.")
                model = existing_model
                threshold = self._load_saved_threshold()

                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)[:, 1]
                    test_predictions = (y_score >= threshold).astype(int)
                else:
                    test_predictions = model.predict(X_test)
                    y_score = test_predictions

                test_accuracy = accuracy_score(y_test, test_predictions)
                test_f1 = f1_score(y_test, test_predictions, zero_division=0)
                test_recall = recall_score(y_test, test_predictions, zero_division=0)
                test_precision = precision_score(y_test, test_predictions, zero_division=0)
                test_roc_auc = roc_auc_score(y_test, y_score)
                cm = confusion_matrix(y_test, test_predictions)

                results = {
                    "test_accuracy": test_accuracy,
                    "test_f1_score": test_f1,
                    "test_recall": test_recall,
                    "test_precision": test_precision,
                    "test_roc_auc": test_roc_auc,
                    "confusion_matrix": cm.tolist(),
                    "threshold": threshold,
                    "model_loaded_from": self.model_trainer_config.trained_model_file_path,
                    "retrained": False
                }

                logging.info(f"Existing model evaluation completed - Results: {results}")
                return results

            # Apply SMOTE to the encoded training features only, after the train/test split.
            logging.info("Applying SMOTE for class imbalance handling")
            try:
                smote = SMOTE(random_state=42, k_neighbors=5)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                logging.info(f"Training data shape after SMOTE: {X_train_resampled.shape}")
            except Exception as e:
                logging.warning(f"SMOTE failed: {e}. Proceeding without resampling.")
                X_train_resampled, y_train_resampled = X_train, y_train
            
            # Initialize and train Random Forest model
            model = RandomForestClassifier(
                n_estimators=200,
                min_samples_split=2,
                min_samples_leaf=2,
                max_features='sqrt',
                max_depth=10,
                random_state=42
            )
            
            logging.info("Training Random Forest model")
            model.fit(X_train_resampled, y_train_resampled)
            
            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")
            
            # Use threshold of 0.2 as per notebook optimization
            threshold = 0.2
            logging.info(f"Using optimized threshold: {threshold}")
            
            # Save the threshold
            np.save(self.model_trainer_config.threshold_file_path, threshold)
            logging.info(f"Threshold saved at {self.model_trainer_config.threshold_file_path}")
            
            # Evaluate on test data
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
                test_predictions = (y_score >= threshold).astype(int)
            else:
                test_predictions = model.predict(X_test)
                y_score = test_predictions
            
            test_accuracy = accuracy_score(y_test, test_predictions)
            test_f1 = f1_score(y_test, test_predictions, zero_division=0)
            test_recall = recall_score(y_test, test_predictions, zero_division=0)
            test_precision = precision_score(y_test, test_predictions, zero_division=0)
            test_roc_auc = roc_auc_score(y_test, y_score)
            
            cm = confusion_matrix(y_test, test_predictions)
            
            results = {
                "test_accuracy": test_accuracy,
                "test_f1_score": test_f1,
                "test_recall": test_recall,
                "test_precision": test_precision,
                "test_roc_auc": test_roc_auc,
                "confusion_matrix": cm.tolist(),
                "threshold": threshold,
                "retrained": True
            }
            
            logging.info(f"Model Training Completed - Results: {results}")
            
            return results
            
        except Exception as e:
            raise CustomException(e, sys)

    def _prepare_features(self, model, input_features):
        expected_features = getattr(model, "feature_names_in_", None)
        if expected_features is not None:
            expected_features = list(expected_features)

        scaler_path = self.model_trainer_config.scaler_file_path
        if not os.path.exists(scaler_path):
            logging.info("No scaler.pkl found. Using raw test features.")
            return input_features.values if hasattr(input_features, "values") else input_features

        scaler = load_object(scaler_path)

        if isinstance(scaler, ColumnTransformer):
            transformed_features = scaler.transform(input_features)
            if hasattr(transformed_features, "toarray"):
                transformed_features = transformed_features.toarray()
            return transformed_features

        categorical_columns = ['Faculty', 'Financial_Support_Source']
        present_categorical = [c for c in categorical_columns if c in input_features.columns]
        if present_categorical:
            input_features = pd.get_dummies(input_features, columns=present_categorical, drop_first=True)

        if expected_features is not None:
            for col in expected_features:
                if col not in input_features.columns:
                    input_features[col] = 0
            input_features = input_features[expected_features]

        transformed_features = scaler.transform(input_features)
        if hasattr(transformed_features, "toarray"):
            transformed_features = transformed_features.toarray()

        return transformed_features

    def _get_threshold(self):
        threshold_path = self.model_trainer_config.threshold_file_path

        if os.path.exists(threshold_path):
            threshold = float(np.load(threshold_path))
            logging.info(f"Loaded saved threshold: {threshold:.4f}")
            return threshold

        threshold = self.model_trainer_config.default_threshold
        logging.info(f"No threshold artifact found. Using default threshold: {threshold:.4f}")
        return threshold

    def _threshold_report(self, y_test, y_score):
        report = {}

        for threshold in np.arange(0.10, 0.91, 0.05):
            y_pred = (y_score >= threshold).astype(int)
            report[round(float(threshold), 2)] = {
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_score),
                "predicted_positive": int(np.sum(y_pred)),
            }

        return report

    def evaluate_model(self, test_path=None):
        try:
            logging.info("Loading saved model")
            model = load_object(self.model_trainer_config.trained_model_file_path)

            test_path = test_path or self.model_trainer_config.test_data_path
            test_df = pd.read_csv(test_path)

            target_column = self.model_trainer_config.target_column_name
            if target_column not in test_df.columns:
                raise ValueError(f"Target column '{target_column}' was not found in {test_path}")

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            X_test = self._prepare_features(model, X_test)

            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
                threshold = self._get_threshold()
                y_pred = (y_score >= threshold).astype(int)
            else:
                threshold = None
                y_pred = model.predict(X_test)
                y_score = y_pred

            labels, counts = np.unique(y_pred, return_counts=True)
            prediction_counts = {
                int(label): int(count) for label, count in zip(labels, counts)
            }

            cm = confusion_matrix(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_score)

            results = {
                "accuracy": accuracy_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc,
                "threshold": threshold,
                "prediction_counts": prediction_counts,
                "confusion_matrix": cm.tolist(),
            }

            if hasattr(model, "classes_"):
                results["model_classes"] = model.classes_.tolist()

            if hasattr(model, "predict_proba"):
                results["probability_summary"] = {
                    "min": float(np.min(y_score)),
                    "mean": float(np.mean(y_score)),
                    "max": float(np.max(y_score)),
                }
                results["threshold_report"] = self._threshold_report(y_test, y_score)

            logging.info(f"Saved Model Evaluation - {results}")

            return results

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.evaluate_model()

    print("Evaluation Completed")
    print(results)
