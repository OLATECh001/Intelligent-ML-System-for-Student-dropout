Pimport os
import sys

# Add the project root to sys.path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class DataTransformationConfig:
    scaler_obj_file_path = os.path.join("artifacts", "scaler.pkl")
    preprocessor_obj_file_path = scaler_obj_file_path


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function creates preprocessing pipeline
        """
        try:
            # =========================
            # NUMERICAL FEATURES
            # =========================
            numerical_columns = [
                "Gender",
                "Age_Range",
                "Level_of_Study",
                "Parental_Level_of_Education",
                "CGPA_Range",
                "Hours_of_Study_per_Week",
                "Class_Attendance",
                "Accommodation_Type",
                "Working_while_studying",
                "Internet_Access",
                "Participation_in_Social_Activities",
                "Scholarship_Status",
                "Academic_support",
                "Health_Challenges",
                "School_Activities_Stress"
            ]

            # =========================
            # CATEGORICAL FEATURES
            # =========================
            categorical_columns = [
                "Faculty",
                "Financial_Support_Source",
            ]

            # =========================
            # NUMERICAL PIPELINE
            # =========================
            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            # =========================
            # CATEGORICAL PIPELINE
            # =========================
            cat_pipeline = Pipeline(
                steps=[
                    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
                ]
            )

            logging.info("Numerical scaling completed")
            logging.info("Categorical encoding completed")

            # =========================
            # COLUMN TRANSFORMER
            # =========================
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ],
                sparse_threshold=0,
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Dropout_Risk"

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on training and test data")

            # =========================
            # FIT + TRANSFORM
            # =========================
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # =========================
            # COMBINE FEATURES + TARGET
            # =========================
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving scaler/preprocessing object")

            save_object(
                file_path=self.data_transformation_config.scaler_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Scaler/preprocessing object saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.scaler_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()

    train_arr, test_arr, preprocessor_path = obj.initiate_data_transformation(
        train_path="artifacts/train.csv",
        test_path="artifacts/test.csv"
    )

    print("Transformation Successful")
    print("Train shape:", train_arr.shape)
    print("Test shape:", test_arr.shape)
