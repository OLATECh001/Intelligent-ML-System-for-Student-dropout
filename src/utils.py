import os
import sys
import joblib

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save object using joblib (better for sklearn objects)
        joblib.dump(obj, file_path)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        # Load object using joblib (better for sklearn objects)
        return joblib.load(file_path)

    except Exception as e:
        raise CustomException(e, sys)