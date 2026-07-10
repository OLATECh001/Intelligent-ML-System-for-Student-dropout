import sys
import os

# Add the project root to sys.path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    try:
        logging.info("Training Pipeline Started")

        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_path, test_path
        )

        # Step 3: Model Training
        # Force retrain to ensure model matches current feature set (Admission_Year removed)
        trainer = ModelTrainer(force_retrain=True)
        results = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"Training Results: {results}")

        print("\n Training pipeline completed successfully!")
        print("Results:", results)

    except Exception as e:
        raise CustomException(e, sys)