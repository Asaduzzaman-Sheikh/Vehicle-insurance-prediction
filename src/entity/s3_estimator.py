from src.cloud_storage.aws_storage import SimpleStorageService
from src.exceptions import CustomException
from src.logger import setup_logger

import sys
import pandas as pd
from src.entity.estimator import MyModel

logger = setup_logger(__name__)


# Define a class to handle S3 interactions and model loading
class S3Estimator:

    def __init__(self, bucket_name: str, model_path: str):
        """
        Initializes the S3Estimator with the specified S3 bucket and model path.

        :param bucket_name: Name of the S3 bucket where the model is stored
        :param model_path: Path to the model file in the S3 bucket
        """
        try:
            self.bucket_name = bucket_name
            self.s3 = SimpleStorageService()
            self.model_path = model_path
            self.loaded_model: MyModel = None
        except Exception as e:
            logger.error(f"Error initializing S3Estimator: {e}")
            raise CustomException(e, sys) from e

    def is_model_present(self, model_path: str) -> bool:
        """
        Checks if the model file exists in the specified S3 bucket.

        :param model_path: Path to the model file in the S3 bucket
        :return: True if the model exists, False otherwise
        """
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except Exception as e:
            logger.error(f"Error checking model presence in S3: {e}")
            raise CustomException(e, sys) from e

    def load_model(self) -> MyModel:
        """
        Loads the model from S3 if it exists.

        :return: An instance of MyModel containing the preprocessing and trained model
        """
        try:
            return self.s3.load_model(self.model_path, self.bucket_name)
        except Exception as e:
            logger.error(f"Error loading model from S3: {e}")
            raise CustomException(e, sys) from e

    def save_model(self, from_file, remove: bool = False) -> None:
        """
        Saves the model to S3.

        :param from_file: Path to the local model file to be uploaded
        :param remove: If True, removes the local file after upload
        """
        try:
            self.s3.upload_file(
                from_filename=from_file,       
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove,
            )
        except Exception as e:
            logger.error(f"Error saving model to S3: {e}")
            raise CustomException(e, sys) from e

    def predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the target variable for the given raw input dataframe using the loaded model.

        :param dataframe: Raw input data as a pandas DataFrame
        :return: Predictions as a pandas DataFrame
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe)
        except Exception as e:
            logger.error(f"Error during prediction in S3Estimator: {e}")
            raise CustomException(e, sys) from e