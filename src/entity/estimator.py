import sys
import pandas as pd
from sklearn.pipeline import Pipeline

from src.exceptions import CustomException
from src.logger import setup_logger

logger = setup_logger(__name__)


class TargetValueMapping:
    """
    Helper class to map the target variable between text (Yes/No) 
    and numbers (1/0) for the machine learning model.
    """
    def __init__(self):
        try:
            self.yes: int = 1
            self.no: int = 0
        except Exception as e:
            logger.error(f"Error initializing TargetValueMapping: {e}")
            raise CustomException(e, sys) from e

    def to_dict(self) -> dict:
        try:
            return self.__dict__
        except Exception as e:
            logger.error(f"Error converting TargetValueMapping to dict: {e}")
            raise CustomException(e, sys) from e

    def reverse_mapping(self) -> dict:
        try:
            return {v: k for k, v in self.to_dict().items()}
        except Exception as e:
            logger.error(f"Error reversing TargetValueMapping: {e}")
            raise CustomException(e, sys) from e


class MyModel:
    """
    A wrapper class that bundles the preprocessing pipeline and the trained 
    machine learning model into a single object. This ensures that when a user 
    passes raw data, it is automatically preprocessed before prediction.
    """
    def __init__(self, preprocessing_object, trained_model_object):
        """
        :param preprocessing_object: The fitted ColumnTransformer/Pipeline
        :param trained_model_object: The fitted ML model (e.g., RandomForestClassifier)
        """
        try:
            self.preprocessing_object = preprocessing_object
            self.trained_model_object = trained_model_object
        except Exception as e:
            logger.error(f"Error initializing MyModel: {e}")
            raise CustomException(e, sys) from e
    def predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the target variable for the given raw input dataframe.
        
        :param dataframe: Raw input data as a pandas DataFrame
        :return: Predictions as a pandas DataFrame
        """
        try:
            # Preprocess the input data
            preprocessed_data = self.preprocessing_object.transform(dataframe)
            
            # Predict using the trained model
            predictions = self.trained_model_object.predict(preprocessed_data)
            return pd.DataFrame(predictions, columns=['Predictions'])
        except Exception as e:
            logger.error(f"Error during prediction in MyModel: {e}")
            raise CustomException(e, sys) from e
        
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"
        
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"