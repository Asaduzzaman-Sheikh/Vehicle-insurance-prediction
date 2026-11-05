# This component is responsible for ingesting data into the training pipeline.
import os 
import sys
import pandas as pd

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.project_data import ProjectDataAccess
from sklearn.model_selection import train_test_split
from src.exceptions import CustomException
from src.logger import setup_logger

logger = setup_logger(__name__)

# Data Ingestion Component
"""
    This component is responsible for ingesting data into the training pipeline.
    It fetches data from the specified collection in the MongoDB database, splits it into training and testing datasets,
    and saves them to the designated file paths.

"""
# Create a class for data ingestion
class DataIngestion:
    
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        Initializes the DataIngestion instance with the provided configuration.

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration for data ingestion.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            logger.info(f"DataIngestion initialized with config: {self.data_ingestion_config}")
        except Exception as e:
            logger.error(f"Error initializing DataIngestion: {e}")
            raise CustomException(e, sys) from e
    
    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Fetches data from the specified collection in the MongoDB database and saves it to the feature store.

        Returns:
            pd.DataFrame: The DataFrame containing the exported data.
        """
        try:
            project_data = ProjectDataAccess()
            df = project_data.export_collection_as_dataframe(collection_name = self.data_ingestion_config.collection_name)
            logger.info(f"Exported data from collection '{self.data_ingestion_config.collection_name}' into DataFrame with shape {df.shape}.")
            # Save the DataFrame to the feature store file path
            os.makedirs(os.path.dirname(self.data_ingestion_config.feature_store_file_path), exist_ok = True)
            logger.info(f"Saving DataFrame to feature store at '{self.data_ingestion_config.feature_store_file_path}'.")
            df.to_csv(self.data_ingestion_config.feature_store_file_path, index = False, header = True)
            logger.info(f"DataFrame saved to feature store successfully.")
            
            return df
        except Exception as e:
            logger.error(f"Error exporting data into feature store: {e}")
            raise CustomException(e, sys) from e
        
    # Define a method to split the data into training and testiong datasets
    def split_data_as_train_test(self, df: pd.DataFrame) -> None:
        """
        Splits the provided DataFrame into training and testing datasets and saves them to the specified file paths.

        Args:
            df (pd.DataFrame): The DataFrame to be split.
        """
        try:
            train_set, test_set = train_test_split(df, test_size = self.data_ingestion_config.train_test_split_ratio, random_state = 42)
            logger.info(f"Split data into training set with shape {train_set.shape} and testing set with shape {test_set.shape}.")
            
            # Save the training and testing datasets tp their respective file paths
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_file_path), exist_ok = True)
            train_set.to_csv(self.data_ingestion_config.train_file_path, index = False, header = True)
            logger.info(f"Training dataset saved successfully.")

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_file_path), exist_ok = True)
            test_set.to_csv(self.data_ingestion_config.test_file_path, index = False, header = True)
            logger.info(f"Testing dataset saved successfully.")
    
        except Exception as e:
            logger.error(f"Error splitting data into training and testing datasets: {e}")
            raise CustomException(e, sys) from e
        
    # Define a method to initiate the data ingestion process
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process by exporting data into the feature store and splitting it into training and testing datasets.

        Returns:
            DataIngestionArtifact: An artifact containing the paths to the training and testing datasets.
        """
        try:
            df = self.export_data_into_feature_store()
            self.split_data_as_train_test(df = df)
            logger.info(f"Data ingestion process completed successfully.")
            
            # Create and return a DataIngestionArtifact with the paths to the training and testing datasets
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path = self.data_ingestion_config.train_file_path,
                test_file_path = self.data_ingestion_config.test_file_path
            )
            logger.info(f"DataIngestionArtifact created with train_file_path: '{data_ingestion_artifact.train_file_path}' and test_file_path: '{data_ingestion_artifact.test_file_path}'.")
            logger.info(f"Data ingestion artifact: {data_ingestion_artifact}")  
            return data_ingestion_artifact
        except Exception as e:
            logger.error(f"Error initiating data ingestion: {e}")
            raise CustomException(e, sys) from e

    