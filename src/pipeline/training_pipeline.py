""" 
This module contains the training pipeline for the machine learning model.
It includes functions for data preprocessing, model training, evaluation, and saving the trained model. 
The pipeline is designed to be modular and easily configurable for different datasets and model architectures.
"""

import sys
import os

from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exceptions import CustomException
from src.logger import setup_logger

logger = setup_logger(__name__)

# Create a class for the training pipeline
class TrainingPipeline:
    
    def __init__(self):
        """
        Initializes the TrainingPipeline instance.
        """
        try:
            self.data_ingestion_config = DataIngestionConfig()
            logger.info(f"TrainingPipeline initialized with DataIngestionConfig: {self.data_ingestion_config}")
        except Exception as e:
            logger.error(f"Error initializing TrainingPipeline: {e}")
            raise CustomException(e, sys) from e
        
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Starts the data ingestion process.

        Returns:
            DataIngestionArtifact: The artifact containing paths to the ingested data files.
        """
        try:
            logger.info("Starting data ingestion process.")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info(f"Data ingestion completed successfully. Artifact: {data_ingestion_artifact}")
            
            return data_ingestion_artifact
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys) from e
        
    # Define a method to start the training pipeline
    def start_training_pipeline(self):
        """
        Starts the training pipeline, which includes data ingestion, preprocessing, model training, and evaluation.
        """
        try:
            logger.info("Starting the training pipeline.")
            data_ingestion_artifact = self.start_data_ingestion()
            # Further steps for preprocessing, model training, and evaluation can be added here
            logger.info("Training pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise CustomException(e, sys) from e