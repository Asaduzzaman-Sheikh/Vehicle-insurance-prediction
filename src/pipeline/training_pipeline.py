""" 
This module contains the training pipeline for the machine learning model.
It includes functions for data preprocessing, model training, evaluation, and saving the trained model. 
The pipeline is designed to be modular and easily configurable for different datasets and model architectures.
"""

import sys
import os

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import DataValidationConfig
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.artifact_entity import DataValidationArtifact
from src.entity.artifact_entity import DataTransformationArtifact
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact
from src.components.model_evaluation import ModelEvaluation
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelEvaluationArtifact
from src.components.model_pusher import ModelPusher
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelPusherArtifact


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

            self.data_validation_config = DataValidationConfig()
            logger.info(f"TrainingPipeline initialized with DataValidationConfig: {self.data_validation_config}")

            self.data_transformation_config = DataTransformationConfig()
            logger.info(f"TrainingPipeline initialized with DataTransformationConfig: {self.data_transformation_config}")

            self.model_trainer_config = ModelTrainerConfig()
            logger.info(f"TrainingPipeline initialized with ModelTrainerConfig: {self.model_trainer_config}")

            self.model_evaluation_config = ModelEvaluationConfig()
            logger.info(f"TrainingPipeline initialized with ModelEvaluationConfig: {self.model_evaluation_config}")

            self.model_pusher_config = ModelPusherConfig()
            logger.info(f"TrainingPipeline initialized with ModelPusherConfig: {self.model_pusher_config}")

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
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info(f"Data ingestion completed successfully. Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Starts the data validation process.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): The artifact containing paths to the ingested data files.

        Returns:
            DataValidationArtifact: The artifact containing the results of the data validation process.
        """
        try:
            logger.info("Starting data validation process")
            data_validation = DataValidation(
                data_validation_config=self.data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact,
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logger.info(f"Data validation completed. Artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            logger.error(f"Error during data validation: {e}")
            raise CustomException(e, sys) from e

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        """
        Starts the data transformation process.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): The artifact containing paths to the ingested data files.
            data_validation_artifact (DataValidationArtifact): The artifact containing the results of data validation.

        Returns:
            DataTransformationArtifact: The artifact containing the paths to the transformed data files.
        """
        try:
            logger.info("Starting data transformation process")
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=self.data_transformation_config,
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logger.info(f"Data transformation completed. Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            logger.error(f"Error during data transformation: {e}")
            raise CustomException(e, sys) from e

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        Starts the model training process.

        Args:
            data_transformation_artifact (DataTransformationArtifact): The artifact containing the paths to the transformed data files.

        Returns:
            ModelTrainerArtifact: The artifact containing the path to the trained model and evaluation metrics.
        """
        try:
            logger.info("Starting model training process...")
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifact=data_transformation_artifact,
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logger.info(f"Model training completed. Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException(e, sys) from e

    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact,   # NEW
    ) -> ModelEvaluationArtifact:
        """
        Starts the model evaluation process.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): The artifact containing paths to the ingested data files.
            model_trainer_artifact (ModelTrainerArtifact): The artifact containing the path to the trained model and evaluation metrics.
            data_transformation_artifact (DataTransformationArtifact): The artifact containing the path to the saved preprocessing object.

        Returns:
            ModelEvaluationArtifact: The artifact containing the results of the model evaluation process.
        """
        try:
            # NEW: propagate the preprocessing pickle path into the eval config so
            # ModelEvaluation can reindex test features to the training schema.
            self.model_evaluation_config.preprocessing_object_file_path = (
                data_transformation_artifact.transformed_object_file_path
            )

            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logger.info(f"Model evaluation completed. Artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException(e, sys) from e

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        Starts the model pushing process to S3.

        Args:
            model_evaluation_artifact (ModelEvaluationArtifact): The artifact containing the results of the model evaluation process.

        Returns:
            ModelPusherArtifact: The artifact containing the S3 bucket name and path where the model is pushed.
        """
        try:
            logger.info("Starting model pushing process...")
            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
                model_evaluation_artifact=model_evaluation_artifact,
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logger.info(f"Model pushing completed. Artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            logger.error(f"Error during model pushing: {e}")
            raise CustomException(e, sys) from e

    # Define a method to start the training pipeline
    def start_training_pipeline(self):
        """
        Starts the training pipeline, which includes data ingestion, validation,
        transformation, model training, evaluation, and pushing.
        """
        try:
            logger.info("Starting the training pipeline.")
            data_ingestion_artifact = self.start_data_ingestion()

            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            if not data_validation_artifact.validation_status:
                logger.error(f"Data validation failed: {data_validation_artifact.error_message}")
                raise Exception(
                    f"Data validation failed for the following reasons:\n"
                    f"{data_validation_artifact.error_message}"
                )
            logger.info("Data validation passed. Proceeding to the next stage.")

            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact, data_validation_artifact
            )
            logger.info(f"Data transformation completed. Artifact: {data_transformation_artifact}")

            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            logger.info(f"Model training completed. Artifact: {model_trainer_artifact}")

            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact,
                model_trainer_artifact,
                data_transformation_artifact,   
            )
            logger.info(f"Model evaluation completed. Artifact: {model_evaluation_artifact}")

            if not model_evaluation_artifact.is_model_accepted:
                logger.error("The model is not accepted for deployment.")
                raise Exception("The model is not accepted for deployment.")

            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact)
            logger.info(f"Model pushing completed. Artifact: {model_pusher_artifact}")

            logger.info("Training pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise CustomException(e, sys) from e