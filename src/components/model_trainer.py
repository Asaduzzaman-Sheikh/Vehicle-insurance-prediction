import sys
import os
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact, ClassificationMetricArtifact, DataTransformationArtifact
from src.utils.main_utils import save_object, load_object, load_numpy_array
from src.entity.estimator import MyModel
    
from src.exceptions import CustomException
from src.logger import setup_logger

logger = setup_logger(__name__)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            logger.info(f"ModelTrainer initialized with config: {self.model_trainer_config} and data transformation artifact: {self.data_transformation_artifact}")
        except Exception as e:
            logger.error(f"Error initializing ModelTrainer: {e}")
            raise CustomException(e, sys) from e
    
    def train_model(self, train_array: np.ndarray, test_array: np.ndarray) -> Tuple[object, ClassificationMetricArtifact]:
        try:
            logger.info("Starting model training...")
            
            # 1. Split features (X) and target (y)
            Xtrain, ytrain = train_array[:, :-1], train_array[:, -1]
            Xtest, ytest = test_array[:, :-1], test_array[:, -1]
            
            # 2. Initialize and train the model (using **kwargs from YAML)
            model = RandomForestClassifier(**self.model_trainer_config.model_params)
            model.fit(Xtrain, ytrain)
            
            # 3. Predict on the test set
            yhat = model.predict(Xtest)
            
            # 4. Evaluate
            accuracy = accuracy_score(ytest, yhat)
            f1 = f1_score(ytest, yhat)
            precision = precision_score(ytest, yhat)
            recall = recall_score(ytest, yhat)
            
            logger.info(f"Model training completed with Accuracy: {accuracy:.4f}, F1 score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            # 5. Package metrics into artifact
            metric_artifact = ClassificationMetricArtifact(
                accuracy=accuracy,
                f1_score=f1, 
                precision_score=precision, 
                recall_score=recall
            )
            
            return model, metric_artifact
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException(e, sys) from e
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logger.info("Initiating model trainer...")
            
            # Load transformed training and testing data
            train_array = load_numpy_array(self.data_transformation_artifact.transformed_train_file_path)
            test_array = load_numpy_array(self.data_transformation_artifact.transformed_test_file_path)
            
            # Train the model
            model, test_metric_artifact = self.train_model(train_array, test_array)
            
            # Load the preprocessing object
            preprocessing_obj = load_object(self.data_transformation_artifact.transformed_object_file_path)
            
            # Evaluate: Check if the model meets the expected score (Accuracy on training data)
            train_accuracy = accuracy_score(train_array[:, -1], model.predict(train_array[:, :-1]))
            
            if train_accuracy < self.model_trainer_config.expected_score:
                logger.error(f"Trained model does not meet the expected accuracy of {self.model_trainer_config.expected_score}.")
                raise CustomException("Trained model does not meet the expected accuracy.", sys)
            
            logger.info(f"Model meets the expected accuracy ({train_accuracy:.4f} >= {self.model_trainer_config.expected_score}).")
            
            # Save the trained model and preprocessing object
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logger.info(f"Trained model and preprocessing object saved at {self.model_trainer_config.trained_model_file_path}")
            
            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metrics_artifact=test_metric_artifact
            )
            
        except Exception as e:
            logger.error(f"Error initiating model trainer: {e}")
            raise CustomException(e, sys) from e