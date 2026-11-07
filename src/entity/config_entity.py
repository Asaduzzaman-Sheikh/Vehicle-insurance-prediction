"""  
    This config entity is used to define the structure of the configuration for the training pipeline.
    It contains the settings and parameters needed to configure the training process.

"""


import os
from dataclasses import dataclass, field
from datetime import datetime
from src.constants import * 
from src.utils.main_utils import read_yaml_file

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Configuration entity for the project
@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config:  TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR_NAME, FILE_NAME)
    train_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR_NAME, TRAIN_FILE_NAME)
    test_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR_NAME, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME
    
@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    validation_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)
    
@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DIR_NAME, DATA_TRANSFORMATION_TRANSFORMED_TRAIN_FILE_NAME)
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DIR_NAME, DATA_TRANSFORMATION_TRANSFORMED_TEST_FILE_NAME)
    transformed_object_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR_NAME, PREPROCESSING_OBJECT_FILE_NAME)
    
@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR_NAME, MODEL_FILE_NAME)
    
    model_config: dict = field(default_factory=lambda: read_yaml_file(MODEL_FILE_PATH))
    
    # The field init=False indicates that these attributes are not expected to be provided during initialization and will be set later in the code
    expected_score: float = field(init=False)
    model_name: str = field(init=False)
    model_params: dict = field(init=False)
    
    # The __post_init__ method is a special method in dataclasses that is automatically called after the instance is created. It allows you to perform additional initialization or validation tasks.
    def __post_init__(self):
        model_selection = self.model_config.get("model_selection", {})
        self.expected_score = model_selection.get("expected_score", 0.6)
        self.model_name = model_selection.get("model_name", "RandomForestClassifier")
        self.model_params = model_selection.get("params", {})