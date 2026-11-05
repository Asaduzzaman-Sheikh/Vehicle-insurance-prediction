"""  
    This config entity is used to define the structure of the configuration for the training pipeline.
    It contains the settings and parameters needed to configure the training process.

"""


import os
from dataclasses import dataclass
from datetime import datetime
from src.constants import * 


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
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME, TIMESTAMP)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR_NAME, FILE_NAME)
    train_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR_NAME, TRAIN_FILE_NAME)
    test_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR_NAME, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME