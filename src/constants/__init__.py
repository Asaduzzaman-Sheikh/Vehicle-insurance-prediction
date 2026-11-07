import os
from datetime import datetime

# Database Constants
DB_NAME: str = "vehicle_insurance"
COLLECTION_NAME: str = "insurance_data"
CONNECTION_STRING: str = os.getenv("CONNECTION_STRING")

# General Project Constants
PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = 'artifacts'

MODEL_FILE_NAME: str = 'model.pkl'

TARGET_COLUMN: str = 'Response'
CURRENT_YEAR: int = datetime.now().year
PREPROCESSING_OBJECT_FILE_NAME: str = 'preprocessing.pkl'

FILE_NAME: str = 'data.csv'
TRAIN_FILE_NAME: str = 'train.csv'
TEST_FILE_NAME: str = 'test.csv'
SCHEMA_FILE_PATH: str = os.path.join('config', 'schema.yaml')
MODEL_FILE_PATH: str = os.path.join('config', 'model.yaml')

# AWS Constants
AWS_ACCESS_KEY_ID_ENV_KEY: str = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY: str = "AWS_SECRET_ACCESS_KEY"
REGION_NAME: str = "ap-south-1"

# Data Ingestion Related Constants
DATA_INGESTION_COLLECTION_NAME: str = "insurance_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR_NAME: str = "feature_store"
DATA_INGESTION_INGESTED_DIR_NAME: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25

# Data Validation Related Constants
DATA_VALIDATION_DIR_NAME: str = 'data_validation'
DATA_VALIDATION_REPORT_FILE_NAME: str = 'report.yaml'

# Data Transformation Related Constants
DATA_TRANSFORMATION_DIR_NAME: str = 'data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DIR_NAME: str = 'transformed'
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR_NAME: str = 'transformed_object'
DATA_TRANSFORMATION_TRANSFORMED_TRAIN_FILE_NAME: str = "train.npy"
DATA_TRANSFORMATION_TRANSFORMED_TEST_FILE_NAME: str = "test.npy"

# Model Training Related Constants
MODEL_TRAINER_DIR_NAME: str = 'model_trainer'
MODEL_TRAINER_TRAINED_MODEL_DIR_NAME: str = 'trained_model'
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME: str = 'model.pkl'