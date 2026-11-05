import os
from datetime import datetime


DB_NAME = "vehicle_insurance"
COLLECTION_NAME = "insurance_data"
CONNECTION_STRING = "CONNECTION_STRING"

PIPELINE_NAME = ""
ARTIFACT_DIR = 'artifacts'

MODEL_FILE_NAME = 'model.pkl'

TARGET_COLUMN = 'Response'
CURRENT_YEAR = datetime.now().year
PREPROCCESING_OBJECT_FILE_NAME = 'preprocessing.pkl'

FILE_NAME = 'data.csv'
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'
SCHEMA_FILE_PATH = os.path.join('config', 'schema.yaml')

AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "ap-south-1"

# Data Ingestion Related Constants
DATA_INGESTION_COLLECTION_NAME = "insurance_data"
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR_NAME = "feature_store"
DATA_INGESTION_INGESTED_DIR_NAME = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.25