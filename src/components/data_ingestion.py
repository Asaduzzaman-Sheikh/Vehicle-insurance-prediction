# This component is responsible for ingesting data into the training pipeline.
import os 
import sys
import pandas as pd

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.project_data import ProjectDataAccess
from src.utils.main_utils import read_yaml_file
from src.constants import SCHEMA_FILE_PATH
from sklearn.model_selection import train_test_split
from src.exceptions import CustomException
from src.logger import setup_logger

logger = setup_logger(__name__)

# Data Ingestion Component
"""
    This component is responsible for ingesting data into the training pipeline.
    It fetches data from the specified collection in the MongoDB database, drops the
    columns listed under `drop_columns` in schema.yaml, splits the result into training
    and testing datasets, and saves them to the designated file paths.
"""
class DataIngestion:
    
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initializes the DataIngestion instance with the provided configuration.

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration for data ingestion.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logger.info(f"DataIngestion initialized with config: {self.data_ingestion_config}")
        except Exception as e:
            logger.error(f"Error initializing DataIngestion: {e}")
            raise CustomException(e, sys) from e

    def _get_drop_columns(self) -> list:
        """
        Returns the list of columns to drop, read from `drop_columns` in schema.yaml.
        Supports both a scalar (e.g. `drop_columns: _id`) and a list
        (e.g. `drop_columns: [_id, id]`).
        """
        drop_columns = self.schema_config.get("drop_columns", [])
        if drop_columns is None:
            return []
        if isinstance(drop_columns, str):
            return [drop_columns]
        if isinstance(drop_columns, (list, tuple, set)):
            return list(drop_columns)
        raise ValueError(
            f"Unsupported type for 'drop_columns' in schema.yaml: "
            f"{type(drop_columns).__name__}"
        )

    def _drop_configured_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the columns listed in `drop_columns` from the DataFrame, ignoring
        any that are not present (so a stale config never crashes ingestion).

        Args:
            df (pd.DataFrame): The input DataFrame.
        Returns:
            pd.DataFrame: The DataFrame with the configured columns dropped.
        """
        try:
            drop_columns = self._get_drop_columns()
            if not drop_columns:
                logger.info("No columns configured for dropping in schema.yaml.")
                return df

            present = [c for c in drop_columns if c in df.columns]
            missing = [c for c in drop_columns if c not in df.columns]

            if missing:
                logger.warning(
                    f"Configured drop columns not found in DataFrame, skipping: {missing}"
                )

            if present:
                df = df.drop(columns=present)
                logger.info(
                    f"Dropped columns {present} from DataFrame. "
                    f"New shape: {df.shape}"
                )
            else:
                logger.info(
                    "None of the configured drop columns were present; "
                    "DataFrame left unchanged."
                )
            return df
        except Exception as e:
            logger.error(f"Error dropping configured columns: {e}")
            raise CustomException(e, sys) from e

    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Fetches data from the specified collection in the MongoDB database, drops
        the configured columns, and saves the result to the feature store.

        Returns:
            pd.DataFrame: The DataFrame containing the exported (and cleaned) data.
        """
        try:
            project_data = ProjectDataAccess()
            df = project_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            logger.info(
                f"Exported data from collection "
                f"'{self.data_ingestion_config.collection_name}' into DataFrame "
                f"with shape {df.shape}."
            )

            # Drop configured columns (e.g. _id, id) before persisting.
            df = self._drop_configured_columns(df)

            # Save the DataFrame to the feature store file path
            os.makedirs(
                os.path.dirname(self.data_ingestion_config.feature_store_file_path),
                exist_ok=True,
            )
            logger.info(
                f"Saving DataFrame to feature store at "
                f"'{self.data_ingestion_config.feature_store_file_path}'."
            )
            df.to_csv(
                self.data_ingestion_config.feature_store_file_path,
                index=False,
                header=True,
            )
            logger.info("DataFrame saved to feature store successfully.")

            return df
        except Exception as e:
            logger.error(f"Error exporting data into feature store: {e}")
            raise CustomException(e, sys) from e
        
    def split_data_as_train_test(self, df: pd.DataFrame) -> None:
        """
        Splits the provided DataFrame into training and testing datasets and saves
        them to the specified file paths.

        Args:
            df (pd.DataFrame): The DataFrame to be split.
        """
        try:
            train_set, test_set = train_test_split(
                df,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
            )
            logger.info(
                f"Split data into training set with shape {train_set.shape} "
                f"and testing set with shape {test_set.shape}."
            )
            
            # Save the training and testing datasets to their respective file paths
            os.makedirs(
                os.path.dirname(self.data_ingestion_config.train_file_path),
                exist_ok=True,
            )
            train_set.to_csv(
                self.data_ingestion_config.train_file_path,
                index=False,
                header=True,
            )
            logger.info("Training dataset saved successfully.")

            os.makedirs(
                os.path.dirname(self.data_ingestion_config.test_file_path),
                exist_ok=True,
            )
            test_set.to_csv(
                self.data_ingestion_config.test_file_path,
                index=False,
                header=True,
            )
            logger.info("Testing dataset saved successfully.")
    
        except Exception as e:
            logger.error(f"Error splitting data into training and testing datasets: {e}")
            raise CustomException(e, sys) from e
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process by exporting data into the feature
        store (with configured columns dropped) and splitting it into training and
        testing datasets.

        Returns:
            DataIngestionArtifact: An artifact containing the paths to the training
            and testing datasets.
        """
        try:
            df = self.export_data_into_feature_store()
            self.split_data_as_train_test(df=df)
            logger.info("Data ingestion process completed successfully.")
            
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path,
            )
            logger.info(
                f"DataIngestionArtifact created with train_file_path: "
                f"'{data_ingestion_artifact.train_file_path}' and "
                f"test_file_path: '{data_ingestion_artifact.test_file_path}'."
            )
            logger.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            logger.error(f"Error initiating data ingestion: {e}")
            raise CustomException(e, sys) from e