import os
import sys
import json
import pandas as pd

from src.utils.main_utils import read_yaml_file, write_yaml_file 
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from src.constants import SCHEMA_FILE_PATH 

from src.exceptions import CustomException
from src.logger import setup_logger

logger = setup_logger(__name__)

class DataValidation:
    
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        """
        Initializes the DataValidation instance with the provided configuration and data ingestion artifact.

        Args:
            data_validation_config (DataValidationConfig): Configuration for data validation.
            data_ingestion_artifact (DataIngestionArtifact): Artifact containing paths to ingested data files.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logger.info(f"DataValidation initialized with config: {self.data_validation_config} and artifact: {self.data_ingestion_artifact}")
        except Exception as e:
            logger.error(f"Error initializing DataValidation: {e}")
            raise CustomException(e, sys) from e
    
    def _expected_columns(self) -> list:
        """
        Returns the canonical list of expected columns:
        numerical features + categorical features + target column.

        This is the single source of truth for both the count check and the
        existence check, so they can never disagree.
        """
        expected = list(self.schema_config['numerical_columns']) \
                + list(self.schema_config['categorical_columns'])
        target = self.schema_config.get('target_column')
        if target:
            expected.append(target)
        return expected

    def validate_number_of_columns(self, dataframe: pd.DataFrame, dataset_name: str = "dataset") -> bool:
        """
        Validates the number of columns in the provided DataFrame against the expected schema.

        Args:
            dataframe (pd.DataFrame): The DataFrame to validate.
            dataset_name (str): Name of the dataset (for logging), e.g. "Train" / "Test".
        Returns:
            bool: True if the number of columns is valid, False otherwise.
        """
        try:
            expected_columns = self._expected_columns()
            actual_columns = dataframe.columns.tolist()
            
            status = len(actual_columns) == len(expected_columns)
            logger.info(
                f"{dataset_name} dataset column count check: "
                f"Expected {len(expected_columns)}, Actual {len(actual_columns)}, Status: {status}"
            )
            if not status:
                logger.warning(
                    f"{dataset_name} dataset actual columns: {actual_columns}"
                )
            return status
        except Exception as e:
            logger.error(f"Error validating number of columns: {e}")
            raise CustomException(e, sys) from e
        
    def is_column_exist(self, dataframe: pd.DataFrame, dataset_name: str = "dataset") -> bool:
        """
        Checks if all expected columns (numerical, categorical, and target) exist
        in the provided DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame to check.
            dataset_name (str): Name of the dataset (for logging), e.g. "Train" / "Test".
        Returns:
            bool: True if all expected columns exist, False otherwise.
        """
        try:
            df_columns = dataframe.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            missing_target = []
            
            for column in self.schema_config['numerical_columns']:
                if column not in df_columns:
                    missing_numerical_columns.append(column)
                    
            for column in self.schema_config['categorical_columns']:
                if column not in df_columns:
                    missing_categorical_columns.append(column)

            target = self.schema_config.get('target_column')
            if target and target not in df_columns:
                missing_target.append(target)
                    
            if missing_numerical_columns:
                logger.warning(f"{dataset_name} dataset missing numerical columns: {missing_numerical_columns}")   
            if missing_categorical_columns:
                logger.warning(f"{dataset_name} dataset missing categorical columns: {missing_categorical_columns}")
            if missing_target:
                logger.warning(f"{dataset_name} dataset missing target column: {missing_target}")
        
            if missing_numerical_columns or missing_categorical_columns or missing_target:
                return False

            # Success path: all expected columns are present.
            return True
        except Exception as e:
            logger.error(f"Error checking if columns exist: {e}")
            raise CustomException(e, sys) from e
        
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads a CSV file and returns a pandas DataFrame.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: The DataFrame read from the CSV file.
        """
        try:
            dataframe = pd.read_csv(file_path)
            logger.info(f"CSV file read successfully from {file_path}")
            return dataframe
        except Exception as e:
            logger.error(f"Error reading CSV file from {file_path}: {e}")
            raise CustomException(e, sys) from e
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiates the data validation process.

        Returns:
            DataValidationArtifact: The artifact containing the results of the data validation.
        """
        try:
            validate_error_message = ""
            logger.info("Starting data validation process.")
            
            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)
            
            # --- Check 1: Number of columns ---------------------------------
            train_columns_status = self.validate_number_of_columns(train_df, dataset_name="Train")
            test_columns_status = self.validate_number_of_columns(test_df, dataset_name="Test")
            
            if not train_columns_status:
                validate_error_message += "Train dataset does not have the expected number of columns.\n"
            if not test_columns_status:
                validate_error_message += "Test dataset does not have the expected number of columns.\n"

            # --- Check 2: Required columns exist ----------------------------
            train_columns_exist_status = self.is_column_exist(train_df, dataset_name="Train")
            test_columns_exist_status = self.is_column_exist(test_df, dataset_name="Test")
            
            if not train_columns_exist_status:
                validate_error_message += "Train dataset is missing some required columns.\n"
            else:
                logger.info("Train dataset contains all the required columns.")
                
            if not test_columns_exist_status:
                validate_error_message += "Test dataset is missing some required columns.\n"    
            else:
                logger.info("Test dataset contains all the required columns.")
                
            validation_status = len(validate_error_message) == 0
            logger.info(f"Data validation completed. Validation status: {validation_status}")
            
            # Create a DataValidationArtifact with the results
            data_validation_artifact = DataValidationArtifact(
                validation_status = validation_status,
                error_message = validate_error_message,
                validation_report_file_path = self.data_validation_config.validation_report_file_path
            )
            
            # Ensure the directory for the validation report exists
            os.makedirs(os.path.dirname(self.data_validation_config.validation_report_file_path), exist_ok = True)
            
            # Save validation report as a JSON file for easier access
            report_dict = {
                "validation_status": validation_status,
                "error_message": validate_error_message.strip()
            }
            
            with open(self.data_validation_config.validation_report_file_path, 'w') as report_file:
                json.dump(report_dict, report_file, indent=4)
                
            logger.info(f"Data validation report saved at {self.data_validation_config.validation_report_file_path}")
            logger.info(f"DataValidationArtifact created: {data_validation_artifact}")
            logger.info("Data validation process completed successfully.")
            return data_validation_artifact
        
        except Exception as e:
            logger.error(f"Error during data validation: {e}")
            raise CustomException(e, sys) from e