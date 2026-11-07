import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTEENN

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.constants import *

from src.utils.main_utils import read_yaml_file, save_numpy_array, save_object
from src.exceptions import CustomException
from src.logger import setup_logger

logger = setup_logger(__name__)


class DataTransformation:

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                data_validation_artifact: DataValidationArtifact,
                data_transformation_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logger.info(
                f"DataTransformation initialized with config: {self.data_transformation_config}, "
                f"ingestion artifact: {self.data_ingestion_artifact}, "
                f"validation artifact: {self.data_validation_artifact}"
            )
        except Exception as e:
            logger.error(f"Error initializing DataTransformation: {e}")
            raise CustomException(e, sys) from e

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            logger.info(f"Reading data from {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error reading data from {file_path}: {e}")
            raise CustomException(e, sys) from e

    def get_data_transformer_object(self) -> Pipeline:
        """
        Builds a ColumnTransformer pipeline that scales numerical columns and
        leaves the rest (already-encoded categorical columns) untouched.
        """
        try:
            numerical_transformer = StandardScaler()
            minmax_transformer = MinMaxScaler()

            # Read from the keys that actually exist in schema.yaml
            numerical_columns = self.schema_config['numerical_columns']
            mm_columns = self.schema_config['mm_columns']

            # Guard: a column cannot be scaled twice. Remove any overlap.
            overlap = set(numerical_columns) & set(mm_columns)
            if overlap:
                numerical_columns = [c for c in numerical_columns if c not in overlap]
                logger.warning(
                    f"Columns {sorted(overlap)} appear in both numerical_columns and "
                    f"mm_columns; using MinMaxScaler for them."
                )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_columns),
                    ('minmax', minmax_transformer, mm_columns),
                ],
                remainder='passthrough'
            )

            final_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
            logger.info("Data transformation pipeline created successfully.")
            return final_pipeline
        except Exception as e:
            logger.error(f"Error creating data transformer object: {e}")
            raise CustomException(e, sys) from e

    def _map_gender_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if 'Gender' in df.columns:
                df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(np.int8)
                logger.info("Mapped 'Gender' column to numerical values.")
            return df
        except Exception as e:
            logger.error(f"Error mapping gender columns: {e}")
            raise CustomException(e, sys) from e

    def _create_dummy_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encodes the *remaining* categorical columns. Gender is mapped
        manually before this step, so it's excluded here to avoid creating a
        Gender_1/Gender_0 pair on top of the manual mapping.
        """
        try:
            categorical_columns = list(self.schema_config['categorical_columns'])
            if 'Gender' in categorical_columns:
                categorical_columns.remove('Gender')

            df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
            logger.info(f"Created dummy variables for categorical columns: {categorical_columns}")
            return df
        except Exception as e:
            logger.error(f"Error creating dummy columns: {e}")
            raise CustomException(e, sys) from e

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.rename(
                columns={
                    'Vehicle_Age_< 1 Year': 'Vehicle_Age_lt_1_Year',
                    'Vehicle_Age_> 2 Years': 'Vehicle_Age_gt_2_Years',
                }
            )
            for column in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
                if column in df.columns:
                    df[column] = df[column].astype(np.int8)
            return df
        except Exception as e:
            logger.error(f"Error renaming columns: {e}")
            raise CustomException(e, sys) from e

    def _drop_id_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the row-identifier column(s) if present.

        `drop_columns` in schema.yaml is a list (["_id", "id"]), and `_id` is
        already gone by the time we get here — so we iterate, dropping only
        what actually exists.
        """
        try:
            drop_columns = self.schema_config.get('drop_columns', [])
            if isinstance(drop_columns, str):
                drop_columns = [drop_columns]

            present = [c for c in drop_columns if c in df.columns]
            if present:
                df = df.drop(columns=present)
                logger.info(f"Dropped identifier columns from DataFrame: {present}")
            else:
                logger.info("No identifier columns present to drop.")
            return df
        except Exception as e:
            logger.error(f"Error dropping identifier column(s): {e}")
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(
                    f"Data validation failed: {self.data_validation_artifact.message}"
                )

            # Load datasets
            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Split features / target
            train_input_feature = train_df.drop(columns=[TARGET_COLUMN])
            train_target_feature = train_df[TARGET_COLUMN]

            test_input_feature = test_df.drop(columns=[TARGET_COLUMN])
            test_target_feature = test_df[TARGET_COLUMN]

            # Encode categoricals (deterministic, no fitting)
            train_input_feature = self._map_gender_columns(train_input_feature)
            test_input_feature = self._map_gender_columns(test_input_feature)

            train_input_feature = self._create_dummy_columns(train_input_feature)
            test_input_feature = self._create_dummy_columns(test_input_feature)

            # Align columns between train and test after one-hot encoding.
            # If train and test have different dummy categories, this prevents
            # shape mismatches in the preprocessor.
            train_input_feature, test_input_feature = train_input_feature.align(
                test_input_feature, join='left', axis=1, fill_value=0
            )
            logger.info(
                f"Aligned train/test feature columns. "
                f"Train shape: {train_input_feature.shape}, Test shape: {test_input_feature.shape}"
            )

            train_input_feature = self._rename_columns(train_input_feature)
            test_input_feature = self._rename_columns(test_input_feature)

            train_input_feature = self._drop_id_column(train_input_feature)
            test_input_feature = self._drop_id_column(test_input_feature)

            logger.info("Data transformation steps completed successfully.")

            # Fit on train, transform both
            preprocessor_obj = self.get_data_transformer_object()
            logger.info("Applying data transformation pipeline to training and testing datasets.")

            processed_train_input_feature = preprocessor_obj.fit_transform(train_input_feature)
            processed_test_input_feature = preprocessor_obj.transform(test_input_feature)
            logger.info("Data transformation applied successfully to training and testing datasets.")

            # Resample ONLY the training set — never the test set.
            smote_enn = SMOTEENN(random_state=42)
            input_feature_train_final, target_feature_train_final = smote_enn.fit_resample(
                processed_train_input_feature, train_target_feature
            )
            logger.info(
                f"SMOTEENN applied to training data: "
                f"{processed_train_input_feature.shape[0]} -> {input_feature_train_final.shape[0]} rows"
            )

            # Test set is left untouched — it must reflect real-world distribution.
            input_feature_test_final = processed_test_input_feature
            target_feature_test_final = np.asarray(test_target_feature)

            # Concatenate features + target
            train_arr = np.c_[input_feature_train_final, target_feature_train_final]
            test_arr = np.c_[input_feature_test_final, target_feature_test_final]

            # Persist
            save_numpy_array(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_obj)
            logger.info("Transformed data and preprocessing object saved successfully.")

            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
            )

        except Exception as e:
            logger.error(f"Error initiating data transformation: {e}")
            raise CustomException(e, sys) from e