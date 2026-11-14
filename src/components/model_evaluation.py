from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact,
)
from sklearn.metrics import f1_score
from src.exceptions import CustomException
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.logger import setup_logger
from src.utils.main_utils import load_object, read_yaml_file
import sys
import pandas as pd
import numpy as np
from typing import Optional
from src.entity.s3_estimator import S3Estimator
from dataclasses import dataclass

logger = setup_logger(__name__)


@dataclass
class ModelEvaluationResponse:
    is_model_accepted: bool
    trained_model_f1_score: float
    best_model_f1_score: float
    improved_f1_score: float


# NOTE: @dataclass removed from ModelEvaluation — the decorator was generating
# an __init__ that the hand-written __init__ silently overwrote.
class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            # FIX: schema_config was never assigned; _create_dummy_columns and
            # _drop_id_column both read it and crashed with AttributeError.
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logger.info(
                f"ModelEvaluation initialized with config: {self.model_evaluation_config} "
                f"and model_trainer_artifact: {self.model_trainer_artifact}"
            )
        except Exception as e:
            logger.error(f"Error initializing ModelEvaluation: {e}")
            raise CustomException(e, sys) from e

    def get_best_model(self) -> Optional[S3Estimator]:
        try:
            bucket_name = self.model_evaluation_config.model_bucket_name
            model_path = self.model_evaluation_config.s3_model_key_path
            s3_estimator = S3Estimator(
                bucket_name=bucket_name,
                model_path=model_path,
            )

            if s3_estimator.is_model_present(model_path=model_path):
                logger.info(f"Best model found in S3 bucket: {bucket_name} at path: {model_path}")
                return s3_estimator
            else:
                logger.info(f"No best model found in S3 bucket: {bucket_name} at path: {model_path}")
                return None
        except Exception as e:
            logger.error(f"Error getting best model: {e}")
            raise CustomException(e, sys) from e

    def _map_gender_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if "Gender" in df.columns:
                df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1}).astype(np.int8)
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
            categorical_columns = list(self.schema_config["categorical_columns"])
            if "Gender" in categorical_columns:
                categorical_columns.remove("Gender")

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
                    "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
                    "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years",
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
            drop_columns = self.schema_config.get("drop_columns", [])
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

    def evaluate_model(self) -> ModelEvaluationResponse:
        try:
            logger.info("Starting model evaluation process.")

            # Load the test dataset
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logger.info(f"Loaded test dataset from {self.data_ingestion_artifact.test_file_path}")

            Xtest = test_df.drop(columns=[TARGET_COLUMN])
            ytest = test_df[TARGET_COLUMN].to_numpy()

            # Reproduce the encoding that DataTransformation applied at training
            # time (Gender map, dummies, rename, drop id).
            Xtest_transformed = self._map_gender_columns(Xtest)
            Xtest_transformed = self._create_dummy_columns(Xtest_transformed)
            Xtest_transformed = self._rename_columns(Xtest_transformed)
            Xtest_transformed = self._drop_id_column(Xtest_transformed)
            logger.info("Encoded test dataset for model evaluation.")

            # FIX: align to the training-time column set/order. The ColumnTransformer
            # was fit on a DataFrame during DataTransformation, so its
            # feature_names_in_ gives us the exact column order the scalers saw.
            # Without this reindex, get_dummies on the test-only CSV can drop
            # categories that were present in train+test together.
            preprocessor = load_object(
                self.model_evaluation_config.preprocessing_object_file_path
            )
            feature_order = list(preprocessor.named_steps["preprocessor"].feature_names_in_)
            Xtest_transformed = Xtest_transformed.reindex(
                columns=feature_order, fill_value=0
            )
            logger.info(
                f"Aligned test features to training schema: "
                f"{Xtest_transformed.shape[1]} columns."
            )

            # FIX: use the F1 computed during training (held-out test, pre-SMOTEENN)
            # rather than a stale/misread value.
            trained_model_f1_score = self.model_trainer_artifact.metrics_artifact.f1_score
            logger.info(f"Trained model F1 score (from training): {trained_model_f1_score}")

            best_model = self.get_best_model()
            best_model_f1_score = 0.0

            if best_model is not None:
                # NOTE: S3Estimator.predict → MyModel.predict → preprocessor.transform
                # internally. Do NOT scale here again.
                best_model_predictions = (
                    best_model.predict(Xtest_transformed).to_numpy().ravel()
                )
                best_model_f1_score = float(
                    f1_score(ytest, best_model_predictions, average="binary")
                )
                logger.info(f"Best model F1 score (recomputed): {best_model_f1_score}")
            else:
                logger.info("No best model in S3; using baseline F1 of 0.0.")

            improved_f1 = trained_model_f1_score - best_model_f1_score

            # Use change_threshold from config instead of accepting any F1 > 0.
            is_model_accepted = improved_f1 >= self.model_evaluation_config.change_threshold
            logger.info(
                f"Acceptance: trained_f1={trained_model_f1_score:.4f}, "
                f"best_f1={best_model_f1_score:.4f}, "
                f"improved_f1={improved_f1:.4f}, "
                f"threshold={self.model_evaluation_config.change_threshold} "
                f"→ accepted={is_model_accepted}"
            )

            result = ModelEvaluationResponse(
                is_model_accepted=is_model_accepted,
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                improved_f1_score=improved_f1,
            )
            logger.info(f"Model evaluation result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logger.info("Initiating model evaluation process")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_evaluation_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                improved_f1_score=evaluate_model_response.improved_f1_score,
                s3_model_path=s3_model_path,
                trained_model_file_path=self.model_trainer_artifact.trained_model_file_path,
            )
            logger.info(f"Model evaluation artifact created: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            logger.error(f"Error initiating model evaluation: {e}")
            raise CustomException(e, sys) from e