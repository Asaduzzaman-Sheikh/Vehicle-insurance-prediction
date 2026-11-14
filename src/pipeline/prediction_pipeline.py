import sys
import pandas as pd

from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import S3Estimator
from src.exceptions import CustomException
from src.logger import setup_logger

logger = setup_logger(__name__)


class VehicleData:
    def __init__(
        self,
        Gender,
        Age,
        Driving_License,
        Region_Code,
        Previously_Insured,
        Annual_Premium,
        Policy_Sales_Channel,
        Vintage,
        Vehicle_Age_lt_1_year,
        Vehicle_Age_gt_2_years,
        Vehicle_Damage_yes,
        Vehicle_Age=None,
        Vehicle_Damage=None,
    ):
        try:
            self.Gender = Gender
            self.Age = Age
            self.Driving_License = Driving_License
            self.Region_Code = Region_Code
            self.Previously_Insured = Previously_Insured
            self.Annual_Premium = Annual_Premium
            self.Policy_Sales_Channel = Policy_Sales_Channel
            self.Vintage = Vintage
            self.Vehicle_Age_lt_1_year = Vehicle_Age_lt_1_year
            self.Vehicle_Age_gt_2_years = Vehicle_Age_gt_2_years
            self.Vehicle_Damage_yes = Vehicle_Damage_yes

            # Derive the raw categorical values from the one-hot flags
            if Vehicle_Age is None:
                if Vehicle_Age_lt_1_year == 1:
                    Vehicle_Age = "< 1 Year"
                elif Vehicle_Age_gt_2_years == 1:
                    Vehicle_Age = "> 2 Years"
                else:
                    Vehicle_Age = "1-2 Year"
            self.Vehicle_Age = Vehicle_Age

            if Vehicle_Damage is None:
                Vehicle_Damage = "Yes" if Vehicle_Damage_yes == 1 else "No"
            self.Vehicle_Damage = Vehicle_Damage

        except Exception as e:
            logger.error(f"Error occurred while initializing VehicleData: {e}")
            raise CustomException(e, sys) from e

    def get_vehicle_data_as_dict(self) -> dict:
        try:
            return {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Vehicle_Age": [self.Vehicle_Age],
                "Vehicle_Damage": [self.Vehicle_Damage],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage],
                # 👇 CHANGED TO MATCH MODEL EXPECTATION
                "Vehicle_Age_lt_1_Year": [self.Vehicle_Age_lt_1_year],
                "Vehicle_Age_gt_2_Years": [self.Vehicle_Age_gt_2_years],
                "Vehicle_Damage_Yes": [self.Vehicle_Damage_yes],
            }
        except Exception as e:
            logger.error(f"Error occurred while creating vehicle data dictionary: {e}")
            raise CustomException(e, sys) from e

    def get_vehicle_data_as_dataframe(self) -> pd.DataFrame:
        try:
            return pd.DataFrame(self.get_vehicle_data_as_dict())
        except Exception as e:
            logger.error(f"Error occurred while creating vehicle data DataFrame: {e}")
            raise CustomException(e, sys) from e


class VehiclePredictor:
    def __init__(self, prediction_pipeline_config: VehiclePredictorConfig | None = None) -> None:
        try:
            self.prediction_pipeline_config = prediction_pipeline_config or VehiclePredictorConfig()
            # Load model once, not per request
            self.model = S3Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
        except Exception as e:
            logger.error(f"Error occurred while initializing VehiclePredictor: {e}")
            raise CustomException(e, sys) from e

    def predict(self, X: pd.DataFrame) -> pd.Series:
        try:
            logger.info("Starting prediction process...")
            return self.model.predict(X)       # ✅ actually return
        except Exception as e:
            logger.error(f"Error occurred during prediction: {e}")
            raise CustomException(e, sys) from e