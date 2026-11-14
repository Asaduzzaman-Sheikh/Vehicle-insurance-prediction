import sys
from typing import Optional

import pandas as pd
from uvicorn import run as app_run
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.pipeline.prediction_pipeline import VehicleData, VehiclePredictor
from src.pipeline.training_pipeline import TrainingPipeline
from src.constants import APP_HOST, APP_PORT
from src.exceptions import CustomException
from src.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Vehicle Insurance Prediction API",
    description="API for predicting vehicle insurance eligibility",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the predictor ONCE — reuses the loaded S3 model across requests
predictor = VehiclePredictor()


class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Gender: Optional[int] = None
        self.Age: Optional[int] = None
        self.Driving_License: Optional[int] = None
        self.Region_Code: Optional[float] = None
        self.Previously_Insured: Optional[int] = None
        self.Annual_Premium: Optional[float] = None
        self.Policy_Sales_Channel: Optional[float] = None
        self.Vintage: Optional[int] = None
        self.Vehicle_Age_lt_1_Year: Optional[int] = None
        self.Vehicle_Age_gt_2_Years: Optional[int] = None
        self.Vehicle_Damage_Yes: Optional[int] = None

    async def load_data_from_form(self):
        form = await self.request.form()
        self.Gender = int(form.get("Gender"))
        self.Age = int(form.get("Age"))
        self.Driving_License = int(form.get("Driving_License"))
        self.Region_Code = float(form.get("Region_Code"))
        self.Previously_Insured = int(form.get("Previously_Insured"))
        self.Annual_Premium = float(form.get("Annual_Premium"))
        self.Policy_Sales_Channel = float(form.get("Policy_Sales_Channel"))
        self.Vintage = int(form.get("Vintage"))
        self.Vehicle_Age_lt_1_Year = int(form.get("Vehicle_Age_lt_1_Year"))
        self.Vehicle_Age_gt_2_Years = int(form.get("Vehicle_Age_gt_2_Years"))
        self.Vehicle_Damage_Yes = int(form.get("Vehicle_Damage_Yes"))


@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"context": None},
    )


@app.get("/train", tags=["training"])
async def train_model():
    try:
        TrainingPipeline().start_training_pipeline()
        return Response(content="Model training completed successfully.", media_type="text/plain")
    except Exception as e:
        logger.error(f"Error occurred during model training: {e}")
        raise CustomException(e, sys) from e


@app.post("/", tags=["form submission for prediction"])
async def handle_form_submission(request: Request):
    try:
        data_form = DataForm(request)
        await data_form.load_data_from_form()

        vehicle_data = VehicleData(
            Gender=data_form.Gender,
            Age=data_form.Age,
            Driving_License=data_form.Driving_License,
            Region_Code=data_form.Region_Code,
            Previously_Insured=data_form.Previously_Insured,
            Annual_Premium=data_form.Annual_Premium,
            Policy_Sales_Channel=data_form.Policy_Sales_Channel,
            Vintage=data_form.Vintage,
            Vehicle_Age_lt_1_year=data_form.Vehicle_Age_lt_1_Year,
            Vehicle_Age_gt_2_years=data_form.Vehicle_Age_gt_2_Years,
            Vehicle_Damage_yes=data_form.Vehicle_Damage_Yes,
        )

        vehicle_df = vehicle_data.get_vehicle_data_as_dataframe()
        prediction = predictor.predict(vehicle_df)

        # ✅ DataFrame-safe extraction (was: int(prediction[0]))
        value = int(prediction.iloc[0, 0])

        status = "Eligible for Insurance" if value == 1 else "Not Eligible for Insurance"

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"context": status},
        )
    except Exception as e:
        logger.error(f"Error occurred during form submission handling: {e}")
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)