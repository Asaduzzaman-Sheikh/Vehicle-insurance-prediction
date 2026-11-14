import sys

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exceptions import CustomException
from src.logger import setup_logger
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import S3Estimator

logger = setup_logger(__name__)

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
            self.s3 = SimpleStorageService()
            self.s3_estimator = S3Estimator(
                bucket_name=self.model_pusher_config.bucket_name,
                model_path=self.model_pusher_config.s3_model_key_path
            )
        except Exception as e:
            logger.error(f"Error initializing ModelPusher: {e}")
            raise CustomException(e, sys) from e
        
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logger.info("Uploading artifacts to S3 bucket...")
            self.s3_estimator.save_model(
                from_file=self.model_evaluation_artifact.trained_model_file_path
            )
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )
            
            logger.info(f"Model artifacts successfully pushed to S3 bucket: {model_pusher_artifact.bucket_name} at path: {model_pusher_artifact.s3_model_path}")
            return model_pusher_artifact
        except Exception as e:
            logger.error(f"Error during model pushing: {e}")
            raise CustomException(e, sys) from e
        
    