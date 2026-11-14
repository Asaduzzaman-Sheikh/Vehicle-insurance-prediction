import boto3
import os

from src.constants import REGION_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
from src.exceptions import CustomException
from src.logger import setup_logger

logger = setup_logger(__name__)

# Define a class to manage AWS connections
class S3Client:
    
    s3_client = None
    s3_resource = None
    def __init__(self, region_name: str = REGION_NAME):
        try:
            if S3Client.s3_client is None or S3Client.s3_resource is None:
                __access_key_id = os.getenv(AWS_ACCESS_KEY_ID)
                __secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY)
                if __access_key_id is None or __secret_access_key is None:
                    raise CustomException("AWS credentials are not set in environment variables.")
                
                S3Client.s3_client = boto3.client(
                    's3',
                    region_name=region_name,
                    aws_access_key_id=__access_key_id,
                    aws_secret_access_key=__secret_access_key
                )
                
                S3Client.s3_resource = boto3.resource(
                    's3',
                    region_name=region_name,
                    aws_access_key_id=__access_key_id,
                    aws_secret_access_key=__secret_access_key
                )
            
            self.s3_client = S3Client.s3_client
            self.s3_resource = S3Client.s3_resource
        except Exception as e:
            logger.error(f"Error initializing S3Client: {e}")
            raise CustomException(e)
                
                