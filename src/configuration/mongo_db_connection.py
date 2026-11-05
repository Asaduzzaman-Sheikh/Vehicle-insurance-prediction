import os
import pymongo
import certifi
import sys

from src.exceptions import CustomException
from src.logger import setup_logger
from src.constants import CONNECTION_STRING, DB_NAME

logger = setup_logger(__name__)

# Establish a connection to the MongoDB database
try:
    # Use certifi to ensure the SSL certificate is valid
    ca = certifi.where()
    logger.info("Establishing connection to MongoDB...")
    
    class MongoDBClient:
        """
        A class to manage the MongoDB connection and provide access to the database.
        
        """
        client = None # Class variable to hold the MongDB client instance
        
        def __init__(self, database_name: str = DB_NAME):
            """
            Initializes the MongoDBClient instance and connects to the specified database.
            
            Args:
                database_name (str): The name of the database to connect to. Defaults to DB_NAME.
            """
            try:
                if MongoDBClient.client is None:
                    CONNECTION_STRING = os.getenv("CONNECTION_STRING")
                    if not CONNECTION_STRING:
                        raise ValueError("MongoDB connection string is not set in environment variables.")
                    MongoDBClient.client = pymongo.MongoClient(CONNECTION_STRING, tlsCAFile=ca)
                    logger.info("MongoDB connection established successfully.")
                
                self.client = MongoDBClient.client
                self.database = self.client[database_name]
                self.database_name = database_name
                logger.info(f"Connected to database: {self.database_name}")
                
            except Exception as e:
                logger.error(f"Error while connecting to MongoDB: {e}")
                raise CustomException(e, sys) from e
            
except Exception as e:
    logger.error(f"Failed to establish MongoDB connection: {e}")
    raise CustomException(e, sys) from e

