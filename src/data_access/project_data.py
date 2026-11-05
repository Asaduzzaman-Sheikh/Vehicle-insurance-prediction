import sys
from typing import Optional
import pandas as pd
import numpy as np

from src.configuration.mongo_db_connection import MongoDBClient
from src.exceptions import CustomException
from src.constants import DB_NAME
from src.logger import setup_logger

logger = setup_logger(__name__)

# Data Access Layer for Project Data
"""
    This module provides a class to interact with the MongoDB database for project data operations.

"""

try:
    class ProjectDataAccess:
        """
        A class to manage data access operations for the project.
        
        This class provides methods to interact with the MongoDB database, including fetching data from collections.
        """
        
        def __init__(self)-> None:
            # Initialize the ProjectDataAccess instance and establish a connection to the MongoDB database.
            try:
                self.mongo_client = MongoDBClient(database_name=DB_NAME)
                logger.info("ProjectDataAccess initialized successfully.")
            except Exception as e:
                logger.error(f"Error initializing ProjectDataAccess: {e}")
                raise CustomException(e, sys) from e
        
        def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
            try:
                # Use the specified database name or default to the instance's database
                db = self.mongo_client.client[database_name] if database_name else self.mongo_client.database
                logger.info(f"Exporting collection '{collection_name}' from database '{db.name}' as DataFrame.")
                
                # Fetch the collection and convert it to a DataFrame
                collection = db[collection_name]
                df = pd.DataFrame(list(collection.find()))
                logger.info(f"Successfully exported collection '{collection_name}' as DataFrame with shape {df.shape}.")
                
                # Drop the '_id' column if it exists
                if '_id' in df.columns.to_list():
                    df.drop(columns=['_id'], inplace=True)
                
                # Replace 'na' strings with np.nan for proper missing value representation
                df.replace({'na': np.nan}, inplace=True)
                logger.info(f"Replaced 'na' strings with np.nan in DataFrame.")
                logger.info(f"All operations completed successfully. Returning DataFrame.")
                return df
            except Exception as e:
                logger.error(f"Error exporting collection '{collection_name}' as DataFrame: {e}")
                raise CustomException(e, sys) from e
except Exception as e:
    logger.error(f"Failed to initialize ProjectDataAccess class: {e}")
    raise CustomException(e, sys) from e
            
                
            
            
            
