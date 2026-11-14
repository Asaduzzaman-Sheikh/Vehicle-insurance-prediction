import os
import sys
import numpy as np
import pandas as pd
import yaml
import dill

from src.exceptions import CustomException
from src.logger import setup_logger

logger = setup_logger(__name__)

# Utility function to read a YAML file and return its contents as a dictionary
def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the YAML file.
    Returns:
        dict: The contents of the YAML file as a dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
            logger.info(f"Successfully read YAML file: {file_path}")
        return content
    except Exception as e:
        logger.error(f"Error reading YAML file: {e}")
        raise CustomException(e, sys) from e  
    
# Utility function to write a dictionary to a YAML file
def write_yaml_file(file_path: str, content: dict, replace: bool = False) -> None:
    """
    Writes a dictionary to a YAML file.

    Args:
        file_path (str): The path to the YAML file.
        content (dict): The dictionary to write to the YAML file.
        replace (bool): Whether to replace the file if it already exists. Default is False.
    """
    try:
        # 1. Handle the replace switch
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Existing file {file_path} removed for replacement.")
        
        # 2. Create the directory if it doesn't exist
        dir_name = os.path.dirname(file_path)
        if dir_name:  # Only create folder if a folder path is provided
            os.makedirs(dir_name, exist_ok=True)
            
        # 3. Write the file
        with open(file_path, 'w') as file:
            yaml.dump(content, file)
            logger.info(f"Successfully wrote content to YAML file: {file_path}")
        
    except Exception as e:
        logger.error(f"Error writing YAML file: {e}")
        raise CustomException(e, sys) from e  
    
# Utility function to save a Python object to a file using dill
def save_object(file_path: str, obj: object) -> None:
    """
    Saves a Python object to a file using dill.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj (object): The Python object to save.
    """
    try:
        # 1. Create the directory if it doesn't exist
        dir_name = os.path.dirname(file_path)
        if dir_name:  # Only create folder if a folder path is provided
            os.makedirs(dir_name, exist_ok = True)
            logger.info(f"Directory {dir_name} created for saving object.")
        # 2. Save the object
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
            logger.info(f"Successfully saved object to file: {file_path}")
    except Exception as e:
        logger.error(f"Error saving object to file: {e}")
        raise CustomException(e, sys) from e
    
# Utility function to load a Python object from a file using dill
def load_object(file_path: str) -> object:
    """
    Loads a Python object from a file using dill.

    Args:
        file_path (str): The path to the file from which the object will be loaded.
    Returns:
        object: The loaded Python object.
    """
    try:
        with open(file_path, 'rb') as file:
            obj = dill.load(file)
            logger.info(f"Successfully loaded object from file: {file_path}")
        return obj
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {file_path}. Error: {fnf_error}")
        raise CustomException(f"File not found: {file_path}", sys) from fnf_error
    except Exception as e:
        logger.error(f"Error loading object from file: {e}")
        raise CustomException(e, sys) from e
    
# Utility function to save a NumPy array to a file
def save_numpy_array(file_path: str, array: np.ndarray) -> None:
    """
    Saves a NumPy array to a file.

    Args:
        file_path (str): The path to the file where the array will be saved.
        array (np.ndarray): The NumPy array to save.
    """
    try:
        # 1. Create the directory if it doesn't exist
        dir_name = os.path.dirname(file_path)
        if dir_name:
            # Only create folder if a folder path is provided
            os.makedirs(dir_name, exist_ok = True)
            logger.info(f"Directory {dir_name} created for saving NumPy array.")
        # 2. Save the NumPy array
        with open(file_path, 'wb') as file:
            np.save(file, array)
            logger.info(f"Successfully saved NumPy array to file: {file_path}")
    except Exception as e:
        logger.error(f"Error saving NumPy array to file: {e}")
        raise CustomException(e, sys) from e
# Utility function to load a NumPy array from a file
def load_numpy_array(file_path: str) -> np.ndarray:
    """
    Loads a NumPy array from a file.

    Args:
        file_path (str): The path to the file from which the array will be loaded.
    Returns:
        np.ndarray: The loaded NumPy array.
    """
    try:
        with open(file_path, 'rb') as file:
            array = np.load(file)
            logger.info(f"Successfully loaded NumPy array from file: {file_path}")
        return array
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {file_path}. Error: {fnf_error}")
        raise CustomException(f"File not found: {file_path}", sys) from fnf_error
    except Exception as e:
        logger.error(f"Error loading NumPy array from file: {e}")
        raise CustomException(e, sys) from e