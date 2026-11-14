import sys
import os
from typing import Optional

# --- FIXED IMPORT FOR SRC STRUCTURE ---
# Try relative import first (works when imported by main.py)
# Fallback to absolute import (works if you run this file directly)
try:
    from ..logger import setup_logger  # Two dots (..) goes up to 'src'
except ImportError:
    from src.logger import setup_logger

# Initialize the logger for this specific module
logger = setup_logger(__name__)


def format_error_message(
    error: Exception, 
    custom_message: str = "An error occurred"
) -> str:
    """
    Extracts the file name and line number from an active exception and 
    returns a highly readable, formatted error string.
    """
    # Get the traceback object from the currently active exception
    exc_type, exc_value, exc_tb = sys.exc_info()
    
    if exc_tb is not None:
        # Extract filename and line number
        filename = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_number = exc_tb.tb_lineno
        
        # Format the output
        formatted_msg = f"[{filename}:{line_number}] {custom_message} -> {str(error)}"
    else:
        # Fallback if the exception was raised manually without a traceback
        formatted_msg = f"{custom_message} -> {str(error)}"
        
    return formatted_msg


class CustomException(Exception):
    """
    A robust custom exception that automatically logs the exact file and line 
    number where the error occurred, making debugging much faster.
    
    Example usage:
        try:
            1 / 0
        except Exception as e:
            raise CustomException("Division failed", e) from e
    """
    
    def __init__(
        self, 
        message: str, 
        original_exception: Optional[Exception] = None
    ):
        """
        Args:
            message (str): Custom context message explaining what failed.
            original_exception (Exception, optional): The original error caught in the except block.
        """
        # 1. Format the detailed error string
        detailed_message = format_error_message(
            error=original_exception or self, 
            custom_message=message
        )
        
        # 2. Pass the detailed message to the base Exception class
        super().__init__(detailed_message)
        
        # 3. Store attributes for easy access if needed later
        self.original_exception = original_exception
        self.custom_message = message

        # 4. Automatically log the error to your file and console!
        # This means you don't have to manually log it in your try/except blocks anymore.
        logger.error(detailed_message)

    def __str__(self) -> str:
        """Returns a clean, readable string representation of the exception."""
        # self.args[0] is the detailed message passed to super().__init__()
        return self.args[0] if self.args else "Unknown CustomException Error"