import sys
from src.logger import setup_logger
from src.exceptions import CustomException
from src.pipeline.training_pipeline import TrainingPipeline

logger = setup_logger(__name__)
logger.info("Logger initialized for demo.py")

# logger.info("This is an info message.")
# logger.debug("This is a debug message.")
# logger.warning("This is a warning message.")
# logger.error("This is an error message.")
# logger.critical("This is a critical message.")
# logger.info("Logging setup complete. Logs will be written to both console and file.")

# try:
#     # Simulate some code that raises an exception
#     1 / 0
# except Exception as e:
#     logger.error(f"An error occurred: {e}", exc_info=True)
#     raise CustomException("Division failed", e) from e

try:
    training_pipeline = TrainingPipeline()
    training_pipeline.start_training_pipeline()
except Exception as e:
    logger.error(f"An error occurred in demo.py: {e}", exc_info=True)
    raise CustomException("An error occurred in demo.py", e) from e
