import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
# Go up 3 levels (__init__.py -> logger -> src -> Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

# ANSI color codes for professional console output
class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to console output based on log level."""
    COLORS = {
        logging.DEBUG: "\033[36m",    # Cyan
        logging.INFO: "\033[32m",     # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
        logging.CRITICAL: "\033[1;31m" # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        log_color = self.COLORS.get(record.levelno, self.RESET)
        formatter = logging.Formatter(
            f"{log_color}%(asctime)s - %(name)s - %(levelname)s - %(message)s{self.RESET}",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return formatter.format(record)


def setup_logger(name: str = "AppLogger", log_level: int = logging.DEBUG) -> logging.Logger:
    """
    Configures and returns a named logger with console and rotating file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers if setup_logger() is called multiple times
    if not logger.handlers:
        # 1. Create log directory safely
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
        except OSError as e:
            sys.stderr.write(f"CRITICAL: Failed to create log directory {LOG_DIR}: {e}\n")

        # 2. Setup Formatters
        # File Formatter: Standard text, includes filename and line number
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # Console Formatter: Colorized for readability
        console_formatter = ColorFormatter()

        # 3. Setup Console Handler (Prints to terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO) # Console gets INFO and above
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # 4. Setup Rotating File Handler (Saves to file)
        try:
            log_file_path = os.path.join(LOG_DIR, LOG_FILE_NAME)
            file_handler = RotatingFileHandler(
                log_file_path, 
                maxBytes=MAX_LOG_SIZE, 
                backupCount=BACKUP_COUNT,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level) # File gets the full log_level (e.g., DEBUG)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            # ==========================================
            # NEW: ADD A SEPARATOR FOR EACH NEW RUN
            # ==========================================
            separator = (
                "\n" + "="*80 + 
                f"\n🚀 NEW APP RUN STARTED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + 
                "="*80 + "\n"
            )
            file_handler.stream.write(separator)
            file_handler.stream.flush() # Force it to write immediately

        except Exception as e:
            logger.warning(f"Could not initialize file logging (file may be locked): {e}")

        # 5. Prevent logs from propagating to the root logger
        logger.propagate = False

    return logger