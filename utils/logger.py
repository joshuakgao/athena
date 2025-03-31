import logging
import os
from datetime import datetime


class Logger:
    def __init__(self, log_dir="logs", log_level=logging.INFO):
        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Generate the log filename using the current date and time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"log_{timestamp}.log"

        # Create full path for the log file
        log_path = os.path.join(log_dir, log_file)

        # Set up logging format
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"

        # Create and configure logger
        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.FileHandler(log_path),  # Log to file
                logging.StreamHandler(),  # Log to console
            ],
        )
        self.logger = logging.getLogger()

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def critical(self, message):
        self.logger.critical(message)


logger = Logger()
