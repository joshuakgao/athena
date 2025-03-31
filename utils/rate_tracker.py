import time
from utils.logger import logger


class RateTracker:
    def __init__(self, unit=""):
        self.start_time = time.time()
        self.total = 0
        self.unit = unit

    def increment(self):
        self.total += 1

    def get_rate(self):
        """Calculate and return the current processing rate in images per hour"""
        elapsed_time = time.time() - self.start_time
        hours = elapsed_time / 3600  # Convert seconds to hours
        rate = self.total / hours if hours > 0 else 0
        return rate

    def log_rate(self):
        """Log the current processing rate with an optional message"""
        rate = self.get_rate()
        logger.info(f"{rate:.2f} {self.unit} per hour")

    def reset(self):
        """Reset the tracker"""
        self.start_time = time.time()
        self.images_processed = 0
