import torch
from utils.logger import logger


def device_selector(device: str = "auto", label=""):
    if device == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

    logger.info(f"Using device {device} for {label}")
    return device
