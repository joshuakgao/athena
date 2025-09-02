"""Parent class used to structure all encoders."""

from abc import ABC, abstractmethod


class BaseEncoder(ABC):
    """Base class for all encoders."""

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def encode(self, *args, **kwargs):
        """Embed the given text and return a list of floats."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def decode(self, *args, **kwargs):
        """Decode the given embedding back to text."""
        raise NotImplementedError("This method should be overridden by subclasses.")
