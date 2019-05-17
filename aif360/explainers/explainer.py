from abc import ABC, abstractmethod


class Explainer(ABC):
    """Base class for explainers."""

    @abstractmethod
    def __init__(self):
        pass
