from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import ABC, abstractmethod


class Explainer(ABC):
    """Base class for explainers."""

    @abstractmethod
    def __init__(self):
        pass
