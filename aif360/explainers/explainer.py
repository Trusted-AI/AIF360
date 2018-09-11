from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import sys

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Explainer(ABC):
    """Base class for explainers."""

    @abc.abstractmethod
    def __init__(self):
        pass
