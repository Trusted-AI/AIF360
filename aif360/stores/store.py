import abc
import copy
import sys
import abc

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Store(ABC):
    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def validate_store(self, **kwargs):
        pass

    @abc.abstractmethod
    def download(self, **kwargs):
        pass

    @abc.abstractmethod
    def existsInDestination(self, **kwargs):
        pass
