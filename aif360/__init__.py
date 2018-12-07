try:
    from .version import version as __version__
except ImportError:
    pass

# Inspired by: http://www.themetabytes.com/2018/06/19/lazy-loading-modules-in-python/
import sys
import importlib
from types import ModuleType


"""A value which signals the attribute has yet to be actually imported."""
sentinal = object()

class LazyModule(ModuleType):
    """A module which allows for lazy importing."""

    def __init__(self, name, mod, fromlist):
        """
        Args:
            name (str): The `__name__` of the `__init__.py` file being modified.
            mod (str): Module where the attributes in `fromlist` may be found.
            fromlist (list): List of attributes to import.
        """
        super(LazyModule, self).__init__(name)
        for cls in fromlist:
            setattr(self, cls, sentinal)
            setattr(self, '__' + cls, mod)

    def __getattribute__(self, attr):
        val = object.__getattribute__(self, attr)
        if val is sentinal:
            mod = object.__getattribute__(self, '__' + attr)
            module = importlib.import_module(mod)
            ret = getattr(module, attr)

            setattr(self, attr, ret)
            return ret

        return val

def lazy_import(name, mod, fromlist):
    """This is roughly equivalent to `from mod import *fromlist` but lazily.

    We use this to bring class names directly into the namespace without
    actually loading them to avoid ImportErrors for uninstalled dependencies.

    Args:
        name (str): The `__name__` of the calling module.
        mod (str): Module where the attributes in `fromlist` may be found.
        fromlist (list): List of attributes to import.

    Returns:
        None: Since this function defers the actual import, it returns None.
    """
    old = sys.modules[name]
    new = LazyModule(name, mod, fromlist)
    new.__path__ = old.__path__
    new.__dict__.update(old.__dict__)

    sys.modules[name] = new
