from abc import ABCMeta


def do_decorate(attr, value):
    return not attr.startswith('_') and callable(value) and getattr(value, '__decorate__', True)

def factory(decorator):
    class ApplyDecoratorMeta(ABCMeta):
        """Metaclass which applies a decorator to all public, non-special
        instance methods.

        Note:
            `decorator` must use @functools.wraps(f) for abstractmethod to work.

        https://stackoverflow.com/questions/10067262/automatically-decorating-every-instance-method-in-a-class
        """
        def __new__(cls, name, bases, dct):
            for attr, value in dct.items():
                if do_decorate(attr, value):
                    dct[attr] = decorator(value)
            return super(ApplyDecoratorMeta, cls).__new__(cls, name, bases, dct)
    return ApplyDecoratorMeta

def dont_decorate(func):
    func.__decorate__ = False
    return func

def ApplyDecorator(decorator):
    return factory(decorator)(str('ApplyDecorator'), (), {})
