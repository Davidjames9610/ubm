# utils to use across all files

"""
Compute the mel scale spectrogram.

Args:
    kls:    (string) : full class name to be used to find either the class or method
    in the module
"""


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m
