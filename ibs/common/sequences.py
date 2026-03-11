"""
Collection of utility functions to process lists and dictionaries.

@author: Holger Schoener <holger.schoener@siemens.com>
"""

import copy

import numpy as np


def is_sequence(arg):
    """Indicate whether arg is a sequence, but not a string."""
    if arg is None:
        return False
    return (
        not hasattr(arg, "strip")
        and not isinstance(arg, np.generic)
        and (hasattr(arg, "__getitem__") or hasattr(arg, "__iter__"))
    )


def addtodict(d, d2={}, **kwargs):
    """
    Add the elements of the provided 'd2' dict and/or the provided keyword arguments to a copy of the dict 'd'.

    A modified copy is returned.
    """
    d = copy.copy(d)
    d.update(d2, **kwargs)
    return d


class DictNamespace(dict):
    """
    Extend dicts by allowing access to their values by attribute notation.

    >>> d = DictNamespace()
    >>> d["a"] = 1
    >>> d.a
    -> 1
    >>> d.b = 2
    >>> d["b"]
    -> 2
    """

    # pylint: disable=too-many-instance-attributes
    def __getattr__(self, name):
        """Return dict element `name` (for names not already defined on dict)."""
        if name in self:
            return self[name]
        raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        """Set dict element `name` from `value` (for names not already defined on dict)."""
        self[name] = value

    def __delattr__(self, name):
        """Remove dict element `name` (for names not already defined on dict)."""
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)
