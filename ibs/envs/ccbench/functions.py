"""
Collection of functions useful for generating or transforming arrays of data.

(c) 2026 Siemens AG

Author:
    Holger Schoener <holger.schoener@siemens.com>
"""

import warnings
from typing import Sequence
from typing import TypeVar

import numpy as np
from scipy import signal

NumberOrVector = TypeVar("NumberOrVector", float, np.ndarray)


def soft_hinge(x: NumberOrVector, d: float = 1, a: float = 1, b: float = 1) -> NumberOrVector:
    """
    Return a smooth approximation of the hinge function f(x) = max(0, x).

    At x=0 the function crosses the Y-axis at d/a; for x>0 the function grows almost linearly (the larger the x, the
    closer to linear), while for x<0 it approaches 0 from above. The larger a is, the sharper the curvature, and the
    closer the similarity to the hinge funtion. `d` scales the function in Y-direction. `b` should stay at 1, otherwise
    the function has a singularity and does not fully resemble a smoothed hinge function any more.

    Generate a plot for visualization of the function using this code:
    ```
    import matplotlib.pyplot as plt
    import warnings
    x = np.linspace(-10, 10, 101)
    y = soft_hinge(x, d=1, a=0.5)
    plt.plot(x, y)
    plt.grid()
    plt.axis(ymin=0)
    ```
    Args:
        x: independent variable of the function.
        d: parameter determining the scaling of the function in Y-direction; for d=1, the function approaches the
            identity for x>0.
        a: parameter determining the sharpness of curvature around x=0 (for higher a, the hinge function approximation
            becomes closer).
        b: should stay at the default value of 1, otherwise the function has a singularity around which the hinge
            function is not well approximated any more; it will assume negative values for some x values, and might
            even approach the identity function instead of the hinge function (except for the singularity).

    Returns:
        The computed function result, as float or np.ndarray.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.where(x == 0, d / a, d * x * np.exp(a * x) / (np.exp(a * x) - b))  # pyright: ignore[reportReturnType]


def norm_filter(filt: Sequence[float]) -> np.ndarray:
    """
    Normalize a provided filter (list of floats) to sum up to 1.0 .

    Args:
        filt: filter values as sequence of floats.
    Returns:
        The `filt` coefficients scaled such that they sum up to 1.0 .
    """
    return np.array(filt) / sum(filt)


# window functions of `scipy.signal.windows` without further parameters, usable as data-only filter kernels
WINDOW_NAMES = ("boxcar", "triang", "bartlett", "hann", "hamming", "blackman")


def window_filter(name: str, length: int, normalize: bool = True) -> np.ndarray:
    """
    Create a FIR filter kernel from a (symmetric) window function of `scipy.signal.windows`.

    Args:
        name: Name of the window function, one of `WINDOW_NAMES`.
        length: Number of filter coefficients.
        normalize: Scale the coefficients to sum up to 1.0 (see `norm_filter`).

    Returns:
        The filter coefficients, e.g. `norm_filter(scipy.signal.windows.hann(length))` for "hann".
    """
    if name not in WINDOW_NAMES:
        raise ValueError(f"unknown window {name!r}, use one of {WINDOW_NAMES}")
    window = getattr(signal.windows, name)(length)
    return norm_filter(window) if normalize else window
