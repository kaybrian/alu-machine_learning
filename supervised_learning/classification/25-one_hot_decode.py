#!/usr/bin/env python3
"""
    defines function that converts a numeric label vector
    into a one-hot matrix
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    one hot decode function
    Args:
        one_hot: one-hot encoded numpy.ndarray with shape (classes, m)
        classes: number of classes
    Returns:
        numpy.ndarray with shape (m,) containing the numeric labels for each
        example, or None on failure
    """
    if type(one_hot) is not np.ndarray:
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception as err:
        return None
