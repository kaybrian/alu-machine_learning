#!/usr/bin/env python3
"""
    defines function that converts a numeric label vector
    into a one-hot matrix
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
        one hot encode function
        Args:
            Y: numpy.ndarray of shape (classes,)
            classes: number of classes
        Returns:
            one-hot encoding of Y with shape (classes, m)
            m: number of examples
    """
    if type(Y) is not np.ndarray or len(Y) == 0:
        return None
    if type(classes) is not int or classes <= 0:
        return None
    
    one_hot = np.zeros((classes, Y.shape[0]))
    for i in range(Y.shape[0]):
        one_hot[Y[i], i] = 1
    return one_hot
