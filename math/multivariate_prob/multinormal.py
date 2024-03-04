#!/usr/bin/env python3
"""
    class MultiNormal that represents
    a Multivariate Normal distribution:
"""


import numpy as np


class MultiNormal:
    """
    class MultiNormal that represents
    a Multivariate Normal distribution:
    """

    def __init__(self, data):
        """
        Args:
            data is a numpy.ndarray of shape (n, d)
            containing the data set
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("X must be a 2D numpy.ndarray")

        if data.shape[0] < 2:
            raise ValueError("data must contain multiple data points")

        self.data = data

        # Set the public instance variables:
        d, n = data.shape
        mean = np.mean(data, axis=1, keepdims=True)
        self.mean = mean
        cov = np.matmul(data - mean, data.T - mean.T) / (n - 1)
        self.cov = cov
