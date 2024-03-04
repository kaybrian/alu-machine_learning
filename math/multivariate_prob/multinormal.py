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
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.data = data
        mean = np.mean(data, axis=1, keepdims=True)
        self.mean = mean
        cov = np.matmul(data - mean, data.T - mean.T) / (n - 1)
        self.cov = cov
