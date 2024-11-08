#!/usr/bin/env python3
"""
A function that computes to policy
with a weight of a matrix
"""


import numpy as np


def policy(matrix, weight):
    """
    Args:
        state is a numpy.ndarray of shape (1, 4)
        weight is a numpy.ndarray of shape (4, 2)

    Returns:
        the policy for the given state and weight
    """
    # for each column of weights, sum (matrix[i] * weight[i]) using dot product
    dot_product = matrix.dot(weight)
    # find the exponent of the calculated dot product
    exp = np.exp(dot_product)
    # policy is exp / sum(exp)
    policy = exp / np.sum(exp)
    return policy
