#!/usr/bin/env python3
"""
    A class DeepNeuralNetwork that defines a deep neural
    network performing binary classification
"""


import numpy as np


class DeepNeuralNetwork:
    """
    A class DeepNeuralNetwork
    """

    def __init__(self, nx, layers):
        if isinstance(nx, int) is False:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if isinstance(layers, list) is False and len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        for i in layers:
            if isinstance(i, int) is False:
                raise TypeError("layers must be a list of positive integers")

            if i < 1:
                raise ValueError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        self.nx = nx
        self.layers = layers

        # init the weights and the biases
        for ii in range(self.L):
            if i == 0:
                self.weights["W1"] = (
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                )
            else:
                self.weights["W" + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]
                ) * np.sqrt(2 / layers[i - 1])
            self.weights["b" + str(i + 1)] = np.zeros((layers[i], 1))
