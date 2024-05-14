#!/usr/bin/env python3
'''
    A class NeuralNetwork that defines a neural network
    with one hidden layer performing binary classification
'''


import numpy as np


class NeuralNetwork:
    '''
        A class NeuralNetwork
    '''

    def __init__(self, nx, nodes):
        '''
            class constructor
        '''
        if type(nx) is not int:
            raise TypeError('nx must be an integer')

        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')

        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.nodes = nodes
        self.nx = nx

        #  weights vector for the hidden layer
        self.W1 = np.random.randn(self.nodes, self.nx)

        # bias for the hidden layer.
        self.b1 = 0

        # Activated output for the hidden layer
        self.A1 = 0

        # weights vector for the output neuron
        self.W2 = np.random.randn(nodes).reshape(1, nodes)

        # bias for the output neuron
        self.b2 = 0

        # Activated output for the output neuron
        self.A2 = 0
