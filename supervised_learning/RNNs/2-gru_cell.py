#!/usr/bin/env python3
'''
    Script that defines a class GRUCell
    GRUCell that represents a gated recurrent unit
'''


import numpy as np


class GRUCell:
    '''
        Class GRUCell that represents a gated recurrent unit

        parameters:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
    '''

    def __init__(self, i, h, o):
        '''
            Class constructor
        '''

        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        '''
            Function that performs forward propagation

            parameters:
                h_prev: contains the previous hidden state
                x_t: contains the data input of the cell

            return:
                h_next: next hidden state
                y: output of the cell
        '''

        concatenation = np.concatenate((h_prev, x_t), axis=1)

        z = np.matmul(concatenation, self.Wz) + self.bz
        r = np.matmul(concatenation, self.Wr) + self.br
        h_hat = np.tanh(np.matmul(concatenation, self.Wh) + self.bh)

        h_next = r * h_hat + (1 - r) * h_prev
        y = np.matmul(h_next, self.Wy)

        return h_next, y
