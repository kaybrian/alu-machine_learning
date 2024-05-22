#!/usr/bin/env python3
'''
    function def train(X_train, Y_train, X_valid, Y_valid,
    layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    that builds, trains, and saves a neural network classifier:
'''


import tensorflow as tf


def train(
    X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
    alpha, iterations, save_path="/tmp/model.ckpt"
):
    '''
        that builds, trains, and saves a neural network classifier:
    '''