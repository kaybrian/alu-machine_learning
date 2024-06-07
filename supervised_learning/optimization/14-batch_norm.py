#!/usr/bin/env python3
"""
    function def create_batch_norm_layer(prev, n, activation):
    that creates a batch normalization layer for a neural network
"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for
    a neural network in tensorflow:

    Args:
        - prev is the activated output of the previous layer
        - n is the number of nodes in the layer to be created
        - activation is the activation function that should be used
            on the output of the layer
        - you should use the tf.layers.Dense layer as the base layer
        with kernal initializer
        tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
        - your layer should incorporate two trainable parameters,
        gamma and beta, initialized as vectors of 1 and 0 respectively
        - you should use an epsilon of 1e-8

    Returns:
        - the activated output of the layer
    """
    layer = tf.layers.Dense(
        n,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"
        ),
        use_bias=False,
    )
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    Z = layer(prev)
    Z_norm = (Z - tf.reduce_mean(Z, axis=0)) / tf.sqrt(
        tf.reduce_mean(tf.square(Z - tf.reduce_mean(Z, axis=0)), axis=0) + 1e-8
    )
    Z_tilde = gamma * Z_norm + beta
    return activation(Z_tilde)
