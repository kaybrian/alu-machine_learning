#!/usr/bin/env python3
"""
    Function def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    that creates the training operation for a neural network in
    tensorflow using the Adam optimization algorithm:
"""


import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow
    using the Adam optimization algorithm

    Args:
        - loss is the loss of the network
        - alpha is the learning rate
        - beta1 is the weight used for the first moment
        - beta2 is the weight used for the second moment
        - epsilon is a small number to avoid division by zero

    Returns:
        The Adam optimization operation
    """
    t = tf.Variable(0, dtype=tf.int32, trainable=False, name='t')
    t_update = tf.assign(t, t + 1)
    m = tf.Variable(tf.zeros_like(loss, dtype=tf.float32), name='m')
    v = tf.Variable(tf.zeros_like(loss, dtype=tf.float32), name='v')
    m_t = beta1 * m + (1 - beta1) * loss
    v_t = beta2 * v + (1 - beta2) * tf.square(loss)
    m_t_c = m_t / (1 - tf.pow(beta1, t))
    v_t_c = v_t / (1 - tf.pow(beta2, t))
    op = tf.assign(m, m_t_c)
    op2 = tf.assign(v, v_t_c)
    with tf.control_dependencies([op, op2]):
        return tf.assign(loss, loss - alpha * m_t_c / (tf.sqrt(v_t_c) + epsilon))
    return op
