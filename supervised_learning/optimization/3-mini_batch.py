#!/usr/bin/env python3
'''
    Function that trains a loaded neural network model
    using mini-batch gradient descent:
'''


import tensorflow as tf
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    '''
        Args
            - X_train is a numpy.ndarray of shape (m, 784) containing the training data
            - m is the number of data points
            - 784 is the number of input features
            - Y_train is a one-hot numpy.ndarray of shape (m, 10) containing the training labels
            - 10 is the number of classes the model should classify
            - X_valid is a numpy.ndarray of shape (m, 784) containing the validation data
            - Y_valid is a one-hot numpy.ndarray of shape (m, 10) containing the validation labels
            - batch_size is the number of data points in a batch
            - epochs is the number of times the training should pass through the whole dataset
            - load_path is the path from which to load the model
            - save_path is the path to where the model should be saved after training
            - x is a placeholder for the input data
            - y is a placeholder for the labels
            - accuracy is an op to calculate the accuracy of the model
            - loss is an op to calculate the cost of the model
            - train_op is an op to perform one pass of gradient descent on the model

        Returns: 
            - The path where the model was saved
    '''
    m = X_train.shape[0]
    classes = Y_train.shape[1]
    X_train, Y_train = shuffle_data(X_train, Y_train)
    X_valid, Y_valid = shuffle_data(X_valid, Y_valid)
    x = tf.placeholder(tf.float32, shape=(None, 784))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
    