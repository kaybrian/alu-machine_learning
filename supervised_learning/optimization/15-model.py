#!/usr/bin/env python3
"""
    Function def model(
        Data_train, Data_valid, layers, activations, alpha=0.001,
        beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
        batch_size=32, epochs=5, save_path='/tmp/model.ckpt'
    ): that builds, trains, and saves a neural network model in tensorflow
    using Adam optimization, mini-batch gradient descent, learning rate decay,
    and batch normalization:
"""


import numpy as np
import tensorflow as tf


def create_placeholders(nx, ny):
    """
    Create placeholders for the input
    data and labels.
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, ny), name="y")
    return x, y


def forward_propagation(x, layers, activations):
    """
    Build the forward propagation part of the
    network with batch normalization.
    """
    A = x
    for i, (layer, activation) in enumerate(zip(layers, activations)):
        layer_name = "layer" + str(i + 1)
        if i == len(layers) - 1:
            A = tf.layers.dense(A, layer, activation=None, name=layer_name)
        else:
            A = tf.layers.dense(A, layer, activation=None, name=layer_name)
            A = tf.layers.batch_normalization(
                A, training=True, name="bn" + str(i + 1)
            )
            A = activation(A)
    return A


def compute_cost(logits, labels):
    """
    Compute the cost using softmax cross entropy.
    """
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=labels
        )
    )
    return cost


def model(
    Data_train,
    Data_valid,
    layers,
    activations,
    alpha=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    decay_rate=1,
    batch_size=32,
    epochs=5,
    save_path="/tmp/model.ckpt",
):
    """Builds, trains, and saves a neural network model."""
    (X_train, Y_train) = Data_train
    (X_valid, Y_valid) = Data_valid
    nx = X_train.shape[1]
    ny = Y_train.shape[1]

    x, y = create_placeholders(nx, ny)
    logits = forward_propagation(x, layers, activations)
    cost = compute_cost(logits, y)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(
        alpha, global_step, decay_steps=1,
        decay_rate=decay_rate, staircase=True
    )

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon
    )
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(cost, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        m = X_train.shape[0]
        for epoch in range(epochs):
            X_train, Y_train = shuffle_data(X_train, Y_train)
            num_minibatches = int(np.ceil(m / batch_size))

            print(f"After {epoch} epochs:")
            train_cost, train_accuracy = sess.run(
                [cost, accuracy], feed_dict={x: X_train, y: Y_train}
            )
            print(f"\tTraining Cost: {train_cost}")
            print(f"\tTraining Accuracy: {train_accuracy}")

            valid_cost, valid_accuracy = sess.run(
                [cost, accuracy], feed_dict={x: X_valid, y: Y_valid}
            )
            print(f"\tValidation Cost: {valid_cost}")
            print(f"\tValidation Accuracy: {valid_accuracy}")

            for step in range(num_minibatches):
                start = step * batch_size
                end = min(start + batch_size, m)
                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]

                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if step % 100 == 0:
                    step_cost, step_accuracy = sess.run(
                        [cost, accuracy], feed_dict={x: X_batch, y: Y_batch}
                    )
                    print(f"\tStep {step}:")
                    print(f"\t\tCost: {step_cost}")
                    print(f"\t\tAccuracy: {step_accuracy}")

        saver.save(sess, save_path)
    return save_path


def shuffle_data(X, Y):
    """Shuffle the data points in two matrices the same way."""
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]
