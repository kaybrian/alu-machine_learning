#!/usr/bin/env python3
'''
    function def train(X_train, Y_train, X_valid, Y_valid,
    layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    that builds, trains, and saves a neural network classifier:
'''


import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(
    X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
    alpha, iterations, save_path="/tmp/model.ckpt"
):
    '''
        that builds, trains, and saves a neural network classifier:
    '''

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    keep_prob = tf.placeholder(tf.float32)
    y_pred = forward_prop(x, layer_sizes, activations)
    cost = calculate_loss(y, y_pred)
    train_op = create_train_op(cost, alpha)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            cost_train, _ = sess.run([cost, train_op], feed_dict={
                x: X_train, y: Y_train, keep_prob: 0.5
            })
            cost_valid = sess.run(cost, feed_dict={
                x: X_valid, y: Y_valid, keep_prob: 1.0
            })
            print("After {} iterations:".format(i))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tValidation Cost: {}".format(cost_valid))
        saver.save(sess, save_path)
        return save_path