#!/usr/bin/env python
import tensorflow as tf
import cv2

ACTIONS = 8 # number of valid actions

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 10, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([4, 4, 64, 128])
    b_conv3 = bias_variable([128])

    W_conv4 = weight_variable([3, 3, 128, 256])
    b_conv4 = bias_variable([256])

    W_conv5 = weight_variable([3, 3, 256, 256])
    b_conv5 = bias_variable([256])

    W_fc1 = weight_variable([2304, 256])
    b_fc1 = bias_variable([256])

    W_fc2 = weight_variable([256, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    #W_fc3 = weight_variable([256, ACTIONS])
    #b_fc3 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 160, 160, 10])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 2) + b_conv4)
    #h_pool4 = max_pool_2x2(h_conv4)

    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)
    #h_pool4 = max_pool_2x2(h_conv4)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv5_flat = tf.reshape(h_conv5, [-1, 2304])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
    #h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1
