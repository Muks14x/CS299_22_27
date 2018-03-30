from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            # print("hiii  " + name)
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name, reuse=tf.AUTO_REUSE)

batchnorm_count = 0
def bn(x):
    global batchnorm_count
    batch_object = batch_norm(name=("bn" + str(batchnorm_count)))
    batchnorm_count += 1
    return batch_object(x)

def conv2d(input_, output_dim,
           kernel_h=5, kernel_w=5, stride_h=2, stride_w=2,
           name="conv2d"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [kernel_h, kernel_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32))
        conv = tf.nn.conv2d(input_, w, strides=[1, stride_h, stride_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        return conv

def deconv2d(input_, output_shape,
             kernel_h=5, kernel_w=5, stride_h=2, stride_w=2,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [kernel_h, kernel_w, output_shape[-1], input_.get_shape()[-1]],
        	initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, stride_h, stride_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(input_, alpha=0.2, name="lrelu"):
    return tf.nn.leaky_relu(input_, alpha=alpha, name=name)

# def lrelu(x, leak=0.2, name="lrelu"):
#   return tf.maximum(x, leak*x)

def dense(input_, output_size=1, activation=None):
	return tf.layers.dense(input_, output_size, activation=activation, reuse=tf.AUTO_REUSE)

# def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
#     shape = input_.get_shape().as_list()
#     with tf.variable_scope(scope or "Linear"):
#         matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
#                                  tf.random_normal_initializer(stddev=stddev))
#         bias = tf.get_variable("bias", [output_size],
#             initializer=tf.constant_initializer(bias_start))
#         if with_w:
#             return tf.matmul(input_, matrix) + bias, matrix, bias
#         else:
#             return tf.matmul(input_, matrix) + bias
