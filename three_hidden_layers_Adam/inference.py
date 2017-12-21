# -*- coding: utf-8 -*-

import tensorflow as tf
import math

INPUT_NODE = 3200
OUTPUT_NODE = 54
HIDDEN1_NODE = 512
HIDDEN2_NODE = 256
HIDDEN3_NODE = 128

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(shape[0])))
    if regularizer !=None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

def inference(input_tensor, regularizer):
    # Hidden 1
    with tf.variable_scope('hidden1'):
        weights = get_weight_variable([INPUT_NODE, HIDDEN1_NODE], regularizer)
        biases = tf.get_variable("biases", [HIDDEN1_NODE], initializer=tf.constant_initializer(0.0))
        hidden1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        
    # Hidden 2
    with tf.variable_scope('hidden2'):
        weights = get_weight_variable([HIDDEN1_NODE, HIDDEN2_NODE], regularizer)
        biases = tf.get_variable("biases", [HIDDEN2_NODE], initializer=tf.constant_initializer(0.0))
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
        
    # Hidden 3
    with tf.variable_scope('hidden3'):
        weights = get_weight_variable([HIDDEN2_NODE, HIDDEN3_NODE], regularizer)
        biases = tf.get_variable("biases", [HIDDEN3_NODE], initializer=tf.constant_initializer(0.0))
        hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)
        
    # Linear
    with tf.variable_scope('linear'):
        weights = get_weight_variable([HIDDEN3_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(hidden3, weights) + biases
    
    # Linear
#    with tf.name_scope('softmax_linear'):
#        weights = tf.Variable(
#        tf.truncated_normal([hidden2_units, NUM_CLASSES],
#                            stddev=1.0 / math.sqrt(float(hidden2_units))),
#        name='weights')
#        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
#                         name='biases')
#        logits = tf.matmul(hidden2, weights) + biases
#    return logits
    
    return logits
