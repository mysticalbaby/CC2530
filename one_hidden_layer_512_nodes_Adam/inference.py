# -*- coding: utf-8 -*-

import tensorflow as tf
import math

INPUT_NODE = 3200
OUTPUT_NODE = 54
HIDDEN1_NODE = 256
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(shape[0])))
    if regularizer !=None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

def inference(input_tensor, train, regularizer):
    # Hidden 1
    with tf.variable_scope('hidden1'):
        weights = get_weight_variable([INPUT_NODE, HIDDEN1_NODE], regularizer)
        biases = tf.get_variable("biases", [HIDDEN1_NODE], initializer=tf.constant_initializer(0.0))
        hidden1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        if train:hidden1 = tf.nn.dropout(hidden1, 0.5)
            
    # Hidden 2
    with tf.variable_scope('hidden2'):
        weights = get_weight_variable([HIDDEN1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        hidden2 = tf.matmul(hidden1, weights) + biases
    
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
    
    return hidden2
