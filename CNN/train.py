# -*- coding: utf-8 -*-
#上述数据均是在matlab里面处理好的，每个文件包含数据和对应的标签，分别读取出来，用于后续使用！
import tensorflow as tf
from input_data import read_data_sets
import inference
import os
import numpy as np


BATCH_SIZE = 100
LEARNING_RATE_BASE = 1e-3
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "../../Identification_model/"
MODEL_NAME= "Identification_model"

def train(CC2530):
    x = tf.placeholder(tf.float32, [None, inference.INPUT_LENGTH, inference.INPUT_WIDE, inference.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, inference.NUM_LABELS], name='y-input')
    
    
    train_feed = {x: np.reshape(CC2530.train.images,[-1,inference.INPUT_LENGTH, inference.INPUT_WIDE, inference.NUM_CHANNELS]),
                 y_: CC2530.train.labels}    
    validate_feed = {x: np.reshape(CC2530.validation.images,[-1,inference.INPUT_LENGTH, inference.INPUT_WIDE, inference.NUM_CHANNELS]),
                 y_: CC2530.validation.labels}    
    test_feed = {x: np.reshape(CC2530.test.images,[-1,inference.INPUT_LENGTH, inference.INPUT_WIDE, inference.NUM_CHANNELS]),
                 y_: CC2530.test.labels}

    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference.inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)
    

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,  global_step, 
                                               CC2530.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True)
    
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name="main")
        
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1                      
    with tf.Session(config=config,) as sess:
        tf.global_variables_initializer().run()
        
        for i in range(TRAINING_STEPS):
            xs, ys = CC2530.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                inference.INPUT_LENGTH, 
                inference.INPUT_WIDE, 
                inference.NUM_CHANNELS))
            _, accuracy_value, loss_value, step = sess.run([train_op, accuracy,loss, global_step], feed_dict={x:reshaped_xs, y_:ys})
            if i%1000 == 0:
                print("After %d training step(s), loss on training batach is %g,trainning accuracy on this batch is %g." % (step, loss_value, accuracy_value)) 
#                train_accuracy_score = sess.run(accuracy, feed_dict=train_feed)
#                print("After %d training step(s), training accuracy = %g" % (step, train_accuracy_score))
#                validation_accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
#                print("After %d training step(s), validation accuracy = %g" % (step, validation_accuracy_score))
#                test_accuracy_score = sess.run(accuracy, feed_dict=test_feed)
#                print("After %d training step(s), test accuracy = %g" % (step, test_accuracy_score))
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))


def main():
    CC2530 = read_data_sets()
    train(CC2530)

if __name__ == "__main__":
    main()               



