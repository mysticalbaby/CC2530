# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from input_data import read_data_sets
import inference
import train

# 加载的时间间隔。
EVAL_INTERVAL_SECS = 5

def evaluate(CC2530):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: CC2530.validation.images, y_: CC2530.validation.labels}
        phased_test_feed = {x:CC2530.phased_test.images, y_: CC2530.phased_test.labels}
#        validate_feed = {x: CC2530.test.images, y_: CC2530.test.labels}
    
        y = inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        

#        while True:
#            with tf.Session(config=config,) as sess:
#                ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
#                if ckpt and ckpt.model_checkpoint_path:
#                    saver.restore(sess, ckpt.model_checkpoint_path)
#                    print(ckpt.model_checkpoint_path)
#                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
#                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
#                else:
#                    print('No checkpoint file found')
#                    return
#            time.sleep(EVAL_INTERVAL_SECS)

        
        with tf.Session(config=config,) as sess:
            saver.restore(sess, "../../Identification_model/Identification_model-159001")
            accuracy_score = sess.run(accuracy, feed_dict=phased_test_feed)
            print("phased_test accuracy = %g" % (accuracy_score))
    
def main(argv=None):
    CC2530 = read_data_sets()
    evaluate(CC2530)

if __name__ == '__main__':
    main()



