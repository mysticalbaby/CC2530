# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from input_data import read_data_sets
import inference
import train
import matplotlib.pyplot as plt
import numpy as np
# 加载的时间间隔。
EVAL_INTERVAL_SECS = 5
classes = np.arange(1,55)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.figure(figsize=(20,20))
#    plt.grid
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluate(CC2530):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, inference.INPUT_LENGTH, inference.INPUT_WIDE, inference.NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, inference.NUM_LABELS], name='y-input')
        train_feed = {x: np.reshape(CC2530.train.images,[-1,inference.INPUT_LENGTH, inference.INPUT_WIDE, inference.NUM_CHANNELS]),
                 y_: CC2530.train.labels}    
        validate_feed = {x: np.reshape(CC2530.validation.images,[-1,inference.INPUT_LENGTH, inference.INPUT_WIDE, inference.NUM_CHANNELS]),
                 y_: CC2530.validation.labels}    
#        test_feed = {x: np.reshape(CC2530.test.images,[-1,inference.INPUT_LENGTH, inference.INPUT_WIDE, inference.NUM_CHANNELS]),
#                 y_: CC2530.test.labels}
#        test_feed = {x: np.reshape(CC2530.test.images[CC2530.test.SNR == 0],[-1,inference.INPUT_LENGTH, inference.INPUT_WIDE, inference.NUM_CHANNELS]),
#                 y_: CC2530.test.labels[CC2530.test.SNR == 0]}
                 
        y = inference.inference(x, False, None)
        
        predited_labels = tf.argmax(y, 1)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        

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
            saver.restore(sess, "../../Identification_model/CNN_model/Identification_model")            
#            train_accuracy_score = sess.run(accuracy, feed_dict=train_feed)
#            print("training accuracy = %g" % (train_accuracy_score))
            
            
            validation_accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
            print("validation accuracy = %g" % (validation_accuracy_score))
            
            
            
#            test_labels, test_accuracy_score = sess.run([predited_labels,accuracy], feed_dict=test_feed)
            
            # Plot confusion matrix
#            conf = np.zeros([len(classes),len(classes)])
#            confnorm = np.zeros([len(classes),len(classes)])
#            for i in range(0,test_feed[x].shape[0]):
#                j = list(test_feed[y_][i,:]).index(1)#the true label
#                k = test_labels[i] # the predicted label
#                conf[j,k] =conf[j,k] + 1
#            for i in range(0,len(classes)):
#                confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
#            plot_confusion_matrix(confnorm, labels=classes,title="ConvNet Confusion Matrix ),test accuracy = %g"%(test_accuracy_score))   
            
#            print("test accuracy = %g" % (test_accuracy_score))
            
            # Plot confusion matrix
#            for snr in range(0,31,5):
#                # extract classes @ SNR
#                test_feed = {x: np.reshape(CC2530.test.images[CC2530.test.SNR == snr],[-1,inference.INPUT_LENGTH, inference.INPUT_WIDE, inference.NUM_CHANNELS]),
#                                           y_: CC2530.test.labels[CC2530.test.SNR == snr]}
#                
#                # estimate classes
#                test_labels, test_accuracy_score = sess.run([predited_labels,accuracy], feed_dict=test_feed)
#            
#                # Plot confusion matrix
#                conf = np.zeros([len(classes),len(classes)])
#                confnorm = np.zeros([len(classes),len(classes)])
#                for i in range(0,test_feed[x].shape[0]):
#                    j = list(test_feed[y_][i,:]).index(1)#the true label
#                    k = test_labels[i] # the predicted label
#                    conf[j,k] =conf[j,k] + 1
#                for i in range(0,len(classes)):
#                    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
#                plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d),test accuracy = %g"%(snr,test_accuracy_score))    
#                savename = '../../'+'ConvNet Confusion Matrix'+ str(snr) + 'dB.jpg'
#                plt.savefig(savename)                        
#                cor = np.sum(np.diag(conf))
#                ncor = np.sum(conf) - cor
#                print "Overall Accuracy: ", cor / (cor+ncor)
#                acc[snr] = 1.0*cor/(cor+ncor)

            for snr in range(0,31,5):
                # extract classes @ SNR
                validation_feed = {x: np.reshape(CC2530.validation.images[CC2530.validation.SNR == snr],[-1,inference.INPUT_LENGTH, inference.INPUT_WIDE, inference.NUM_CHANNELS]),
                                           y_: CC2530.validation.labels[CC2530.validation.SNR == snr]}
                
                # estimate classes
                validation_labels, validation_accuracy_score = sess.run([predited_labels,accuracy], feed_dict=validation_feed)
            
                # Plot confusion matrix
                conf = np.zeros([len(classes),len(classes)])
                confnorm = np.zeros([len(classes),len(classes)])
                for i in range(0,validation_feed[x].shape[0]):
                    j = list(validation_feed[y_][i,:]).index(1)#the true label
                    k = validation_labels[i] # the predicted label
                    conf[j,k] =conf[j,k] + 1
                for i in range(0,len(classes)):
                    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
                plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d),validation accuracy = %g"%(snr,validation_accuracy_score))    
                savename = '../../'+'ConvNet Validation Confusion Matrix'+ str(snr) + 'dB.jpg'
                plt.savefig(savename)
           
        
    
def main(argv=None):
    CC2530 = read_data_sets()
    evaluate(CC2530)

if __name__ == '__main__':
    main()



