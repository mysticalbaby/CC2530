# -*- coding: utf-8 -*-

"""Functions for downloading and reading CC2530 Preamble data."""

import tensorflow as tf
from scipy.io import loadmat as load
import numpy
import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel() - 1] = 1#由于是从index0开始的
    return labels_one_hot

class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 one_hot=False,
                 dtype=tf.float32,
                 seed=None):
        
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
    
        numpy.random.seed(seed1 if seed is None else seed2)
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        
        self._num_examples = images.shape[0]        
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
        
    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
    
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

#Dataset = collections.namedtuple('Dataset', ['data', 'target'])


def read_data_sets(dtype=dtypes.float32,
                   seed=None,
                   num_classes = 54
                   ):
    train_data = load('../../预处理/前导提取/Syn_Header_Datasets/train_data.mat')
#    train_phased_data = load('../../预处理/前导提取/Syn_Header_Datasets/train_phased_data.mat')
#    train_phased_data = load('../../预处理/前导提取/Syn_Header_Datasets/cost207_train_phased_data.mat')
    train_phased_data = load('../../预处理/前导提取/Syn_Header_Datasets/No_Noise_4simloops_train_phased_data.mat')
    test_data = load('../../预处理/前导提取/Syn_Header_Datasets/test_data.mat')
    
    validation_data = load('../../预处理/前导提取/Syn_Header_Datasets/validation_data.mat')
    test_phased_data = load('../../预处理/前导提取/Syn_Header_Datasets/test_phased_data.mat')
    cost207_test_phased_data = load('../../预处理/前导提取/Syn_Header_Datasets/cost207_test_phased_data.mat')
    
    train_x_data = train_data['train_x_data']
    train_y_labels = train_data['train_y_labels']
    #读取训练集中的数据和对应标签

    train_phased_x_data = train_phased_data['train_phased_x_data']
    train_phased_y_labels = train_phased_data['train_phased_y_labels']
    #读取训练集中的数据和对应标签
    
    test_x_data = test_data['test_x_data']
    test_y_labels = test_data['test_y_labels']
    #读取测试集中的数据和对应标签
    
    validation_x_data = validation_data['validation_x_data']
    validation_y_labels = validation_data['validation_y_labels']
    
    test_x_phased_data = test_phased_data['test_x_data']
    test_y_phased_labels = test_phased_data['test_y_labels']
    
    cost207_test_x_phased_data = cost207_test_phased_data['test_x_data']
    cost207_test_y_phased_labels = cost207_test_phased_data['test_y_labels']

    
    train_images = train_x_data
    train_labels = dense_to_one_hot(train_y_labels, num_classes) 
    
    train_phased_images = train_phased_x_data
    train_phased_labels = dense_to_one_hot(train_phased_y_labels, num_classes) 
    
    validation_images = validation_x_data
    validation_labels = dense_to_one_hot(validation_y_labels, num_classes)
    
    test_images = test_x_data
    test_labels = dense_to_one_hot(test_y_labels, num_classes) 
    
    test_phased_images = test_x_phased_data
    test_phased_labels = dense_to_one_hot(test_y_phased_labels, num_classes)
    
    cost207_test_phased_images = cost207_test_x_phased_data
    cost207_test_phased_labels = dense_to_one_hot(cost207_test_y_phased_labels, num_classes)
    

    options = dict(dtype=dtype, seed=seed)
    train = DataSet(train_images, train_labels, **options)
    phased_train = DataSet(train_phased_images, train_phased_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)
    phased_test = DataSet(test_phased_images, test_phased_labels, **options)
    cost207_phased_test = DataSet(cost207_test_phased_images, cost207_test_phased_labels, **options)
    Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test', 'phased_test','cost207_phased_test'])
    return Datasets(train=phased_train, validation=validation, test=test, phased_test=phased_test,cost207_phased_test = cost207_phased_test)

