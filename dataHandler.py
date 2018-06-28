# -*- coding: utf-8 -*-

from mnist import MNIST
import numpy as np
from random import shuffle


class DataHandler():

    def __init__(self, src_dir):

        """ Initializes the mnist handler and sets gz to True so it can handle
            .gz files """
        self.mndata = MNIST(src_dir)
        self.mndata.gz = True
        self.batch_counter = 0
        self.example_counter = 0
        self.current_miniBatch = None
        self.data_loaded = False

    def load_training(self, validation=True):
        """ Loads the train set from disk and splits it into train and validation """
        self.im, self.lb = self.mndata.load_training()
        self.data_loaded = True
        indexes = range(0, len(self.im))
        shuffle(indexes)

        if validation == True:
            """ Splits the train set in 95% train set and 5% validation set """
            self.train_set = indexes[0: 57000]
            self.val_set = indexes[57000:]
        else:
            self.train_set = indexes

    def load_validation(self):
        """ Takes the validation set and puts in the current minibatch """
        self.current_miniBatch = self.val_set
        self.example_counter = 0

    def get_mini_batches(self, minBatch_size=None, minBatch_num=None):
        """ Generates a list of random minibatches. It outputs a list of n minibatches of
            size minBatch_size, or outputs a list of minBatch_num minibatches """
        if not self.data_loaded:
            self.load_training()
            self.data_loaded = True
        else:
            shuffle(self.train_set)

        k = len(self.train_set)

        if not(minBatch_num is None):
            minBatch_size = int (k / minBatch_num)

        p, r = 0, minBatch_size
        self.mini_batches = []
        while(p < k):
            if p+r > k:
                self.mini_batches.append(self.train_set[p:])
            else:
                self.mini_batches.append(self.train_set[p:r])
            p = r
            r += minBatch_size

        self.current_miniBatch = self.mini_batches[0]
        return self.mini_batches

    def move_to_next_batch(self):
        """ Just updates the batch counter and the current minibatch """
        if self.batch_counter > len(self.train_set):
            self.batch_counter = 0
        else:
            self.batch_counter += 1

        self.current_miniBatch = self.mini_batches[self.batch_counter]

    def get_example(self, update_batch=False):
        """ Gets the real index of an example inside the current mini batch """        
        ex = self.current_miniBatch[self.example_counter]
        self.example_counter += 1

        if self.example_counter >= len(self.current_miniBatch):
            if update_batch == True:
                self.move_to_next_batch()
            self.example_counter = 0

        return self.get_example_data(ex)


    def get_example_data(self, example):
        y = 10 * [0]
        y[self.lb[example]] = 1
        return np.asarray(self.im[example]), np.asarray(y).reshape(len(y),1)

if __name__ == '__main__':
    data = DataHandler('./mnist_dataset')
    data.load_training()
    data.get_mini_batches(minBatch_size=10)

    for i in range(len(data.current_miniBatch)):
        im, lb = data.get_example()
        print im, lb
        print "\n\n"
