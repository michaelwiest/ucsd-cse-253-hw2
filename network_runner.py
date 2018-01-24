from mnist import MNIST
import numpy as np
import pylab as plt
from helper import *


class NetworkRunner(object):
    def __init__(self, mnist_directory, lr0=None, lr_dampener=None,
                 minibatch_size=128):
        self.mnist_directory = mnist_directory
        self.lr_dampener = lr_dampener
        self.holdout_data = None
        self.holdout_data = None
        self.holdout_labels = None
        self.target = None
        self.load_data(self.mnist_directory)

        if lr0 == None:
            self.lr0 = 0.001 / self.train_data.shape[0]
        else:
            self.lr0 = lr0
        self.minibatch_index = 0
        self.minibatch_size = minibatch_size


    def load_data(self, mnist_directory):
        mndata = MNIST(mnist_directory)
        tr_data, tr_labels = mndata.load_training()
        te_data, te_labels = mndata.load_testing()
        train_temp = np.array(tr_data)
        self.train_data = np.concatenate(
                                        (np.ones((train_temp.shape[0], 1)),
                                         train_temp
                                        ), axis=1
                                        )
        self.train_labels = np.array(tr_labels)
        test_temp = np.array(te_data)
        self.test_data = np.concatenate(
                                        (np.ones((test_temp.shape[0], 1)),
                                         test_temp
                                        ), axis=1
                                        )
        self.test_labels = np.array(te_labels)
        self.num_categories = len(list(set(self.train_labels)))
        self.possible_categories = list(set(self.train_labels))
        self.possible_categories.sort()
        print 'Loaded data...'

    def subset_data(self, train_amount, test_amount):
        if train_amount > 0:
            self.train_data = self.train_data[:train_amount]
            self.train_labels = self.train_labels[:train_amount]
        else:
            self.train_data = self.train_data[-train_amount:]
            self.train_labels = self.train_labels[-train_amount:]
        if test_amount > 0:
            self.test_data = self.test_data[:test_amount]
            self.test_labels = self.test_labels[:test_amount]
        else:
            self.test_data = self.test_data[-test_amount:]
            self.test_labels = self.test_labels[-test_amount:]
        print 'Subsetted data.'


    def assign_holdout(self, percent):
        percent /= 100.0
        num_held = int(self.train_data.shape[0] * percent)
        self.train_data = self.train_data[:-num_held]
        self.train_labels = self.train_labels[:-num_held]
        self.holdout_data = self.train_data[-num_held:]
        self.holdout_labels = self.train_labels[-num_held:]
        print 'Assigned holdout data'

    def get_next_mini_batch(self):
        td = self.train_data[self.minibatch_index * self.minibatch_size : (self.minibatch_index + 1) * self.minibatch_size]
        tl = self.train_labels[self.minibatch_index * self.minibatch_size : (self.minibatch_index + 1) * self.minibatch_size]
        self.minibatch_index += 1
        return td, tl
