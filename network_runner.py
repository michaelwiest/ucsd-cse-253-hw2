from __future__ import print_function
from mnist import MNIST
import numpy as np
import pylab as plt
from helper import *
import random
from visible_to_hidden import *
from hidden_to_output import *
import time

np.set_printoptions(threshold=np.nan)

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
            self.lr0 = 100.0 / self.train_data.shape[0]
        else:
            self.lr0 = lr0
        self.minibatch_index = 0
        self.minibatch_size = minibatch_size


    def load_data(self, mnist_directory):
        mndata = MNIST(mnist_directory)
        tr_data, tr_labels = mndata.load_training()
        te_data, te_labels = mndata.load_testing()
        train_temp = np.array(tr_data) / 175.0

        self.train_data = train_temp - 1.0
        self.train_labels = np.array(tr_labels)
        test_temp = np.array(te_data)

        self.test_data = test_temp
        self.test_labels = np.array(te_labels)

        self.possible_categories = list(set(self.train_labels))
        self.possible_categories.sort()
        self.num_categories = len(self.possible_categories)
        print('Loaded data...')

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
        print('Subsetted data.')


    def update_learning_rate(self, iteration):
        if self.lr_dampener is not None:
            return self.lr0 / (1.0 + iteration / self.lr_dampener)
        else:
            return self.lr0

    def assign_holdout(self, percent):
        percent /= 100.0
        num_held = int(self.train_data.shape[0] * percent)
        self.train_data = self.train_data[:-num_held]
        self.train_labels = self.train_labels[:-num_held]
        self.holdout_data = self.train_data[-num_held:]
        self.holdout_labels = self.train_labels[-num_held:]
        print('Assigned holdout data')

    def get_next_mini_batch(self, shuffle=False):
        if not shuffle:
            if (self.minibatch_index + 1) * self.minibatch_size > self.train_data.shape[0]:
                self.minibatch_index = 0

            td = self.train_data[self.minibatch_index * self.minibatch_size : (self.minibatch_index + 1) * self.minibatch_size]
            tl = self.train_labels[self.minibatch_index * self.minibatch_size : (self.minibatch_index + 1) * self.minibatch_size]
            self.minibatch_index += 1
        else:
            indices = random.sample(xrange(self.train_data.shape[0]), self.minibatch_size)
            td = self.train_data[indices]
            tl = self.train_labels[indices]

        return td, tl

    def train(self, iterations, num_hidden, reset_batches=True, epochs_per_batch=1):
        self.train_loss_log = []
        self.train_classification_log = []
        if reset_batches:
            self.minibatch_index = 0

        eta = self.lr0

        d, l = self.get_next_mini_batch()
        self.sigmoid_layer = SigmoidLayer(self.train_data.shape[1] + 1, num_hidden)
        self.softmax_layer = SoftmaxLayer(num_hidden + 1, self.num_categories)

        for iteration in xrange(iterations):

            for i in xrange(epochs_per_batch):
                intermediate = self.sigmoid_layer.forward_prop(d)

                preds = self.softmax_layer.forward_prop(intermediate)

                self.softmax_layer.update_weights(eta, l, preds)
                self.sigmoid_layer.update_weights(self.softmax_layer, eta, l, preds)

            eta = self.update_learning_rate(iteration)
            self.train_loss_log.append(norm_loss_function(
                             softmax(
                                np.dot(self.softmax_layer.last_input, self.softmax_layer.weights)), l))
            self.train_classification_log.append(evaluate(preds, l))
            d, l = self.get_next_mini_batch(shuffle=True)
