from __future__ import print_function
from mnist import MNIST
import numpy as np
import pylab as plt
from helper import *
import random
from sigmoid_layer import *
from softmax_layer import *
import time

np.set_printoptions(threshold=np.nan)

class NeuralNetwork(object):
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
            self.lr0 = 500.0 / self.train_data.shape[0]
        else:
            self.lr0 = lr0
        self.minibatch_index = 0
        self.minibatch_size = minibatch_size

        self.forward_props = []

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

    def log(self, labels):
        self.train_loss_log.append(norm_loss_function(
                         self.layers[-1].last_output, labels))
        self.train_classification_log.append(evaluate(self.layers[-1].last_output, labels))

        if self.holdout_data is not None:
            # Do forward prop with the holdout data.
            self.__forward_prop(self.holdout_data)
            self.holdout_loss_log.append(norm_loss_function(
                             self.layers[-1].last_output, self.holdout_labels))
            self.holdout_classification_log.append(evaluate(self.layers[-1].last_output, self.holdout_labels))


    def __build_layers(self, hidden_layers):
        self.layers = []
        self.layers.append(SigmoidLayer(self.train_data.shape[1] + 1, hidden_layers[0]))
        for i in xrange(len(hidden_layers) - 1):
            self.layers.append(SigmoidLayer(hidden_layers[i] + 1, hidden_layers[i + 1]))
        self.layers.append(SoftmaxLayer(hidden_layers[-1] + 1, self.num_categories))


    def __forward_prop(self, data, save=True):
        temp = data
        for layer in self.layers:
            temp = layer.forward_prop(temp, save_input=save, save_output=save)
            # self.forward_props.append(temp)

    def __back_prop(self, labels, eta):
        future_delta = None
        future_weights = None
        for layer in reversed(self.layers):
            layer.update_weights(future_delta, eta, labels, layer.last_output,
                                 future_weights)
            future_delta = layer.delta
            future_weights = layer.prev_weights


    def train(self, iterations, hidden_layers, reset_batches=True, epochs_per_batch=1):
        # Reset logs.
        self.train_loss_log = []
        self.train_classification_log = []
        self.holdout_loss_log = []
        self.holdout_classification_log = []
        if reset_batches:
            self.minibatch_index = 0

        eta = self.lr0
        data, labels = self.get_next_mini_batch()
        self.__build_layers(hidden_layers)

        for iteration in xrange(iterations):
            self.__forward_prop(data)
            self.__back_prop(labels, eta)
            self.log(labels)

            eta = self.update_learning_rate(iteration)
            data, labels = self.get_next_mini_batch(shuffle=True)







pass