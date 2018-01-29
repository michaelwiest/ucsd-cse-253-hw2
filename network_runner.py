from __future__ import print_function
from mnist import MNIST
import numpy as np
import pylab as plt
from helper import *
import random
from visible_to_hidden import *
from hidden_to_output import *
import time
import pdb

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

    def compare_gradients(self, epsilon, dat, lab, weights_sigmoid, weights_softmax):
        intermediate = self.sigmoid_layer.forward_prop(dat)
        preds = self.softmax_layer.forward_prop(intermediate)

        shape_perturb = weights_softmax.shape
        data_ind = random.randrange(0,self.minibatch_size) # from 1 to 128
        perturb_ind = [random.randrange(1,shape_perturb[0]),random.randrange(0,shape_perturb[1])]
        # ^ from [65x10], so [between 1 and 64][between 0 and 9]

        # for wjk
        # only find gradient for one data point
        grad_real = self.softmax_layer.grad([lab[data_ind]],preds[data_ind], data_ind) # 65x10
        # grad_real = grad_real[perturb_ind[0]][perturb_ind[1]] # 1
        grad_real = grad_real[perturb_ind[0]]

        g_diff=[]
        num_loss=[]
        # perturb weight with index jk
        perturb(weights_softmax, epsilon, perturb_ind)
        preds = self.softmax_layer.forward_prop(intermediate)
        # only find cross entropy loss for one data point
        cross_ent_l = cross_ent_loss(preds[data_ind],[lab[data_ind]]) # get 1x10
        # cross_ent_l = cross_ent_l[perturb_ind[1]]

        perturb(weights_softmax, -2*epsilon, perturb_ind)
        preds = self.softmax_layer.forward_prop(intermediate)
        cross_ent_u = cross_ent_loss(preds[data_ind],[lab[data_ind]]) # get 1x10
        # cross_ent_u = cross_ent_u[perturb_ind[1]]

        perturb(weights_softmax, epsilon, perturb_ind)
        grad = abs(cross_ent_u - cross_ent_l)/float(epsilon)
        g_diff.append(np.sqrt(np.sum(grad - grad_real)**2))


        # if perturb_wij:
        # grad_real = self.sigmoid_layer.grad(self.softmax_layer,lab,preds, data_ind)
        grad_real = self.softmax_layer.grad([lab[data_ind]],preds[data_ind], data_ind) # 65x10
        grad_ind = random.randrange(1,65)

        shape_perturb = weights_sigmoid.shape
        perturb_ind = [random.randrange(0,shape_perturb[0]),random.randrange(0,shape_perturb[1])]
        grad_real = grad_real[grad_ind] # now this is 1x10
        # this is one number, corresponding to i and j

        perturb(weights_sigmoid, epsilon, perturb_ind)
        intermediate = self.sigmoid_layer.forward_prop(dat)
        preds = self.softmax_layer.forward_prop(intermediate)
        cross_ent_l = cross_ent_loss(preds[data_ind],[lab[data_ind]]) # get 1x10
        cross_ent_l = cross_ent_l

        perturb(weights_sigmoid, -2*epsilon, perturb_ind)
        intermediate = self.sigmoid_layer.forward_prop(dat)
        preds = self.softmax_layer.forward_prop(intermediate)
        cross_ent_u = cross_ent_loss(preds[data_ind],[lab[data_ind]]) # get 1x10
        cross_ent_u = cross_ent_u
        grad = (cross_ent_u - cross_ent_l)/epsilon
        # but this is a 1x 10 vector--confusion....
        perturb(weights_sigmoid, epsilon, perturb_ind)

        g_diff.append(np.sqrt(np.sum(grad - grad_real)**2))
        return g_diff

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
        self.grad_diff_f = np.empty((0,2))
        for i in range(10):
            grad_diff = self.compare_gradients(10**-2, d, l, self.sigmoid_layer.weights, self.softmax_layer.weights)
            self.grad_diff_f = np.vstack((self.grad_diff_f, grad_diff))

