from __future__ import print_function
from neural_network import *
from sigmoid_layer import *
from softmax_layer import *
from helper import *
import pylab as plt

nn = NeuralNetwork('mnist', lr_dampener=10000)
nn.assign_holdout(10)
d, l = nn.get_next_mini_batch()


iters = 1000
num_hidden = [64]
nn.train(iters, num_hidden)


plt.plot(xrange(iters), nn.train_loss_log, alpha=0.7, label='Train')
plt.plot(xrange(iters), nn.holdout_loss_log, alpha=0.7, label='Holdout')

plt.legend(loc='upper right')
plt.show()

plt.plot(xrange(iters), nn.train_classification_log, alpha=0.7, label='Train')
plt.plot(xrange(iters), nn.holdout_classification_log, alpha=0.7, label='Holdout')
plt.legend(loc='lower right')
plt.show()
