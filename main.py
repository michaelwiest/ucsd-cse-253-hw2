from __future__ import print_function
from neural_network import *
from sigmoid_layer import *
from softmax_layer import *
from helper import *
import pylab as plt

nn = NeuralNetwork('mnist', lr_dampener=10000)
d, l = nn.get_next_mini_batch()


iters = 1000
num_hidden = [64, 64]
nn.train(iters, num_hidden)

plt.plot(xrange(iters), nn.train_loss_log)
plt.show()

plt.plot(xrange(iters), nn.train_classification_log)
plt.show()
