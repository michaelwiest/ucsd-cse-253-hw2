from __future__ import print_function
from network_runner import *
from hidden_to_output import *
from visible_to_hidden import *
from helper import *
import pylab as plt

nn = NeuralNetwork('mnist', lr_dampener=10000)
d, l = nn.get_next_mini_batch()


iters = 10000
num_hidden = 64
nn.train(iters, num_hidden)

plt.plot(xrange(iters), nn.train_loss_log)
plt.show()

plt.plot(xrange(iters), nn.train_classification_log)
plt.show()
