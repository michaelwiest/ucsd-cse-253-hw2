from __future__ import print_function
from network_runner import *
from hidden_to_output import *
from visible_to_hidden import *
from helper import *
import pylab as plt

nr = NetworkRunner('mnist')
d, l = nr.get_next_mini_batch()


iters = 1500
num_hidden = 64
nr.train(iters, num_hidden)

plt.plot(xrange(iters), nr.train_loss_log)
plt.show()

plt.plot(xrange(iters), nr.train_classification_log)
plt.show()
