from __future__ import print_function
from network_runner import *
from hidden_to_output import *
from visible_to_hidden import *
from helper import *
import pylab as plt
import pdb

nr = NetworkRunner('mnist', lr_dampener=10000)
d, l = nr.get_next_mini_batch()


iters = 1000
num_hidden = 64
nr.train(iters, num_hidden)


plt.plot(xrange(iters), nr.train_loss_log)
plt.show()

plt.plot(xrange(iters), nr.train_classification_log)
plt.show()

plt.figure()
plt.plot(xrange(10),nr.grad_diff_f[:,0])
plt.ylabel('Gradient Difference')
plt.xlabel('Iterations')
plt.plot(xrange(10),nr.grad_diff_f[:,1])
plt.plot(xrange(10),[10**-2]*10)
# plt.legend(('dE/dwjk','dE/dwij','epsilon'))
plt.show()