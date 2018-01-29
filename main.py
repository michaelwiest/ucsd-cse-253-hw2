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
a = plt.plot(xrange(30),nr.grad_diff_f[:,0],label='dE/dwjk')
plt.ylabel('Gradient Difference')
plt.xlabel('Iterations')
b = plt.plot(xrange(30),nr.grad_diff_f[:,1],label='dE/dwij')
c = plt.plot(xrange(30),[10**-4]*30,label='epsilon*10^-2')
plt.legend()
plt.show()
