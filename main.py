from __future__ import print_function
from neural_network import *
from sigmoid_layer import *
from softmax_layer import *
from helper import *
import pylab as plt
import pdb

nn = NeuralNetwork('mnist', lr_dampener=10)
nn.assign_holdout(10)
d, l = nn.get_next_mini_batch()


iters = 1000
num_hidden = [64]
# nn.train(iters, num_hidden)


# plt.plot(xrange(iters), nn.train_loss_log, alpha=0.7, label='Train')
# plt.plot(xrange(iters), nn.holdout_loss_log, alpha=0.7, label='Holdout')

# plt.legend(loc='upper right')
# plt.show()

# plt.plot(xrange(iters), nn.train_classification_log, alpha=0.7, label='Train')
# plt.plot(xrange(iters), nn.holdout_classification_log, alpha=0.7, label='Holdout')
# plt.legend(loc='lower right')
# plt.show()


iters = 30
epsilon = 10**-2
pdb.set_trace()
nn.evaluate_gradient(iters, epsilon)

plt.figure()
plt.plot(xrange(iters),nn.gd_final[0,:],label='Change wjk')
plt.ylabel('Gradient Difference')
plt.xlabel('Iterations')
plt.plot(xrange(iters),nn.gd_final[1,:], label = 'Change wjk bias')
plt.plot(xrange(iters),[epsilon**-2]*iters, 'epsilon*10^-2')
plt.plot(xrange(iters),nn.gd_final[2,:],label='Change wij')
plt.plot(xrange(iters),nn.gd_final[3,:],label='Change wij bias')
plt.legend()
plt.show()
