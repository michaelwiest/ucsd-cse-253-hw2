from __future__ import print_function
from neural_network import *
from sigmoid_layer import *
from softmax_layer import *
from helper import *
import pylab as plt

nn = NeuralNetwork('mnist',
                   lr_dampener=2000,
                   magic_sigma=False
                   # , alpha=0.1
                   )
nn.assign_holdout(10)
d, l = nn.get_next_mini_batch()


iters = 3000
num_hidden = [64]
nn.train(iters, num_hidden)


plt.plot(nn.iterations, nn.train_loss_log, alpha=0.7, label='Train')
plt.plot(nn.iterations, nn.holdout_loss_log, alpha=0.7, label='Holdout')
plt.plot(nn.iterations, nn.test_loss_log, alpha=0.7, label='Test')

plt.legend(loc='upper right')
plt.show()

plt.plot(nn.iterations, nn.train_classification_log, alpha=0.7, label='Train')
plt.plot(nn.iterations, nn.holdout_classification_log, alpha=0.7, label='Holdout')
plt.plot(nn.iterations, nn.test_classification_log, alpha=0.7, label='Test')
plt.legend(loc='lower right')
plt.show()
