from __future__ import print_function
from neural_network import *
from sigmoid_layer import *
from softmax_layer import *
from helper import *
import pylab as plt

nn = NeuralNetwork('mnist',
                   lr_dampener=1000
                   # , magic_sigma=True
                   # , alpha=0.1
                   )
nn.assign_holdout(16.6)
d, l = nn.get_next_mini_batch()


iters = 3000
num_hidden = [64]
nn.train(iters, num_hidden)
nn.set_to_optimal_weights()
nn.forward_prop(nn.test_data)
optimal_error = evaluate(nn.layers[-1].last_output, nn.test_labels)
optimal_loss = norm_loss_function(nn.layers[-1].last_output, nn.test_labels)


plt.plot(nn.iterations, nn.train_loss_log, alpha=0.7, label='Train')
plt.plot(nn.iterations, nn.holdout_loss_log, alpha=0.7, label='Holdout')
plt.plot(nn.iterations, nn.test_loss_log, alpha=0.7, label='Test')
plt.xlabel('Iterations')
plt.ylabel('Normalized Loss')
plt.title('Progression of Normalized Loss Over Training')
plt.annotate('Optimal Test Loss: {}'.format(optimal_loss),
             horizontalalignment='right',
             fontsize=10,
             xy = (iters * 0.92, -0.03))
plt.legend(loc='upper right')
plt.show()

plt.plot(nn.iterations, nn.train_classification_log, alpha=0.7, label='Train')
plt.plot(nn.iterations, nn.holdout_classification_log, alpha=0.7, label='Holdout')
plt.plot(nn.iterations, nn.test_classification_log, alpha=0.7, label='Test')
plt.xlabel('Iterations')
plt.ylabel('Percent Accuracy of Classification')
plt.title('Progression of Classification Accuracy Over Training')
plt.annotate('Optimal Test Error: {}%'.format(optimal_error),
             horizontalalignment='right',
             fontsize=10,
             xy = (iters * 0.92, 30))
plt.legend(loc='lower right')
plt.show()
