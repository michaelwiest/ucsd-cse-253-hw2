from network_runner import *
from hidden_to_output import *
from visible_to_hidden import *
from helper import *
import pylab as plt

nr = NetworkRunner('mnist')
d, l = nr.get_next_mini_batch()


iters = 2000
num_hidden = 64
nr.train(iters, num_hidden)

# plt.plot(xrange(iters), errs)
# plt.show()
#
# plt.plot(xrange(iters), preds)
# plt.show()
