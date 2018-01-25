from network_runner import *
from hidden_to_output import *
from visible_to_hidden import *
from helper import *
import pylab as plt

nr = NetworkRunner('mnist', lr_dampener=50)
d, l = nr.get_next_mini_batch()

SL = SigmoidLayer(785, 64)
SML = SoftmaxLayer(65, 10, l)
errs = []
preds = []
eta = nr.lr0
iters = 100
for i in xrange(iters):
    d, l = nr.get_next_mini_batch()
    for i in xrange(2):
        out1 = SL.forward_prop(d)
        out2 = SML.forward_prop(out1)
        w1 = SML.weights
        SML.update_weights(eta)
        SL.update_weights(SML, eta)
    errs.append(norm_loss_function(
                     softmax(
                        np.dot(SML.last_input, SML.weights)),
                     l
                     ))
    eta = nr.update_learning_rate(i)

    preds.append(evaluate(out2, l))

plt.plot(xrange(iters), errs)
plt.show()

plt.plot(xrange(iters), preds)
plt.show()
