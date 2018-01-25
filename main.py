from network_runner import *
from hidden_to_output import *
from visible_to_hidden import *
from helper import *


nr = NetworkRunner('mnist')
d, l = nr.get_next_mini_batch()
# a = nr.get_next_mini_batch()
# print get_one_hot(a[1])
# a = nr.get_next_mini_batch()
# print get_one_hot(a[1])

SL = SigmoidLayer(785, 64)
SML = SoftmaxLayer(65, 10, l)
out = SL.forward_prop(d)
out = SML.forward_prop(out)
SML.update_weights(nr.lr0)
