from network_runner import *
from hidden_to_output import *
from visible_to_hidden import *
from helper import *


nr = NetworkRunner('mnist')
a = nr.get_next_mini_batch()
print get_one_hot(a[1])
a = nr.get_next_mini_batch()
print get_one_hot(a[1])
a = nr.get_next_mini_batch()
print get_one_hot(a[1])
