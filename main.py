from network_runner import *
from hidden_to_output import *
from visible_to_hidden import *

nr = NetworkRunner('mnist')
print nr.get_next_mini_batch()
