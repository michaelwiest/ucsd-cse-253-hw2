from helper import *

class HiddenToVisible(object):
    def __init__(self, input_shape, num_out):
        self.input_shape = input_shape
        self.num_out = num_out
        self._setup()

    def _setup(self):
        self.W = np.random.rand(self.input_shape[1], self.num_out)
        self.b = np.zeros(self.num_out)


    def forward_prop(self, input_data):
        self.last_input = input_data
        return np.dot(input_data, self.W)

    def back_prop(self, output_grad):
        return np.dot(output_grad, np.transpose(self.W))
