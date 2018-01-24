from helper import *

class SigmoidLayer(object):
    def __init__(self, input_shape, num_out, learn_rate):
        self.input_shape = input_shape
        self.num_out = num_out
        self._setup()

    def _setup(self):
        self.W = np.random.rand(self.input_shape[1], self.num_out)
        self.b = np.zeros(self.num_out)


    def forward_prop(self, input_data):
        self.last_input = input_data
        return sigmoid(input_data)

    def back_prop(self, output_grad):
        return output_grad * sigma_d(self.last_input)
