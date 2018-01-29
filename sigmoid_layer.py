from __future__ import print_function
from helper import *

class SigmoidLayer(object):
    def __init__(self, num_in, num_out):
        # num out is 64
        self.num_out = num_out
        # self.eta = eta
        # num in is 785
        self.num_in = num_in
        self.weights = None

    def set_random_weights(self):
        print('Initialized weights of shape: [{}, {}]'.format(self.num_in,
                                                              self.num_out))
        dev = 1.0 / (np.sqrt(self.num_in))
        self.prev_weights = np.zeros((self.num_in, self.num_out))
        self.weights = (np.random.normal(0, dev, (self.num_in, self.num_out)))


    def forward_prop(self, input_data, add_bias=True, save_input=True,
                     save_output=True):
        if self.weights is None:
            self.set_random_weights()
        if add_bias:
            input_data = prefix_ones(input_data)

        if save_input:
            self.last_input = input_data
        output = sigma(np.dot(input_data, self.weights))

        if save_output:
            self.last_output = output

        return output

    def get_delta(self, future_delta, future_weights):
        aj = np.dot(self.last_input, self.weights)
        self.delta = sigma_d(aj) * (np.dot(future_delta, np.transpose(future_weights)))[:, 1:]
        return self.delta

    # labels and predictions aren't used. Just for ease of calling.
    def update_weights(self, future_delta, eta, labels, predictions, future_weights):
        delta = self.get_delta(future_delta, future_weights)
        self.prev_weights = self.weights
        self.weights = self.weights + eta * np.dot(np.transpose(self.last_input), delta)
        return self.weights
