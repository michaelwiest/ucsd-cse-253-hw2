from __future__ import print_function
from helper import *


class SoftmaxLayer(object):
    def __init__(self, num_in, num_out):
        self.num_in = num_in
        self.num_out = num_out
        self.weights = None

    def set_random_weights(self):
        print('Initialized weights of shape: [{}, {}]'.format(self.num_in,
                                                              self.num_out))
        dev = 1.0 / (np.sqrt(self.num_in))
        self.prev_weights = np.zeros((self.num_in, self.num_out))
        self.weight_delta = np.zeros((self.num_in, self.num_out))
        self.weights = (np.random.normal(0, dev, (self.num_in, self.num_out)))

    def forward_prop(self, input_data, add_bias=True, save_input=True,
                     save_output=True):
        if self.weights is None:
            self.set_random_weights()

        if add_bias:
            input_data = prefix_ones(input_data)

        if save_input:
            self.last_input = input_data

        output = softmax(np.dot(input_data, self.weights))

        if save_output:
            self.last_output = output

        return output

    def get_delta(self, predictions, labels):
        labels = get_one_hot(labels)
        self.delta = (labels - predictions)
        return self.delta

    # future_delta isn't used. It's just for ease of calling the function.
    def update_weights(self, future_delta, eta, labels, predictions,
                       future_weights, alpha=None):
        delta = self.get_delta(predictions, labels)
        self.prev_weights = self.weights
        if alpha is not None:
            self.weights = self.weights + (alpha * self.weight_delta + eta * np.dot(np.transpose(self.last_input), delta))
        else:
            self.weights = self.weights + eta * np.dot(np.transpose(self.last_input), delta)
        self.weight_delta = self.weights - self.prev_weights
        return self.weights
