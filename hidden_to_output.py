from __future__ import print_function
from helper import *


class SoftmaxLayer(object):
    def __init__(self, num_in, num_out):
        # self._setup()
        # num_in should be 65
        self.num_in = num_in
        # num_out should be 10
        self.num_out = num_out
        # self.labels = labels
        self.weights = None

    def set_random_weights(self):
        print('Initialized weights of shape: [{}, {}]'.format(self.num_in,
                                                              self.num_out))
        dev = 1.0 / (np.sqrt(self.num_in))
        self.prev_weights = np.zeros((self.num_in, self.num_out))
        self.weights = (np.random.normal(0, dev, (self.num_in, self.num_out)))

    def forward_prop(self, input_data, add_bias=True, save_input=True):
        if self.weights is None:
            self.set_random_weights()

        if add_bias:
            input_data = prefix_ones(input_data)

        if save_input:
            self.last_input = input_data
        return softmax(np.dot(input_data, self.weights))

    def get_delta_k(self, predictions, labels):
        labels = get_one_hot(labels)
        return (labels - predictions)

    def update_weights(self, eta, labels, predictions):
        delta_k = self.get_delta_k(predictions, labels)
        self.prev_weights = self.weights
        self.weights = self.weights + eta * np.dot(np.transpose(self.last_input), delta_k)
        return self.weights
