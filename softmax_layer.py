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
    def update_weights(self, future_delta, eta, labels, predictions, future_weights):
        delta = self.get_delta(predictions, labels)
        self.prev_weights = self.weights
        self.weights = self.weights + eta * np.dot(np.transpose(self.last_input), delta)
        return self.weights

    def grad(self, labels, predictions, data_ind):
        delta_k = self.get_delta_k(predictions, labels)
        delta_k = delta_k.reshape((1,delta_k.shape[0]))
        last_in = self.last_input[data_ind]
        last_in = last_in.reshape((last_in.shape[0],1))
        return np.dot(last_in, delta_k)
