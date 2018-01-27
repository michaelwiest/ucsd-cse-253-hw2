from __future__ import print_function
from helper import *
from random import randrange

class SigmoidLayer(object):
    def __init__(self, num_in, num_out):
        # num out is 64
        self.num_out = num_out
        # self.eta = eta
        # num in is 785
        self.num_in = num_in
        self.weights = None

    def set_random_weights(self):
        scale = 2
        print('Initialized weights of shape: [{}, {}]'.format(self.num_in,
                                                              self.num_out))
        self.weights = (np.random.rand(self.num_in, self.num_out) * scale) - (scale - 1)


    def forward_prop(self, input_data, add_bias=True, save_input=True):
        if self.weights is None:
            self.set_random_weights()
        if add_bias:
            input_data = prefix_ones(input_data)
        if save_input:
            self.last_input = input_data
        # print(self.last_input)
        # print(self.weights)
        return sigma(np.dot(input_data, self.weights))

    def get_delta_j(self, SoftmaxLayer, labels, predictions):
        aj = np.dot(self.last_input, self.weights)
        delta_k = SoftmaxLayer.get_delta_k(labels, predictions)
        wjk = SoftmaxLayer.prev_weights

        # k = 10; wjk = 65*10

        # Ignore the first row because it is bias.
        return sigma_d(aj) * (np.dot(delta_k, np.transpose(wjk)))[:, 1:]

    def update_weights(self, SoftmaxLayer, eta, labels, predictions):
        delta_j = self.get_delta_j(SoftmaxLayer, predictions, labels)
        self.weights = self.weights + eta * np.dot(np.transpose(self.last_input), delta_j)
        return self.weights

    def grad(self, SoftmaxLayer, labels, predictions):
        delta_j = self.get_delta_j(SoftmaxLayer, predictions, labels)
        return np.dot(np.transpose(self.last_input), delta_j)
