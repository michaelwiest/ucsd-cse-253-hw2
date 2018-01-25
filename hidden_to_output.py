from helper import *


class SoftmaxLayer(object):
    def __init__(self, num_in, num_out, labels):
        # self._setup()
        # num_in should be 65
        self.num_in = num_in
        # num_out should be 10
        self.num_out = num_out
        self.labels = labels
        self.weights = None

    def _add_bias_to_weights(self, weights):
        return np.concatenate(
                        (np.ones((temp.shape[1], 1)),
                         weights), axis=0
                        )
    def set_random_weights(self):
        print 'Initialized weights of shape: [{}, {}]'.format(self.num_in,
                                                              self.num_out)
        self.weights = np.random.rand(self.num_in, self.num_out)


    def forward_prop(self, input_data, add_bias=True, save_input=True):
        if self.weights is None:
            self.set_random_weights()
        if add_bias:
            input_data = prefix_ones(input_data)
        if save_input:
            self.last_input = input_data
        return softmax(np.dot(input_data, self.weights))

    def get_delta_k(self):
        pred = self.forward_prop(self.last_input, add_bias=False)
        labels = get_one_hot(self.labels)
        return (labels - pred)

    def update_weights(self, eta):
        delta_k = self.get_delta_k()
        self.weights = self.weights + eta * np.dot(np.transpose(self.last_input), delta_k)
        return self.weights
