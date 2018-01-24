from helper import *


class HiddenToOutput(object):
    def __init__(self, num_in, num_out, eta, labels):
        self._setup()
        self.eta = eta
        self.num_in = num_in
        self.num_out = num_out
        self.labels = labels

    def _add_bias_to_weights(self, weights):
        return np.concatenate(
                        (np.ones((temp.shape[1], 1)),
                         weights), axis=0
                        )
    def _setup(self):
        temp = np.random.rand(self.num_in, self.num_out)
        # Load in the weights with a bias of 1
        self.weights = self._add_bias_to_weights(temp)

    def forward_prop(self, input_data):
        self.last_input = input_data
        return softmax(np.dot(input_data, self.weights))

    def get_delta_k(self):
        pred = self.forward_prop(self.last_input)
        labels = get_one_hot(self.labels)
        return (labels - pred)

    def update_weights(self):
        g = softmax(self.forward_prop(input_data))
        delta_k = self.get_delta_k()
        self.weights = self.weights + self.eta * delta_k * norm_loss_function(self.weights, self.last_input, self.labels)

    # def get_approx_gradient(self, input_data, weights, labels):
    #     norm_loss_function(weights, last_input, labels)


    # def approx_gradient
