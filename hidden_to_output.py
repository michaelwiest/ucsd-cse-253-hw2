from helper import *


class HiddenToOutput(object):
    def __init__(self, num_in, num_out, eta, labels):
        self._setup()
        self.eta = eta
        self.num_in = num_in
        self.num_out = num_out
        self.labels = labels

    def _setup(self):
        temp = np.random.rand(self.num_in, self.num_out)
        # Load in the weights with a bias of 1
        self.weights = np.concatenate(
                        (np.ones((temp.shape[1], 1)),
                         temp), axis=0
                        )

    def forward_prop(self, input_data):
        self.last_input = input_data
        return np.dot(input_data, self.weights)

    # def back_prop(self, output_grad):
    #     return np.dot(output_grad, np.transpose(self.W))

    def get_delta_k(self):
        pred = self.forward_prop(last_input)
        labels = get_one_hot(self.labels)
        return (labels - pred)

    def update_weights(self):
        g = softmax(self.forward_prop(input_data))
        delta_k = self.get_delta_k
        self.weights = self.weights + self.eta * delta_k * g

    def approx_gradient
