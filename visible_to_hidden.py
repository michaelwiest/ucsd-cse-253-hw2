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
        return sigma(np.dot(input_data, self.weights))

    def get_delta_j(self, SoftmaxLayer):
        aj = np.dot(self.last_input, self.weights)
        delta_k = SoftmaxLayer.get_delta_k()
        wjk = SoftmaxLayer.weights

        # k = 10; wjk = 65*10
        return sigma_d(aj) * np.dot(delta_k, np.transpose(wjk))

    def update_weights(self, SoftmaxLayer, eta):
        delta_j = self.get_delta_j(SoftmaxLayer)
        z_j = self.forward_prop(self.last_input, add_bias=False)
        self.weights = self.weights + eta * delta_j * z_j * self.last_input
        return self.weights


#    def back_prop(self, output_grad):
#        # get_delta_k and get_output_weights come from michaels code
 #       output_grad = np.dot(get_delta_k, get_output_weights)
#        return output_grad * sigma_d(self.last_input)
