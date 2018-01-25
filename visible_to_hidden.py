from helper import *

class SigmoidLayer(object):
    def __init__(self, num_in, num_out, eta, labels_in):
        # num out is 64
        self.num_out = num_out
        self.eta = eta
        # num in is 784
        self.num_in = num_in
        self.labels_in = labels_in
        self._setup()

    def _setup(self):
        temp = np.random.rand(self.num_in, self.num_out)
        # add one to weights
        self.weights = np.concatenate(temp,np.ones((self.num_in,1)),0)


    def forward_prop(self, input_data):
        self.last_input = input_data
        return sigma(np.dot(input_data,self.weights))

    def get_delta_j(self):
        aj = np.dot(self.last_input, self.weights)
        delta_k = LinearLayer.get_delta_k()
        wjk = LinearLayer.get_output_weights()

        # k = 10; wjk = 65*10
        return sigma_d(aj)*np.dot(delta_k,np.transpose(wjk))

    def update_weights_j(self, input_data):
        delta_j = self.get_delta_j()
        z_j = self.forward_prop(input_data)
        self.weights = self.weights + eta*delta_j*z_j


#    def back_prop(self, output_grad):
#        # get_delta_k and get_output_weights come from michaels code
 #       output_grad = np.dot(get_delta_k, get_output_weights)
#        return output_grad * sigma_d(self.last_input)
