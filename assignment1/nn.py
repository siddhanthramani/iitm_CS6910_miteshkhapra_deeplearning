import numpy as np

class neural_network():
    def __init__(self, num_neurons_dict, activation_dict, activation_type="constant", nn_init = np.random.rand(), nn_init_random_max=1):
        self.num_neurons_dict = num_neurons_dict
        self.activation_dict = activation_dict
        self.activation_type = activation_type

        self.layers = list(num_neurons_dict.keys())
        self.W = dict()
        self.b = dict()

        for layer, num_neurons in sorted(num_neurons_dict.items()):
            if layer == 0:
                continue
            self.W[layer] = nn_init(num_neurons_dict[layer], num_neurons) * nn_init_random_max
            self.b[layer] = nn_init(num_neurons) * nn_init_random_max
    
    def __activation_function(self, layer, input):
        if self.activation_type == "constant":
            if layer == max(self.layers):
                activation = self.activation_dict[1]
            else:
                activation = self.activation_dict[0]
        else:
            activation = self.activation_dict[layer]

        if activation == "linear":
            return input
        elif activation == "logistic":
            return (1/ (1 + np.exp(-input)))

    def forward_pass(self, x_i):
        for layer in self.layers:
            x_i = np.dot(np.transpose(self.W[layer]), x_i) + self.b[layer]
            x_i = self.__activation_function(x_i)
        
        return x_i



        
    