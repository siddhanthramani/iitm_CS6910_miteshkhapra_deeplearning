import numpy as np

class neural_network():
    def __init__(self, num_neurons_dict, activation_dict, activation_type="constant", nn_init = np.random.rand, nn_init_random_max=1):
        self.num_neurons_dict = num_neurons_dict
        self.activation_dict = activation_dict
        self.activation_type = activation_type

        self.layers = list(range(1, len(num_neurons_dict.keys())))
        self.W = dict()
        self.b = dict()

        for layer, num_neurons in sorted(num_neurons_dict.items()):
            if layer == 0:
                continue
            self.W[layer] = nn_init(num_neurons_dict[layer-1], num_neurons) * nn_init_random_max
            self.b[layer] = nn_init(num_neurons, 1) * nn_init_random_max
    
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

    def forward_pass(self, x_i, print_transforms=1):
        for layer in self.layers:
            ip_shape = x_i.shape
            x_i = np.add(np.dot(np.transpose(self.W[layer]), x_i), self.b[layer])
            x_i = self.__activation_function(layer, x_i)
            op_shape = x_i.shape
            if print_transforms:
                print("Layer : {} - Input_Shape : {}, W shape : {}, b shape : {}, and output_shape : {}"
                      .format(layer, ip_shape, self.W[layer].shape, self.b[layer].shape, op_shape))
        return x_i



        
    