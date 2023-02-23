import numpy as np

class neural_network():
    
    def __init__(self, num_neurons_dict, activation_dict, activation_type="constant", loss_function = "cross_entropy", nn_init = np.random.rand, nn_init_random_max=1):
        self.num_neurons_dict = num_neurons_dict
        self.activation_dict = activation_dict
        self.activation_type = activation_type
        self.loss_function = loss_function

        self.layers = list(range(1, len(num_neurons_dict.keys())))
        self.max_layer = max(self.layers)
        self.W = dict()
        self.b = dict()

        self.__gradient_descent_type_dict = {
            "sgd" : self.__sgd_update
        }
        self.grad_loss_update_first_time = 0
        self.param_dict = dict()

        for layer, num_neurons in sorted(num_neurons_dict.items()):
            if layer == 0:
                continue
            self.W[layer] = nn_init(num_neurons_dict[layer-1], num_neurons) * nn_init_random_max
            self.b[layer] = nn_init(num_neurons, 1) * nn_init_random_max

        print("hi")
            
    def __get_activation_function(self, layer):
        if self.activation_type == "constant":
            if layer == max(self.layers):
                activation = self.activation_dict[1]
            else:
                activation = self.activation_dict[0]
        else:
            activation = self.activation_dict[layer]
        return activation 
    

    def __activation_function(self, input, activation):
        if activation == "linear":
            return input
        elif activation == "logistic":
            return (1/ (1 + np.exp(-input)))
        elif activation == "tanh":
            return np.tanh(input)
        elif activation == "softmax":
            return (np.exp(-input)/ np.exp(-input).sum())


    def __forward_pass(self, x_i):
        a_i = dict()
        h_i = dict()
        h_i[0] = x_i
        for layer in self.layers:
            a_i[layer] = np.add(np.dot(np.transpose(self.W[layer]), h_i[layer-1]), self.b[layer])
            activation = self.__get_activation_function(layer)
            h_i[layer] = self.__activation_function(a_i[layer], activation)

        return a_i, h_i
    

    def __grad_activation(self, ak, activation):
        if activation == "logistic":
            inter = self.__activation_function(ak, activation)
            return (inter/(1-inter))
        elif activation == "tanh":
            inter = self.__activation_function(ak, activation)
            return (1 - np.square(inter))
            
    def __get_loss(self, a_i, h_i, y):
        if self.loss_function == "cross_entropy":
            if not self.activation_dict[self.max_layer] == "softmax":
                raise Exception("Cross entropy can be used only if the final activation function is softmax")

            loss = -np.sum(np.multiply(y, np.log(h_i[self.max_layer])))
        return loss
    
    def __common_backward_pass(self, a_i, h_i, y):
        if self.loss_function == "cross_entropy":
            if not self.activation_dict[self.max_layer] == "softmax":
                raise Exception("Cross entropy can be used only if the final activation function is softmax")

            grad_loss_h = dict()
            grad_loss_W = dict()
            grad_loss_b = dict()

            grad_loss_a = dict()
            grad_loss_a[self.max_layer] = -(y-h_i[self.max_layer])

            for layer in self.layers:
                grad_layer = self.max_layer - layer + 1
                grad_loss_W[grad_layer] = np.dot(h_i[grad_layer-1], np.transpose(grad_loss_a[grad_layer]))
                grad_loss_b[grad_layer] = grad_loss_a[grad_layer]
                if grad_layer == 1:
                    break
                grad_loss_h[grad_layer-1] = np.dot(self.W[grad_layer], grad_loss_a[grad_layer])
                grad_loss_a[grad_layer-1] = np.multiply(grad_loss_h[grad_layer-1], 
                                                self.__grad_activation(a_i[grad_layer-1], 
                                                                        self.__get_activation_function(grad_layer)))

        elif self.loss_function == "squared_error":
            loss = np.sum(np.square(h_i[max(self.layers)] - y))
            grad_loss_h = dict()
            grad_loss_W = dict()
            grad_loss_b = dict()
            print("THIS HAS NOT BEEN CODED YET")

        return grad_loss_W, grad_loss_b 


    def __gradient_loss_update(self, global_grad_loss_W, global_grad_loss_b, grad_loss_W, grad_loss_b):
        for layer in self.layers:
            if self.grad_loss_update_first_time:
                global_grad_loss_W[layer] += grad_loss_W[layer]
                global_grad_loss_b[layer] += grad_loss_b[layer]
            else:
                global_grad_loss_W[layer] = grad_loss_W[layer]
                global_grad_loss_b[layer] = grad_loss_b[layer]
        self.grad_loss_update_first_time = 1

        return global_grad_loss_W, global_grad_loss_b
        

    def __sgd_update(self, global_grad_loss_W, global_grad_loss_b):
        for layer in self.layers:
            W_eta_shape = self.W[layer].shape
            b_eta_shape = self.b[layer].shape
            self.W[layer] -= np.multiply(np.full(W_eta_shape, self.param_dict["eta"]), global_grad_loss_W[layer])
            self.b[layer] -= np.multiply(np.full(b_eta_shape, self.param_dict["eta"]), global_grad_loss_b[layer])

        
    def fit(self, all_x, all_y, minibatch_size=0, epochs=1, gradient_descent_type="sgd", **kwargs):
        self.grad_loss_update_first_time = 0
        for param, param_val in kwargs.items():
            self.param_dict[param] = param_val
        
        if not minibatch_size:
            minibatch_size = len(all_y)

        global_grad_loss_b = dict()
        global_grad_loss_W = dict()
        num_points_seen = 0
        for i in range(epochs):            
            for x_i, y in zip(all_x, all_y):
                y = np.array([1 if i==(y-1) else 0 for i in range(10)]).reshape(10, 1)
                a_i, h_i = self.__forward_pass(x_i)
                grad_loss_W, grad_loss_b = self.__common_backward_pass(a_i, h_i, y)
                global_grad_loss_W, global_grad_loss_b =  self.__gradient_loss_update(global_grad_loss_W, global_grad_loss_b, 
                                                                            grad_loss_W, grad_loss_b)
                num_points_seen += 1

                if num_points_seen % minibatch_size == 0:            
                    self.__gradient_descent_type_dict[gradient_descent_type](global_grad_loss_W, global_grad_loss_b)
                    num_points_seen = 0
                    print("Loss is : ", self.__get_loss(a_i, h_i, y))

        print("Model fitting is over.")


    def predict(self, x_i, print_transforms=1):
        for layer in self.layers:
            ip_shape = x_i.shape
            x_i = np.add(np.dot(np.transpose(self.W[layer]), x_i), self.b[layer])
            activation = self.__get_activation_function(layer)
            x_i = self.__activation_function(x_i, activation)
            op_shape = x_i.shape
            if print_transforms:
                print("Layer : {} - Input_Shape : {}, W shape : {}, b shape : {}, and output_shape : {}"
                      .format(layer, ip_shape, self.W[layer].shape, self.b[layer].shape, op_shape))
        return x_i
        