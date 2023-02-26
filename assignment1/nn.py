import numpy as np
import json 

class neural_network():
    # Epsilon is a small value which is added when we do not want a value to go to zero
    epsilon = 0.0001

    # The init function contains the instance variables and methods
    def __init__(self, num_neurons_dict, activation_dict, activation_type="constant"
                 , loss_function="cross_entropy" , nn_init=np.random.rand, nn_init_random_max=1
                 , **nn_init_params):
        # contains the neural architecture
        self.num_neurons_dict = num_neurons_dict
        # contains the activation function of each layer
        self.activation_dict = activation_dict
        # contains the metadata of activation_dict
        self.activation_type = activation_type
        # contains the loss function to be used
        self.loss_function = loss_function

        # setting up frequently required instance variables
        self.layers = list(range(1, len(num_neurons_dict.keys())))
        self.max_layer = max(self.layers)
        # contains the weights of the model
        self.W = dict()
        # contains the biases of the model
        self.b = dict()

        # a dict which maps gradient algo name to respective algo function
        self.__gradient_descent_type_dict = {
            "sgd" : self.__sgd_step_update,
            "momentum" : self.__momentum_step_update
        }
        # a dict which contains parameters which can be passed for fitting model
        self.param_dict = dict()

        # sets up the initial weights of the neural network as per user input
        for layer, num_neurons in sorted(num_neurons_dict.items()):
            if layer == 0:
                continue
            self.W[layer] = nn_init(num_neurons, num_neurons_dict[layer-1], **nn_init_params) * nn_init_random_max
            self.b[layer] = nn_init(num_neurons, 1, **nn_init_params) * nn_init_random_max


    # A helper function used to get the activation function of a particular layer
    def __get_activation_function(self, layer):
        # checks that layer value is correct
        if layer < 1 or layer > self.max_layer:
            raise Exception("Activation function exists only for layers between 1 and max layer")
        
        # For constant case
        if self.activation_type == "constant":
            # if last activation is asked for, val for key=1 is returned
            if layer == max(self.layers):
                activation = self.activation_dict[1]
            # if any other activation is asked for, val for key=0 is returned
            else:
                activation = self.activation_dict[0]
        # if activation_type is not constant, activation for a specfic layer is returned
        else:
            activation = self.activation_dict[layer]
        return activation
    

    # This gets the loss between y_pred and y(y_actual)
    def __get_loss(self, y_pred, y):
        if self.loss_function == "cross_entropy":
            if not self.__get_activation_function(self.max_layer) == "softmax":
                raise Exception("Cross entropy can be used only if the final activation function is softmax")

            loss = -np.sum(np.multiply(y, np.log(y_pred + self.epsilon)))
    
        elif self.loss_function == "squared_error":
            loss = np.sum(np.square(y_pred - y))
        
        return loss
    
    # A set of two logistic functions to ensure stable outputs 
    # for positive and negative inputs respectively 
    def __positive_logistic(self, input):
        return 1 / (1 + np.exp(-input))
    def __negative_logistic(self, input):
        exp_inp = np.exp(input)
        return exp_inp / (1 + exp_inp)

    # Returns an activated input as per activation value 
    def __forward_activation(self, input, activation):
        if activation == "linear":
            return input
        
        # Numerically stable logistic - https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
        elif activation == "logistic":
            pos_inputs = input >=0
            neg_inputs = ~pos_inputs

            result = np.empty_like(input, dtype=np.float)
            result[pos_inputs] = self.__positive_logistic(input[pos_inputs])
            result[neg_inputs] = self.__negative_logistic(input[neg_inputs])
            return result
        
        elif activation == "tanh":
            return np.tanh(input)
        
        # Numerically stable softmax - https://cs231n.github.io/linear-classify/#softmax
        elif activation == "softmax":
            softmax_num = np.exp(input - np.max(input))
            return (softmax_num/ np.sum(softmax_num))


    # Forward propogation - computes each pre-activation and activation till predicted output
    def __forward_prop(self, x_i):
        a_i = dict()
        h_i = dict()
        # For clean code - Activated 0th layer equated to input
        h_i[0] = x_i
        for layer in self.layers:
            a_i[layer] = np.add(np.dot(self.W[layer], h_i[layer-1]), self.b[layer])
            activation = self.__get_activation_function(layer)
            h_i[layer] = self.__forward_activation(a_i[layer], activation)

        return a_i, h_i
    

    # Calculated gradients of activated values
    def __grad_activation(self, ak, activation):
        if activation == "logistic":
            activated_val = self.__forward_activation(ak, activation)
            return (activated_val * (1 - activated_val))
        elif activation == "tanh":
            activated_val = self.__forward_activation(ak, activation)
            return (1 - np.square(activated_val))
    

    # Calculated the gradient of pre-activation max_layer wrt output
    def __grad_wrt_output(self, y_pred, y, activation):
        if activation == "softmax":
            return - (y - y_pred)
    

    def __back_prop(self, a_i, h_i, y):
        if self.loss_function == "cross_entropy":
            # Check - our cross_entropy algo is optimized for softmax
            if not self.__get_activation_function(self.max_layer) == "softmax":
                raise Exception("Cross entropy can be used only if the final activation function is softmax")

            # Inits values
            grad_loss_W = dict()
            grad_loss_b = dict()

            grad_loss_a = dict()
            grad_loss_h = dict()

            # Inits the pre_activation grad wrt output for max layer (final layer)
            grad_loss_a[self.max_layer] = self.__grad_wrt_output(h_i[self.max_layer], y
                                                , self.__get_activation_function(self.max_layer))            
            # Loops through each layer in reverse
            for grad_layer in self.layers[::-1]:
                grad_loss_W[grad_layer] = np.dot(grad_loss_a[grad_layer], np.transpose(h_i[grad_layer-1]))
                grad_loss_b[grad_layer] = grad_loss_a[grad_layer]
                
                # If layer becomes equal to 1, we do not have to calc the 0th layer h, a
                if grad_layer == 1:
                    break
                
                grad_loss_h[grad_layer-1] = np.dot(np.transpose(self.W[grad_layer]), grad_loss_a[grad_layer])
                grad_loss_a[grad_layer-1] = np.multiply(grad_loss_h[grad_layer-1], 
                                                self.__grad_activation(a_i[grad_layer-1], 
                                                                        self.__get_activation_function(grad_layer-1)))

        elif self.loss_function == "squared_error":
            loss = np.sum(np.square(h_i[max(self.layers)] - y))
            grad_loss_h = dict()
            grad_loss_W = dict()
            grad_loss_b = dict()
            print("THIS HAS NOT BEEN CODED YET")

        return grad_loss_W, grad_loss_b 


    def __grad_update(self, accumulated_grad_loss_W, accumulated_grad_loss_b, grad_loss_W, grad_loss_b):
        for layer in self.layers:
            # For first time update, we init accumalated value
            if self.grad_update_first_time:
                accumulated_grad_loss_W[layer] = grad_loss_W[layer]
                accumulated_grad_loss_b[layer] = grad_loss_b[layer]
            # For all other times, we accumulate the gradient
            else:
                accumulated_grad_loss_W[layer] += grad_loss_W[layer]
                accumulated_grad_loss_b[layer] += grad_loss_b[layer]
        self.grad_update_first_time = 0

        return accumulated_grad_loss_W, accumulated_grad_loss_b
        

    def __sgd_step_update(self, global_grad_loss_W, global_grad_loss_b):
        for layer in self.layers:
            W_eta_shape = self.W[layer].shape
            b_eta_shape = self.b[layer].shape
            
            self.W[layer] -= np.multiply(np.full(W_eta_shape, self.param_dict["eta"]), global_grad_loss_W[layer])
            self.b[layer] -= np.multiply(np.full(b_eta_shape, self.param_dict["eta"]), global_grad_loss_b[layer])
    

    def __momentum_step_update(self, global_grad_loss_W, global_grad_loss_b):
        if self.step_update_first_time == 1:
            self.prev_global_grad_loss_W = dict()
            self.prev_global_grad_loss_b = dict()
            for layer in self.layers:
                self.prev_global_grad_loss_W[layer] = np.zeros(self.W[layer].shape)
                self.prev_global_grad_loss_b[layer] = np.zeros(self.b[layer].shape)
            self.step_update_first_time == 0
        
        u_W = dict()
        u_b = dict()
        for layer in self.layers:
            W_shape = self.W[layer].shape
            b_shape = self.b[layer].shape

            u_W[layer] = np.add(np.multiply(np.full(W_shape, self.param_dict["beta"]), self.prev_global_grad_loss_W[layer])
                            , global_grad_loss_W[layer])
            u_b[layer] = np.add(np.multiply(np.full(b_shape, self.param_dict["beta"]), self.prev_global_grad_loss_b[layer])
                            , global_grad_loss_b[layer])
            
            self.W[layer] -= np.multiply(np.full(W_shape, self.param_dict["eta"]), u_W[layer])
            self.b[layer] -= np.multiply(np.full(b_shape, self.param_dict["eta"]), u_b[layer])
            
        self.prev_global_grad_loss_W = u_W.copy()
        self.prev_global_grad_loss_b = u_b.copy()
        

    def fit(self, all_x, all_y, minibatch_size=0, epochs=1, gradient_descent_type="sgd", **kwargs):
        self.grad_update_first_time = 1
        self.step_update_first_time = 1
        
        for param, param_val in kwargs.items():
            self.param_dict[param] = param_val
        
        if not minibatch_size:
            minibatch_size = len(all_y)

        global_grad_loss_W = dict()
        global_grad_loss_b = dict()
        
        num_points_seen = 0

        for epoch in range(epochs):            
            for x_i, y in zip(all_x, all_y):
                y = np.array([1 if i==(y-1) else 0 for i in range(10)]).reshape(10, 1)
                a_i, h_i = self.__forward_prop(x_i)
    
                grad_loss_W, grad_loss_b = self.__back_prop(a_i, h_i, y)
                global_grad_loss_W, global_grad_loss_b =  self.__grad_update(global_grad_loss_W, global_grad_loss_b, 
                                                                            grad_loss_W, grad_loss_b)
                num_points_seen += 1

                if num_points_seen % minibatch_size == 0:         
                    self.__gradient_descent_type_dict[gradient_descent_type](global_grad_loss_W, global_grad_loss_b)
                    num_points_seen, global_grad_loss_W, global_grad_loss_b = self.__step_reset()
            
            epoch_loss = list()
            for x_i, y in zip(all_x, all_y):
                epoch_loss.append(self.__get_loss(self.predict(x_i), y))
            print("Epoch Loss - {} is : {}".format(epoch+1, np.average(np.array(epoch_loss))))
        print("Model fitting is over.")


    def __step_reset(self):
        num_points_seen = 0
        global_grad_loss_b = dict()
        global_grad_loss_W = dict()
        self.grad_update_first_time = 1
        return num_points_seen, global_grad_loss_W, global_grad_loss_b


    def predict(self, x_i, print_transforms=0):
        for layer in self.layers:
            ip_shape = x_i.shape
            x_i = np.add(np.dot(self.W[layer], x_i), self.b[layer])
            activation = self.__get_activation_function(layer)
            x_i = self.__forward_activation(x_i, activation)
            op_shape = x_i.shape
            if print_transforms:
                print("Layer : {} - Input_Shape : {}, W shape : {}, b shape : {}, and output_shape : {}"
                      .format(layer, ip_shape, self.W[layer].shape, self.b[layer].shape, op_shape))
        return x_i


    def save_model(self, file_location):
        model_json = {
            "num_neurons_dict" : self.num_neurons_dict,
            "activation_dict" : self.activation_dict,
            "activation_type" : self.activation_type,
            "loss_function" : self.loss_function,
            "param_dict" : self.param_dict,
            "W" : self.W,
            "b" : self.b
        }
        model_object = json.dumps(model_json)
        with open(file_location, "w") as outfile:
            outfile.write(model_object)


    def load_model(self, file_location):
        with open(file_location, 'r') as infile:   
            model_object = json.load(infile)
            if not self.num_neurons_dict == model_object.num_neurons_dict:
                print("WARNING : num_neurons_dict do not match.")
            if not self.activation_dict == model_object.activation_dict:
                print("WARNING : activation_dict do not match.")
            if not self.activation_type == model_object.activation_type:
                print("WARNING : activation_type do not match.")
            if not self.loss_function == model_object.loss_function:
                print("WARNING : loss_function do not match.")
            if not self.param_dict == model_object.param_dict:
                print("WARNING : param_dict do not match.")
            
            self.W = model_object.W
            self.b = model_object.b

        