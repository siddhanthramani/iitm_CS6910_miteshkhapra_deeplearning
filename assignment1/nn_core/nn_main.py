import numpy as np
import json
from nn_core.nn_activations import *
from nn_core.nn_loss import *
from nn_core.nn_optimizer import *
from nn_utils.output_utils import get_accuracy_metrics
from nn_utils import constants

class neural_network():
    # Epsilon is a small value which is added when we do not want a value to go to zero
    epsilon = 0.0001

    # The init function contains the instance variables and methods
    def __init__(self, num_neurons_dict, activation_dict, activation_type="constant"
                 , loss_function="cross_entropy" , nn_init=np.random.rand, nn_init_random_max=1
                 , **nn_init_params):
        
        # setting up the neural architecture
        self.num_neurons_dict = num_neurons_dict
        self.activation_dict = activation_dict
        self.activation_type = activation_type
        self.loss_function = loss_function

        # setting up frequently required instance variables
        self.layers = list(range(1, len(num_neurons_dict.keys())))
        self.max_layer = max(self.layers)
        self.number_of_classes = self.num_neurons_dict[self.max_layer] 
                
        # sets up the initial weights of the neural network as per user input
        self.W = dict()
        self.b = dict()
        for layer, num_neurons in sorted(num_neurons_dict.items()):
            if layer == 0:
                continue
            self.W[layer] = nn_init(num_neurons, num_neurons_dict[layer-1], **nn_init_params) * nn_init_random_max
            self.b[layer] = nn_init(num_neurons, 1, **nn_init_params) * nn_init_random_max
        

    # A helper function used to get the activation function of a particular layer
    def __helper_get_activation_function(self, layer):
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


    def __helper_get_one_hot_encoded_y(self, y):
        return np.array([1 if i==(y-1) else 0 for i in range(self.number_of_classes)]).reshape(self.number_of_classes, 1)


    # Forward propogation - computes each pre-activation and activation till predicted output
    def __forward_prop(self, x_i):
        a_i = dict()
        h_i = dict()
        # For clean code - Activated 0th layer equated to input
        h_i[0] = x_i
        for layer in self.layers:
            a_i[layer] = np.add(np.dot(self.W[layer], h_i[layer-1]), self.b[layer])
            activation = self.__helper_get_activation_function(layer)
            h_i[layer] = forward_activation(a_i[layer], activation)

        return a_i, h_i


    # Backward propogation
    def __back_prop(self, a_i, h_i, y):
        # Inits values
        grad_loss_W = dict()
        grad_loss_b = dict()

        grad_loss_a = dict()
        grad_loss_h = dict()

        # Inits the pre_activation grad wrt output for max layer (final layer)
        grad_loss_a[self.max_layer] = grad_wrt_output(y, h_i[self.max_layer], self.loss_function
                                            , self.__helper_get_activation_function(self.max_layer))            
        # Loops through each layer in reverse
        for grad_layer in self.layers[::-1]:
            grad_loss_W[grad_layer] = np.dot(grad_loss_a[grad_layer], np.transpose(h_i[grad_layer-1]))
            grad_loss_b[grad_layer] = grad_loss_a[grad_layer]
            
            # If layer becomes equal to 1, we do not have to calc the 0th layer h, a
            if grad_layer == 1:
                break
            
            grad_loss_h[grad_layer-1] = np.dot(np.transpose(self.W[grad_layer]), grad_loss_a[grad_layer])
            grad_loss_a[grad_layer-1] = np.multiply(grad_loss_h[grad_layer-1], 
                                                    grad_activation(a_i[grad_layer-1], 
                                                                    self.__helper_get_activation_function(grad_layer-1)))

        return grad_loss_W, grad_loss_b 


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


    def __train(self, train_x, train_y, optimizer, minibatch_size):
        if not minibatch_size:
            minibatch_size = len(train_y)

        for x_i, y in zip(train_x, train_y):
            y = self.__helper_get_one_hot_encoded_y(y)
            
            a_i, h_i = self.__forward_prop(x_i)
            grad_loss_W, grad_loss_b = self.__back_prop(a_i, h_i, y)

            optimizer.grad_update(grad_loss_W, grad_loss_b)
    
            if optimizer.num_points_seen % minibatch_size == 0:   
                optimizer.step_update()


    def __validate(self, train_x, train_y, val_x=None, val_y=None):
        # Epoch init
        epoch_loss = list()
        if val_x is not None and val_y is not None:
            epoch_x = val_x
            epoch_y = val_y
        else:
            epoch_x = train_x
            epoch_y = train_y
        
        for x_i, y in zip(epoch_x, epoch_y): 
            y = self.__helper_get_one_hot_encoded_y(y)
            epoch_loss.append(get_loss(y, self.__predict_single_input(x_i), self.loss_function, constants.epsilon))
        
        y_pred = self.predict(epoch_x)
        accuracy_metrics = get_accuracy_metrics(epoch_y, y_pred)

        return (round(np.average(np.array(epoch_loss)), 4)
                , round(accuracy_metrics[0], 4))


    def fit(self, optimizer, train_x, train_y, val_x=None, val_y=None
            , minibatch_size=0, epochs=1):
        list_validation_loss = []
        list_validation_accuracy = []
        for epoch in range(epochs):   
            # Train
            self.__train(train_x, train_y, optimizer, minibatch_size)
            
            # Validate
            validation_loss, validation_accuracy = self.__validate(train_x, train_y, val_x, val_y)
            print("Epoch - {}\tValidation Loss - {}\tValidation Accuracy - {}"
                  .format(epoch+1, validation_loss, validation_accuracy))
            list_validation_loss.append(validation_loss)
            list_validation_accuracy.append(validation_accuracy)

        print("Model fitting is over.")
        return list_validation_loss, list_validation_accuracy


    def __predict_single_input(self, single_input_x):
        x_i = single_input_x.copy()
        for layer in self.layers:
            x_i = np.add(np.dot(self.W[layer], x_i), self.b[layer])
            activation = self.__helper_get_activation_function(layer)
            x_i = forward_activation(x_i, activation)
        return x_i
        

    def predict(self, vec_x):
        y_pred = []
        for x_i in vec_x:
            for layer in self.layers:
                x_i = np.add(np.dot(self.W[layer], x_i), self.b[layer])
                activation = self.__helper_get_activation_function(layer)
                x_i = forward_activation(x_i, activation)
            y_pred.append(np.argmax(x_i) + 1)
        return np.array(y_pred)


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

        