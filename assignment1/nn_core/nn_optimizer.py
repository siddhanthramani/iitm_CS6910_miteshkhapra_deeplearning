import numpy as np
from copy import deepcopy
import math

class regular_gradient_descent:

    def __init__(self, neural_network_instance, eta):
        self.eta = eta
        self.nn_instance = neural_network_instance
        self.step_reset()


    def grad_update(self, grad_loss_W, grad_loss_b):
        for layer in self.nn_instance.layers:
            # For first time update, we init accumalated value
            if self.grad_update_first_time:
                self.total_grad_loss_W[layer] = grad_loss_W[layer]
                self.total_grad_loss_b[layer] = grad_loss_b[layer]
            # For all other times, we accumulate the gradient
            else:
                self.total_grad_loss_W[layer] += grad_loss_W[layer]
                self.total_grad_loss_b[layer] += grad_loss_b[layer]
        self.grad_update_first_time = 0
        self.num_points_seen += 1        


    def step_update(self):
        for layer in self.nn_instance.layers:
            W_eta_shape = self.nn_instance.W[layer].shape
            b_eta_shape = self.nn_instance.b[layer].shape
            
            self.nn_instance.W[layer] -= np.multiply(np.full(W_eta_shape, self.eta), self.total_grad_loss_W[layer])
            self.nn_instance.b[layer] -= np.multiply(np.full(b_eta_shape, self.eta), self.total_grad_loss_b[layer])
        
        self.step_reset()


    def step_reset(self):
        self.num_points_seen = 0
        self.total_grad_loss_b = dict()
        self.total_grad_loss_W = dict()
        self.grad_update_first_time = 1




class momentum_gradient_descent:

    def __init__(self, neural_network_instance, eta, beta):
        if (beta < 0 or beta >= 1):
            print("WARNING : Beta value is not between 0 (inclusive) and 1")
        self.beta = beta
        self.eta = eta
        self.nn_instance = neural_network_instance
        self.step_reset()


    def grad_update(self, grad_loss_W, grad_loss_b):
        for layer in self.nn_instance.layers:
            # For first time update, we init accumalated value
            if self.grad_update_first_time:
                self.total_grad_loss_W[layer] = grad_loss_W[layer]
                self.total_grad_loss_b[layer] = grad_loss_b[layer]
            # For all other times, we accumulate the gradient
            else:
                self.total_grad_loss_W[layer] += grad_loss_W[layer]
                self.total_grad_loss_b[layer] += grad_loss_b[layer]
        self.grad_update_first_time = 0
        self.num_points_seen += 1        


    def step_update(self):
        for layer in self.nn_instance.layers:
            W_shape = self.nn_instance.W[layer].shape
            b_shape = self.nn_instance.b[layer].shape

            self.prev_total_grad_loss_W[layer] = np.add(np.multiply(np.full(W_shape, self.beta), self.prev_total_grad_loss_W[layer])
                                                        , np.multiply(np.full(W_shape, self.eta), self.total_grad_loss_W[layer]))
            self.prev_total_grad_loss_b[layer] = np.add(np.multiply(np.full(b_shape, self.beta), self.prev_total_grad_loss_b[layer])
                                                        , np.multiply(np.full(b_shape, self.eta), self.total_grad_loss_b[layer]))
            
            self.nn_instance.W[layer] -= self.prev_total_grad_loss_W[layer]
            self.nn_instance.b[layer] -= self.prev_total_grad_loss_b[layer]
        
        self.step_reset()


    def step_reset(self):
        self.num_points_seen = 0
        self.total_grad_loss_b = dict()
        self.total_grad_loss_W = dict()
        self.prev_total_grad_loss_W = dict()
        self.prev_total_grad_loss_b = dict()
        for layer in self.nn_instance.layers:
                self.prev_total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
                self.prev_total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)
        self.grad_update_first_time = 1





class nestrov_accelerated_gradient_descent:

    def __init__(self, neural_network_instance, eta, beta):
        if (beta < 0 or beta >= 1):
            print("WARNING : Beta value is not between 0 (inclusive) and 1")
        self.beta = beta
        self.eta = eta
        self.nn_instance = neural_network_instance
        self.step_update_first_time = 1
        self.step_reset()


    def grad_update(self, grad_loss_W, grad_loss_b):
        for layer in self.nn_instance.layers:
            # For first time update, we init accumalated value
            if self.grad_update_first_time:
                self.total_grad_loss_W[layer] = grad_loss_W[layer]
                self.total_grad_loss_b[layer] = grad_loss_b[layer]
            # For all other times, we accumulate the gradient
            else:
                self.total_grad_loss_W[layer] += grad_loss_W[layer]
                self.total_grad_loss_b[layer] += grad_loss_b[layer]
        self.grad_update_first_time = 0
        self.num_points_seen += 1        


    def step_update(self):
        if self.step_update_first_time:
            self.step_update_first_time = 0
        else:
            self.nn_instance.W = deepcopy(self.actual_W)
            self.nn_instance.b = deepcopy(self.actual_b)

        for layer in self.nn_instance.layers:
            W_shape = self.nn_instance.W[layer].shape
            b_shape = self.nn_instance.b[layer].shape

            self.prev_total_grad_loss_W[layer] = np.add(np.multiply(np.full(W_shape, self.beta), self.prev_total_grad_loss_W[layer])
                                                        , np.multiply(np.full(W_shape, self.eta), self.total_grad_loss_W[layer]))
            self.prev_total_grad_loss_b[layer] = np.add(np.multiply(np.full(b_shape, self.beta), self.prev_total_grad_loss_b[layer])
                                                        , np.multiply(np.full(b_shape, self.eta), self.total_grad_loss_b[layer]))
            
            self.nn_instance.W[layer] -= self.prev_total_grad_loss_W[layer]
            self.nn_instance.b[layer] -= self.prev_total_grad_loss_b[layer]
            
        self.actual_W = deepcopy(self.nn_instance.W)
        self.actual_b = deepcopy(self.nn_instance.b)

        for layer in self.nn_instance.layers:
            W_shape = self.nn_instance.W[layer].shape
            b_shape = self.nn_instance.b[layer].shape
            self.nn_instance.W[layer] -= np.multiply(np.full(W_shape, self.beta), self.prev_total_grad_loss_W[layer])
            self.nn_instance.b[layer] -= np.multiply(np.full(b_shape, self.beta), self.prev_total_grad_loss_b[layer])

        self.step_reset()


    def step_reset(self):
        self.num_points_seen = 0
        self.total_grad_loss_b = dict()
        self.total_grad_loss_W = dict()
        self.prev_total_grad_loss_W = dict()
        self.prev_total_grad_loss_b = dict()
        for layer in self.nn_instance.layers:
                self.prev_total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
                self.prev_total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)
        self.local_W = dict()
        self.local_b = dict()
        self.grad_update_first_time = 1




class RMSProp:

    def __init__(self, neural_network_instance, eta, beta):
        if (beta < 0 or beta >= 1):
            print("WARNING : Beta value is not between 0 (inclusive) and 1")
        self.beta = beta
        self.eta = eta
        self.nn_instance = neural_network_instance
        self.step_reset()


    def grad_update(self, grad_loss_W, grad_loss_b):
        for layer in self.nn_instance.layers:
            # For first time update, we init accumalated value
            if self.grad_update_first_time:
                self.total_grad_loss_W[layer] = grad_loss_W[layer]
                self.total_grad_loss_b[layer] = grad_loss_b[layer]
            # For all other times, we accumulate the gradient
            else:
                self.total_grad_loss_W[layer] += grad_loss_W[layer]
                self.total_grad_loss_b[layer] += grad_loss_b[layer]
        self.grad_update_first_time = 0
        self.num_points_seen += 1        


    def step_update(self):
        for layer in self.nn_instance.layers:
            W_shape = self.nn_instance.W[layer].shape
            b_shape = self.nn_instance.b[layer].shape

            self.prev_total_grad_loss_W[layer] = np.add(np.multiply(np.full(W_shape, self.beta), self.prev_total_grad_loss_W[layer])
                                                        , np.multiply(np.full(W_shape, (1-self.beta)), np.square(self.total_grad_loss_W[layer])))
            self.prev_total_grad_loss_b[layer] = np.add(np.multiply(np.full(b_shape, self.beta), self.prev_total_grad_loss_b[layer])
                                                        , np.multiply(np.full(b_shape, (1-self.beta)), np.square(self.total_grad_loss_b[layer])))
            
            self.nn_instance.W[layer] -= np.sqrt(self.prev_total_grad_loss_W[layer]) + 
            self.nn_instance.b[layer] -= np.sqrt(self.prev_total_grad_loss_b[layer]) + 
        
        self.step_reset()


    def step_reset(self):
        self.num_points_seen = 0
        self.total_grad_loss_b = dict()
        self.total_grad_loss_W = dict()
        self.prev_total_grad_loss_W = dict()
        self.prev_total_grad_loss_b = dict()
        for layer in self.nn_instance.layers:
                self.prev_total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
                self.prev_total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)
        self.grad_update_first_time = 1