import numpy as np
from copy import deepcopy
from nn_utils import constants
import math

class regular_gradient_descent:

    def __init__(self, neural_network_instance, eta, weight_decay=0.0):
        self.eta = eta
        self.weight_decay = weight_decay
        self.nn_instance = neural_network_instance
        self.step_reset()


    def grad_update(self, grad_loss_W, grad_loss_b):
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] += grad_loss_W[layer]
            self.total_grad_loss_b[layer] += grad_loss_b[layer]
        self.num_points_seen += 1        


    def step_update(self):
        for layer in self.nn_instance.layers:
            W_shape = self.nn_instance.W[layer].shape
            b_shape = self.nn_instance.b[layer].shape
            
            self.nn_instance.W[layer] -= np.multiply(np.full(W_shape, self.eta), self.total_grad_loss_W[layer])\
                                        - np.multiply(np.full(W_shape, self.eta * self.weight_decay), self.nn_instance.W[layer])
            self.nn_instance.b[layer] -= np.multiply(np.full(b_shape, self.eta), self.total_grad_loss_b[layer])
            

        self.step_reset()


    def step_reset(self):
        self.num_points_seen = 0
        self.total_grad_loss_W = dict()
        self.total_grad_loss_b = dict()
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)




class momentum_gradient_descent:

    def __init__(self, neural_network_instance, eta, beta, weight_decay=0.0):
        if (beta < 0 or beta >= 1):
            print("WARNING : Beta value is not between 0 (inclusive) and 1")
        self.beta = beta
        self.eta = eta
        self.weight_decay = weight_decay
        self.nn_instance = neural_network_instance
        self.setup()


    def setup(self):
        self.num_points_seen = 0
        self.total_grad_loss_W = dict()
        self.total_grad_loss_b = dict()
        self.prev_total_grad_loss_W = dict()
        self.prev_total_grad_loss_b = dict()
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)
            self.prev_total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.prev_total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)


    def grad_update(self, grad_loss_W, grad_loss_b):
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] += grad_loss_W[layer]
            self.total_grad_loss_b[layer] += grad_loss_b[layer]
        self.num_points_seen += 1


    def step_update(self):
        for layer in self.nn_instance.layers:
            W_shape = self.nn_instance.W[layer].shape
            b_shape = self.nn_instance.b[layer].shape

            self.prev_total_grad_loss_W[layer] = np.add(np.multiply(np.full(W_shape, self.beta), self.prev_total_grad_loss_W[layer])
                                                        , np.multiply(np.full(W_shape, self.eta), self.total_grad_loss_W[layer]))\
                                                        - np.multiply(np.full(W_shape, self.eta * self.weight_decay), self.nn_instance.W[layer])
            self.prev_total_grad_loss_b[layer] = np.add(np.multiply(np.full(b_shape, self.beta), self.prev_total_grad_loss_b[layer])
                                                        , np.multiply(np.full(b_shape, self.eta), self.total_grad_loss_b[layer]))
            
            self.nn_instance.W[layer] -= self.prev_total_grad_loss_W[layer]
            self.nn_instance.b[layer] -= self.prev_total_grad_loss_b[layer]
        
        self.step_reset()


    def step_reset(self):
        self.num_points_seen = 0
        self.total_grad_loss_W = dict()
        self.total_grad_loss_b = dict()
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)





class nestrov_accelerated_gradient_descent:

    def __init__(self, neural_network_instance, eta, beta, weight_decay=0.0):
        if (beta < 0 or beta >= 1):
            print("WARNING : Beta value is not between 0 (inclusive) and 1")
        self.beta = beta
        self.eta = eta
        self.weight_decay = weight_decay
        self.nn_instance = neural_network_instance
        self.step_update_first_time = 1
        self.setup()


    def setup(self):
        self.num_points_seen = 0
        self.total_grad_loss_W = dict()
        self.total_grad_loss_b = dict()
        self.prev_total_grad_loss_W = dict()
        self.prev_total_grad_loss_b = dict()
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)
            self.prev_total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.prev_total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)


    def grad_update(self, grad_loss_W, grad_loss_b):
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] += grad_loss_W[layer]
            self.total_grad_loss_b[layer] += grad_loss_b[layer]
        self.num_points_seen += 1   


    def step_update(self):
        if self.step_update_first_time:
            self.actual_W = dict()
            self.actual_b = dict()
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
            
            self.nn_instance.W[layer] -= self.prev_total_grad_loss_W[layer]\
                                        - np.multiply(np.full(W_shape, self.eta * self.weight_decay), self.nn_instance.W[layer])

            self.nn_instance.b[layer] -= self.prev_total_grad_loss_b[layer]
            
            self.actual_W[layer] = deepcopy(self.nn_instance.W[layer])
            self.actual_b[layer] = deepcopy(self.nn_instance.b[layer])
            self.nn_instance.W[layer] -= np.multiply(np.full(W_shape, self.beta), self.prev_total_grad_loss_W[layer])
            self.nn_instance.b[layer] -= np.multiply(np.full(b_shape, self.beta), self.prev_total_grad_loss_b[layer])

        self.step_reset()

            
    def step_reset(self):
        self.num_points_seen = 0
        self.total_grad_loss_W = dict()
        self.total_grad_loss_b = dict()
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)




class RMSProp:

    def __init__(self, neural_network_instance, eta, beta, weight_decay=0.0):
        if (beta < 0 or beta >= 1):
            print("WARNING : Beta value is not between 0 (inclusive) and 1")
        self.beta = beta
        self.eta = eta
        self.weight_decay = weight_decay
        self.nn_instance = neural_network_instance
        self.setup()


    def setup(self):
        self.num_points_seen = 0
        self.total_grad_loss_W = dict()
        self.total_grad_loss_b = dict()
        self.prev_total_grad_loss_W = dict()
        self.prev_total_grad_loss_b = dict()
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)
            self.prev_total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.prev_total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)


    def grad_update(self, grad_loss_W, grad_loss_b):
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] += grad_loss_W[layer]
            self.total_grad_loss_b[layer] += grad_loss_b[layer]
        self.num_points_seen += 1        


    def step_update(self):
        for layer in self.nn_instance.layers:
            W_shape = self.nn_instance.W[layer].shape
            b_shape = self.nn_instance.b[layer].shape

            self.prev_total_grad_loss_W[layer] = np.add(np.multiply(np.full(W_shape, self.beta), self.prev_total_grad_loss_W[layer])
                                                        , np.multiply(np.full(W_shape, (1-self.beta)), np.square(self.total_grad_loss_W[layer])))
            self.prev_total_grad_loss_b[layer] = np.add(np.multiply(np.full(b_shape, self.beta), self.prev_total_grad_loss_b[layer])
                                                        , np.multiply(np.full(b_shape, (1-self.beta)), np.square(self.total_grad_loss_b[layer])))
            
            self.nn_instance.W[layer] -= np.multiply(self.eta, self.total_grad_loss_W[layer]) / (np.sqrt(self.prev_total_grad_loss_W[layer]) + constants.epsilon)\
                                        - np.multiply(np.full(W_shape, self.eta * self.weight_decay), self.nn_instance.W[layer])
            
            self.nn_instance.b[layer] -= np.multiply(self.eta, self.total_grad_loss_b[layer]) / (np.sqrt(self.prev_total_grad_loss_b[layer]) + constants.epsilon)
        
        self.step_reset()


    def step_reset(self):
        self.num_points_seen = 0
        self.total_grad_loss_W = dict()
        self.total_grad_loss_b = dict()
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)




class Adadelta:

    def __init__(self, neural_network_instance, eta, beta, weight_decay=0.0):
        if (beta < 0 or beta >= 1):
            print("WARNING : Beta value is not between 0 (inclusive) and 1")
        self.beta = beta
        self.eta = eta
        self.weight_decay = weight_decay
        self.nn_instance = neural_network_instance
        self.setup()


    def setup(self):
        self.num_points_seen = 0
        self.total_grad_loss_W = dict()
        self.total_grad_loss_b = dict()
        self.total_grad_loss_vW = dict()
        self.total_grad_loss_vb = dict()
        self.total_grad_loss_uW = dict()
        self.total_grad_loss_ub = dict()
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)
            self.total_grad_loss_vW[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_vb[layer] = np.zeros(self.nn_instance.b[layer].shape)
            self.total_grad_loss_uW[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_ub[layer] = np.zeros(self.nn_instance.b[layer].shape)


    def grad_update(self, grad_loss_W, grad_loss_b):
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] += grad_loss_W[layer]
            self.total_grad_loss_b[layer] += grad_loss_b[layer]
        self.num_points_seen += 1        


    def step_update(self):
        this_total_grad_loss_W = dict()
        this_total_grad_loss_b = dict()

        for layer in self.nn_instance.layers:
            W_shape = self.nn_instance.W[layer].shape
            b_shape = self.nn_instance.b[layer].shape

            self.total_grad_loss_vW[layer] = np.add(np.multiply(np.full(W_shape, self.beta), self.total_grad_loss_vW[layer])
                                                        , np.multiply(np.full(W_shape, (1-self.beta)), np.square(self.total_grad_loss_W[layer])))
            self.total_grad_loss_vb[layer] = np.add(np.multiply(np.full(b_shape, self.beta), self.total_grad_loss_vb[layer])
                                                        , np.multiply(np.full(b_shape, (1-self.beta)), np.square(self.total_grad_loss_b[layer])))
            
            this_total_grad_loss_W[layer] = np.multiply(self.total_grad_loss_W[layer], np.sqrt(self.total_grad_loss_uW[layer] + constants.epsilon)) / np.sqrt(self.total_grad_loss_vW[layer] + constants.epsilon) 
            this_total_grad_loss_b[layer] = np.multiply(self.total_grad_loss_b[layer], np.sqrt(self.total_grad_loss_ub[layer] + constants.epsilon)) / np.sqrt(self.total_grad_loss_vb[layer] + constants.epsilon)

            self.total_grad_loss_uW[layer] = np.add(np.multiply(np.full(W_shape, self.beta), self.total_grad_loss_uW[layer])
                                                        , np.multiply(np.full(W_shape, (1-self.beta)), np.square(this_total_grad_loss_W[layer])))
            self.total_grad_loss_ub[layer] = np.add(np.multiply(np.full(b_shape, self.beta), self.total_grad_loss_ub[layer])
                                                        , np.multiply(np.full(b_shape, (1-self.beta)), np.square(this_total_grad_loss_b[layer])))
            
            self.nn_instance.W[layer] -= this_total_grad_loss_W[layer]\
                                        - np.multiply(np.full(W_shape, self.eta * self.weight_decay), self.nn_instance.W[layer])
            self.nn_instance.b[layer] -= this_total_grad_loss_b[layer]
        
        self.step_reset()


    def step_reset(self):
        self.num_points_seen = 0
        self.total_grad_loss_W = dict()
        self.total_grad_loss_b = dict()
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)




class Adam:

    def __init__(self, neural_network_instance, eta, beta1, beta2, weight_decay=0.0):
        if ((beta1 < 0 or beta1 >= 1) or (beta2 < 0 or beta2 >= 1)):
            print("WARNING : Beta value is not between 0 (inclusive) and 1")
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.weight_decay = weight_decay
        self.nn_instance = neural_network_instance
        self.step_update_count = 0
        self.setup()


    def setup(self):
        self.num_points_seen = 0
        self.total_grad_loss_W = dict()
        self.total_grad_loss_b = dict()
        self.total_grad_loss_mW = dict()
        self.total_grad_loss_mb = dict()
        self.total_grad_loss_vW = dict()
        self.total_grad_loss_vb = dict()
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)
            self.total_grad_loss_mW[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_mb[layer] = np.zeros(self.nn_instance.b[layer].shape)
            self.total_grad_loss_vW[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_vb[layer] = np.zeros(self.nn_instance.b[layer].shape)
                

    def grad_update(self, grad_loss_W, grad_loss_b):
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] += grad_loss_W[layer]
            self.total_grad_loss_b[layer] += grad_loss_b[layer]
        self.num_points_seen += 1        


    def step_update(self):
        this_total_grad_loss_mW_hat = dict()
        this_total_grad_loss_mb_hat = dict()
        this_total_grad_loss_vW_hat = dict()
        this_total_grad_loss_vb_hat = dict()

        for layer in self.nn_instance.layers:
            W_shape = self.nn_instance.W[layer].shape
            b_shape = self.nn_instance.b[layer].shape

            self.total_grad_loss_mW[layer] = np.add(np.multiply(np.full(W_shape, self.beta1), self.total_grad_loss_mW[layer])
                                                        , np.multiply(np.full(W_shape, (1-self.beta1)), self.total_grad_loss_W[layer]))
            self.total_grad_loss_mb[layer] = np.add(np.multiply(np.full(b_shape, self.beta1), self.total_grad_loss_mb[layer])
                                                        , np.multiply(np.full(b_shape, (1-self.beta1)), self.total_grad_loss_b[layer]))

            self.total_grad_loss_vW[layer] = np.add(np.multiply(np.full(W_shape, self.beta2), self.total_grad_loss_vW[layer])
                                                        , np.multiply(np.full(W_shape, (1-self.beta2)), np.square(self.total_grad_loss_W[layer])))
            self.total_grad_loss_vb[layer] = np.add(np.multiply(np.full(b_shape, self.beta2), self.total_grad_loss_vb[layer])
                                                        , np.multiply(np.full(b_shape, (1-self.beta2)), np.square(self.total_grad_loss_b[layer])))
            
            this_total_grad_loss_mW_hat[layer] = self.total_grad_loss_mW[layer]/ (1 - np.power(self.beta1, self.step_update_count + 1))
            this_total_grad_loss_mb_hat[layer] = self.total_grad_loss_mb[layer]/ (1 - np.power(self.beta1, self.step_update_count + 1))

            this_total_grad_loss_vW_hat[layer] = self.total_grad_loss_vW[layer]/ (1 - np.power(self.beta2, self.step_update_count + 1))
            this_total_grad_loss_vb_hat[layer] = self.total_grad_loss_vb[layer]/ (1 - np.power(self.beta2, self.step_update_count + 1))
            
            self.nn_instance.W[layer] -= (self.eta * this_total_grad_loss_mW_hat[layer])/(np.sqrt(this_total_grad_loss_vW_hat[layer]) + constants.epsilon)\
                                        - np.multiply(np.full(W_shape, self.eta * self.weight_decay), self.nn_instance.W[layer])
            self.nn_instance.b[layer] -= (self.eta * this_total_grad_loss_mb_hat[layer])/(np.sqrt(this_total_grad_loss_vb_hat[layer]) + constants.epsilon)
        
        self.step_update_count += 1
        self.step_reset()


    def step_reset(self):
        self.num_points_seen = 0
        self.total_grad_loss_W = dict()
        self.total_grad_loss_b = dict()
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)




class NAdam:

    def __init__(self, neural_network_instance, eta, beta1, beta2, weight_decay=0.0):
        if ((beta1 < 0 or beta1 >= 1) or (beta2 < 0 or beta2 >= 1)):
            print("WARNING : Beta value is not between 0 (inclusive) and 1")
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.weight_decay = weight_decay
        self.nn_instance = neural_network_instance
        self.step_update_count = 0
        self.setup()


    def setup(self):
        self.num_points_seen = 0
        self.total_grad_loss_W = dict()
        self.total_grad_loss_b = dict()
        self.total_grad_loss_mW = dict()
        self.total_grad_loss_mb = dict()
        self.total_grad_loss_vW = dict()
        self.total_grad_loss_vb = dict()
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)
            self.total_grad_loss_mW[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_mb[layer] = np.zeros(self.nn_instance.b[layer].shape)
            self.total_grad_loss_vW[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_vb[layer] = np.zeros(self.nn_instance.b[layer].shape)


    def grad_update(self, grad_loss_W, grad_loss_b):
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] += grad_loss_W[layer]
            self.total_grad_loss_b[layer] += grad_loss_b[layer]
        self.num_points_seen += 1        


    def step_update(self):
        this_total_grad_loss_mW_hat = dict()
        this_total_grad_loss_mb_hat = dict()
        this_total_grad_loss_vW_hat = dict()
        this_total_grad_loss_vb_hat = dict()

        for layer in self.nn_instance.layers:
            W_shape = self.nn_instance.W[layer].shape
            b_shape = self.nn_instance.b[layer].shape

            self.total_grad_loss_mW[layer] = np.add(np.multiply(np.full(W_shape, self.beta1), self.total_grad_loss_mW[layer])
                                                        , np.multiply(np.full(W_shape, (1-self.beta1)), self.total_grad_loss_W[layer]))
            self.total_grad_loss_mb[layer] = np.add(np.multiply(np.full(b_shape, self.beta1), self.total_grad_loss_mb[layer])
                                                        , np.multiply(np.full(b_shape, (1-self.beta1)), self.total_grad_loss_b[layer]))

            self.total_grad_loss_vW[layer] = np.add(np.multiply(np.full(W_shape, self.beta2), self.total_grad_loss_vW[layer])
                                                        , np.multiply(np.full(W_shape, (1-self.beta2)), np.square(self.total_grad_loss_W[layer])))
            self.total_grad_loss_vb[layer] = np.add(np.multiply(np.full(b_shape, self.beta2), self.total_grad_loss_vb[layer])
                                                        , np.multiply(np.full(b_shape, (1-self.beta2)), np.square(self.total_grad_loss_b[layer])))
            
            this_total_grad_loss_mW_hat[layer] = self.total_grad_loss_mW[layer]/ (1 - np.power(self.beta1, self.step_update_count + 1))
            this_total_grad_loss_mb_hat[layer] = self.total_grad_loss_mb[layer]/ (1 - np.power(self.beta1, self.step_update_count + 1))

            this_total_grad_loss_vW_hat[layer] = self.total_grad_loss_vW[layer]/ (1 - np.power(self.beta2, self.step_update_count + 1))
            this_total_grad_loss_vb_hat[layer] = self.total_grad_loss_vb[layer]/ (1 - np.power(self.beta2, self.step_update_count + 1))
            
            self.nn_instance.W[layer] -= ((self.eta/np.sqrt(this_total_grad_loss_vW_hat[layer] + constants.epsilon)) *
                                          (np.add(self.beta1 * this_total_grad_loss_mW_hat[layer], 
                                            ((1-self.beta1) * self.total_grad_loss_W[layer]) / (1 - np.power(self.beta1, self.step_update_count + 1))
                                            )))\
                                        - np.multiply(np.full(W_shape, self.eta * self.weight_decay), self.nn_instance.W[layer])
            self.nn_instance.b[layer] -= ((self.eta/np.sqrt(this_total_grad_loss_vb_hat[layer] + constants.epsilon)) *
                                          (np.add(self.beta1 * this_total_grad_loss_mb_hat[layer], 
                                            ((1-self.beta1) * self.total_grad_loss_b[layer]) / (1 - np.power(self.beta1, self.step_update_count + 1)))))
        
        self.step_update_count += 1
        self.step_reset()


    def step_reset(self):
        self.num_points_seen = 0
        self.total_grad_loss_W = dict()
        self.total_grad_loss_b = dict()
        for layer in self.nn_instance.layers:
            self.total_grad_loss_W[layer] = np.zeros(self.nn_instance.W[layer].shape)
            self.total_grad_loss_b[layer] = np.zeros(self.nn_instance.b[layer].shape)
