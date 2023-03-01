import numpy as np
from nn_utils.math_utils import numerically_stable_logistic, numerically_stable_softmax

# Forward activation functions
def forward_activation(a_input, activation):
    if activation == "linear":
        return a_input
    
    elif activation == "logistic":
        return numerically_stable_logistic(a_input)
    
    elif activation == "tanh":
        return np.tanh(a_input)
    
    elif activation == "softmax":
        return numerically_stable_softmax(a_input)


# Gradients wrt activation layer
def grad_activation(a_input, activation):
    if activation == "linear":
        return 1
    
    elif activation == "logistic":
        activated_val = forward_activation(a_input, activation)
        return (activated_val * (1 - activated_val))
    
    elif activation == "tanh":
        activated_val = forward_activation(a_input, activation)
        return (1 - np.square(activated_val))


# Gradients wrt output activation layers
def grad_wrt_output(y, y_pred, activation):
    if activation == "softmax":
        return - (y - y_pred)