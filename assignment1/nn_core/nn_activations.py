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

    elif activation == "relu":
        a_input[a_input < 0] = 0
        return a_input

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

    elif activation == "relu":
        a_input[a_input >= 0] = 1
        a_input[a_input < 0] = 0
        return a_input

# Gradients wrt output activation layers
def grad_wrt_output(y, y_pred, loss_function, activation):
    if loss_function == "cross_entropy":
        if activation == "softmax":
            return - (y - y_pred)
    elif loss_function == "squared_error":
        if activation == "softmax":
            return (y_pred - y) * y_pred * (1 - y_pred)