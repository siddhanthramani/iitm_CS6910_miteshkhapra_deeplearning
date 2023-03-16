import numpy as np
from nn_utils import constants
# This gets the loss between y_pred and y(y_actual)
def get_loss(y, y_pred, loss_function, epsilon):
    if loss_function == "cross_entropy":
        loss = -np.sum(np.multiply(y, np.log(y_pred + epsilon)))

    elif loss_function == "mean_squared_error":
        loss = np.sum(np.square(np.subtract(y_pred, y)).mean())
    
    return loss