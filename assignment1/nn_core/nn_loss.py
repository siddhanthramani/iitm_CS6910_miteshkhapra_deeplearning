import numpy as np

# This gets the loss between y_pred and y(y_actual)
def get_loss(y, y_pred, loss_function, epsilon = 1e-8):
    if loss_function == "cross_entropy":
        loss = -np.sum(np.multiply(y, np.log(y_pred + epsilon)))

    elif loss_function == "squared_error":
        loss = np.sum(np.square(y_pred - y))
    
    return loss