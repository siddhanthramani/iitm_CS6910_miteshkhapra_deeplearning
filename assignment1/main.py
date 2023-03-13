# Imports required modules
from nn_utils.data_utils import load, plot_random_image_per_class, do_checks
from nn_utils.output_utils import get_accuracy_metrics
from nn_core.nn_main import neural_network
from nn_core.nn_optimizer import *
from nn_user.weight_init import *
from nn_user.augment_data import augment_data
import numpy as np
from sklearn.model_selection import train_test_split
import json
import wandb
wandb_initializer = {
    "random" : random_init
    , "xavier" : xavier_init
}
wandb_optimizer = {
    "regular_gradient_descent" : regular_gradient_descent
    , "momentum_gradient_descent" : momentum_gradient_descent
    , "nestrov_accelerated_gradient_descent" : nestrov_accelerated_gradient_descent
    , "rmsprop" : RMSProp
    , "adam" : Adam
    , "nadam" : NAdam
}
wandb_optimizer_params = {
    "regular_gradient_descent" : {}
    , "momentum_gradient_descent" : {"beta" : 0.9}
    , "nestrov_accelerated_gradient_descent" : {"beta" : 0.9}
    , "rmsprop" : {"beta" : 0.9}
    , "adam" : {"beta1" : 0.9, "beta2" : 0.99}
    , "nadam" : {"beta1" : 0.9, "beta2" : 0.99}
}

def main():
    # Start wandb
    wandb.init()

    # Initialize wandb params
    epochs = wandb.config.epochs
    number_of_hidden_layers = wandb.config.number_of_hidden_layers
    size_of_every_hidden_layer = wandb.config.size_of_every_hidden_layer
    weight_decay = wandb.config.weight_decay
    learning_rate = wandb.config.learning_rate
    optimizer = wandb.config.optimizer
    batch_size = wandb.config.batch_size
    weight_init = wandb.config.weight_init
    activation_function = wandb.config.activation_function

    # Defining the network structure
    dict_neural_network_structure = {}
    dict_neural_network_structure[0] = 28*28
    dict_neural_network_structure[number_of_hidden_layers + 1] = 10
    for hidden_layer in range(1, number_of_hidden_layers + 1):
        dict_neural_network_structure[hidden_layer] = size_of_every_hidden_layer
    
    # Defining the activations
    activation_dict = {0 : activation_function, 1 : "softmax"}
    
    # Creating the neural network instance
    nn1 = neural_network(dict_neural_network_structure, activation_dict, nn_init=wandb_initializer[weight_init])
    
    # Defining the optimizer
    optimizer = wandb_optimizer[optimizer](nn1, eta=learning_rate, **wandb_optimizer_params[optimizer])
    
    # Fitting the model
    nn1.fit(wandb, optimizer, X_train, y_train, X_val, y_val, epochs=epochs, minibatch_size = batch_size)
    
    # Checking for accuracy
    output = nn1.predict(X_test)
    accuracy_metrics = get_accuracy_metrics(y_test, output)
    for val in accuracy_metrics:
        print(val*100)
        wandb.log({"test_accuracy" : val*100})

if __name__ == "__main__":

    # Loading the data
    data = load()
    X = data["train_X"]/255
    y = data["train_y"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
    X_test = data["test_X"]/255
    y_test = data["test_y"]

    # Checking the data for infinities or nans
    do_checks(X_train, y_train)
    do_checks(X_val, y_val)
    do_checks(X_test, y_test)

    # Augment data - if required
    # X_train, y_train = augment_data(X_train, y_train, mode="replace")
    
    # WandB sweep
    with open("./assignment1/wandb_config.json") as f:
        sweep_config = json.load(f)
    sweep_id = wandb.sweep(project="sweep_test", sweep=sweep_config)
    wandb.agent(sweep_id, function=main, count=2)
# data, X_train, X_val, X_test, y_train, y_val, y_test