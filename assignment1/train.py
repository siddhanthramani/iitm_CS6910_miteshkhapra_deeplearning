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
from argparse import ArgumentParser
from nn_utils import constants

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

def cli_parser():
    parser = ArgumentParser(prog="train"
                            , description="Train and test a neural network. Track it via wandb.")


    wandb_parser = parser.add_argument_group("Wandb Group", "Options related to wandb.")
    general_params_parser = parser.add_argument_group("General Parameters Group", "Options related to general training parameters.")
    optimizer_parser = parser.add_argument_group("Optimizer Group", "Options related to optimizers and its parameters.")
    nn_arch_parser = parser.add_argument_group("Neural Network Architecture Group"
                                            , "Options related to defining the network structure.")


    wandb_parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname"
                        , help="Project name used to track experiments in Weights & Biases dashboard."
                        , metavar="project_name")
    wandb_parser.add_argument("-we", "--wandb_entity", type=str, default="myname"
                        , help="Wandb Entity used to track experiments in the Weights & Biases dashboard."
                        , metavar="entity_name")

    general_params_parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"]
                        , help="Choose the dataset you would like to model."
                        , metavar="dataset")
    general_params_parser.add_argument("-e", "--epochs", type=int, default=1
                        , help="Number of epochs to train neural network."
                        , metavar="number_of_epochs")
    general_params_parser.add_argument("-b", "--batch_size", type=int, default=4
                        , help="Batch size used to train neural network."
                        , metavar="batch_size")
    general_params_parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"]
                        , help="Loss function to optimize the model on."
                        , metavar="loss_function")
    general_params_parser.add_argument("-w_i", "--weight_init", type=str, default="random", choices=["random", "xavier"]
                        , help="Weight and bias initialization method."
                        , metavar="weight_init")

    optimizer_parser.add_argument("-o", "--optimizer", type=str, default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
                        , help="Optimization algorithm"
                        , metavar="optimzer")
    optimizer_parser.add_argument("-lr", "--learning_rate", type=float, default=0.1
                        , help="Learning rate used to optimize model parameters."
                        , metavar="learning_rate")
    optimizer_parser.add_argument("-m", "--momentum", type=float, default=0.5
                        , help="Momentum used by momentum and nag optimizers."
                        , metavar="momentum")
    optimizer_parser.add_argument("-beta", "--beta", type=float, default=0.5
                        , help="Beta used by rmsprop optimizer."
                        , metavar="beta")
    optimizer_parser.add_argument("-beta1", "--beta1", type=float, default=0.5
                        , help="Beta1 used by adam and nadam optimizers."
                        , metavar="beta1")
    optimizer_parser.add_argument("-beta2", "--beta2", type=float, default=0.5
                        , help="Beta2 used by adam and nadam optimizers."
                        , metavar="beta2")
    optimizer_parser.add_argument("-eps", "--epsilon", type=float, default=0.000001
                        , help="Epsilon used by optimizers."
                        , metavar="epsilon")
    optimizer_parser.add_argument("-w_d", "--weight_decay", type=float, default=0
                        , help="Weight decay used by optimizers."
                        , metavar="weight_decay")

    nn_arch_parser.add_argument("-nhl", "--num_layers", type=int, default=1
                        , help="Number of hidden layers used in feedforward neural network."
                        , metavar="number_of_hidden_layers")
    nn_arch_parser.add_argument("-sz", "--hidden_size", type=int, default=4
                        , help="Number of hidden neurons in a feedforward layer."
                        , metavar="size_of_hidden_layer")
    nn_arch_parser.add_argument("-a", "--activation", type=str, default="logistic", choices=["identity", "logistic", "tanh", "relu"]
                        , help="Activation function of each layer."
                        , metavar="activation_function")


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = cli_parser()

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

    # Get CLI params
    wandb_project = args["wandb_project"]
    wandb_entity = args["wandb_entity"]
    dataset = args["dataset"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    loss = args["loss"]
    weight_init = args["weight_init"]
    optimizer = args["optimizer"]
    learning_rate = args["learning_rate"]
    momentum = args["momentum"]
    beta = args["beta"]
    beta1 = args["beta1"]
    beta2 = args["beta2"]
    epsilon = args["epsilon"]
    weight_decay = args["weight_decay"]
    num_layers = args["num_layers"]
    hidden_size = args["hidden_size"]
    activation = args["activation"]

    # Setting optimizer hyperparameters as required
    if optimizer in ["adam", "nadam"]:
        wandb_optimizer_params[optimizer]["beta1"] = beta1
        wandb_optimizer_params[optimizer]["beta2"] = beta2
    else:
        wandb_optimizer_params[optimizer]["beta"] = beta

    # Setting epsilon value
    constants.epsilon = epsilon

    # Start wandb
    wandb.init(project=wandb_project, entity=wandb_entity)

    # Defining the network structure
    dict_neural_network_structure = {}
    dict_neural_network_structure[0] = 28*28
    dict_neural_network_structure[num_layers + 1] = 10
    for hidden_layer in range(1, num_layers + 1):
        dict_neural_network_structure[hidden_layer] = hidden_size
    
    # Defining the activations
    activation_dict = {0 : activation, 1 : "softmax"}
    
    # Creating the neural network instance
    nn1 = neural_network(dict_neural_network_structure, activation_dict, nn_init=wandb_initializer[weight_init])
    
    # Defining the optimizer
    optimizer = wandb_optimizer[optimizer](nn1, eta=learning_rate, weight_decay=weight_decay, **wandb_optimizer_params[optimizer])
    
    # Fitting the model
    nn1.fit(wandb, optimizer, X_train, y_train, X_val, y_val, epochs=epochs, minibatch_size=batch_size)
    
    # Checking for accuracy
    output = nn1.predict(X_test)
    accuracy_metrics = get_accuracy_metrics(y_test, output)
    for val in accuracy_metrics:
        print(val*100)
        wandb.log({"test_accuracy" : val*100})

    wandb.finish()