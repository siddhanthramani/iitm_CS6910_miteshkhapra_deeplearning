# Imports required modules
from nn_utils.data_utils import load, plot_random_image_per_class, do_data_checks
from nn_utils.output_utils import get_accuracy_metrics, plot_confusion_matrix
from nn_core.nn_main import neural_network
from nn_core.nn_optimizer import *
from nn_user.weight_init import *
from nn_user.augment_data import augment_data
from sklearn.model_selection import train_test_split
import wandb
import os
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
    , "momentum_gradient_descent" : {}
    , "nestrov_accelerated_gradient_descent" : {}
    , "rmsprop" : {}
    , "adam" : {}
    , "nadam" : {}
}

# Parser for command line arguments
def cli_parser():
    parser = ArgumentParser(prog="train"
                            , description="Train and test a neural network. Track it via wandb. Use python assignment1\\train.py @wandb_expt_args.text to read flags from text file."
                            , fromfile_prefix_chars="@")

    wandb_parser = parser.add_argument_group("Wandb Group", "Options related to wandb.")
    general_params_parser = parser.add_argument_group("General Parameters Group", "Options related to general training parameters.")
    optimizer_parser = parser.add_argument_group("Optimizer Group", "Options related to optimizers and its parameters.")
    nn_arch_parser = parser.add_argument_group("Neural Network Architecture Group"
                                            , "Options related to defining the network structure.")

    # Parser for WandB related arguments
    wandb_parser.add_argument("-wp", "--wandb_project", type=str, default="Assignment1_test"
                        , help="Project name used to track experiments in Weights & Biases dashboard."
                        , metavar="project_name")
    wandb_parser.add_argument("-we", "--wandb_entity", type=str, default="None"
                        , help="Wandb Entity used to track experiments in the Weights & Biases dashboard."
                        , metavar="entity_name")

    # Parser for general arguments
    general_params_parser.add_argument("-d", "--dataset", type=str, default="fashionmnist", choices=["mnist", "fashionmnist"]
                        , help="Choose the dataset you would like to model."
                        , metavar="dataset")
    general_params_parser.add_argument("-e", "--epochs", type=int, default=100
                        , help="Number of epochs to train neural network."
                        , metavar="number_of_epochs")
    general_params_parser.add_argument("-b", "--batch_size", type=int, default=64
                        , help="Batch size used to train neural network."
                        , metavar="batch_size")
    general_params_parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"]
                        , help="Loss function to optimize the model on."
                        , metavar="loss_function")
    general_params_parser.add_argument("-w_i", "--weight_init", type=str, default="xavier", choices=["random", "xavier"]
                        , help="Weight and bias initialization method."
                        , metavar="weight_init")
    general_params_parser.add_argument("-wimf", "--weight_init_multiplication_factor", type=int, default=1
                        , help="Multiply the weight initialzation with this factor to scale init values"
                        , metavar="weight_init_multiplication_factor")
    general_params_parser.add_argument("-ado", "--augment_data_on", type=int, default=0, choices=[0, 1]
                        , help="Does data need to be augmented? 0-No, 1-Yes"
                        , metavar="augment_data_on")
    
    # Parser for optimizer related arguments
    optimizer_parser.add_argument("-o", "--optimizer", type=str, default="momentum_gradient_descent", choices=["regular_gradient_descent", "momentum_gradient_descent", "nestrov_accelerated_gradient_descent", "rmsprop", "adam", "nadam"]
                        , help="Optimization algorithm"
                        , metavar="optimzer")
    optimizer_parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4
                        , help="Learning rate used to optimize model parameters."
                        , metavar="learning_rate")
    optimizer_parser.add_argument("-beta", "--beta", type=float, default=0.9
                        , help="Beta used by momentum, nag optimizers, and rmsprop optimizer."
                        , metavar="beta")
    optimizer_parser.add_argument("-beta1", "--beta1", type=float, default=0.9
                        , help="Beta1 used by adam and nadam optimizers."
                        , metavar="beta1")
    optimizer_parser.add_argument("-beta2", "--beta2", type=float, default=0.99
                        , help="Beta2 used by adam and nadam optimizers."
                        , metavar="beta2")
    optimizer_parser.add_argument("-eps", "--epsilon", type=float, default=1e-8
                        , help="Epsilon used by optimizers."
                        , metavar="epsilon")
    optimizer_parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0001
                        , help="Weight decay used by optimizers."
                        , metavar="weight_decay")

    # Parser for neural network architecture related arguments
    nn_arch_parser.add_argument("-nhl", "--number_of_hidden_layers", type=int, default=1
                        , help="Number of hidden layers used in feedforward neural network."
                        , metavar="number_of_hidden_layers")
    nn_arch_parser.add_argument("-sz", "--size_of_every_hidden_layer", type=int, default=64
                        , help="Number of hidden neurons in a feedforward layer."
                        , metavar="size_of_every_hidden_layer")
    nn_arch_parser.add_argument("-a", "--activation", type=str, default="tanh", choices=["identity", "logistic", "tanh", "relu"]
                        , help="Activation function of each layer."
                        , metavar="activation_function")

    args = vars(parser.parse_args())
    print(args)
    return args

if __name__ == "__main__":
    # Get CLI params
    args = cli_parser()

    wandb_project = args["wandb_project"]
    wandb_entity = args["wandb_entity"]
    dataset = args["dataset"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    loss = args["loss"]
    weight_init = args["weight_init"]
    weight_init_multiplication_factor = args["weight_init_multiplication_factor"]
    augment_data_on = args["augment_data_on"]
    optimizer = args["optimizer"]
    learning_rate = args["learning_rate"]
    beta = args["beta"]
    beta1 = args["beta1"]
    beta2 = args["beta2"]
    epsilon = args["epsilon"]
    weight_decay = args["weight_decay"]
    number_of_hidden_layers = args["number_of_hidden_layers"]
    size_of_every_hidden_layer = args["size_of_every_hidden_layer"]
    activation = args["activation"]

    # Setting optimizer hyperparameters as required
    if optimizer in ["adam", "nadam"]:
        wandb_optimizer_params[optimizer]["beta1"] = beta1
        wandb_optimizer_params[optimizer]["beta2"] = beta2
    elif optimizer in ["momentum_gradient_descent", "nestrov_accelerated_gradient_descent", "rmsprop"]:
        wandb_optimizer_params[optimizer]["beta"] = beta
    elif optimizer in ["regular_gradient_descent"]:
        pass

    # Setting epsilon value
    global_constants = constants.global_constants(epsilon=epsilon)

    # Loading the data
    data = load(dataset=dataset)
    X = data["train_X"]/255
    y = data["train_y"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
    X_test = data["test_X"]/255
    y_test = data["test_y"]

    # Checking the data for infinities or nans
    do_data_checks(X_train, y_train)
    do_data_checks(X_val, y_val)
    do_data_checks(X_test, y_test)

    # Augment data
    if augment_data_on:
        X_train, y_train = augment_data(X_train, y_train, mode="append")

    # Start wandb
    if wandb_entity == "None":
        wandb.init(project=wandb_project)
    else:
        wandb.init(project=wandb_project, entity=wandb_entity)
    
    # Defining the network structure
    dict_neural_network_structure = {}
    dict_neural_network_structure[0] = 28*28
    dict_neural_network_structure[number_of_hidden_layers + 1] = 10
    for hidden_layer in range(1, number_of_hidden_layers + 1):
        dict_neural_network_structure[hidden_layer] = size_of_every_hidden_layer
    dict_neural_network_structure = {0:28*28, 1:64, 2:32, 3:16, 4:10}
    # Defining the activations
    activation_dict = {0 : activation, 1 : "softmax"}
    
    # Creating the neural network instance
    nn1 = neural_network(global_constants, dict_neural_network_structure, activation_dict, loss_function=loss
                         , weight_init=wandb_initializer[weight_init], weight_init_multiplication_factor=weight_init_multiplication_factor)
    
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
    
    # Plotting the confusion matrix and logging it on wandb
    plot_confusion_matrix(wandb, y_test, output)

    # Finishing the wandb experiment
    wandb.finish()
