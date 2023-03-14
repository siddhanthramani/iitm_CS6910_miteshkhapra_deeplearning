from argparse import ArgumentParser

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
general_params_parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mean_mean_squared_error", "cross_entropy"]
                    , help="Loss function to optimize the model on."
                    , metavar="loss_function")
general_params_parser.add_argument("-w_i", "--weight_init", type=str, default="random", choices=["random", "Xavier"]
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