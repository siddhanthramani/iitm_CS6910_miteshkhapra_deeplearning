from __future__ import unicode_literals, print_function, division
from io import open
import json
import wandb

from seq2seq import *
from preprocessor import *
from train import *

from constants import get_device

# Device is a cuda device if compatible NVidia GPU is found.
device = get_device()

def main():
    # Start wandb
    with wandb.init() as run:

        # Initialize wandb params
        epochs = wandb.config.epochs
        input_embedding_size = wandb.config.input_embedding_size
        number_of_encoder_layers = wandb.config.number_of_encoder_layers
        number_of_decoder_layers = wandb.config.number_of_decoder_layers
        hidden_layer_size = wandb.config.hidden_layer_size
        cell_type = wandb.config.cell_type
        decoder_type = wandb.config.decoder_type
        bidirectional = wandb.config.bidirectional
        dropout = wandb.config.dropout
        
        # Definng the name of the run
        run_name = "ies{}_ne{}_nd{}_hls{}_ct{}_dt{}_bd{}_dp{}".format(input_embedding_size, number_of_encoder_layers, number_of_decoder_layers
                                                 , hidden_layer_size, cell_type, decoder_type
                                                 , bidirectional, dropout)
        run.name = run_name
        # Defining the network structure
        input_lang, output_lang, pairs = prepareData("../../data/Aksharantar/aksharantar_sampled/aksharantar_sampled/tam/tam_train.csv", 'eng', 'tam')

        encoder1 = EncoderRNN(rnn_type="gru", input_embedding_dict_size=input_lang.n_chars, input_embedding_size=hidden_layer_size, hidden_size=hidden_layer_size).to(device)
        # attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_chars, dropout_p=0.1).to(device)
        decoder1 = DecoderRNN(rnn_type="gru", output_embedding_dict_size=output_lang.n_chars, output_embedding_size=hidden_layer_size, hidden_size=hidden_layer_size).to(device)
        trainIters(encoder1, decoder1, 5, print_every=1)
        
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


if __name__ == "__main__":

    # Loading the data - Done before sweeps to prevent multiple time consuming calls
    data = load(dataset="mnist")
    X = data["train_X"]/255
    y = data["train_y"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
    X_test = data["test_X"]/255
    y_test = data["test_y"]

    # Checking the data for infinities or nans
    do_data_checks(X_train, y_train)
    do_data_checks(X_val, y_val)
    do_data_checks(X_test, y_test)

    # Augment data - if required
    # X_train, y_train = augment_data(X_train, y_train, mode="replace")
    
    # WandB sweep
    with open("./assignment1/wandb_sweep_config.json") as f:
        sweep_config = json.load(f)
    sweep_id = wandb.sweep(project="crossentropy_sweep_mnist_2", sweep=sweep_config)
    wandb.agent(sweep_id, function=main, count=20)
# data, X_train, X_val, X_test, y_train, y_val, y_test