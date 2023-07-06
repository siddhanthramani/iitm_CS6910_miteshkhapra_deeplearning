from __future__ import unicode_literals, print_function, division
from io import open
import json
import wandb

from seq2seq import *
from preprocessor import *
from train import *
from evaluate import *
from metrics import *

from constants import  get_device

# Device is a cuda device if compatible NVidia GPU is found.
device = get_device()
decoder_type_dict = {
    "regular" : DecoderRNN,
    "attention" : AttnDecoderRNN
}

def main():
    print("RUN STARTS")
    # Start wandb
    with wandb.init() as run:

        # Initialize wandb params
        n_iters = wandb.config.n_iters
        input_embedding_size = wandb.config.input_embedding_size
        number_of_encoder_layers = wandb.config.number_of_encoder_layers
        number_of_decoder_layers = wandb.config.number_of_decoder_layers
        hidden_layer_size = wandb.config.hidden_layer_size
        cell_type = wandb.config.cell_type
        decoder_type = wandb.config.decoder_type
        bidirectional = wandb.config.bidirectional
        dropout = wandb.config.dropout
        learning_rate = wandb.config.learning_rate
        
        # Definng the name of the run
        run_name = "ies{}_ne{}_nd{}_hls{}_ct{}_dt{}_bd{}_dp{}".format(input_embedding_size, number_of_encoder_layers, number_of_decoder_layers
                                                 , hidden_layer_size, cell_type, decoder_type
                                                 , bidirectional, dropout)
        run.name = run_name
        # Defining the network structure
        encoder = EncoderRNN(rnn_type=cell_type, input_embedding_dict_size=input_lang.n_chars
                              , input_embedding_size=input_embedding_size, hidden_size=hidden_layer_size
                              , num_layers=number_of_encoder_layers).to(device)
        
        decoder = decoder_type_dict[decoder_type](rnn_type=cell_type, output_embedding_dict_size=output_lang.n_chars\
                                                   , output_embedding_size=hidden_layer_size, hidden_size=hidden_layer_size
                                                   , num_layers=number_of_decoder_layers).to(device)
        trainIters(trainpairs, input_lang, output_lang, encoder, decoder, n_iters, learning_rate)
        
        # Checking for accuracy
        predictions, targets = evaluate_batch(valpairs, encoder, decoder)
        accuracy = get_accuracy(predictions, targets)
        wandb.log({"val_accuracy" : accuracy * 100})


if __name__ == "__main__":

    # Loading the data - Done before sweeps to prevent multiple time consuming calls
    input_lang, output_lang, trainpairs = prepareData("../data/Aksharantar/aksharantar_sampled/aksharantar_sampled/tam/tam_train.csv", 'eng', 'tam')
    _, _, valpairs = prepareData("../data/Aksharantar/aksharantar_sampled/aksharantar_sampled/tam/tam_valid.csv", 'eng', 'tam')
    _, _, testpairs = prepareData("../data/Aksharantar/aksharantar_sampled/aksharantar_sampled/tam/tam_test.csv", 'eng', 'tam')

    # WandB sweep
    with open("./assignment3/wandb_sweep_config.json") as f:
        sweep_config = json.load(f)
    sweep_id = wandb.sweep(project="assignment3_attemptv1", sweep=sweep_config)
    wandb.agent(sweep_id, function=main, count=20)