import torch
import torch.nn as nn
from seq2seq import *

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 50
teacher_forcing_ratio = 0.5


rnn_type_dict = {
    "rnn" : nn.RNN,
    "gru" : nn.GRU,
    "lstm" : nn.LSTM
}

decoder_type_dict = {
    "regular" : DecoderRNN,
    "attention" : AttnDecoderRNN
}


def get_device():
    # Device is a cuda device if compatible NVidia GPU is found.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

