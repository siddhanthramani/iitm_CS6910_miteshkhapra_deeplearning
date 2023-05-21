import torch
import torch.nn as nn

SOS_token = 0
EOS_token = 1
UNKNOWN_token = 2
MAX_LENGTH = 50
teacher_forcing_ratio = 0.5


rnn_type_dict = {
    "rnn" : nn.RNN,
    "gru" : nn.GRU,
    "lstm" : nn.LSTM
}


def get_device():
    # Device is a cuda device if compatible NVidia GPU is found.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

