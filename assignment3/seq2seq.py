import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import get_device, rnn_type_dict, MAX_LENGTH


# Cuda or CPU
device = get_device()


class EncoderRNN(nn.Module):
    def __init__(self, rnn_type, input_embedding_dict_size, input_embedding_size, hidden_size, num_layers=1, dropout=0, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.input_embedding_dict_size = input_embedding_dict_size
        self.embedding_size = input_embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(input_embedding_dict_size, input_embedding_size)
        self.encoder = rnn_type_dict[rnn_type](input_embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input, hidden):
        embedded_input = self.embedding(input).view(1, 1, -1)
        output, hidden = self.encoder(embedded_input, hidden)
        if self.rnn_type == "lstm":
            hidden = hidden[0]
        return output, hidden
    
    def initHidden(self):
        if self.bidirectional:
            return torch.zeros(2*(self.num_layers), 1, self.hidden_size, device=device)
        else:
            return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
    

class DecoderRNN(nn.Module):
    def __init__(self, rnn_type, output_embedding_dict_size, output_embedding_size, hidden_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_embedding_dict_size = output_embedding_dict_size
        self.output_embedding_size = output_embedding_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_embedding_dict_size, output_embedding_size)
        self.rnn_type = rnn_type
        self.decoder = rnn_type_dict[rnn_type](output_embedding_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_embedding_dict_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, output, hidden):
        embedded_output = self.embedding(output).view(1, 1, -1)
        embedded_output_relued = F.relu(embedded_output)
        output, hidden = self.decoder(embedded_output_relued, hidden)
        if self.rnn_type == "lstm":
            hidden = hidden[0]
        final_output = self.softmax(self.out(output[0]))
        return final_output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
    

class AttnDecoderRNN(nn.Module):
    def __init__(self, rnn_type, output_embedding_dict_size, output_embedding_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_embedding_dict_size = output_embedding_dict_size
        self.output_embedding_size = output_embedding_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_embedding_dict_size, self.output_embedding_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.decoder = rnn_type_dict[rnn_type](self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_embedding_dict_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.decoder(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)