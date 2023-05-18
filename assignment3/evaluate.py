import torch

import random

from constants import SOS_token, EOS_token, MAX_LENGTH,  get_device
from train_eval_helpers import *


device = get_device()


def evaluate(input_lang, output_lang, encoder, decoder, word, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromword(input_lang, word)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_chars = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            # decoder_output, decoder_hidden, decoder_attention = decoder(
            #     decoder_input, decoder_hidden, encoder_outputs)
            # decoder_attentions[di] = decoder_attention.data
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_chars.append('<EOS>')
                break
            else:
                decoded_chars.append(output_lang.index2char[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_chars, decoder_attentions[:di + 1]
    

def evaluateRandomly(pairs, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        # output_chars, attentions = evaluate(encoder, decoder, pair[0])
        output_chars = evaluate(encoder, decoder, pair[0])
        output_word = ' '.join(output_chars)
        print('<', output_chars)
        print('')