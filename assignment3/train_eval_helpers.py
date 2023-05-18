import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import time
import math

from constants import SOS_token, EOS_token, get_device


# GPU or CPU
device = get_device()


def indexesFromword(lang, word):
    return [lang.char2index[char] for char in list(word)]


def tensorFromword(lang, word):
    indexes = indexesFromword(lang, word)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromword(input_lang, pair[0])
    target_tensor = tensorFromword(output_lang, pair[1])
    return (input_tensor, target_tensor)


plt.switch_backend('agg')
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))