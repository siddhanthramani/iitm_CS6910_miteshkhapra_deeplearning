import random

from constants import SOS_token, EOS_token


class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {"SOS": 0, "EOS": 1, "UNKNOWN": 2}
        self.char2count = {}
        self.index2char = {0: "SOS", 1: "EOS", 2: "UNKNOWN"}
        self.n_chars = 2  # Count SOS and EOS

    def addword(self, word):
        # Splits a word into individual characters and adds them to the object instance
        for char in list(word):
            self.addchar(char)

    def addchar(self, char):
        # If the char is being encountered for the first time
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            # Increment count of unique chars
            self.n_chars += 1
        else:
            # Increment count to depict how many times this char has occured
            self.char2count[char] += 1


def readLangs(data_path, lang1, lang2):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(data_path, encoding="utf-8").\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[s for s in l.split(",")] for l in lines]

    # Make Lang instances
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def  prepareData(data_path, lang1, lang2):
    input_lang, output_lang, pairs = readLangs(data_path, lang1, lang2)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting chars...")
    for pair in pairs:
        input_lang.addword(pair[0])
        output_lang.addword(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, pairs



