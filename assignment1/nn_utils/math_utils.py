import numpy as np

# A set of two logistic functions to ensure stable outputs 
# for positive and negative inputs respectively 
def positive_logistic(input):
    return 1 / (1 + np.exp(-input))
def negative_logistic(input):
    exp_inp = np.exp(input)
    return exp_inp / (1 + exp_inp)


# Numerically stable logistic - https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
def numerically_stable_logistic(input):
    pos_inputs = input >=0
    neg_inputs = ~pos_inputs

    result = np.empty_like(input, dtype=np.float)
    result[pos_inputs] = positive_logistic(input[pos_inputs])
    result[neg_inputs] = negative_logistic(input[neg_inputs])
    return result


# Numerically stable softmax - https://cs231n.github.io/linear-classify/#softmax
def numerically_stable_softmax(input):
    softmax_num = np.exp(input - np.max(input))
    return (softmax_num/ np.sum(softmax_num))