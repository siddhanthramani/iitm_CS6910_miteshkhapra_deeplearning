# One of the best things you can do test if a new code is working is to run it line by line in debug mode and observe the changes.
# For this you will need your code and also a reference code to compare your intermediary results with.
# The test.py contains this reference code for the regular_gradient_descent_algorithm
# There are 2 inputs, 2 outputs and a single hidden layer with 4 neurons
# The weights are initialized to prevent randomness across multiple debug sessions.

# The main.py - your code should contain the following 
# The data input for implementing the AND function
# data = {}
# data["train_X"] = np.array([np.array([-255, 255]), np.array([-255, -255]) 
#                             , np.array([255, -255]), np.array([255, 255])]).reshape(4, 2, 1)
# data["train_y"] = np.array([1, 1, 1, 2]).reshape(4, 1)
# The neural network structure as below
# num_neurons_dict = {0:2, 1:4, 3:2}
# activation_dict = {0 : "logistic", 1 : "softmax"}


# Run your code for a specific step and compare it with the below reference model
# If all the variables match, your code is ideally working fine.
# Also at the end of the main.py run, the model should have learned the AND function.

# test.py
# Imports
import numpy as np

# Defining math.utils functions agian for easy reference
def positive_logistic(input):
    return 1 / (1 + np.exp(-input))
def negative_logistic(input):
    exp_inp = np.exp(input)
    return exp_inp / (1 + exp_inp)

def logistic(input):
    pos_inputs = input >=0
    neg_inputs = ~pos_inputs

    result = np.empty_like(input, dtype=np.float)
    result[pos_inputs] = positive_logistic(input[pos_inputs])
    result[neg_inputs] = negative_logistic(input[neg_inputs])
    return result

def softmax(input):
    softmax_num = np.exp(input - np.max(input))
    return (softmax_num/ np.sum(softmax_num))

# Learning rate used
eta = 0.00005
# Input (based on which input has been passed to your main.py model for that specific step) 2x1
x = np.array([-1, 1]).reshape(2, 1)
# Output (based on which input has been passed to your main.py model for that specific step) 2x1
y = np.array([1, 0]).reshape(2, 1)

# Defining weights
# 4x2
w1s = np.array([np.array([0.22491641, 0.02180481])
                    , np.array([0.26076326, 0.12215034])
                    , np.array([0.25546389, 0.21189626])
                    , np.array([0.10628627, 0.01291387])])
# 2x4
w2s = np.array([np.array([0.00461026, 0.33884685, 0.25269835, 0.10613222])
                    , np.array([0.21316539, 0.01781372, 0.03532867, 0.25390077])])

# 4x1
b1s = np.array([np.array([0.05422128])
                      , np.array([0.37212847])
                      , np.array([0.16308096])
                      , np.array([0.01445151])])

# 2x1
b2s = np.array([np.array([0.54795226])
                      , np.array([0.50381235])])

# Forward pass
h0s = x
a1s = np.add(np.dot(w1s, h0s), b1s)
h1s = logistic(a1s)
a2s = np.add(np.dot(w2s, h1s), b2s)
h2s = softmax(a2s)

# Backward pass
da2s = -(np.array([1, 0]).reshape(2, 1) - h2s)
dw2s = np.dot(da2s, np.transpose(h1s))
db2s = da2s
dh1s = np.dot(np.transpose(w2s), da2s)
da1s = dh1s * (h1s * (1-h1s))
dw1s = np.dot(da1s, np.transpose(h0s))
db1s = da1s

# Update
w1n = w1s - eta * dw1s
w2n = w2s - eta * dw2s
b1n = b1s - eta * db1s
b2n = b2s - eta * db2s

# Prediction pass
h0n = x
a1n = np.add(np.dot(w1n, h0n), b1n)
h1n = logistic(a1n)
a2n = np.add(np.dot(w2n, h1n), b2n)
h2n = softmax(a2n)

# Printing the values of reference model to compare 
print("a1s")
print(a1s)
print("a2s")
print(a2s)
print("h1s")
print(h1s)
print("h2s")
print(h2s)
