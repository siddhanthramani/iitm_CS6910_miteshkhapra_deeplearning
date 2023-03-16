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

import numpy as np
eta = 0.00005
# 2x1
x = np.array([-1, 1]).reshape(2, 1)
# 2x1
y = np.array([1, 0]).reshape(2, 1)
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

h0s = x
a1s = np.add(np.dot(w1s, h0s), b1s)
h1s = logistic(a1s)
a2s = np.add(np.dot(w2s, h1s), b2s)
h2s = softmax(a2s)

da2s = -(np.array([1, 0]).reshape(2, 1) - h2s)
dw2s = np.dot(da2s, np.transpose(h1s))
db2s = da2s
dh1s = np.dot(np.transpose(w2s), da2s)
da1s = dh1s * (h1s * (1-h1s))

dw1s = np.dot(da1s, np.transpose(h0s))
db1s = da1s

w1n = w1s - eta * dw1s
w2n = w2s - eta * dw2s
b1n = b1s - eta * db1s
b2n = b2s - eta * db2s

h0n = x
a1n = np.add(np.dot(w1n, h0n), b1n)
h1n = logistic(a1n)
a2n = np.add(np.dot(w2n, h1n), b2n)
h2n = softmax(a2n)

print("a1s")
print(a1s)
print("a2s")
print(a2s)


print("h1s")
print(h1s)
print("h2s")
print(h2s)


num_neurons_dict = {0:2, 1:4, 2:4, 2:2}
activation_dict = {0 : "logistic", 1 : "softmax"}

nn1 = neural_network(num_neurons_dict, activation_dict, weight_init=xavier_init, weight_type="w")
# nn1 = neural_network(num_neurons_dict, activation_dict, weight_init=np.random.rand) #, weight_type="w"
data["train_X"] = np.array([np.array([-255, 255]), np.array([-255, -255]) 
                            , np.array([255, -255]), np.array([255, 255])]).reshape(4, 2, 1)
data["train_y"] = np.array([1, 1, 1, 2]).reshape(4, 1)

print(data["train_X"].shape)
print(data["train_y"].shape)

nn1.fit(data["train_X"]/255, data["train_y"], eta=1, epochs=100, minibatch_size = 0)
# nn1.fit(data["train_X"]/255, data["train_y"], gradient_descent_type = "momentum", eta=0.0001, beta = 0.09, epochs=20, minibatch_size = 128)
print(nn1.W)
print(nn1.b)

# output = nn1.predict(data["train_X"][random_sample_indices][0])
# print(data["train_y"][random_sample_indices][0])
# print(np.argmax(output) + 1)

# output = nn1.predict(data["train_X"][random_sample_indices][1])
# print(data["train_y"][random_sample_indices][1])
# print(np.argmax(output) + 1)

# accuracy = 0
# total = 0

# for index, x in enumerate(data["train_X"]):
#     output = nn1.predict(x, 0)
#     prediction = np.argmax(output) + 1
#     y = data["train_y"][index]
#     if prediction == y:
#         accuracy += 1
#     total+=1
# print(accuracy*100/total)

output = nn1.predict(np.array([-1, -1]).reshape(2, 1), 0)
print(np.argmax(output) + 1)
output = nn1.predict(np.array([1, -1]).reshape(2, 1), 0)
print(np.argmax(output) + 1)
output = nn1.predict(np.array([-1, 1]).reshape(2, 1), 0)
print(np.argmax(output) + 1)
output = nn1.predict(np.array([1, 1]).reshape(2, 1), 0)
print(np.argmax(output) + 1)


        # self.W[1] = np.array([np.array([0.22491641, 0.02180481])
        #             , np.array([0.26076326, 0.12215034])
        #             , np.array([0.25546389, 0.21189626])
        #             , np.array([0.10628627, 0.01291387])])
        # # 2x4
        # self.W[2] = np.array([np.array([0.00461026, 0.33884685, 0.25269835, 0.10613222])
        #                     , np.array([0.21316539, 0.01781372, 0.03532867, 0.25390077])])

        # # 4x1
        # self.b[1] = np.array([np.array([0.05422128])
        #                     , np.array([0.37212847])
        #                     , np.array([0.16308096])
        #                     , np.array([0.01445151])])

        # # 2x1
        # self.b[2] = np.array([np.array([0.54795226])
        #                     , np.array([0.50381235])])

        # print(self.W)
        # print(self.b)