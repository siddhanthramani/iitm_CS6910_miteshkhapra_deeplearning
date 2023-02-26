from utils import load
from nn import neural_network
import numpy as np

data = load()
num_neurons_dict = {0:28*28, 1:10}
activation_dict = {0 : "logistic", 1 : "softmax"}
print(data["train_X"].shape)
print(data["train_y"].shape)
print(data["train_X"][0:10].shape)


nn1 = neural_network(num_neurons_dict, activation_dict, nn_init_random_max=0.01)
# output = nn1.predict(data["train_X"][0])
# print(output)

nn1.fit(data["train_X"]/255, data["train_y"], eta=0.00001, epochs=10, minibatch_size = 64)
# output = nn1.predict(data["train_X"][0])
# print(data["train_y"][0])
# print(output)

# output = nn1.predict(data["train_X"][1])
# print(data["train_y"][1])
# print(output)

accuracy = 0
total = 0
for index, x in enumerate(data["test_X"]):
    output = nn1.predict(x, 0)
    prediction = np.argmax(output) + 1
    y_hat = data["test_y"][index]
    if prediction == y_hat:
        accuracy += 1
    total+=1
print(accuracy*100/total)
