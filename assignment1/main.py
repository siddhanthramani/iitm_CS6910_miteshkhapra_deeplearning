from utils import load
from nn import neural_network


data = load()
num_neurons_dict = {0 : 28*28, 1 : 40, 2 : 10}
activation_dict = {0 : "linear", 1 : "logistic", 2 : "softmax"}
print(data["train_X"].shape)
print(data["train_y"].shape)
print(data["train_X"][0:10].shape)

nn1 = neural_network(num_neurons_dict, activation_dict, nn_init_random_max=100)
# output = nn1.predict(data["train_X"][0])
# print(output)

nn1.fit(data["train_X"][0:10], data["train_y"][0:10], eta=0.1, epochs=20)
output = nn1.predict(data["train_X"][0])
print(output)