from utils import load
from nn import neural_network


data = load()
num_neurons_dict = {0 : 28*28, 1 : 20, 2 : 20, 3:20, 4:20, 5:20, 6:10}
activation_dict = {0 : "logistic", 1 : "softmax"}
print(data["train_X"].shape)
print(data["train_y"].shape)
print(data["train_X"][0:10].shape)

nn1 = neural_network(num_neurons_dict, activation_dict, nn_init_random_max=0.01)
# output = nn1.predict(data["train_X"][0])
# print(output)

nn1.fit(data["train_X"]/255, data["train_y"], eta=0.000001, epochs=5)
output = nn1.predict(data["train_X"][0])
print(data["train_y"][0])
print(output)