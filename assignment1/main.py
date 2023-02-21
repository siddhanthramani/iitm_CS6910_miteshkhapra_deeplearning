from utils import load
from nn import neural_network


data = load()
num_neurons_dict = {0 : 28*28, 1 : 40, 2 : 10}
activation_dict = {0 : "linear", 1 : "logistic", 2 : "softmax"}
print(data["train_X"].shape)
print(data["train_y"].shape)
nn1 = neural_network(num_neurons_dict, activation_dict)
output = nn1.predict(data["train_X"][0])
# print(output)

nn1.fit(data["train_X"], data["train_y"], eta=0.1)
output = nn1.predict(data["train_X"][0])
print(output)