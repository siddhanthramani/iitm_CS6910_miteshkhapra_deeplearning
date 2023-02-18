from utils import load
from nn import neural_network


data = load()
num_neurons_dict = {0 : 28*28, 1 : 40, 2 : 10}
activation_dict = {0 : "linear", 1 : "logistic"}
nn1 = neural_network(num_neurons_dict, activation_dict)
output = nn1.forward_pass(data["train_X"][0].reshape(28*28, 1))
print(output)