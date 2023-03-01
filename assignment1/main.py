# Imports required modules
from utils import load, get_random_class_indices, do_checks
from nn import neural_network
from helper import xavier_init
import numpy as np
from sklearn.model_selection import train_test_split

# Loading the data
data = load()
X = data["train_X"]/255
y = data["train_y"]
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
X_test = data["test_X"]/255
y_test = data["test_y"]

# Checking for infinities or nans
do_checks(X_train, y_train)
do_checks(X_val, y_val)
do_checks(X_test, y_test)

# Defining the network structure
num_neurons_dict = {0:28*28, 1:10, 2:10}
activation_dict = {0 : "tanh", 1 : "softmax"}
nn1 = neural_network(num_neurons_dict, activation_dict, nn_init=xavier_init, weight_type="w")
# nn1 = neural_network(num_neurons_dict, activation_dict, nn_init=np.random.rand) #, weight_type="w"

# Fitting the data to the model
nn1.fit(X_train, y_train, X_val, y_val, eta=0.00001, epochs=50, minibatch_size = 0)
# nn1.fit(data["train_X"]/255, data["train_y"], gradient_descent_type = "momentum", eta=0.0001, beta = 0.09, epochs=20, minibatch_size = 128)

# Checking for accuracy

output = nn1.predict(X_test)
accuracy_metrics = nn1.get_accuracy_metrics(y_test, output)
for val in accuracy_metrics:
    print(val)
