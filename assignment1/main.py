# Imports required modules
from nn_utils.data_utils import load, get_random_class_indices, do_checks
from nn_utils.output_utils import get_accuracy_metrics
from nn_core.nn_main import neural_network
from nn_core.nn_optimizer import *
from nn_user.weight_init import xavier_init
from nn_user.augment_data import augment_data
import numpy as np
from sklearn.model_selection import train_test_split
import json

import wandb

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

# WandB
with open('./assignment1/wandb_config.json') as f:
    wandb_config = json.load(f)
wandb.init(project="test", config=wandb_config)


# Defining the network structure
num_neurons_dict = {0:28*28, 1:10, 2:10}
activation_dict = {0 : "tanh", 1 : "softmax"}
nn1 = neural_network(num_neurons_dict, activation_dict, nn_init=xavier_init, weight_type="w")
# nn1 = neural_network(num_neurons_dict, activation_dict, nn_init=np.random.rand) #, weight_type="w"

# Fitting the data to the model
# optimizer = regular_gradient_descent(nn1, eta=0.00001)
# optimizer = nestrov_accelerated_gradient_descent(nn1, eta=0.00001, beta=0.9)
# optimizer = Adadelta(nn1, eta=0.001, beta=0.9)
optimizer = NAdam(nn1, eta=0.001, beta1=0.9, beta2=0.99)
print(len(X_train), len(y_train))
# X_train, y_train = augment_data(X_train, y_train, mode="replace")

print(len(X_train), len(y_train))
nn1.fit(wandb, optimizer, X_train, y_train, X_val, y_val, epochs=200, minibatch_size = 0)

# Checking for accuracy

output = nn1.predict(X_test)
accuracy_metrics = get_accuracy_metrics(y_test, output)
for val in accuracy_metrics:
    print(val*100)
    wandb.log({"test_accuracy" : val*100})

wandb.finish()