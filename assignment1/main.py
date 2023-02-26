from utils import load, get_random_class_indices
from nn import neural_network
from helper import xavier_init
import numpy as np



data = load()
print(data["train_X"].shape)
print(data["train_y"].shape)
print(data["train_X"][0:10].shape)

random_sample_indices = get_random_class_indices(data["train_y"])
random_sample_indices = np.array(random_sample_indices).reshape(-1)

for x, y in zip(data["train_X"]/255, data["train_y"]):
    y = np.array([1 if i==(y-1) else 0 for i in range(10)]).reshape(10, 1)
    if not (np.isfinite(x).all() or np.isfinite(y).all()):
        print(np.isfinite(x), np.isfinite(y))
        break

# num_neurons_dict = {0:28*28, 1:10, 2:10}
num_neurons_dict = {0:2, 1:4, 2:4, 3:2}
activation_dict = {0 : "logistic", 1 : "softmax"}

nn1 = neural_network(num_neurons_dict, activation_dict, nn_init=xavier_init, weight_type="w")
# nn1 = neural_network(num_neurons_dict, activation_dict, nn_init=np.random.rand) #, weight_type="w"
data["train_X"] = np.array([np.array([0, 0]), np.array([0, 255])
                            , np.array([255, 0]), np.array([255, 255])]).reshape(4, 2, 1)
data["train_y"] = np.array([1, 1, 1, 2]).reshape(4, 1)

print(data["train_X"].shape)
print(data["train_y"].shape)

nn1.fit(data["train_X"]/255, data["train_y"], eta=0.00000001, epochs=1000, minibatch_size = 0)
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

output = nn1.predict(data["train_X"][0], 0)
print(np.argmax(output) + 1)
output = nn1.predict(data["train_X"][1], 0)
print(np.argmax(output) + 1)
output = nn1.predict(data["train_X"][2], 0)
print(np.argmax(output) + 1)
output = nn1.predict(data["train_X"][3], 0)
print(np.argmax(output) + 1)
