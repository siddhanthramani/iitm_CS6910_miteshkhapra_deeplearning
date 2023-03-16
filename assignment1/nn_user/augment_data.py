import numpy as np
from scipy.ndimage import rotate
from copy import deepcopy
import random
from tqdm import tqdm
import albumentations as A

# Hard codes the shape of input
x_shape = (1, 784, 1)

# Albumentations allows you to do some great transforms quite easily
# Edit this to change your transformations
# Visualize the transform here - https://demo.albumentations.ai/
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

# Augment data
def augment_data(X, y, mode="replace", albumentation_transformer=transform):
    first_index = 1
    for img, true_value in tqdm(zip(X, y)):
        new_img = albumentation_transformer(image=img)['image']
        if first_index:
            X_new = new_img.reshape(x_shape)
            y_new = true_value
            first_index = 0
        else:
            X_new = np.append(X_new, new_img.reshape(x_shape), axis=0)
            y_new = np.append(y_new, true_value)
    
    if mode == "append":
        X = np.append(X, X_new)
        y = np.append(y, y_new)
        return X, y
    else:
        return X_new, y_new
