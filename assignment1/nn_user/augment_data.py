import numpy as np
import scipy
from scipy.ndimage import rotate
from copy import deepcopy
import random

def rotate_img(X, y, max_angle=30, percent=0.1, bg_patch=(5,5)):
    X_new = np.zeros(X.shape())
    y_new = np.zeros(y.shape())
    
    for img, true_value in zip(X, y):
        rint = random.randint(0, 1)
        if not rint < percent:
            continue
        else:
            angle = random.randint(0, max_angle)

        assert len(img.shape) <= 3, "Incorrect image shape"
        rgb = len(img.shape) == 3
        if rgb:
            bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
        else:
            bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
        
        img_copy = deepcopy(img)
        img_copy = rotate(img_copy, angle, reshape=False)
        mask = [img_copy <= 0, np.any(img_copy <= 0, axis=-1)][rgb]
        img_copy[mask] = bg_color
        X = np.append(X, img_copy)
        y = np.append(y, true_value)

    return X, y