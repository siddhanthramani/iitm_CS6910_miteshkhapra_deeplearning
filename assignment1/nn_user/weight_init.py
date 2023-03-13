import numpy as np

def random_init(shape0, shape_1, worb):
    weights = np.random.rand(shape0, shape_1)
    return weights

def xavier_init(shape_0, shape_1, worb):
    if worb == "w":
        nodes_in = shape_0 * shape_1
        weights = np.random.rand(shape_0, shape_1) * np.sqrt(1/nodes_in)
    elif worb == "b":
        weights = np.zeros((shape_0, shape_1))
    else:
        raise Exception("Weight type should either be w or b")
    
    return weights