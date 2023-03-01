import numpy as np

def xavier_init(shape_0, shape_1, **nn_init_params):
    weight_type = nn_init_params["weight_type"]
    if weight_type == "w":
        nodes_in = shape_0 * shape_1
        weights = np.random.rand(shape_0, shape_1) * np.sqrt(1/nodes_in)
    elif weight_type == "b":
        weights = np.zeros(shape_0, shape_1)
    else:
        raise Exception("Weight type should either be w or b")
    
    return weights