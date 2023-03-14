import gzip
import numpy as np
import time
import wandb
import random
import matplotlib.pyplot as plt

dimensions = (28, 28)

def load(path = "../data/", dataset="fashionmnist"):
    t0 = time.time()
    print(f"Loading {dataset}")

    with gzip.open(path + f"{dataset}/train-labels.gz", "rb") as lbpath:
        train_labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(path + f"{dataset}/train-images.gz", "rb") as lbpath:
        train_images = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=16)
    train_images = train_images.reshape((-1, 1, 28, 28)).astype("float32")

    with gzip.open(path + f"{dataset}/test-labels.gz", "rb") as lbpath:
        test_labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(path + f"{dataset}/test-images.gz", "rb") as lbpath:
        test_images = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=16)
    test_images = test_images.reshape((-1, 1, 28, 28)).astype("float32")
    
    data = {
        "train_X": train_images.reshape(-1, 28*28, 1),
        "train_y": train_labels,
        "test_X": test_images.reshape(-1, 28*28, 1),
        "test_y": test_labels,
    }

    print("Dataset mnist loaded in {0:.2f}s.".format(time.time() - t0))

    return data


def plot_image(image):
    plt.show(image)


def get_random_class_indices(label_data):
    random_sample_indices = []
    for unique_class in np.unique(label_data):
        random_sample = random.choice(np.argwhere(label_data==unique_class))
        random_sample_indices.append(random_sample)
    return random_sample_indices


def plot_random_image_per_class(data, from_dataset="train"):
    label_data = data["{}_y".format(from_dataset)]
    random_sample_indices = get_random_class_indices(label_data)
    max_columns = 5
    row_index = 0
    col_index = 0
    fig, ax = plt.subplots(nrows=int(len(random_sample_indices)/max_columns), ncols=max_columns)
    print(len(random_sample_indices))
    for random_sample in random_sample_indices:
        if not col_index < max_columns:
            row_index +=1
            col_index = 0
        print(row_index, col_index)
        ax[row_index, col_index].imshow(data["{}_X".format(from_dataset)][random_sample].reshape(dimensions))
        col_index += 1
        
            
    return plt


def do_checks(X_check, y_check):
    for x, y in zip(X_check, y_check):
        y = np.array([1 if i==(y-1) else 0 for i in range(10)]).reshape(10, 1)
        if not (np.isfinite(x).all() or np.isfinite(y).all()):
            print(np.isfinite(x), np.isfinite(y))
            break

if __name__ == "__main__":
    wandb.init(project="sweep_test")

    # WandB plot sample image
    data = load()
    plt = plot_random_image_per_class(data)
    wandb.log({"img": [wandb.Image(plt)]})

    wandb.finish()