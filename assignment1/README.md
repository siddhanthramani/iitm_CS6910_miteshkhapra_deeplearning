## **Motivation :**
This is my submission for assignment 1 of Professor Mitesh Khapra's course.<br>
While this is a graded assignment, I approached it from a different perspective - to just code a really easy to understand neural network code which could be used as a reference for students and professionals looking to learn deep learning.


I believe my code will benefit two kinds of people:
1. those looking to use and experiment with neural networks
2. those looking to get their hands dirty and expand on this code


Hence, this code has been written for with three guiding principles
1. Easy to understand and modular
2. Easy to expand and experiment with
3. Empowering both programmers and users

<br>

## **Get started :**
You will need three things to get started:
1. This main code
2. The data
3. The environment

The software development process ensures that all three of these can be easily setup by you.<br>
**Note : The process and all the code has been tested both on Windows 11 and on Native Ubuntu 22.04.**<br>
**It works!**

<br>


### **Getting the main code :**
Create a folder (say deep_learning) where you want the code and data to reside. 
On your command line, cd into this folder, and run
```
git clone https://github.com/siddhanthramani/iitm_CS6910_miteshkhapra_deeplearning.git
```
The code will be synced to your local PC.

<br>

### **Getting the data :**
In your folder (say deep_learning) create a folder called "data".
Under the data folder, create subfolders called fashionmnist and mnist: 
This is the place you will download the data. 
Your folder structure should resemble this:
```bash
.
├── data
│   ├── fashionmnist
│   └── mnist
└── iitm_CS6910_miteshkhapra_deeplearning
```

For each dataset, download the required data files and store it in the appropriate subfolder:

- For fashionmnist - download the data from here (copy paste the link to new tab - clicking does not work sometimes):
    <ol type="a">
     <li>http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz</li>
     <li>http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz</li>
     <li>http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz</li>
     <li>http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz</li>
     <br>
     </ol>


- For mnist - download the data from the first file links mentioned here:
    http://yann.lecun.com/exdb/mnist/

<br>
For both fashionmnist and mnist, name the files appropriately as train-images.gz, test-images.gz, train-labels.gz and test-labels.gz.

Your final folder structure will resemble this:
```bash
.
├── data
│   ├── fashionmnist
│   │   ├── test-images.gz
│   │   ├── test-labels.gz
│   │   ├── train-images.gz
│   │   └── train-labels.gz
│   └── mnist
│       ├── test-images.gz
│       ├── test-labels.gz
│       ├── train-images.gz
│       └── train-labels.gz
└── iitm_CS6910_miteshkhapra_deeplearning
```

<br>

### **Getting/setting up the environment :**
cd into the iitm_CS6910_miteshkhapra_deeplearning folder. 
Note that all code runs must be performed from this folder.

To set up the environment, perform the following three steps:
1. Install Python 3.7 - the interpretor which has been used.
Download the version for your system from here : https://www.python.org/downloads/release/python-379/
2. Install pipenv - python package manager to easily install libraries
On the command line, run
```
pip install pipenv
```
3. Install all required packages - a virtual environment will be automatically created
On the command line, run 
```
pipenv install
```

<br>
<br>

## **User or developer?**
**Note : Both these files have commands which work on Windows CMD. Adapt them suitably to use them on Linux - ensure to replace \ with / in file paths if on Linux or Windows powershell.**
- Are you looking to use and experiment with this package?<br>
    [Read the README for users](./README_users.md)

- Are you looking to get your hands dirty and expand on this code<br>
    [Read the README for developers](./README_developers.md)

<br>

## Useful tips for both users and developers 
a. If the model is not getting trained :
1. try experimenting with the initialization
2. try experimenting with the learning rate
3. try running the code for more epochs
4. use the test.py to debug the code

b. If you are receiving NaNs :
1. Ensure the input or output data has no NaNs
2. experiment with the factor of your initial weights - if they become too small or too large they might cause issues
3. search the internet for numerically stable functions - especially for activations

c. If loss/accuracy is not improving : 
1. Check if the loss function and accuracy are written correctly - check with numbers if possible
2. Use regularisation if test accuracy is poor when compared to train - try image augmentation, l2, dropout

<br>

## **References :** 
1. [CS6910](http://www.cse.iitm.ac.in/~miteshk/CS6910.html) - the lecture notes were a constant reference and has influenced a majority of my code.
2. [Previous student's work - ArupDas15](https://github.com/ArupDas15/Fundamentals_Of_Deep_Learning/blob/master/cs6910_assignment1/optimiser.py) - This repository helped me understand that optimizers can be defined as a seperate class. I improved on this by removing redundancies. For example, the optimizer class written requested for the layer information of neural network while the neural network class also did. I made it cleaner by passing the neural network instance to the optimizer and the optimizer instance to the fit method of the neural network. This interdependency removed redundancies. Also my naming conventions are smaller and enhances readablility.
3. [Previous student's work - NiteshMethani](https://github.com/NiteshMethani/Deep-Learning-CS7015/blob/4c280b1bf8af2b1335a0409de87348230d260cc0/FeedForwardNN/src/FeedForwardNetwork.py) - This repository helped me understand that some functions could be made more stable, for example softmax. I improved on this by transfering the ideas to have stable logistic functions. Also, my modular code (separate math.utils file) enhances readability.
4. [Reading fashionmnist dataset](https://numpy-datasets.readthedocs.io/en/master/_modules/numpy_datasets/images/fashionmnist.html) - Downloading and reading fashionmnist dataset.
5. [MNIST dataset](http://yann.lecun.com/exdb/mnist/) - Downloading mnist dataset.
6. [Reading MNIST dataset](https://github.com/hsjeong5/MNIST-for-Numpy/blob/master/mnist.py) - Reading mnist dataset.
7. [Numerically stable softmax](https://www.sharpsightlabs.com/blog/numpy-softmax/) - Learned how to define a numerically stable softmax function.
8. [Numerically stable sigmoid(logistic)](https://stackoverflow.com/a/64717799) - Learned how to define a numerically stable logistic function.
9. [Markdown guide](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) - Taught me how to write a markdown file.
10. [Representing a folder structure](https://stackoverflow.com/a/47795759) - Taught me how to create folder structures on github.
11. [Argparse guide](https://realpython.com/command-line-interfaces-python-argparse/#creating-command-line-interfaces-with-pythons-argparse) - Fantastic guide which taught me how to use argparse module and also how to make it easier to read flags from a text file.
12. [WandB sweeps](https://docs.wandb.ai/guides/sweeps/add-w-and-b-to-your-code) - Learned how to run a sweep.
13. [WandB sweep search method](https://docs.wandb.ai/guides/sweeps) - Learned the difference between random, grid and bayes.
13. [Albumentations data transform](https://albumentations.ai/docs/getting_started/image_augmentation/) - Learned how to transform data using the albumentations library.
14. [Gradient of mse loss wrt to softmax](https://stats.stackexchange.com/questions/153285/derivative-of-softmax-and-squared-error)
15. [Gradient of softmax wrt to input](https://www.mldawn.com/the-derivative-of-softmaxz-function-w-r-t-z/)<br>
Used 14 and 15 together to derive the gradient of mse loss wrt to the input last layer a(L).
16. [Gradient of mse loss wrt to input](https://book.huihoo.com/deep-learning/version-30-03-2015/mlp.html) - My derivation did not calculate the gradients currently whereas the deep learning book's derivative was doing a good job. So sticking with that.
