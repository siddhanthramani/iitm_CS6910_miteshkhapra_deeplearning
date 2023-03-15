This is my submission for assignment 1 of Professor Mitesh Khapra's course. 
While this is a graded assignment, I approached it from a different perspective - to just code a really easy to understand
neural network code which could be used as a reference for students and professionals looking to learn deep learning. 

I believe two kinds of people will benefit from my code
1. those looking to use and experiment with neural networks
2. those looking to get their hands dirty and expand on this code

Hence, this code has been written for with three guiding principles
1. Easy to understand and modular
2. Easy to expand and experiment with
3. Empowering both programmers and users

With that, let me show how the code has been structured to ensure modularity

Important files in the main directory
main.py -> Is used to run sweeps. The configuration for which is present in wandb_sweep_config.json
train.py -> Is used to run a single experiment. The configuration for which is sent via the CLI.
test.py -> Is used to debug the nn_code to check if the output is as expected.


Get started
1. Install Python 3.7 - the interpretor which has been used to
Download the version for your system from here : 
2. Install pipenv - python package manager to easily install libraries
pip install pipenv
3. Install all required packages - a virtual environment will be automatically created
pipenv install

Note : The code has been tested both on Windows 11  and native Ubuntu 22.0.4. It works!

Useful tips for users
Most libraries out there do not allow you to include your own algorithms and they default many hyperparameters.

But there are three things which I believe can have drastic changes in the accuracy.
And the user should have the power to influence these things - consciously.
Thus my code allows the user to influence and define the following three things
1. The neural network structure - quite obvious. The user defines the structure.
2. The data itself - maybe obvious. The user can define the transformations which they will like to include in their data to make the model more robust. Albumentations implementation has been added for reference.
3. The weight initialization method - not so obvious. The user can include their custom algorithm based on the dataset they are dealing with. Xavier implementation has been added for reference.

I will urge you to experiment with activations, optimization algorithms and loss functions also.


Useful tips for programmers 
a. if the model is not getting trained,
1. try experimenting with the initialization
2. try experimenting with the learning rate
3. try running the code for more epochs
4. use the test.py to debug the code

b. if you are receiving NaNs
1. Ensure the input or output data has no NaNs
2. experiment with the factor of your initial weights - if they become too small or too large they might cause issues
3. search the internet for numerically stable functions - especially for activations

c. loss/accuracy is not improving
1. Check if the loss function and accuracy are written correctly - check with numbers if possible
2. Use regularisation if test accuracy is poor when compared to train - try image augmentation, l2, dropout




References : 
A constant reference which has influenced a majority of my code are Professor Mitesh Khapra's CS6910's lecture materials.
