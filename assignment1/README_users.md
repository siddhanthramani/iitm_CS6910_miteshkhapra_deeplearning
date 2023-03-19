## Important files in the main directory
- main.py -> Is used to run sweeps, the configuration for which is present in wandb_sweep_config.json
- train.py -> Is used to run a single experiment, the configuration for which is sent via the CLI.
    To make it easier, you can also send the arguments via wandb_expt_args.txt
- test.py -> Is used to debug the nn_code to check if the output is as expected.

<br>

## Training the model
While training, you can log your important metrics with [WandB](https://wandb.ai/site).<br>
In wandb, running an experiment will by default track all important parameters.

- To run a batch of experiments (training multiple different models) use the main.py file. You do this with different sets of hyperparameters to see which set performs best.
- To run a single experiment (training your model) use the train.py file. You do this when you know what your best performing set of hyperparameters are.

Note: For all runs, 
1. cd into the iitm_cs6910_miteshkhapra_deeplearing folder
2. activate the virtual environment with
```
pipenv shell
```

<br>

### To run a batch of experiments
1. Setting the configuration - i.e choosing the different sets of hyperparameters you wish to run your model on, you use the wandb_sweep_config.json.<br>
Here, for each hyperparameter, include the list of hyperparameter values you'd like to try out.<br>
You can also choose between three methods - "random" for a random search across hyperparemeters, and "grid" to search each possible hyperparemeter combination in an orderly manner, and "bayes" to search for hyperparameters which would optimize for the metric.
Know more about this [here](https://docs.wandb.ai/guides/sweeps)<br>

2. Run the main.py with
```
python assignment1/main.py
```

<br>

### To run a single experiment
Two options are provided to run a single experiment - both are via the CLI.<br>
To know which flags are expected and what their defaults are, run
```
python assignment1/train.py --help
```

- By entering the flags directly in the command line (if this is cumbersome, look at option 2).<br>
    You can do this by running the following commands with the appropriate flags
    ```
    python assignment1/train.py -flag1=flag1_value -flag2=flag2_value
    ```
- By entering the flags in a text file. This is easier when the flag list is too long, like in our case.
    ```
    python assignment1/train.py @path_to_argument_list_text_file
    ```
    where "path_to_argument_list_text_file" is a text file which contains the flag value pairs.<br>
    Note that the text file should have one flag value pair per line and there should be no additional spaces either in between or at the end of the line.<br>
    For example, have a look at the assignment1/wandb_expt_args.txt <br>
    To train the models with the flags in assignment1/wandb_expt_args.txt, run
    ```
    python assignment1/train.py @assignment1/wandb_expt_args.txt
    ```
    Note that the "@" symbol should be present before the text file path.

<br>

## User editable code
Most libraries out there do not allow you to include your own algorithms and they default many hyperparameters. But there are three things which I believe can have drastic changes in the accuracy, and that you should have the power to influence these things - consciously.<br>
Thus my code allows you to influence and define the following three things:
1. The neural network structure - quite obvious. You define the structure.
    The neural network structure definition was designed to be highly easy and visual to understand.
    To define the structure you have to pass in two dictionaries
    - A dict with key:value pairs where the key identifies the layer and the value defines the number of neurons to be present in that layer. The smallest key should be 0 and whose value should be the number of inputs, and the largest key will be the equal to the total number of layers in the network with the value being the size of the output.
    For example, for a neural network which has 2 inputs, a single hidden layer with 4 neurons and 5 outputs, the dict will be :<br>
    {0:2, 1:4, 2:5}
    - A dict which defines the activation for each layer. This is of two types:
        1. When the activation functions are constant for all layers except the last layer, the dict will contain two key:value pairs. 
        Key 1 will contain the activation function of the output layer, and Key 0 will contain the activation function of all other layers.
        2. When the activation function varies from layer to layer, the dict will contain multiple key:value pairs where keys will identify the layer and the values will contain the activation function for that particular layer. The key set will contain values from 1 to the number_of_layers in the network. 
2. The data itself - maybe obvious. You can define the transformations which you will like to include in the data to make the model more robust. An albumentations implementation has been added for reference.
3. The weight initialization method - not so obvious. You can include your own custom algorithm based on the dataset you are dealing with. Xavier implementation has been added for reference.

You can write the codes to define the data transformation and the weight initialization in the nn_core folder.
#### **Structure and functionality of nn_users**
<img src="../assignment1/documentation/nnuser.png" width="500">

How does the nn_users code work:

- augment_data = Add a new data transformation to the list parameter of the A.compose function. You can have a look at how a particular transformation will affect your data [here](https://demo.albumentations.ai/).
- weight_init = Add a new weight initialization function which takes in three inputs - Shape of axis 0 of input, Shape of axis 1 of input and worb which can either be "w" for weights init and "b" for bias init.

<br>

**I will urge you to experiment with activations, optimization algorithms and loss functions also.**

<br>

