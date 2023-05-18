We have initially created a class called Lang. 
This is a helper class which helps us store details about each character which can be used for character embedding (a numerical representation of the character) later on. This class has two predefined chars - SOS and EOS to depict the start and end of a sentence (in our case, a word) respectively.

This class has three important dictionaries
1. The char2index (dict 1) and index2char (dict 2) dicts act as a two way dict and provides a mapping between a character and a unique index (and vice versa).
2. The char2count dict (dict 3), which stores how many times a character has occured in our dataset.


We have then created a function called readLangs which opens our file, and splits it into a set of lines. In our file each line contains a english, native lang pair (in this particular case tamil), both of which are separated by a comma.
We split each line into a list of english transliteration word and native lang word.
We then instantiaite two Lang objects, one each for each of the two languages.
The input lang object, the output lang object, and the list of list of english, tamil pairs (var aptly called pairs), are returned.

We then have a function called prepareData which calls the readLangs method to get two lang objects and list of list of english, tamil pairs (var pairs). The function then iterates through each pair and add the first word to the input lang object and the second word to the output lang object. Not the calling the addword function automatically calls the add char method on the Lang object which effectively creates the three important dictionaries for each of the Lang object.


The encoder module extends the nn.Module class which contains some basic functionality available.
The class then takes in a set of paramenters all of which are passed on to as parameters to nn.Embeddeding and nn.rnncell.
nn.Embedding requires two inputs - how big the embedding dictionary (mapping of character to numerical embedding vector), and what should be the size of the vector embedding. 
The rnn cell - irrrespective of whether it is RNN, GRU, or LSTM, take in the following parameters
1. input_size - number of input features
2. Hidden_size - number of hidden layer features
3. num_layers - number of layers stacked vertically
(Note horizontal layers are just for visual understanding. In reality there is only one layer)
4. dropout - applies dropout to all output layers except last layer
5. bidirectional - makes the RNN bidirectional (once the output layer is generated, the learning continues in the opposite direction)
For other parameters, you can check https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN

nn.embedding takes in two parameters
1. size of embedding dictionary 
2. the size of each embedding 
For more info, check https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
