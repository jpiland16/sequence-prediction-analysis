print("\nLoading PyTorch...\n")

import torch
from torch import nn, Tensor

from tqdm import tqdm

from sequence_utils import one_hot_encode

def get_train_set(train_data: list, seq_len: int, 
        num_bands: int) -> 'tuple[Tensor]':
    """
    Generate a pair of training sequence Tensors (X[], y[]) where y is one 
    time-step ahead of X.

    A note on the shape of the input - 3 dimensions:

    For each training example:
        For each timestep in the sequence:
           There should be a value for each input
           (there should be a one-hot vector here)
    """

    train_x = [] # a set of little sequences
    train_y = []
    
    # Subtract 1 here to accomodate for predictions
    for i in range(len(train_data) - seq_len - 1):

        # Notice: the input is one-hot-encoded, but the target is not
        # Also note: we are subtracting 1 from the values because
        # the presentation used 1-based indices rather than 0-based
        train_x.append([one_hot_encode(value - 1, vector_size = num_bands) 
            for value in train_data[i : i + seq_len]]) # a little sequence
        train_y.append([value - 1 
            for value in train_data[i + 1 : i + seq_len + 1]])

    return ( Tensor(train_x), Tensor(train_y) )


class SimpleRNN_01(nn.Module):
    """
    A simple recurrent neural network plus 1 hidden layer.
    Based on the tutorial available at 
    https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
    """
    def __init__(self, params: dict):
        """
        Initializes a neural network with recurrent layer(s) and 1 fully 
        connected layer.
        """
        super(SimpleRNN_01, self).__init__()

        # Defining some parameters ------------------------
        self.hidden_layer_size = params["HIDDEN_DIM"]
        self.n_layers = params["NUM_LAYERS"]
        self.input_size = params["NUM_BANDS"]
        self.output_size = params["NUM_BANDS"]
        self.params = params

        #Defining the layers ------------------------------
        # RNN Layer
        self.rnn = nn.RNN(self.input_size, self.hidden_layer_size, 
            self.n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_layer_size, self.output_size)
    
    def forward(self, input_data: torch.Tensor):
        """
        Return the output(s) from each timestep in the input sequence.
        """

        # Passing in the input and hidden state into the model and 
        # obtaining outputs
        output, _ = self.rnn(input_data)
        
        # Reshaping the outputs such that it can be fit into the 
        # fully connected layer
        output = output.contiguous().view(-1, self.hidden_layer_size)   
        output = self.fc(output)
        return output

    def train(self, train_list: 'list[int]', show_stats: bool):

        # Define Loss, Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), 
            lr=self.params["LEARNING_RATE"])

        # In this case, we are using the same batch 
        # for each training epoch
        training_input, training_target_output = get_train_set(
            train_list, seq_len = self.params["SUBSEQ_LEN"], 
            num_bands = self.params["NUM_BANDS"])
    

        iter = range(self.params["NUM_EPOCHS"])
        if show_stats:
            iter = tqdm(iter)
        for _ in iter:
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            output = self(training_input)
            training_target_output = training_target_output.view(-1).long()
            loss = criterion(output, training_target_output)
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

            if show_stats:
                iter.set_description(f"loss: {round(loss.item(), 3):5}")

        if show_stats:
            print("\nDone training!\n")

    def predict(self, input_list: 'list[int]') -> 'list[int]':
        # Surround input_seq with square brackets because it is a single batch 
        # Subtract 1 from value to convert to 0-based indexing
        input_seq = Tensor([[one_hot_encode(value - 1, 
            vector_size = self.params["NUM_BANDS"]) for value in input_list]])
        output = self(input_seq)

        # Select the index of maximum probability
        # Add 1 to value due to 1-based indexing
        return [torch.max(probs, dim=0)[1].item() + 1
            for probs in output ]


class RNN_SingleOutput(nn.Module):
    """
    A simple recurrent neural network plus 1 hidden layer.
    Outputs a single value in the range (0, +infinity).
    Is clamped at both ends and rounded before accuracy is tested.
    """
    def __init__(self, params: dict):
        """
        Initializes a neural network with recurrent layer(s) and 1 fully 
        connected layer.
        """
        super(RNN_SingleOutput, self).__init__()

        # Defining some parameters ------------------------
        self.hidden_layer_size = params["HIDDEN_DIM"]
        self.n_layers = params["NUM_LAYERS"]
        self.input_size = params["NUM_BANDS"]
        self.output_size = 1
        self.params = params

        # Defining the layers ------------------------------
        # RNN Layer
        self.rnn = nn.RNN(self.input_size, self.hidden_layer_size, 
            self.n_layers, batch_first=True, nonlinearity="relu")   
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_layer_size, self.output_size)
    
    def forward(self, input_data: torch.Tensor):
        """
        Return the output(s) from each timestep in the input sequence.
        """

        # Passing in the input and hidden state into the model and 
        # obtaining outputs
        output, _ = self.rnn(input_data)
        
        # Reshaping the outputs such that it can be fit into the 
        # fully connected layer
        output = output.contiguous().view(-1, self.hidden_layer_size)
        output = self.fc(output)
        return output.view(-1)

    def train(self, train_list: 'list[int]', show_stats: bool):

        # Define Loss, Optimizer
        criterion = nn.SmoothL1Loss(beta = self.params["BETA"])
        optimizer = torch.optim.Adam(self.parameters(), 
            lr=self.params["LEARNING_RATE"])

        # In this case, we are using the same batch 
        # for each training epoch
        training_input, training_target_output = get_train_set(
            train_list, seq_len = self.params["SUBSEQ_LEN"], 
            num_bands = self.params["NUM_BANDS"])

        # get_train_set uses 0-based indexing. Let's use 
        # 1-based indexing instead.
        training_target_output += 1

        # Rescale the targets to be between 0 and 1
        # training_target_output /= self.params["NUM_BANDS"]

        iter = range(self.params["NUM_EPOCHS"])
        if show_stats:
            iter = tqdm(iter)
        for _ in iter:
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            output = self(training_input)
            training_target_output = training_target_output.view(-1).float()
            loss = criterion(output, training_target_output)
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

            if show_stats:
                iter.set_description(f"loss: {round(loss.item(), 3):5}")

        if show_stats:
            print("\nDone training!\n")

    def predict(self, input_list: 'list[int]') -> 'list[int]':
        # Surround input_seq with square brackets because it is a single batch 
        input_seq = Tensor([[one_hot_encode(value - 1, 
            vector_size = self.params["NUM_BANDS"]) for value in input_list]])
        output = self(input_seq)

        return output.detach().numpy()

class SimpleLSTM_01(nn.Module):
    """
    A simple LSTM plus 1 hidden layer. Code is very similar to that of 
    SimpleRNN_01. The one difference is marked with a multiline comment.
    """
    def __init__(self, params: dict):
        """
        Initializes a neural network with LSTM layer(s) and 1 fully 
        connected layer.
        """
        super(SimpleLSTM_01, self).__init__()

        # Defining some parameters ------------------------
        self.hidden_layer_size = params["HIDDEN_DIM"]
        self.n_layers = params["NUM_LAYERS"]
        self.input_size = params["NUM_BANDS"]
        self.output_size = params["NUM_BANDS"]
        self.params = params

        #Defining the layers ------------------------------
        # LSTM Layer
        """
        ~ THE NEXT LINE IS DIFFERENT FROM SimpleRNN_01 ~~~~~~~~~~~~
        (only in the fact that it says LSTM instead of RNN) 
        """
        self.lstm = nn.LSTM(self.input_size, self.hidden_layer_size, 
            self.n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_layer_size, self.output_size)
    
    def forward(self, input_data: torch.Tensor):
        """
        Return the output(s) from each timestep in the input sequence.
        """

        # Passing in the input and hidden state into the model and 
        # obtaining outputs
        output, _ = self.lstm(input_data)

        # Reshaping the outputs such that it can be fit into the 
        # fully connected layer
        output = output.contiguous().view(-1, self.hidden_layer_size)
        output = self.fc(output)
        return output

    def train(self, train_list: 'list[int]', show_stats: bool):

        # Define Loss, Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), 
            lr=self.params["LEARNING_RATE"])

        # In this case, we are using the same batch 
        # for each training epoch
        training_input, training_target_output = get_train_set(
            train_list, seq_len = self.params["SUBSEQ_LEN"], 
            num_bands = self.params["NUM_BANDS"])
    

        iter = range(self.params["NUM_EPOCHS"])
        if show_stats:
            iter = tqdm(iter)
        for _ in iter:
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            output = self(training_input)
            training_target_output = training_target_output.view(-1).long()
            loss = criterion(output, training_target_output)
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

            if show_stats:
                iter.set_description(f"loss: {round(loss.item(), 3):5}")

        if show_stats:
            print("\nDone training!\n")

    def predict(self, input_list: 'list[int]') -> 'list[int]':
        # Surround input_seq with square brackets because it is a single batch 
        # Subtract 1 from value to convert to 0-based indexing
        input_seq = Tensor([[one_hot_encode(value - 1, 
            vector_size = self.params["NUM_BANDS"]) for value in input_list]])
        output = self(input_seq)

        # Select the index of maximum probability
        # Add 1 to value due to 1-based indexing
        return [torch.max(probs, dim=0)[1].item() + 1
            for probs in output ]