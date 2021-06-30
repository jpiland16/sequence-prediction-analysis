# Third-party libraries
import matplotlib.pyplot as plt

# Personal code
from patterns import get_patterns
from interface_utils import select_option_from, confirm
from sequence_utils import get_sequence_from_pattern
from nn_parameters import get_parameters
from stats import get_percent_correct, roundify
from networks import SimpleRNN_01, RNN_SingleOutput, SimpleLSTM_01

# Constants that affect neural network performace

ADV_SEES_ENTIRE_DATASET = False # Determines whether training input is allowed
                                # to preface the testing input

SELECTED_NETWORK = -1 # set this to a non-negative index to suppress
                      # the prompt and choose the network with that index 

SELECTED_PATTERN = -1 # set this to a non-negative index to suppress
                      # the prompt and choose the pattern with that index 

FORECAST_LOOKBACK = -1 # set this to a non-negative index to force the network
                       # to make new predictions at each timestep based on the 
                       # last FORECAST_LOOKBACK timesteps, if data is available.
                       # If negative, network will run a single time, using all
                       # the available information.

USE_DEFAULT_PARAMS = True

# Constants that affect displayed output (i.e. the look of the graphs)
USE_MARKERS = False


# Collect all the networks available
class Network():
    """
    Wrapper class for a network, so that it is easier for the user to select
    using the command line.
    """
    def __init__(self, nnet, name: str) -> None:
        self.nnet = nnet
        self.name = name
    def __str__(self):
        return self.name

network_list = [
    Network(SimpleRNN_01, "Simple RNN with one-hot vector output"), # 0
    Network(RNN_SingleOutput, "RNN that outputs a single value"),   # 1
    Network(SimpleLSTM_01, "Simple LSTM with one-hot output")       # 2
]

def main():
    
    params = get_parameters("TRAINING", 
        default=USE_DEFAULT_PARAMS or confirm("Use the default parameters " + 
            "when training the neural network?")
    )

    if SELECTED_NETWORK == -1:
        print("\nWhich neural network would you like to use?\n")
        neural_net = select_option_from(network_list).nnet(params)
    else:
        neural_net = network_list[SELECTED_NETWORK].nnet(params)
    
    available_pattterns = get_patterns()

    if SELECTED_PATTERN == -1:
        print("Which pattern would you like to use?\n")
        pattern = select_option_from(available_pattterns)
    else:
        pattern = available_pattterns[SELECTED_PATTERN]
    print()

    # Get a list of bandwidths as determined by the selected pattern
    entire_transmission = get_sequence_from_pattern(pattern, 
        num_bands = params["NUM_BANDS"], 
        length = params["TRAIN_LENGTH"] + params["TEST_LENGTH"])

    train_data = entire_transmission[:params["TRAIN_LENGTH"]]
    test_data = entire_transmission[params["TRAIN_LENGTH"]:]

    neural_net.train(train_data)

    # Don't make a prediction for the last value in test_data (hence the -1)
    prediction_input = test_data[:-1] if not ADV_SEES_ENTIRE_DATASET \
        else (train_data + test_data)[:-1]

    if FORECAST_LOOKBACK >= 0:
        # Only look at the last FORECAST_LOOKBACK timesteps for information
        output = [

            neural_net.predict(prediction_input[
                # We can't take information from timesteps before 0 (hence max)
                max(0, i - FORECAST_LOOKBACK) : i + 1
            ])[-1] # Take the last prediction, ignore any intermediate
                   # predictions (hence the -1)

            for i in range(len(prediction_input))
        ]

    else:
        # Use all the information that is available
        output = neural_net.predict(prediction_input)

    # Only use predictions made in the testing section of the data
    # (if we didn't show the whole dataset, this is irrelevant)
    if ADV_SEES_ENTIRE_DATASET:
        output = output[params["TRAIN_LENGTH"]:]


    # Compare the predictions to the test data advanced by one timestep
    accuracy = get_percent_correct(test_data[1:], 
        roundify(output, min_v = 1, max_v = params["NUM_BANDS"] ))
    
    # print(f"Accuracy on testing set: {round(accuracy * 100, 2)}%")
    show_attack_success_graph(pattern, params, output, test_data, accuracy)


def show_attack_success_graph(pattern, params, output, test_data, accuracy):
    # Plot the results
    timesteps = [t + 1 for t in range(params["TRAIN_LENGTH"], 
        params["TRAIN_LENGTH"] + params["TEST_LENGTH"] - 1)]
    
    tx_line_format = 'bs-' if USE_MARKERS else 'b'
    adv_line_format = 'r^-' if USE_MARKERS else 'r'
    
    plt.plot(timesteps, test_data[1:], tx_line_format, 
        label="Transmitted Slots")
    plt.plot(timesteps, output, adv_line_format, label="Predicted Slots")

    plt.yticks([ i + 1 for i in range(params["NUM_BANDS"]) ])

    plt.title(f"{pattern.name}: {round(accuracy * 100, 2)}% accuracy")
    plt.xlabel("Time (s)")
    plt.ylabel("The Transmission Slots")
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
    print()