# Third-party libraries
import matplotlib.pyplot as plt

# Personal code
from patterns import get_patterns
from interface_utils import select_option, confirm
from sequence_utils import get_sequence_from_pattern
from nn_parameters import get_parameters
from stats import get_percent_correct
from networks import SimpleRNN_01

# Constants that affect neural network performace
ADV_SEES_ENTIRE_DATASET = True # Determines whether training input is allowed
                                # to preface the testing input

# Constants that affect displayed output
USE_MARKERS = False

def main():
    print("Which pattern would you like to use?\n")
    pattern = select_option(get_patterns())
    
    params = get_parameters("TRAINING", 
        default=confirm("\nDo you want to use the default parameters?")
    )
    
    print()

    # Get a list of bandwidths as determined by the selected pattern
    entire_transmission = get_sequence_from_pattern(pattern, 
        num_bands = params["NUM_BANDS"], 
        length = params["TRAIN_LENGTH"] + params["TEST_LENGTH"])

    train_data = entire_transmission[:params["TRAIN_LENGTH"]]
    test_data = entire_transmission[params["TRAIN_LENGTH"]:]

    neural_net = SimpleRNN_01(params)
    neural_net.train(train_data)

    # Don't make a prediction for the last value in test_data
    output = neural_net.predict(test_data[:-1]) if not ADV_SEES_ENTIRE_DATASET \
        else neural_net.predict((test_data + train_data)[:-1])

    if ADV_SEES_ENTIRE_DATASET:
        output = output[params["TRAIN_LENGTH"]:]

    # Compare the predictions to the test data advanced by one timestep
    accuracy = get_percent_correct(test_data[1:], output)
    
    # print(f"Accuracy on testing set: {round(accuracy * 100, 2)}%")

    # Plot the results
    timesteps = [t + 1 for t in range(params["TRAIN_LENGTH"], 
        params["TRAIN_LENGTH"] + params["TEST_LENGTH"] - 1)]
    
    tx_line_format = 'bs-' if USE_MARKERS else 'b'
    adv_line_format = 'r^-' if USE_MARKERS else 'r'
    
    plt.plot(timesteps, test_data[1:], tx_line_format, 
        label="Transmitted Slots")
    plt.plot(timesteps, output, adv_line_format, label="Predicted Slots")

    plt.yticks([1, 2, 3, 4, 5])

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