# Python library imports
import math, os, pickle
import matplotlib.pyplot as plt
from datetime import datetime

# Personal code
from nn_parameters import get_parameters
from main import train_and_test

# Choose the network
from networks import SimpleRNN_01, SimpleLSTM_01, RNN_SingleOutput
NETWORK_TO_TEST = SimpleRNN_01

# Choose the pattern
from patterns import meta_switch
PATTERN_TO_TEST = meta_switch

SHOW_STATUS = True
OUTPUT_DIR_NAME = "output"

class ChangingParameter:
    """
    Wrapper class that holds the name of one parameter and a 
    corresponding function that takes a stepping value and converts it 
    to a parameter value. For example, suppose you wanted to increase the 
    number of input nodes stepwise from 5 to 10, but wanted to simulataneously 
    increase the  hidden layer size from 20 to 40. You could use two 
    ChangingParameter objects, one with step_function: `lambda x: x` (DEFAULT) 
    and another with step_function: `lambda x: 4 * x`. Then you would pass 
    both of the ChangingParameter objects in a list to a 
    ParameterVariationScheme with min 5 and max 10. 
    """
    def __init__(self, parameter_name: str, step_function: 'callable' = None):
        self.parameter_name = parameter_name
        if step_function == None:
            step_function = lambda x: x
        self.step_function = step_function

class ParameterVariationScheme:
    """
    A class that changes a set of parameters in accordance with the 
    functions defined in their respective ChangingParameter objects.

    Uses a single stepping value to modify 1 or more parameters.
    """
    def __init__(self, min, max, step_size,
            changing_parameters: 'list[ChangingParameter]',
            stepper_name: str, full_description: str = ""):
        self.changing_parameters = changing_parameters
        self.min = min
        self.max = max
        self.step_size = step_size
        self.stepper_name = stepper_name
        self.full_description = full_description

class Test:
    """
    A class that containts all the information needed to run a single "test"
    of the model, which consists of one or more simulations. Takes as input
    a list of type ParameterVariationScheme, such that the last received 
    scheme is varied the most frequently. For example, if the first 
    scheme in the list varies from 5 to 10 with step size 1, and the second 
    scheme varies from 0.10 to 0.12 with step size 0.01, the resulting 
    simulations would run as follows:

     1) S1 = 5, S2 = 0.10   \n
     2) S1 = 5, S2 = 0.11   \n
     3) S1 = 5, S2 = 0.12   \n
  
     4) S1 = 6, S2 = 0.10   \n
     5) S1 = 6, S2 = 0.11   \n
        . . .               \n

     16) S1 = 10, S2 = 0.10 \n
     17) S1 = 10, S2 = 0.11 \n
     18) S1 = 10, S2 = 0.12
    """
    def __init__(self, parameter_variations: 'list[ParameterVariationScheme]'):
        self.parameter_variations = parameter_variations

    def run(self, num_sim_repeats: int):
        """
        Initiate the test simulations. `num_sim_repeats` is the number of
        times to repeat a simulation, using the same parameter set.
        """
        # Get the default parameters as specified in nn_parameters.py
        simulation_params = get_parameters("TRAINING")

        result_set = self.run_internal(num_sim_repeats, "", 
            self.parameter_variations, simulation_params, "")

        # last_varied_step_name is the name of value that will be graphed 
        # on the x-axis.
        last_varied_step_name = self.parameter_variations[-1].stepper_name

        return {
            "results": result_set,
            "x-axis-name": last_varied_step_name
        }

    def run_internal(self, num_sim_repeats: int, run_name_prefix: str,
            parameter_variations: 'list[ParameterVariationScheme]',
            simulation_params: dict, long_description: str, 
            depth: int = 1) -> 'list[dict]':
        """
        Do not call this method outside this class - call `start_run` instead. 
        This method is only to be called within this class to recursively 
        iterate over all possible simulations.
        """

        variation = parameter_variations[0]
        stepper_value = variation.min
        run_count = 0 # For this level only
        num_steps = ((variation.max - variation.min) // variation.step_size) + 1

        results = []

        while stepper_value <= variation.max:

            if (SHOW_STATUS): 
                print()
                print("-- " * (depth - 1) + 
                    f"Running:   step {run_count + 1} of {num_steps} " + 
                    f"for scheme level {depth}", end="")

            # Change each specified parameter in accordance with its step
            # function
            for changing_param in variation.changing_parameters:
                simulation_params[changing_param.parameter_name] =  \
                     changing_param.step_function(stepper_value)

            if len(parameter_variations) == 1: # If we are on the last parameter
                                               # that gets varied (see function
                                               # docstring)

                sim_repeat_group = []
                end_text = f" of {num_sim_repeats}...)"

                for i in range(num_sim_repeats):
                    
                    if i > 0:
                        backspace_amount = int(math.log10(i))
                        print("\b" * (backspace_amount + len(end_text) + 1), 
                            end="", flush=True)
                        print(f"{i + 1}" + end_text, end="", flush=True)
                    else:
                        print(f" (repeat {i + 1}" + end_text, 
                            end="", flush=True)

                    sim_repeat_group.append(run_test(
                            simulation_params, 
                            name = run_name_prefix 
                                + f"{variation.stepper_name}_" 
                                + f"{stepper_value}",
                            graph_title = run_name_prefix,
                            description = long_description + 
                                f"VARYING SCHEME #{depth} ----\n" 
                                + variation.full_description    
                        )
                    )

                results.append({
                    "group": sim_repeat_group,
                    "step_value": stepper_value
                })

            else:
                results.append(self.run_internal(
                    num_sim_repeats, 
                    run_name_prefix = run_name_prefix + 
                        f"{variation.stepper_name}_{stepper_value} ",
                    parameter_variations = parameter_variations[1:], 
                    simulation_params = simulation_params, 
                    long_description = long_description + 
                        f"VARYING SCHEME #{depth} ----\n" 
                        + variation.full_description + "\n\n",
                    depth = depth + 1)
                )

            stepper_value += variation.step_size
            run_count += 1

            if (SHOW_STATUS): 
                print()
                print("-- " * (depth - 1) + 
                    f"Completed: step {run_count} of {num_steps} " + 
                    f"for scheme level {depth}", end="")

        return results
        
def run_test(params, name, graph_title, description):
    neural_net = NETWORK_TO_TEST(params)
    sim_res = train_and_test(neural_net, PATTERN_TO_TEST, params, 
        show_stats = False)
    sim_res.name = name
    sim_res.description = description
    sim_res.graph_title = graph_title
    return sim_res

def save_output_graphs(results, x_axis_name: str, agg_group_function: callable, 
        target_dir: str):
    """
    Saves a single image of a graph if `results` is a `list` of `dict`s.
    Otherwise, recursively calls itself to create several images.

    `agg_group_function` should take as input a list of `SimulationResult`s 
    and return some aggregate statistic (such as max, average, etc.)
    """
    if type(results[0]) == dict:

        x_axis_set = [results[i]["step_value"] for i in range(len(results))]

        accuracy_set = [agg_group_function(results[i]["group"])
            for i in range(len(results))]

        # Take the first simulation of several repeats
        # in the group of repeated simulations
        # in the first step value of the results
        graph_title = results[0]["group"][0].graph_title

        plt.clf()
        plt.plot(x_axis_set, accuracy_set)
        plt.title(f"accuracy vs {x_axis_name}")
        plt.savefig(os.path.join(target_dir, graph_title + ".png"))

    else: 
        for result_set in results:
            save_output_graphs(result_set, x_axis_name, agg_group_function, 
                target_dir)

def main():
    demo_test_01()
    print()

def demo_test_01():
    test = Test([
        ParameterVariationScheme(
            min = 5,
            max = 20,
            step_size = 1,
            changing_parameters = [
                ChangingParameter("NUM_BANDS")
            ],
            stepper_name = "num-of-bands",
            full_description = "Vary the number of bandwidths (unique items)"
        ), 
        ParameterVariationScheme(
            min = 20,
            max = 40,
            step_size = 1,
            changing_parameters = [
                ChangingParameter("HIDDEN_DIM")
            ],
            stepper_name = "hidden-dim",
            full_description = "Vary the size of the hidden layers"
        )
    ])

    agg_func = lambda sim_list: \
        sum([sim.accuracy for sim in sim_list]) / len(sim_list)

    test_results = test.run(num_sim_repeats = 25)

    save_dir = get_save_dir()

    save_output_graphs(test_results["results"], test_results["x-axis-name"], 
        agg_func, save_dir)

    # Also save all the results in a pickle for future reference
    with open(os.path.join(save_dir, "all-data.pickle"), 'wb') as file:
        pickle.dump(test_results, file)

def get_save_dir():
    if not os.path.exists(OUTPUT_DIR_NAME):
        os.mkdir(OUTPUT_DIR_NAME)
    sub_dir_name = "Tests {:%Y.%m.%d %H-%M-%S}".format(datetime.now())
    os.mkdir(os.path.join(OUTPUT_DIR_NAME, sub_dir_name))
    return os.path.join(OUTPUT_DIR_NAME, sub_dir_name)
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
    print()