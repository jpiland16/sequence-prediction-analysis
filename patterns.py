import random as r
from sequence_utils import Pattern

# Complex transmission patterns
class AutoRegressive():
    def __init__(self):
        self.last_value = 0
    def __call__(self):
        if self.last_value % 2 == 0:
            return 

# Simple transmission patterns
random = lambda t, N: r.randint(1, N)
mod_with_noise = lambda t, N: (t % N) + 1 + ((
        (r.randint(0, 1) * 2) - 1 # returns -1 or 1
    ) if t % N == 2 else 0)
random_spike = lambda t, N: r.randint(1, N) if t % 10 == 0 else 1
sawtooth = lambda t, N: t % N + 1
pulse = lambda t, N: 5 if t % N <= 2 else 1

def get_patterns() -> 'list[Pattern]':
    """
    Returns all the lambda functions present in this file as a list of
    Pattern objects.
    """
    return [Pattern(var, var_name) for var_name, var in globals().items() 
        if type(var).__name__ == "function" and var.__name__ == "<lambda>"]