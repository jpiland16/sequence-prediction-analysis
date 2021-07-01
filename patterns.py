import random as r

class Pattern():
    """
    Wrapper class for pattern-generating lambda functions as shown below.
    """
    def __init__(self, generator: 'function', name: str) -> None:
        self.name = name
        self.generator = generator
    def __call__(self, *args):
        return self.generator(*args)
    def __str__(self):
        return self.name
    def __repr__(self):
        return f"patterns.Pattern: {self.name}"

# Complex transmission patterns
class AutoRegressive():
    """
    I'm not completely sure if this is the function we were looking for.
    """
    def __init__(self):
        self.last_value = 1
    def __call__(self, t: int, N: int) -> int:
        if self.last_value % 2 == 0:
            next = self.last_value + t % N
        else:
            next = self.last_value - t % N

        if next >= N:
            next = r.choices([1, N], weights=[0.1, 0.9])[0]
        elif next < 1:
            next = r.choices([1, N], weights=[0.9, 0.1])[0]

        self.last_value = next
        return next

auto_regressor_func = AutoRegressive()

# Simple transmission patterns
random = lambda t, N: r.randint(1, N)
mod_with_noise = lambda t, N: (t % N) + 1 + ((
        (r.randint(0, 1) * 2) - 1 # returns -1 or 1
    ) if t % N == N // 2 else 0)
random_spike = lambda t, N: r.randint(1, N) if t % 10 == 0 else 1
auto_regressive = lambda t, N: auto_regressor_func(t, N)
sawtooth = lambda t, N: t % N + 1
pulse = lambda t, N: N if t % N <= N // 2 else 1

def get_patterns() -> 'list[Pattern]':
    """
    Returns all the lambda functions present in this file as a list of
    Pattern objects.
    """
    return [Pattern(var, var_name) for var_name, var in globals().items() 
        if type(var).__name__ == "function" and var.__name__ == "<lambda>"]

# Pattern that includes other patterns
class MetaPattern():
    def __init__(self):
        self.SW_PROB = 0.05 # switch pattern with probability P
        self.pattern_list = get_patterns()
        self.current_pattern_index = r.randint(0, len(self.pattern_list) - 1)
    def __call__(self, t: int, N: int) -> int:
        if r.random() < self.SW_PROB:
            self.current_pattern_index = r.randint(0, 
                len(self.pattern_list) - 1)
        return self.pattern_list[self.current_pattern_index](t, N)

meta_switch_func = MetaPattern()
meta_switch = lambda t, N: meta_switch_func(t, N)

