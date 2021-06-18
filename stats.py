def get_percent_correct(list_1: 'list[int]', list_2: 'list[int]') -> float:
    """
    Returns the percent of the values of list_1 and list_2 that match, as a 
    decimal between 0 and 1. Lists must have the same length.
    """
    if len(list_1) != len(list_2):
        raise IndexError("Lists are not the same length! " + 
            f"(Received lists of length {len(list_1)} and {len(list_2)})")

    total = len(list_1)
    correct = 0

    for a, b in zip(list_1, list_2):
        if a == b:
            correct += 1
    
    return correct / total