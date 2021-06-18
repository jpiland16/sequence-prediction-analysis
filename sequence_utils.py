from patterns import Pattern

def get_sequence_from_pattern(pattern: Pattern, num_bands: int, 
        length: int) -> 'list[int]':
    """
    Returns a list of the specified length, where pattern is the 
    Pattern to obtain the sequence values from and num_bands is the number 
    of bandwidths.
    """
    return [ pattern(t, num_bands) for t in range(length) ]

def one_hot_encode(value: int, vector_size: int) -> 'list[int]':
    """
    Converts a value to a sparse, binary vector of the given size 
    where there is a 1 at the index `value` and zeroes everywhere else.
    """
    return [ (1 if index == value else 0) for index in range(vector_size) ]