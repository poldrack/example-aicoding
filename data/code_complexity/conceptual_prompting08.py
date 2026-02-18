def big_o(algorithm_type, input_size):   

    """

    Returns the Big O time complexity of some common algorithms.

    

    Args:

    algorithm_type (str): the type of algorithm, such as "linear" or "quadratic"

    input_size (int): the size of the input for the algorithm

    

    Returns:

    int: the calculated Big O complexity for the given algorithm type and input size

    """



    if algorithm_type == "constant":

        return 1

    elif algorithm_type == "linear":

        return input_size

    elif algorithm_type == "quadratic":

        return input_size ** 2

    elif algorithm_type == "cubic":

        return input_size ** 3

    elif algorithm_type == "logarithmic":

        import math

        return int(math.log2(input_size))

    elif algorithm_type == "linearithmic":

        import math

        return input_size * int(math.log2(input_size))

    else:

        raise ValueError("Unsupported algorithm type")
