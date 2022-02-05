from tensorflow import keras


CALIFORNIA_INPUT_DIM = 8


def get_dnn(n_layers: int, n_units_list: list, activation_function_list: list, input_dim: int):
    """
    Get DNN whose the number of layers is n_layers, and
    each layer has some units specified by n_units_list and
    an activation function specified by activation_function_list.

    Parameters
    ----------
    n_layers : int
        the number of layers
    n_units_list : list[int]
        the numbers of units of each layer
        The length has to be same as n_layers and the length of activation_function_list.
    activation_function_list : list[str]
        the list of activation functions of each layer
        The length has to be same as n_layers and the length of n_units_list.
    input_dim : int
        the dimension of input data

    Return
    ------
    model : keras.engine.sequential.Sequential
        the dnn whose the structure is specified by arguments

    Raise
    -----
    ValueError: if n_layers, the length of n_units_list, and the length of activation_function_list are not same
    """
    if not n_layers == len(n_units_list) == len(activation_function_list):
        raise ValueError(f"n_layers, the length of n_units_list, and the length of activation_function_list must be same, but n_layers is {n_layers}, the length of n_units_list is {n_units_list}, and the length of activation_function_list is {activation_function_list}.")

    model = keras.models.Sequential()
    
    # Add an input layer
    model.add(keras.layers.Dense(n_units_list[0], input_dim=input_dim, activation=activation_function_list[0]))
    n_layers -= 1

    # Add hidden layers
    for index in range(n_layers - 1):
        model.add(keras.layers.Dense(n_units_list[index+1], activation=activation_function_list[index]))

    # Add an output layer
    model.add(keras.layers.Dense(n_units_list[-1], activation=activation_function_list[-1]))

    return model


def get_california_dnn(n_layers: int, n_units_list: list, activation_function_list: list):
    """
    Get DNN for california housing. See get_dnn function for more information.

    Parameters
    ----------
    n_layers : int
        the number of layers
    n_units_list : list[int]
        the numbers of units of each layer
    activation_function_list : list[str]
        the list of activation functions of each layer

    Return
    ------
    model : keras.engine.sequential.Sequential
        the dnn for california houseing, whose the structure is specified by arguments
    """
    model = get_dnn(n_layers=n_layers, n_units_list=n_units_list, activation_function_list=activation_function_list, input_dim=CALIFORNIA_INPUT_DIM)

    return model
