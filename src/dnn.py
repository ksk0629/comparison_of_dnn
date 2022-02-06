from typing import List, Optional, Union
import yaml

import mlflow
from tensorflow import keras

import definition
import utils


CALIFORNIA_INPUT_DIM = 8


def get_dnn(n_layers: int, n_units_list: List[int], activation_function_list: List[str], input_dim: int, seed: Optional[int]=57):
    """
    Get a DNN whose the number of layers is n_layers, and
    each layer has some units specified by n_units_list and
    an activation function specified by activation_function_list.

    Parameters
    ----------
    n_layers : int
        the number of layers
    n_units_list : List[int]
        the numbers of units of each layer
        The length has to be same as n_layers and the length of activation_function_list.
    activation_function_list : List[str]
        the list of activation functions of each layer
        The length has to be same as n_layers and the length of n_units_list.
    input_dim : int
        the dimension of input data
    seed : Optional[int]

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

    # Fix seed if it's not None
    if seed is not None:
        utils.fix_seed(seed)

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


def get_california_dnn(n_layers: int, n_units_list: List[int], activation_function_list: List[str], seed: Optional[int]=57):
    """
    Get a DNN for california housing. See get_dnn function for more information.

    Parameters
    ----------
    n_layers : int
        the number of layers
    n_units_list : List[int]
        the numbers of units of each layer
    activation_function_list : List[str]
        the list of activation functions of each layer
    seed : Optional[int]

    Return
    ------
    model : keras.engine.sequential.Sequential
        the dnn for california houseing, whose the structure is specified by arguments
    """
    model = get_dnn(n_layers=n_layers, n_units_list=n_units_list, activation_function_list=activation_function_list, input_dim=CALIFORNIA_INPUT_DIM, seed=seed)

    return model


def train_california_dnn(n_layers: int, n_units_list: List[int], activation_function_list: List[str], epochs: int, batch_size: int, patience: int=5,
                         seed: Optional[int]=57, eval_size: Optional[Union[float, int]]=None, test_size: Optional[Union[float, int]]=None,
                         train_size: Optional[Union[float, int]]=None, shuffle: bool=True):
    """
    Train a DNN for california housing, whose the structure is specified by arguments.

    Parameters
    ----------
    n_layers : int
        the number of layers
    n_units_list : List[int]
        the numbers of units of each layer
    activation_function_list : List[str]
        the list of activation functions of each layer
    epochs : int
    batch_size : int
    patience : int
        number of epochs with no improvement after which training will be stopped
    seed : Optional[int]
    eval_size : float or int, default None
    test_size : float or int, default None
    train_size: float or int, degault None
    random_state : int, default 57
    shuffle : bool, default True

    Return
    ------
    model : keras.engine.sequential.Sequential
        the dnn for california houseing, whose the structure is specified by arguments
    history : keras.callbacks.History
        the training history
    """
    # Load California dataset
    california_train_df, california_eval_df, california_test_df = utils.load_splitting_california_dataset_with_eval(
        eval_size=eval_size, test_size=test_size, train_size=train_size, random_state=seed, shuffle=shuffle)

    # Build the model
    model = get_california_dnn(n_layers=n_layers, n_units_list=n_units_list, activation_function_list=activation_function_list, seed=seed)

    # Compile the model with Adam optimizer and mean squared error
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
    model.compile(loss=loss, optimizer=optimizer)

    # Train model
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='min')]
    validation_data = (california_eval_df.drop([definition.CALIFORNIA_TARGET], axis=1), california_eval_df[definition.CALIFORNIA_TARGET])
    history = model.fit(x=california_train_df.drop([definition.CALIFORNIA_TARGET], axis=1), y=california_train_df[definition.CALIFORNIA_TARGET], epochs=epochs, batch_size=batch_size, callbacks=callbacks, validation_data=validation_data)
    print("Finished training.")

    evaluation_loss = model.evaluate(x=california_test_df.drop([definition.CALIFORNIA_TARGET], axis=1), y=california_test_df[definition.CALIFORNIA_TARGET])
    print(f"Evaluation mean squared error: {evaluation_loss}")

    return model, history


def train_california_dnn_with_mlflow(config_yaml_path: str):
    """
    Train a DNN for california housing, whose the structure is specified by arguments with mlflow.

    Parameters
    ----------
    config_yaml_path : str
        the config about the model structure, training, and experiment information

    Return
    ------
    model : keras.engine.sequential.Sequential
        the dnn for california houseing, whose the structure is specified by arguments
    history : keras.callbacks.History
        the training history
    """
    # Load config
    with open(config_yaml_path, "r") as yaml_f:
        config = yaml.safe_load(yaml_f)
    config_mlflow = config["mlflow"]
    config_dataset = config["dataset"]
    dnn_parameters = config["dnn"]
    dnn_train_parameters = config["dnn_train"]
    
    mlflow.set_experiment(config_mlflow["experiment_name"])
    with mlflow.start_run(run_name=config_mlflow["run_name"]) as run:
        mlflow.keras.autolog()
        mlflow.log_artifact(config_yaml_path)

        model, history = train_california_dnn(**dnn_parameters, **dnn_train_parameters, **config_dataset)

    return model, history
