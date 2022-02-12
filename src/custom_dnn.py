from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union

from tensorflow import keras

from custom_dataset import CustomDataset


class CustomDNN(metaclass=ABCMeta):
    """Abstract base class for creating DNN"""

    def __init__(self):
        pass

    @property
    @abstractmethod
    def dataset(self) -> CustomDataset:
        raise NotImplementedError()

    @property
    @abstractmethod
    def input_dimension(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def target_name(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def loss(self):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def run_all_process_with_mlflow(config_yaml_path: str):
        """
        Build, train and evaluate the DNN, whose the structure is specified by config file, with mlflow.

        Parameters
        ----------
        config_yaml_path : str
            the config about the model structure, training, and experiment information

        Return
        ------
        custom_dnn : CustomDNN
            trained CustomDNN
        """
        raise NotImplementedError()

    @property
    def has_built_model(self) -> bool:
        try:
            self.model
            return True
        except AttributeError:
            return False

    def build(self, n_layers: int, n_units_list: List[int], activation_function_list: List[str], seed: Optional[int]=57) -> None:
        """
        Build a DNN whose the number of layers is n_layers, and
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
        seed : Optional[int]

        Raise
        -----
        ValueError: if n_layers, the length of n_units_list, and the length of activation_function_list are not same
        """
        if not n_layers == len(n_units_list) == len(activation_function_list):
            raise ValueError(f"n_layers, the length of n_units_list, and the length of activation_function_list must be same, but n_layers is {n_layers}, the length of n_units_list is {n_units_list}, and the length of activation_function_list is {activation_function_list}.")

        # Fix seed if it's not None
        if seed is not None:
            self.dataset.fix_seed(seed)

        model = keras.models.Sequential()
        
        # Add an input layer
        model.add(keras.layers.Dense(n_units_list[0], input_dim=self.input_dimension, activation=activation_function_list[0]))
        n_layers -= 1

        # Add hidden layers
        for index in range(n_layers - 1):
            model.add(keras.layers.Dense(n_units_list[index+1], activation=activation_function_list[index]))

        # Add an output layer
        model.add(keras.layers.Dense(n_units_list[-1], activation=activation_function_list[-1]))

        self.model = model
        
        self.model.summary()

    def train(self, epochs: int, batch_size: int, patience: int=5, seed: Optional[int]=57,
              eval_size: Optional[Union[float, int]]=None, test_size: Optional[Union[float, int]]=None,
              train_size: Optional[Union[float, int]]=None, shuffle: bool=True):
        """
        Train and evaluate the DNN.

        Parameters
        ----------
        epochs : int
        batch_size : int
        patience : int
            number of epochs with no improvement after which training will be stopped
        seed : Optional[int]
        eval_size : float or int, default None
        test_size : float or int, default None
        train_size: float or int, degault None
        shuffle : bool, default True

        Raise
        -----
        AttributeError : if there is no model
        """
        if not self.has_built_model:
            raise AttributeError("There is no model.")

        # Load dataset
        train_dataset, eval_dataset, test_dataset = self.dataset.load_splitted_dataset_with_eval(
            eval_size=eval_size, test_size=test_size, train_size=train_size, random_state=seed, shuffle=shuffle)

        # Compile the model with Adam optimizer and mean squared error
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss=self.loss, optimizer=optimizer)

        # Train model
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='min')]
        validation_data = (eval_dataset.drop([self.target_name], axis=1), eval_dataset[self.target_name])
        self.history = self.model.fit(
            x=train_dataset.drop([self.target_name], axis=1), y=train_dataset[self.target_name],
            epochs=epochs, batch_size=batch_size, callbacks=callbacks, validation_data=validation_data
            )
        print("Finished training.")

        self.test_loss = self.model.evaluate(x=test_dataset.drop([self.target_name], axis=1), y=test_dataset[self.target_name])
        print(f"test loss: {self.test_loss}")
