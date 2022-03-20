from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union

from tensorflow import keras

from custom_dataset import CustomDataset


class CustomDNN(metaclass=ABCMeta):
    """Abstract base class for creating DNN"""

    def __init__(self):
        self.__model = None

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
    def run_all_process_with_mlflow(config_yaml_path: str) -> object:
        """Build, train and evaluate a DNN, whose the structure is specified by config file, with mlflow.

        :param str config_yaml_path: the config about the model structure, training, and experiment information
        :return object: trained CustomDNN
        """
        raise NotImplementedError()

    @property
    def model(self) -> Optional[keras.model.Sequential]:
        """
        :return Optional[keras.model.Sequential]: model object
        """
        return self.__model

    @property
    def has_built_model(self) -> bool:
        """
        :return bool: whether a model has already built or not
        """
        if self.model is None:
            return False
        else:
            return True

    def build(self, n_layers: int, n_units_list: List[int], activation_function_list: List[str], seed: Optional[int]=57) -> None:
        """Build a DNN whose number of layers is n_layers, and each layer has some units specified by n_units_list and each
        activation function specified by activation_function_list.

        :param int n_layers: the number of layers
        :param List[int] n_units_list: the numbers of units of each layer (The length has to be same as n_layers and the length of activation_function_list.)
        :param List[str] activation_function_list: the list of activation functions of each layer (The length has to be same as n_layers and the length of n_units_list.)
        :param Optional[int] seed: random seed (Random seed will not be fixed if this is None.), defaults to 57
        :raises ValueError: if n_layers, the length of n_units_list, and the length of activation_function_list are not same
        """
        # Check arguments validation
        if not n_layers == len(n_units_list) == len(activation_function_list):
            msg_1 = f"n_layers, the length of n_units_list, and the length of activation_function_list must be same."
            msg_2 = f"n_layers is {n_layers},the length of n_units_list is {n_units_list}, and the length of activation_function_list is {activation_function_list}."
            msg = msg_1 + msg_2
            raise ValueError(msg)

        # Fix seed if it's not None
        if seed is not None:
            self.dataset.fix_seed(seed)

        # Create a model
        model = keras.models.Sequential()
        
        # Add an input layer to the model
        model.add(keras.layers.Dense(n_units_list[0], input_dim=self.input_dimension, activation=activation_function_list[0]))
        n_layers -= 1

        # Add hidden layers to the model
        for index in range(n_layers - 1):
            model.add(keras.layers.Dense(n_units_list[index+1], activation=activation_function_list[index+1]))

        # Add an output layer to the model
        model.add(keras.layers.Dense(n_units_list[-1], activation=activation_function_list[-1]))

        # Store the model
        self.__model = model
        
        self.model.summary()

    def train(self, epochs: int, batch_size: int, patience: int=5, seed: Optional[int]=57,
              eval_size: Optional[Union[float, int]]=None, test_size: Optional[Union[float, int]]=None,
              train_size: Optional[Union[float, int]]=None, shuffle: bool=True) -> None:
        """Train and evaluate the DNN.

        :param int epochs: the number of epochs
        :param int batch_size: the training batch size
        :param int patience: the number of epochs with no improvement after which training will be stopped, defaults to 5
        :param Optional[int] seed: random seed, defaults to 57
        :param Optional[Union[float, int]] eval_size: the ecaluating dataset size, defaults to None
        :param Optional[Union[float, int]] test_size: the testing dataset size, defaults to None
        :param Optional[Union[float, int]] train_size: the training dataset size, defaults to None
        :param bool shuffle: whether the dataset shuffles or not, defaults to True
        :raises AttributeError: if there is no model
        """
        # Check the exsitence of the model
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
