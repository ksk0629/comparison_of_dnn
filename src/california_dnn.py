import yaml
import mlflow

import pandas as pd

from california_dataset import CaliforniaDataset
from custom_dnn import CustomDNN


CALIFORNIA_INPUT_DIMENSION = 8
CALIFORNIA_TARGET = "MedHouseVal"


class CaliforniaDNN(CustomDNN):
    """DNN for california housing dataset"""
    
    @property
    def dataset(self) -> CaliforniaDataset:
        return CaliforniaDataset()

    @property
    def input_dimension(self) -> int:
        return CALIFORNIA_INPUT_DIMENSION

    @property
    def target_name(self) -> str:
        return CALIFORNIA_TARGET

    @staticmethod
    def run_all_process_with_mlflow(config_yaml_path: str) -> CaliforniaDNN:
        """
        Build, train and evaluate the DNN, whose the structure is specified by config file, with mlflow.

        Parameters
        ----------
        config_yaml_path : str
            the config about the model structure, training, and experiment information

        Return
        ------
        california_dnn : CaliforniaDNN
            trained CaliforniaDNN
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

            california_dnn = CaliforniaDNN()

            california_dnn.build(**dnn_parameters)
            california_dnn.train(**dnn_train_parameters, **config_dataset)

        return california_dnn
