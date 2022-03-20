import argparse

import mlflow
import pandas as pd
from tensorflow import keras
import yaml

from california_dataset import CaliforniaDataset
from custom_dnn import CustomDNN
from definition import CALIFORNIA_INPUT_DIMENSION, CALIFORNIA_TARGET


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

    @property
    def loss(self) -> keras.losses.MeanSquaredError:
        return keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")

    @staticmethod
    def run_all_process_with_mlflow(config_yaml_path: str) -> object:
        """Build, train and evaluate the DNN, whose the structure is specified by config file, with mlflow.

        :param str config_yaml_path: the config about the model structure, training, and experiment information
        :return obejct: trained CaliforniaDNN
        """
        # Load configs
        with open(config_yaml_path, "r") as yaml_f:
            config = yaml.safe_load(yaml_f)
        config_mlflow = config["mlflow"]
        config_dataset = config["dataset"]
        dnn_parameters = config["dnn"]
        dnn_train_parameters = config["dnn_train"]

        # Start training and evaluating with mlflow
        mlflow.set_experiment(config_mlflow["experiment_name"])
        with mlflow.start_run(run_name=config_mlflow["run_name"]):
            # Log automatically
            mlflow.keras.autolog()
            
            # Log the used config file
            mlflow.log_artifact(config_yaml_path)

            # Create and train california dnn class
            california_dnn = CaliforniaDNN()
            california_dnn.build(**dnn_parameters)
            california_dnn.train(**dnn_train_parameters, **config_dataset)

            # Log the accuracy
            mlflow.log_metric("test_loss", california_dnn.test_loss)

        return california_dnn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate california DNN")

    parser.add_argument("-c", "--config_yaml_path", required=False, type=str, default="./config_california.yaml")
    args = parser.parse_args()

    CaliforniaDNN.run_all_process_with_mlflow(args.config_yaml_path)
