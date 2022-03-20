import argparse

import mlflow
from tensorflow import keras
import pandas as pd
import yaml

from custom_dnn import CustomDNN
from definition import IRIS_INPUT_DIMENSION, IRIS_TARGET
from iris_dataset import IrisDataset


class IrisDNN(CustomDNN):
    """DNN for iris dataset"""
    
    @property
    def dataset(self) -> IrisDataset:
        return IrisDataset()

    @property
    def input_dimension(self) -> int:
        return IRIS_INPUT_DIMENSION

    @property
    def target_name(self) -> str:
        return IRIS_TARGET

    @property
    def loss(self) -> keras.losses.SparseCategoricalCrossentropy:
        return keras.losses.SparseCategoricalCrossentropy()

    @staticmethod
    def run_all_process_with_mlflow(config_yaml_path: str) -> object:
        """Build, train and evaluate the DNN, whose the structure is specified by config file, with mlflow.

        :param str config_yaml_path: the config about the model structure, training, and experiment information
        :return _type_: trained IrisDNN
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
        with mlflow.start_run(run_name=config_mlflow["run_name"]) as run:
            # Log automatically
            mlflow.keras.autolog()

            # Log the used config file
            mlflow.log_artifact(config_yaml_path)

            # Create and train a IrisDNN model
            iris_dnn = IrisDNN()
            iris_dnn.build(**dnn_parameters)
            iris_dnn.train(**dnn_train_parameters, **config_dataset)

            # Log the accuracy
            mlflow.log_metric("test_loss", iris_dnn.test_loss)

        return iris_dnn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate iris DNN")

    parser.add_argument("-c", "--config_yaml_path", required=False, type=str, default="./config_iris.yaml")
    args = parser.parse_args()

    IrisDNN.run_all_process_with_mlflow(args.config_yaml_path)
