from abc import ABCMeta, abstractmethod
import os
import random
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection
import tensorflow as tf


class CustomDataset(metaclass=ABCMeta):
    """Abstract base class for dataset"""

    def __init__(self) -> None:
        self.default_eval_size = 0.25

    @abstractmethod
    def load_dataset(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def stratify(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def target_name(self):
        raise NotImplementedError()

    def fix_seed(self, seed: int=57) -> None:
        """Fix random seed. (Ref.: https://qiita.com/bee2/items/08eab7a899c9ff56eb35)

        :param int seed: random seed, defaults to 57
        """
        os.environ["PYTHONHASHSEED"] = f"{seed}"
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

    def load_splitted_dataset(self,
                              test_size: Optional[Union[float, int]]=None,
                              train_size: Optional[Union[float, int]]=None,
                              random_state: Optional[int]=57,
                              shuffle: bool=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load datasets splitted into train and test.
        See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html to know more about parameters.

        :param Optional[Union[float, int]] test_size: the test dataset size (This will be 0.25 if test_size and train_size are None.), defaults to None
        :param Optional[Union[float, int]] train_size: the training dataset size, defaults to None
        :param Optional[int] random_state: random seed, defaults to 57
        :param bool shuffle: whether the dataset shuffles or not, defaults to True
        :return Tuple[pandas.DataFrame, pandas.DataFrame]: datasets for training and testing
        """
        dataset = self.load_dataset()
        train_dataset, test_dataset = sklearn.model_selection.train_test_split(
            dataset, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=self.stratify)

        return train_dataset, test_dataset

    def load_splitted_dataset_with_eval(self, eval_size: Optional[Union[float, int]]=None,
                                        test_size: Optional[Union[float, int]]=None,
                                        train_size: Optional[Union[float, int]]=None,
                                        random_state: Optional[int]=57,
                                        shuffle: bool=True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load datasets splitted into train, eval, and test.
        See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html to know more about parameters.

        :param Optional[Union[float, int]] eval_size: the evaluating dataset size (This will be self.default_eval_size if eval_size is None.), defaults to None
        :param Optional[Union[float, int]] test_size: the test dataset size (This will be 0.25 if test_size and train_size are None.), defaults to None
        :param Optional[Union[float, int]] train_size: the training dataset size, defaults to None
        :param Optional[int] random_state: random seed, defaults to 57
        :param bool shuffle: whther the dataset shuffles or not, defaults to True
        :return Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: datasets for training, evaluating, and testing
        """
        train_and_eval_dataset, test_dataset = self.load_splitted_dataset(test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle)

        stratify = None if self.stratify is None else train_and_eval_dataset[self.target_name]
        
        actual_eval_size = self.default_eval_size if eval_size is None else eval_size
        train_dataset, eval_dataset = sklearn.model_selection.train_test_split(
            train_and_eval_dataset, test_size=actual_eval_size, train_size=None, random_state=random_state, shuffle=shuffle, stratify=stratify)

        return train_dataset, eval_dataset, test_dataset
