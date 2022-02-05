import pandas as pd
import sklearn
from sklearn import datasets, model_selection


def load_california_housing():
    """
    Load sklearn.dataset.california_housing dataset.

    Return
    ------
    california_df : pandas.DataFrame
        california dataset
    """
    california_dataset = sklearn.datasets.fetch_california_housing()

    california_data = california_dataset.data
    california_targets = california_dataset.target
    california_feature_names = california_dataset.feature_names

    california_data_df = pd.DataFrame({feature_name: feature for feature_name, feature in zip(california_feature_names, california_data.T)})
    california_targets_df = pd.DataFrame({"target": california_targets})
    california_df = pd.concat([california_data_df, california_targets_df], axis=1)

    return california_df


def load_splitting_california_dataset(test_size=None, train_size=None, random_state=57, shuffle=True):
    """
    Load california dataset splitted into train and test.

    Parameters (See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
    ----------
    test_size : float or int, default None
        This will be 0.25 if test_size and train_size are None.
    train_siez: float or int, degault None
    random_state : int, default 57
    shuffle : bool, default True

    Returns
    -------
    train_df : pandas.DataFrame
    test_df : pandas.DataFrame
    """
    california_df = load_california_housing()
    train_df, test_df = sklearn.model_selection.train_test_split(
        california_df, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle)

    return train_df, test_df


def load_splitting_california_dataset_with_eval(eval_size=None, test_size=None, train_size=None, random_state=57, shuffle=True):
    """
    Load california dataset splitted into train, eval, and test.

    Parameters (See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
    ----------
    eval_size : float or int, default None
        This will be 0.25 if eval_size is None.
        Note that, eval_size is size of evaluation dataset.
        The dataset is extracted from train dataset,
        which is extracted from original dataset with train_size.
    test_size : float or int, default None
        This will be 0.25 if test_size and train_size are None.
    train_siez: float or int, degault None
    random_state : int, default 57
    shuffle : bool, default True

    Returns
    -------
    train_df : pandas.DataFrame
    eval_df : pandas.DataFrame
    test_df : pandas.DataFrame
    """
    train_and_eval_df, test_df = load_splitting_california_dataset(
        test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle)
    actual_eval_size = 0.25 if eval_size is None else eval_size
    train_df, eval_df = sklearn.model_selection.train_test_split(
        train_and_eval_df, test_size=actual_eval_size, train_size=None, random_state=random_state, shuffle=shuffle)

    return train_df, eval_df, test_df
