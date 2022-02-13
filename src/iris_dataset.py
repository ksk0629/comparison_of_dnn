import pandas as pd
import sklearn
from sklearn import datasets

from custom_dataset import CustomDataset
from definition import IRIS_TARGET


class IrisDataset(CustomDataset):
    """Iris dataset class"""

    def load_dataset(self) -> pd.DataFrame:
        """
        Load iris dataset using sklearn.datasets.load_iris() function.

        Return
        ------
        iris_dataset : pandas.DataFrame
        """
        iris_dataset = sklearn.datasets.load_iris(as_frame=True)["frame"]

        return iris_dataset

    @property
    def stratify(self):
        return self.load_dataset()[IRIS_TARGET]

    @property
    def target_name(self):
        return IRIS_TARGET
