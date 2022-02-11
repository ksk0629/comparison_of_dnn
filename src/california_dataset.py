import pandas as pd
import sklearn
from sklearn import datasets

from custom_dataset import CustomDataset
from definition import CALIFORNIA_TARGET


class CaliforniaDataset(CustomDataset):
    """Caifornia dataset class"""

    def load_dataset(self) -> pd.DataFrame:
        """
        Load california housing dataset using sklearn.datasets.fetch_california_housing() function.

        Return
        ------
        california_dataset : pandas.DataFrame
        """
        california_dataset = sklearn.datasets.fetch_california_housing(as_frame=True)["frame"]

        return california_dataset

    @property
    def stratify(self):
        return CALIFORNIA_TARGET