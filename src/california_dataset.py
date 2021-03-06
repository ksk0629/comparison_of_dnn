import pandas as pd
import sklearn
from sklearn import datasets

from custom_dataset import CustomDataset


class CaliforniaDataset(CustomDataset):
    """Caifornia dataset class"""

    def load_dataset(self) -> pd.DataFrame:
        """Load california housing dataset using sklearn.datasets.fetch_california_housing() function.

        :return pandas.DataFrame: California dataset
        """
        california_dataset = sklearn.datasets.fetch_california_housing(as_frame=True)["frame"]

        return california_dataset

    @property
    def stratify(self) -> None:
        return None

    @property
    def target_name(self) -> None:
        return None
