import pandas as pd
import sklearn
from sklearn import datasets

from custom_dataset import CustomDataset

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
