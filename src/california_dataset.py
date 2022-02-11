import sklearn
from sklearn import datasets

import custom_dataset

class CliforniaDataset(custom_dataset):
    """Caifornia dataset class"""

    def load_dataset() -> pd.DataFrame:
        """
        Load california housing dataset using sklearn.datasets.fetch_california_housing() function.

        Return
        ------
        california_dataset : pandas.DataFrame
        """
        california_dataset = sklearn.datasets.fetch_california_housing(as_frame=True)

        return california_dataset
