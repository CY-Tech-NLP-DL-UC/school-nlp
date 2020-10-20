import pandas as pd
from pathlib import Path
import time

class DataManager:
    """ ...
    """
    def __init__(self):
        self.path = Path(__file__).resolve().parent / "aclImdb"
    
    def load(self, path):
        """ TODO
        """
        pos_data_bearer = {
            'label': [
                "pos"
                for child in (path / "pos").iterdir()
            ],
            'features': [
                child.open().read()
                for child in (path / "pos").iterdir()
            ]
        }

        neg_data_bearer = {
            'label': [
                "neg"
                for child in (path / "neg").iterdir()
            ],
            'features': [
                child.open().read()
                for child in (path / "neg").iterdir()
            ]
        }

        pos_data = pd.DataFrame(pos_data_bearer)
        neg_data = pd.DataFrame(neg_data_bearer)

        return pd.concat([pos_data, neg_data]).reset_index(drop=True)

    def get(self):
        train_path = self.path / "train"
        test_path = self.path / "test"
        train_data = self.load(train_path)
        test_data = self.load(test_path)

        return (
            train_data.sample(frac=1).reset_index(drop=True),
            test_data.sample(frac=1).reset_index(drop=True)
        )

if __name__ == "__main__":
    start = time.time()
    # dm = DataManager()
    # train_data, test_data = dm.get()
    # print(train_data.head())
    # print(test_data.head())
    print("Execution time : " + str(time.time() - start))
    # train_data.to_csv("train_dataset.csv", index=False)
    # test_data.to_csv("test_dataset.csv", index=False)