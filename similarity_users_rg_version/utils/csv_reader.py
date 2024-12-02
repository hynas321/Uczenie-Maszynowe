import pandas as pd
from typing import Tuple


def load_csv_data() -> Tuple[ pd.DataFrame, pd.DataFrame]:

    task_data = pd.read_csv(
        'csv_files/task.csv',
        delimiter=';',
        index_col=0,
        usecols=[0, 1, 2, 3],
        names=["index", "user_id", "movie_id", "rating"],
        header=None
    )

    train_data = pd.read_csv(
        'csv_files/train.csv',
        delimiter=';',
        index_col=0,
        usecols=[0, 1, 2, 3],
        names=["index", "user_id", "movie_id", "rating"],
        header=None
    )

    return train_data, task_data
