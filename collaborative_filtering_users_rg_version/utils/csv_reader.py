import pandas as pd
from typing import Tuple


def load_csv_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        movie_data = pd.read_csv(
            'csv_files/movie.csv',
            delimiter=';',
            index_col=0,
            usecols=[0, 1, 2],
            names=["movie_id", "tmdb_movie_id", "title"],
            header=None
        )

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

    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure the CSV files exist in the specified directory.")
        raise
    except pd.errors.ParserError as e:
        print(f"Error parsing the CSV files: {e}")
        raise

    return movie_data, task_data, train_data
