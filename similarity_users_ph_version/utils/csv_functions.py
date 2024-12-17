from typing import Dict
from pandas import DataFrame
import pandas as pd

def load_csv_data() -> tuple[DataFrame, DataFrame, DataFrame]:
    movie_data_df = pd.read_csv('csv_files/movie.csv', delimiter=';', index_col=0, usecols=[0, 1, 2], names=["movie_id", "tmdb_movie_id", "title"])
    task_data_df = pd.read_csv('csv_files/task.csv', delimiter=';', index_col=0, usecols=[0, 1, 2, 3], names=["index", "user_id", "movie_id", "rating"])
    train_data_df = pd.read_csv('csv_files/train.csv', delimiter=';', index_col=0, usecols=[0, 1, 2, 3], names=["index", "user_id", "movie_id", "rating"])
    return movie_data_df, task_data_df, train_data_df

def save_accuracies_to_csv(user_accuracies: Dict[int, float], filename: str) -> None:
    with open(filename, 'w') as f:
        f.write("user_id,accuracy\n")

        for user_id, accuracy in user_accuracies.items():
            f.write(f"{user_id},{accuracy:.3f}\n")

        if user_accuracies:
            average_accuracy = sum(user_accuracies.values()) / len(user_accuracies)
            f.write(f"Average,{average_accuracy:.3f}\n")

def save_predictions_to_csv(task_data_df: DataFrame, filename: str) -> None:
    task_data_df['rating'] = task_data_df['rating'].round().astype(int)
    task_data_df.to_csv(filename, sep=';', index=True, header=False)