import pickle
from typing import Dict
from pandas import DataFrame
import pandas as pd
from tqdm import tqdm

from collaborative_filtering_ph_version.class_models.movie_features import MovieFeatures
from collaborative_filtering_ph_version.services.tmdb_api_service import TmdbApiService


def load_csv_data() -> tuple[DataFrame, DataFrame, DataFrame]:
    movie_data_df = pd.read_csv('csv_files/movie.csv', delimiter=';', index_col=0, usecols=[0, 1, 2], names=["movie_id", "tmdb_movie_id", "title"])
    task_data_df = pd.read_csv('csv_files/task.csv', delimiter=';', index_col=0, usecols=[0, 1, 2, 3], names=["index", "user_id", "movie_id", "rating"])
    train_data_df = pd.read_csv('csv_files/train.csv', delimiter=';', index_col=0, usecols=[0, 1, 2, 3], names=["index", "user_id", "movie_id", "rating"])
    return movie_data_df, task_data_df, train_data_df

def load_or_fetch_movie_features(movie_id_tmdb_ids: Dict[int, int],
                                 tmdb_api_service: TmdbApiService) -> Dict[int, MovieFeatures]:
    try:
        with open('movie_features.pkl', 'rb') as f:
            movie_features_dict: Dict[int, MovieFeatures] = pickle.load(f)
            print("Loaded movie features from 'movie_features.pkl'")
            return movie_features_dict
    except (FileNotFoundError, ModuleNotFoundError):
        movie_features_dict: Dict[int, MovieFeatures] = {}

    for movie_id, tmdb_id in tqdm(movie_id_tmdb_ids.items(), desc="Fetching movie features", unit="movie"):
        if movie_id not in movie_features_dict:
            try:
                features = tmdb_api_service.fetch_movie_details(tmdb_id)
                movie_features_dict[movie_id] = features
            except Exception as e:
                print(f"Could not fetch features of the TMDB movie {tmdb_id}: {e}")
                return {}

    with open('movie_features.pkl', 'wb') as f:
        pickle.dump(movie_features_dict, f)
        print("Movie features have been saved to 'movie_features.pkl'")

    return movie_features_dict

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