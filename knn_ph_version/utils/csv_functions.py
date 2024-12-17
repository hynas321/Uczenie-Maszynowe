import csv
import pickle
from typing import Dict, Tuple, Hashable, Any, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from knn_ph_version.class_models.movie_features import MovieFeatures
from knn_ph_version.services.tmdb_api_service import TmdbApiService


def load_csv_data() -> Tuple[DataFrame, DataFrame, DataFrame]:
    movie_data_df = pd.read_csv('csv_files/movie.csv', delimiter=';', index_col=0,
                                usecols=[0, 1, 2], names=["movie_id", "tmdb_movie_id", "title"])
    task_data_df = pd.read_csv('csv_files/task.csv', delimiter=';', index_col=0,
                               usecols=[0, 1, 2, 3], names=["index", "user_id", "movie_id", "rating"])
    train_data_df = pd.read_csv('csv_files/train.csv', delimiter=';', index_col=0,
                                usecols=[0, 1, 2, 3], names=["index", "user_id", "movie_id", "rating"])

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

def save_predictions_to_csv(task_data_df: DataFrame, predictions: dict[Hashable, int | Any],
                            train_data_df: DataFrame) -> None:
    submission_df = task_data_df.copy()

    for i in submission_df.index:
        if i in predictions:
            submission_df.at[i, 'rating'] = predictions[i]
        else:
            submission_df.at[i, 'rating'] = int(round(train_data_df['rating'].mean()))

    submission_df['rating'] = submission_df['rating'].astype(int).clip(0, 5)
    submission_df.to_csv('submission.csv', sep=';', index=True, header=False)

def save_k_report(k_scores: Dict[int, List[float]], best_k_counts: Dict[int, int], report_filename: str) -> None:
    with open(report_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["k", "Count of Best k", "Average Accuracy"])

        for k in sorted(k_scores.keys()):
            accuracies = k_scores[k]
            average_accuracy = np.mean(accuracies) if accuracies else 0
            count = best_k_counts.get(k, 0)
            writer.writerow([k, count, average_accuracy])

    print(f"K value report saved to {report_filename}")